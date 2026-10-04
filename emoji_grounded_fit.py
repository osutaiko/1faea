"""Train reconstruction and direct emoji answering using a map and premade Q&A."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from conversation_model import ConversationReader, pad
from emoji_grounded_data import DIRECTORY as DATA
from emoji_grounded_model import GroundedEncoder, GroundedReply, MeaningLexicon, TrainingTextDecoder, discrete_vectors, ordered_content_loss
from emoji_grounded_semantics import SemanticReader
from emoji_meanings import DIRECTORY as MEANINGS
from run import ROOT


def initialize_vectors(reader, rows):
    embedding = reader.backbone.get_input_embeddings()
    values = []
    with torch.no_grad():
        for row in rows:
            core = embedding(torch.tensor(reader.tokenizer.encode(row['core_meaning'], add_special_tokens=False))).float().mean(0)
            if row['associations']:
                related = torch.stack([embedding(torch.tensor(reader.tokenizer.encode(term, add_special_tokens=False))).float().mean(0) for term in row['associations']]).mean(0)
                core = .8 * core + .2 * related
            values.append(core)
    return torch.stack(values)


@torch.no_grad()
def prepare(reader, semantic_reader, lexicon, output, slots):
    groups = {}
    rejected = {}
    for split in ('train', 'validation', 'test'):
        rows = [json.loads(line) for line in (DATA / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
        selected = []
        for row in rows:
            texts = [row['question'], row['answer']]
            formatted = [reader.tokenizer.apply_chat_template([dict(role='user', content=text)], tokenize=False, add_generation_prompt=False) for text in texts]
            if (max(len(reader.tokenizer.encode(text, add_special_tokens=False)) for text in formatted) <= 128 and
                    max(len(semantic_reader.tokenizer.encode(text)) for text in texts) <= 256):
                selected.append(row)
        rejected[split] = len(rows) - len(selected)
        groups[split] = []
        for start in range(0, len(selected), 2):
            records = selected[start:start + 2]
            path = output / f'{split}-{start // 2:04d}.pt'
            if not path.exists():
                data = dict(rows=records)
                for prefix, field in (('q', 'question'), ('a', 'answer')):
                    texts = [row[field] for row in records]
                    features, mask = semantic_reader.text_inputs(texts)
                    data[prefix + 'features'] = features.to(torch.float16)
                    data[prefix + 'mask'] = mask
                    data[prefix + 'tokens'] = pad([reader.tokenizer.encode(text, add_special_tokens=False) + [reader.tokenizer.eos_token_id] for text in texts], -100)
                    anchors = torch.full((len(records), slots), -100, dtype=torch.long)
                    for index, text in enumerate(texts):
                        values = lexicon.anchors(text, slots)
                        anchors[index, :len(values)] = torch.tensor(values, dtype=torch.long)
                    data[prefix + 'anchors'] = anchors
                torch.save(data, path)
            elif torch.load(path, weights_only=True)['rows'] != records:
                raise ValueError('Cached grounded features differ from dataset')
            groups[split].append(path)
        if not groups[split]:
            raise ValueError('No fitting rows in ' + split)
        print(json.dumps(dict(prepared=split, pairs=len(selected), token_limit_rejected=rejected[split])), flush=True)
    return groups, rejected


def grounding_loss(logits, anchors):
    if not (anchors >= 0).any():
        return logits.sum() * 0
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), anchors.reshape(-1), ignore_index=-100)


def reply_targets(ids, end):
    targets = torch.cat((ids, torch.full_like(ids[:, :1], end)), dim=1)
    prefix = torch.cat((torch.full_like(ids[:, :1], end), ids), dim=1)
    return prefix, targets


@torch.no_grad()
def evaluate(encoder, reply, paths, meanings):
    encoder.eval()
    reply.eval()
    total, count, exact, anchor_hits, anchor_count = 0., 0, 0, 0, 0
    examples = []
    code_ids = set()
    reply_symbols, states_seen, replies_seen = set(), set(), set()
    for path in paths:
        data = torch.load(path, weights_only=True)
        _, _, states = encoder(data['qfeatures'], data['qmask'])
        _, _, answer_ids = encoder(data['afeatures'], data['amask'])
        prefix, targets = reply_targets(answer_ids, reply.end)
        logits = reply(reply.vectors[states], prefix)
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction='sum').item()
        count += targets.numel()
        generated = reply.generate(states)
        code_ids.update(states.flatten().tolist())
        code_ids.update(answer_ids.flatten().tolist())
        for index, row in enumerate(data['rows']):
            output = generated[index].tolist()
            visible = output[:output.index(reply.end)] if reply.end in output else output
            expected = answer_ids[index].tolist()
            states_seen.add(tuple(states[index].tolist()))
            replies_seen.add(tuple(visible))
            reply_symbols.update(visible)
            exact += visible == expected
            anchors = {value for value in data['aanchors'][index].tolist() if value >= 0}
            anchor_hits += len(set(visible) & anchors)
            anchor_count += len(anchors)
            examples.append(dict(id=row['id'], question=row['question'], reference_answer=row['answer'],
                                 state=''.join(meanings[value]['symbol'] for value in states[index].tolist()),
                                 reply=''.join(meanings[value]['symbol'] for value in visible),
                                 reference_state=''.join(meanings[value]['symbol'] for value in expected),
                                 terminated=reply.end in output))
    return dict(reply_loss=total / count, count=len(examples), exact_reference_state=exact / len(examples),
                answer_lexicon_anchor_recall=anchor_hits / anchor_count if anchor_count else None,
                symbols_used=len(code_ids), generated_symbols_used=len(reply_symbols),
                distinct_question_states=len(states_seen), distinct_replies=len(replies_seen), examples=examples,
                metric_limit='Reference emoji states are learned latent codes, not verified emoji answers. Anchor recall checks words, not factual correctness.')


@torch.no_grad()
def reconstruction_audit(encoder, text_decoder, english_vectors, paths):
    losses = dict(correct=0., reversed=0., empty=0., masked_correct=0., masked_reversed=0., masked_empty=0.)
    count = 0
    for path in paths[:2]:
        data = torch.load(path, weights_only=True)
        ids = encoder(data['afeatures'], data['amask'])[2]
        selected = english_vectors[ids]
        losses['correct'] += text_decoder(selected, data['atokens']).item()
        losses['reversed'] += text_decoder(selected.flip(1), data['atokens']).item()
        losses['empty'] += text_decoder(torch.zeros_like(selected), data['atokens']).item()
        prefix_mask = torch.ones_like(data['atokens'], dtype=torch.bool)
        for key, states in (('correct', selected), ('reversed', selected.flip(1)), ('empty', torch.zeros_like(selected))):
            losses['masked_' + key] += text_decoder(states, data['atokens'], prefix_mask).item()
        count += 1
    return dict(losses={key: value / count for key, value in losses.items()},
                note='Teacher-forced English reconstruction only; compare zero and reversed states to detect bottleneck bypass.')


def train(args):
    torch.set_num_threads(2)
    torch.manual_seed(719)
    rng = random.Random(719)
    map_path = MEANINGS / 'map.json'
    meanings = json.loads(map_path.read_text(encoding='utf-8'))
    output = ROOT / 'runs' / 'emoji-grounded' / args.name
    output.mkdir(parents=True, exist_ok=True)
    semantic_source = json.loads((DATA / 'semantic-reader.json').read_text(encoding='utf-8'))
    manifest = dict(map_sha256=hashlib.sha256(map_path.read_bytes()).hexdigest(),
                    dataset_sha256={split: hashlib.sha256((DATA / f'{split}.jsonl').read_bytes()).hexdigest() for split in ('train', 'validation', 'test')},
                    slots=args.slots, seed=719, grounding_version=6, semantic_source=semantic_source,
                    base=json.loads((ROOT / 'data/conversation/sources.json').read_text(encoding='utf-8'))['model_revision'])
    manifest_path = output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Experiment inputs changed; use a new experiment name')
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    reader = ConversationReader()
    semantic_reader = SemanticReader(semantic_source['model'], semantic_source['revision'])
    english_vectors = initialize_vectors(reader, meanings)
    vectors = semantic_reader.meaning_vectors(meanings)
    lexicon = MeaningLexicon(meanings)
    groups, rejected = prepare(reader, semantic_reader, lexicon, output, args.slots)
    encoder = GroundedEncoder(vectors, slots=args.slots)
    reply = GroundedReply(vectors, slots=args.slots)
    text_decoder = TrainingTextDecoder(reader.backbone, reader.tokenizer.eos_token_id)
    cached = [torch.load(path, weights_only=True) for path in groups['train']]
    content_projection = torch.nn.Linear(vectors.shape[1], vectors.shape[1])
    parameters = list(encoder.parameters()) + list(text_decoder.adapter.parameters()) + list(content_projection.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=.0003)
    started = time.monotonic()
    with (output / 'training.jsonl').open('w', encoding='utf-8') as journal:
        for step in range(1, args.grounding_steps + args.reconstruction_steps + 1):
            encoder.train()
            data = rng.choice(cached)
            prefix = 'q' if step % 2 else 'a'
            logits, selected, _ = encoder(data[prefix + 'features'], data[prefix + 'mask'])
            anchors = grounding_loss(logits, data[prefix + 'anchors'])
            english_selected = discrete_vectors(logits, english_vectors)[0]
            content = ordered_content_loss(selected, data[prefix + 'features'], data[prefix + 'mask'], content_projection)
            reconstruction = selected.sum() * 0
            contrast = selected.sum() * 0
            if step > args.grounding_steps:
                targets = data[prefix + 'tokens']
                prefix_mask = torch.rand(targets.shape) < .5
                reconstruction = text_decoder(english_selected, targets, prefix_mask)
                with torch.no_grad():
                    reversed_loss = text_decoder(english_selected.flip(1), targets, prefix_mask)
                    empty_loss = text_decoder(torch.zeros_like(english_selected), targets, prefix_mask)
                contrast = F.relu(.1 + reconstruction - reversed_loss) + F.relu(.1 + reconstruction - empty_loss)
            distribution = logits.softmax(-1).mean((0, 1))
            coverage = (distribution * distribution.clamp_min(1e-9).log()).sum()
            loss = reconstruction + contrast + content + 3 * anchors + .1 * coverage
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            if step % 20 == 0 or step > args.grounding_steps:
                entry = dict(phase='reconstruction' if step > args.grounding_steps else 'grounding', step=step, loss=loss.item(), reconstruction=reconstruction.item(), grounding=anchors.item(), ordered_content=content.item(), state_contrast=contrast.item(), elapsed=time.monotonic() - started)
                journal.write(json.dumps(entry) + '\n')
                journal.flush()
                print(json.dumps(entry), flush=True)
        # Freeze the learned language before fitting replies, so target codes cannot drift.
        encoder.eval().requires_grad_(False)
        audit = reconstruction_audit(encoder, text_decoder, english_vectors, groups['validation'])
        (output / 'reconstruction-audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        for data in cached:
            with torch.no_grad():
                data['states'] = encoder(data['qfeatures'], data['qmask'])[2]
                data['answers'] = encoder(data['afeatures'], data['amask'])[2]
        baseline = evaluate(encoder, reply, groups['validation'], meanings)
        (output / 'baseline-validation.json').write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding='utf-8')
        if baseline['symbols_used'] < 4:
            raise ValueError('Emoji bottleneck collapsed; reply training is blocked')
        optimizer = torch.optim.AdamW(reply.parameters(), lr=.0003)
        best = float('inf')
        for step in range(1, args.reply_steps + 1):
            reply.train()
            data = rng.choice(cached)
            prefix, targets = reply_targets(data['answers'], reply.end)
            logits = reply(vectors[data['states']], prefix)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reply.parameters(), 1.)
            optimizer.step()
            if step % 100 == 0 or step == args.reply_steps:
                report = evaluate(encoder, reply, groups['validation'], meanings)
                entry = dict(phase='reply', step=step, loss=loss.item(), validation_loss=report['reply_loss'], elapsed=time.monotonic() - started)
                journal.write(json.dumps(entry) + '\n')
                journal.flush()
                print(json.dumps(entry), flush=True)
                if report['reply_loss'] < best:
                    best = report['reply_loss']
                    # English reconstruction decoder is deliberately absent from this export.
                    torch.save(dict(encoder=encoder.state_dict(), encoder_config=encoder.config,
                                    reply=reply.state_dict(), reply_config=reply.config, meanings=meanings,
                                    semantic_model=semantic_source, training_text_model=reader.model_name,
                                    manifest=manifest, selected_reply_step=step), output / 'runtime.pt')
                    (output / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    checkpoint = torch.load(output / 'runtime.pt', weights_only=True)
    reply.load_state_dict(checkpoint['reply'])
    test = evaluate(encoder, reply, groups['test'], meanings)
    test['test_used_for_selection'] = False
    (output / 'test.json').write_text(json.dumps(test, ensure_ascii=False, indent=2), encoding='utf-8')
    result = dict(grounding_steps=args.grounding_steps, reconstruction_steps=args.reconstruction_steps, reply_steps=args.reply_steps,
                  selected_reply_step=checkpoint['selected_reply_step'], elapsed_seconds=time.monotonic() - started,
                  token_limit_rejected=rejected, reconstruction_audit=audit,
                  test={key: value for key, value in test.items() if key != 'examples'},
                  generated_qa_labels=0, map_embeddings_frozen=True, pretrained_text_backbone_frozen=True,
                  runtime_english_decoder=False, promoted=False, general_use_ready=False)
    (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='objective-v3')
    parser.add_argument('--slots', type=int, default=8)
    parser.add_argument('--reconstruction-steps', type=int, default=120)
    parser.add_argument('--grounding-steps', type=int, default=200)
    parser.add_argument('--reply-steps', type=int, default=800)
    train(parser.parse_args())
