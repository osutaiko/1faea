"""Bounded unattended training using immutable meanings and premade English Q&A."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import time

import torch
from torch.nn import functional as F

from conversation_model import ConversationReader, pad
from emoji_general_model import GeneralEncoder, compact
from emoji_grounded_fit import grounding_loss, initialize_vectors
from emoji_grounded_model import GroundedReply, MeaningLexicon, TrainingTextDecoder, discrete_vectors, ordered_content_loss
from emoji_grounded_semantics import SemanticReader
from run import ROOT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def full_meanings():
    rows = json.loads((ROOT / 'data/emoji-meanings/map.json').read_text(encoding='utf-8'))
    known = {row['symbol'] for row in rows}
    official = []
    original_count = len(rows)
    source = ROOT / 'data/unicode/emoji-test-18.txt'
    for line in source.read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#'):
            continue
        codes, rest = line.split(';', 1)
        status, description = rest.split('#', 1)
        if status.strip() not in ('fully-qualified', 'component'):
            continue
        symbol = ''.join(chr(int(code, 16)) for code in codes.split())
        name = re.split(r' E\d+\.\d+ ', description.strip(), maxsplit=1)[1]
        official.append(symbol)
        if symbol not in known:
            rows.append(dict(id=len(rows), symbol=symbol, core_meaning=name, associations=[]))
            known.add(symbol)
    if set(official) != known:
        raise ValueError('Meaning vocabulary differs from the complete official emoji set')
    for index, row in enumerate(rows):
        row['id'] = index
    return rows, dict(version='18.0', official_symbols=len(official), missing=[],
                     source='https://www.unicode.org/Public/emoji/latest/emoji-test.txt',
                     source_sha256=digest(source), new_core_only_symbols=len(rows) - original_count,
                     note='Fully qualified sequences plus standalone components; spelling variants are not separate meanings. Literal definitions do not prove learned competence.')


@torch.no_grad()
def prepare(reader, semantic, meanings, output):
    lexicon = MeaningLexicon(meanings)
    rows = []
    source = ROOT / 'data/emoji-grounded/dolly-15k.jsonl'
    for index, line in enumerate(source.read_text(encoding='utf-8').splitlines()):
        item = json.loads(line)
        question = item['instruction'].strip()
        if item['context'].strip():
            question = item['context'].strip() + '\n\n' + question
        rows.append(dict(id='dolly-' + str(index), question=question, answer=item['response'].strip(), source='Dolly 15k', source_index=index))
    for split in ('train', 'validation', 'test'):
        for line in (ROOT / 'data/emoji-grounded' / (split + '.jsonl')).read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            rows.append(dict(**item, source='UltraChat 200k', original_split=split))
    groups = {key: [] for key in ('train', 'validation', 'test')}
    seen = set()
    from emoji_grounded_eval import CASES
    reserved = {' '.join(case['question'].casefold().split()) for case in CASES}
    for row in rows:
        key = ' '.join(row['question'].casefold().split())
        if key in seen or key in reserved or not row['answer'] or len(row['question']) > 650 or len(row['answer']) > 650:
            continue
        texts = [row['question'], row['answer']]
        if max(len(reader.tokenizer.encode(text)) for text in texts) > 160 or max(len(semantic.tokenizer.encode(text)) for text in texts) > 256:
            continue
        seen.add(key)
        value = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 20
        split = row.get('original_split') or ('test' if value == 0 else 'validation' if value == 1 else 'train')
        groups[split].append(row)
    # Bounded validation/test subsets chosen by hash, not by observed performance.
    for split in ('validation', 'test'):
        groups[split] = sorted(groups[split], key=lambda row: hashlib.sha256(row['question'].encode()).hexdigest())[:100]
    data = {}
    for split, records in groups.items():
        path = output / (split + '-features.pt')
        (output / (split + '.jsonl')).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records), encoding='utf-8')
        if path.exists():
            cached = torch.load(path, weights_only=True)
            if cached['rows'] != records:
                raise ValueError('Training feature cache changed')
            data[split] = cached
            continue
        cached = dict(rows=records)
        for prefix, field in (('q', 'question'), ('a', 'answer')):
            features = []
            anchors = []
            tokens = []
            for start in range(0, len(records), 8):
                texts = [row[field] for row in records[start:start + 8]]
                values, mask = semantic.text_inputs(texts)
                features.append(compact(values, mask))
                for text in texts:
                    ids = lexicon.anchors(text, 8)
                    anchors.append(ids + [-100] * (8 - len(ids)))
                    tokens.append(reader.tokenizer.encode(text, add_special_tokens=False) + [reader.tokenizer.eos_token_id])
                if start % 400 == 0:
                    print(json.dumps(dict(preparing=split, field=field, count=start, total=len(records))), flush=True)
            cached[prefix + 'features'] = torch.cat(features)
            cached[prefix + 'anchors'] = torch.tensor(anchors, dtype=torch.long)
            cached[prefix + 'tokens'] = pad(tokens, -100)
        torch.save(cached, path)
        data[split] = cached
    write(output / 'dataset-report.json', dict(counts={key: len(value['rows']) for key, value in data.items()},
          sources=[dict(name='Dolly 15k', license='CC-BY-SA-3.0', sha256=digest(source), url='https://huggingface.co/datasets/databricks/databricks-dolly-15k'),
                   dict(name='UltraChat 200k', license='MIT', url='https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k')],
          generated_qa_labels=0, standalone=True, deduplication='Normalized exact question only; paraphrase overlap not exhaustively audited',
          files_sha256={split: digest(output / (split + '.jsonl')) for split in groups}))
    return data


def targets(ids, anchors, end):
    sequences = []
    for values, literal in zip(ids.tolist(), anchors.tolist()):
        codes = [code for code in literal if code >= 0]
        codes = list(dict.fromkeys(codes + values))[:6]
        sequences.append(codes + [end])
    labels = pad(sequences, -100)
    prefix = torch.cat((torch.full_like(labels[:, :1], end), labels[:, :-1].masked_fill(labels[:, :-1] < 0, end)), dim=1)
    return prefix, labels


@torch.no_grad()
def reply_evaluate(reply, states, labels, prefix, vectors):
    reply.eval()
    loss = 0.
    count = 0
    for start in range(0, len(states), 8):
        logits = reply(vectors[states[start:start + 8]], prefix[start:start + 8])
        batch = labels[start:start + 8]
        loss += F.cross_entropy(logits.flatten(0, 1), batch.flatten(), ignore_index=-100, reduction='sum').item()
        generated = reply.generate(states[start:start + 8])
        rollout = torch.full_like(prefix[start:start + 8], reply.end)
        length = min(generated.shape[1], rollout.shape[1] - 1)
        rollout[:, 1:length + 1] = generated[:, :length]
        free_logits = reply(vectors[states[start:start + 8]], rollout)
        loss += F.cross_entropy(free_logits.flatten(0, 1), batch.flatten(), ignore_index=-100, reduction='sum').item()
        count += (batch >= 0).sum().item()
    return loss / (2 * count)


@torch.no_grad()
def coverage(encoder, reply, features, meanings, output):
    encoder.eval()
    reply.eval()
    hits = 0
    output_hits = 0
    misses = []
    for start in range(0, len(meanings), 16):
        values = features[start:start + 16]
        mask = torch.ones(values.shape[:2], dtype=torch.bool)
        ids = encoder(values, mask)[2]
        expected = torch.arange(start, start + len(values))
        found = (ids == expected[:, None]).any(1)
        # Read a single exact emoji state: distinct from understanding an English description.
        states = expected[:, None].expand(-1, 8)
        prefix = torch.full((len(values), 1), reply.end, dtype=torch.long)
        generated = reply(reply.vectors[states], prefix)[:, 0].argmax(-1)
        hits += found.sum().item()
        output_hits += (generated == expected).sum().item()
        misses.extend(meanings[start + index]['symbol'] for index in range(len(values)) if not found[index])
    report = dict(vocabulary=len(meanings), literal_description_state_hits=hits, exact_state_first_output_hits=output_hits,
                  literal_description_recall=hits / len(meanings), exact_state_first_output_recall=output_hits / len(meanings),
                  missed_description_symbols=misses, note='Training-map recall, not an independent generalization or chat test.')
    write(output / 'coverage.json', report)
    return report


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(719)
    rng = random.Random(719)
    output = ROOT / 'runs/emoji-general' / args.name
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'status.json'
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding='utf-8'))
        started = status['started_unix']
        deadline = status['deadline_unix']
    else:
        started = time.time()
        deadline = started + args.hours * 3600
        status = dict(started_unix=started, deadline_unix=deadline, requested_hours=args.hours)
    status.update(state='running', phase='preparation', updated_unix=time.time())
    write(status_path, status)
    meanings, unicode_report = full_meanings()
    write(output / 'meanings.json', meanings)
    write(output / 'unicode-coverage.json', unicode_report)
    source = json.loads((ROOT / 'data/emoji-grounded/semantic-reader.json').read_text())
    reader = ConversationReader()
    semantic = SemanticReader(source['model'], source['revision'])
    vector_path = output / 'initialization.pt'
    if vector_path.exists():
        initialized = torch.load(vector_path, weights_only=True)
    else:
        values = []
        for start in range(0, len(meanings), 16):
            texts = [row['core_meaning'] for row in meanings[start:start + 16]]
            features, mask = semantic.text_inputs(texts)
            values.append(compact(features, mask))
        initialized = dict(vectors=semantic.meaning_vectors(meanings), english=initialize_vectors(reader, meanings), descriptions=torch.cat(values))
        torch.save(initialized, vector_path)
    vectors = initialized['vectors']
    data = prepare(reader, semantic, meanings, output)
    encoder = GeneralEncoder(vectors)
    reply = GroundedReply(vectors)
    text = TrainingTextDecoder(reader.backbone, reader.tokenizer.eos_token_id)
    projection = torch.nn.Linear(vectors.shape[1], vectors.shape[1])
    encoder_parameters = list(encoder.parameters()) + list(text.adapter.parameters()) + list(projection.parameters())
    encoder_optimizer = torch.optim.AdamW(encoder_parameters, lr=.0003)
    reply_optimizer = torch.optim.AdamW(reply.parameters(), lr=.0003)
    snapshot = output / 'training.pt'
    step = 0
    best = float('inf')
    if snapshot.exists():
        saved = torch.load(snapshot, weights_only=True)
        encoder.load_state_dict(saved['encoder'])
        reply.load_state_dict(saved['reply'])
        text.adapter.load_state_dict(saved['adapter'])
        projection.load_state_dict(saved['projection'])
        encoder_optimizer.load_state_dict(saved['encoder_optimizer'])
        reply_optimizer.load_state_dict(saved['reply_optimizer'])
        step, best = saved['step'], saved['best']
    encoder_end = started + args.hours * 3600 * .45
    cached_states = None
    literal_seen = set()
    def save_training():
        temporary = snapshot.with_suffix('.tmp')
        torch.save(dict(encoder=encoder.state_dict(), reply=reply.state_dict(), adapter=text.adapter.state_dict(),
                        projection=projection.state_dict(), encoder_optimizer=encoder_optimizer.state_dict(),
                        reply_optimizer=reply_optimizer.state_dict(), step=step, best=best), temporary)
        temporary.replace(snapshot)
    def export():
        path = output / 'runtime.pt'
        temporary = path.with_suffix('.tmp')
        torch.save(dict(encoder=encoder.state_dict(), encoder_config=encoder.config, reply=reply.state_dict(), reply_config=reply.config,
                        meanings=meanings, semantic_model=source, selected_step=step, runtime_english_decoder=False), temporary)
        temporary.replace(path)
    journal = (output / 'training.jsonl').open('a', encoding='utf-8')
    while time.time() < deadline:
        step += 1
        phase = 'encoder' if time.time() < encoder_end else 'reply'
        if phase == 'encoder':
            encoder.train()
            # Every other batch grounds literal definitions across the entire vocabulary.
            if step % 2:
                indices = torch.tensor([(step // 2 * 16 + index) % len(meanings) for index in range(16)])
                literal_seen.update(indices.tolist())
                features = initialized['descriptions'][indices]
                mask = torch.ones(features.shape[:2], dtype=torch.bool)
                logits, selected, ids = encoder(features, mask)
                loss = F.cross_entropy(logits.flatten(0, 1), indices[:, None].expand(-1, 8).flatten())
            else:
                indices = torch.tensor(rng.sample(range(len(data['train']['rows'])), 2))
                prefix = 'q' if step % 4 else 'a'
                features = data['train'][prefix + 'features'][indices]
                mask = torch.ones(features.shape[:2], dtype=torch.bool)
                logits, selected, ids = encoder(features, mask)
                loss = 3 * grounding_loss(logits, data['train'][prefix + 'anchors'][indices])
                loss = loss + ordered_content_loss(selected, features, mask, projection)
                if step > 500:
                    labels = data['train'][prefix + 'tokens'][indices]
                    labels = labels[:, :(labels >= 0).sum(1).max()]
                    english = discrete_vectors(logits, initialized['english'])[0]
                    prefix_mask = torch.rand(labels.shape) < .5
                    reconstruction = text(english, labels, prefix_mask)
                    with torch.no_grad():
                        empty = text(torch.zeros_like(english), labels, prefix_mask)
                        reversed_loss = text(english.flip(1), labels, prefix_mask)
                    loss = loss + reconstruction + F.relu(.1 + reconstruction - empty) + F.relu(.1 + reconstruction - reversed_loss)
            encoder_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder_parameters, 1., error_if_nonfinite=True)
            encoder_optimizer.step()
        else:
            if cached_states is None:
                encoder.eval().requires_grad_(False)
                cached_states = {}
                with torch.no_grad():
                    for split, group in data.items():
                        qids, aids = [], []
                        for start in range(0, len(group['rows']), 16):
                            for prefix, storage in (('q', qids), ('a', aids)):
                                features = group[prefix + 'features'][start:start + 16]
                                storage.append(encoder(features, torch.ones(features.shape[:2], dtype=torch.bool))[2])
                        prefix, labels = targets(torch.cat(aids), group['aanchors'], reply.end)
                        cached_states[split] = dict(states=torch.cat(qids), prefix=prefix, labels=labels)
                export()
            reply.train()
            if step % 20 == 0:
                # Definition replay prevents everyday data from erasing rare Unicode symbols.
                indices = torch.tensor([(step // 20 * 16 + index) % len(meanings) for index in range(16)])
                states = indices[:, None].expand(-1, 8)
                prefix = torch.stack((torch.full_like(indices, reply.end), indices), dim=1)
                labels = torch.stack((indices, torch.full_like(indices, reply.end)), dim=1)
            else:
                group = cached_states['train']
                indices = torch.tensor(rng.sample(range(len(group['states'])), 8))
                states, prefix, labels = (group[key][indices] for key in ('states', 'prefix', 'labels'))
            with torch.no_grad():
                predicted = reply(vectors[states], prefix).argmax(-1)
                sampled_prefix = prefix.clone()
                replace = torch.rand(prefix[:, 1:].shape) < .5
                sampled_prefix[:, 1:] = torch.where(replace, predicted[:, :-1], prefix[:, 1:])
            logits = reply(vectors[states], sampled_prefix)
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
            with torch.no_grad():
                wrong = reply(vectors[states.roll(1, 0)], sampled_prefix)
                wrong_loss = F.cross_entropy(wrong.flatten(0, 1), labels.flatten(), ignore_index=-100)
            loss = loss + .2 * F.relu(.1 + loss - wrong_loss)
            reply_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reply.parameters(), 1., error_if_nonfinite=True)
            reply_optimizer.step()
            if step % 500 == 0:
                group = cached_states['validation']
                validation_loss = reply_evaluate(reply, vectors=vectors, **group)
                if validation_loss < best:
                    best = validation_loss
                    export()
                write(output / 'validation.json', dict(step=step, loss=validation_loss, best_loss=best, test_used_for_selection=False))
        if step % 20 == 0:
            entry = dict(step=step, phase=phase, loss=loss.item(), elapsed_seconds=time.time() - started,
                         remaining_seconds=max(0, deadline - time.time()), literal_symbols_seen=len(literal_seen))
            journal.write(json.dumps(entry, allow_nan=False) + '\n')
            journal.flush()
            status.update(entry, updated_unix=time.time())
            write(status_path, status)
            print(json.dumps(entry), flush=True)
        if step % 100 == 0:
            save_training()
    save_training()
    if not (output / 'runtime.pt').exists():
        export()
    selected = torch.load(output / 'runtime.pt', weights_only=True)
    encoder.load_state_dict(selected['encoder'])
    reply.load_state_dict(selected['reply'])
    audit = coverage(encoder, reply, initialized['descriptions'], meanings, output)
    from emoji_grounded_eval import CASES, score
    from emoji_general_chat import answer
    checks = []
    for case in CASES:
        result = answer(encoder.eval(), reply.eval(), semantic, meanings, case['question'])
        # Preserve complete catalog symbols and output order via actual generated IDs.
        features, mask = semantic.text_inputs([case['question']])
        features = compact(features, mask)
        mask = torch.ones(features.shape[:2], dtype=torch.bool)
        with torch.no_grad():
            states = encoder(features, mask)[2]
            ids = reply.generate(states)[0].tolist()
        ids = ids[:ids.index(reply.end)] if reply.end in ids else ids
        checks.append(dict(**case, **result, passed=score([meanings[index]['symbol'] for index in ids], case)))
    examples = []
    for row in data['test']['rows'][:30]:
        examples.append(dict(question=row['question'], reference_answer=row['answer'], **answer(encoder.eval(), reply.eval(), semantic, meanings, row['question'])))
    result = dict(elapsed_seconds=time.time() - started, selected_step=selected['selected_step'], coverage=audit,
                  fixed_checks_passed=sum(row['passed'] for row in checks), fixed_checks_total=len(checks), fixed_checks=checks,
                  heldout_examples=examples, runtime_english_decoder=False, generated_qa_labels=0,
                  general_use_ready=False, promoted=False, note='Check semantic quality manually; this run does not automatically certify general-chat competence.')
    write(output / 'result.json', result)
    status.update(state='complete', phase='evaluation', elapsed_seconds=time.time() - started, updated_unix=time.time())
    write(status_path, status)
    journal.close()
    print(json.dumps({key: value for key, value in result.items() if key not in ('coverage', 'fixed_checks', 'heldout_examples')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours', type=float, default=5)
    parser.add_argument('--name', default='five-hour')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        output = ROOT / 'runs/emoji-general' / args.name
        if output.exists():
            write(output / 'error.json', dict(error=type(error).__name__, message=str(error), time_unix=time.time()))
        raise
