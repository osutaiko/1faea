"""Answer-side information test; English reconstruction exists only in training."""

import argparse
import hashlib
import json
import random
import time

import torch
from torch.nn import functional as F

from conversation_model import ConversationReader
from emoji_general_model import GeneralEncoder
from emoji_grounded_model import TrainingTextDecoder, discrete_vectors
from run import ROOT


class BlindReconstruction(TrainingTextDecoder):
    def __init__(self, backbone, end_token, teacher_prefix=False):
        super().__init__(backbone, end_token)
        self.teacher_prefix = teacher_prefix

    def forward(self, states, labels):
        embedding = self.backbone.get_input_embeddings()
        # Default: only length/padding is visible. Prefix control shifts labels.
        queries = torch.full_like(labels, self.end_token)
        if self.teacher_prefix:
            queries[:, 1:] = labels[:, :-1].clamp_min(0)
            queries = queries.masked_fill(labels < 0, self.end_token)
        values = torch.cat((self.adapter(states).to(embedding.weight.dtype), embedding(queries)), 1)
        mask = torch.cat((torch.ones(states.shape[:2], dtype=torch.bool), labels >= 0), 1)
        hidden = self.backbone(inputs_embeds=values, attention_mask=mask, use_cache=False).last_hidden_state[:, states.shape[1]:]
        return F.linear(hidden, embedding.weight).float()


def loss(text, states, labels):
    return F.cross_entropy(text(states, labels).flatten(0, 1), labels.flatten(), ignore_index=-100)


def state_vectors(encoder, features, english, continuous=False):
    logits, _, ids = encoder(features, torch.ones(features.shape[:2], dtype=torch.bool))
    values = logits.softmax(-1) @ english if continuous else discrete_vectors(logits, english)[0]
    return values, ids


@torch.no_grad()
def evaluate(encoder, text, data, english, count, continuous=False):
    encoder.eval()
    totals = dict(real=0., reversed=0., mismatched=0., empty=0.)
    first = dict(real=0., reversed=0., mismatched=0., empty=0.)
    tokens = 0
    for start in range(0, count, 2):
        indices = torch.arange(start, min(start + 2, count))
        labels = data['atokens'][indices]
        labels = labels[:, :(labels >= 0).sum(1).max()]
        values, _ = state_vectors(encoder, data['afeatures'][indices], english, continuous)
        wrong, _ = state_vectors(encoder, data['afeatures'][(indices + count // 2) % count], english, continuous)
        for key, states in dict(real=values, reversed=values.flip(1), mismatched=wrong, empty=torch.zeros_like(values)).items():
            logits = text(states, labels)
            totals[key] += F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction='sum').item()
            first[key] += F.cross_entropy(logits[:, 0], labels[:, 0], reduction='sum').item()
        tokens += (labels >= 0).sum().item()
    return dict(**{key: value / tokens for key, value in totals.items()}, first_token={key: value / count for key, value in first.items()})


def train(args):
    torch.set_num_threads(2)
    torch.manual_seed(719)
    rng = random.Random(719)
    base = ROOT / 'runs/emoji-general/five-hour'
    output = ROOT / 'runs/emoji-reconstruction' / args.name
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(base / 'runtime.pt', weights_only=True)
    initialized = torch.load(base / 'initialization.pt', weights_only=True)
    data = {split: torch.load(base / (split + '-features.pt'), weights_only=True) for split in ('train', 'validation', 'test')}
    encoder = GeneralEncoder(checkpoint['encoder']['vectors'], **checkpoint['encoder_config'])
    encoder.load_state_dict(checkpoint['encoder'])
    reader = ConversationReader()
    text = BlindReconstruction(reader.backbone, reader.tokenizer.eos_token_id, args.teacher_prefix)
    parameters = list(encoder.parameters()) + list(text.adapter.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=.0003)
    english = initialized['english']
    snapshot = output / 'training.pt'
    manifest = dict(steps=args.steps, continuous=args.continuous, state_slots=encoder.config['slots'], preserve_repetition=True,
                    answer_prefix_visible=args.teacher_prefix, answer_length_visible=True, diagnostic_only=True,
                    generated_qa_labels=0, runtime_english_answer_generation=False,
                    base_checkpoint_sha256=hashlib.sha256((base / 'runtime.pt').read_bytes()).hexdigest(),
                    dataset=json.loads((base / 'dataset-report.json').read_text()),
                    validation_examples=16, test_examples=32, seed=719)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    step, best, elapsed, seen = 0, float('inf'), 0., set()
    if snapshot.exists():
        saved = torch.load(snapshot, weights_only=True)
        encoder.load_state_dict(saved['encoder'])
        text.adapter.load_state_dict(saved['adapter'])
        optimizer.load_state_dict(saved['optimizer'])
        step, best, elapsed = saved['step'], saved['best'], saved['elapsed_seconds']
        seen = set(saved['definition_ids'])
        rng.setstate(saved['rng'])
        torch.set_rng_state(saved['torch_rng'])
    else:
        baseline = evaluate(encoder, text, data['validation'], english, 16, args.continuous)
        (output / 'baseline.json').write_text(json.dumps(baseline, indent=2))
        baseline_test = evaluate(encoder, text, data['test'], english, 32, args.continuous)
        (output / 'baseline-test.json').write_text(json.dumps(baseline_test, indent=2))
    started = time.monotonic()
    with (output / 'training.jsonl').open('a') as journal:
        while step < args.steps:
            step += 1
            encoder.train()
            if step % 4 == 0:
                indices = torch.tensor([((step // 4 - 1) * 32 + i) % len(english) for i in range(32)])
                seen.update(indices.tolist())
                features = initialized['descriptions'][indices]
                logits = encoder(features, torch.ones(features.shape[:2], dtype=torch.bool))[0]
                objective = F.cross_entropy(logits.flatten(0, 1), indices[:, None].expand(-1, encoder.config['slots']).flatten())
            else:
                indices = torch.tensor(rng.sample(range(len(data['train']['rows'])), 2))
                labels = data['train']['atokens'][indices]
                labels = labels[:, :(labels >= 0).sum(1).max()]
                values, _ = state_vectors(encoder, data['train']['afeatures'][indices], english, args.continuous)
                objective = loss(text, values, labels)
                with torch.no_grad():
                    empty = loss(text, torch.zeros_like(values), labels)
                    reversed_loss = loss(text, values.flip(1), labels)
                objective = objective + F.relu(.1 + objective - empty) + F.relu(.1 + objective - reversed_loss)
            optimizer.zero_grad()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            if step % 50 == 0 or step == args.steps:
                validation = evaluate(encoder, text, data['validation'], english, 16, args.continuous)
                entry = dict(step=step, loss=objective.item(), validation=validation, elapsed_seconds=elapsed + time.monotonic() - started)
                print(json.dumps(entry), flush=True)
                journal.write(json.dumps(entry) + '\n'); journal.flush()
                if validation['real'] < best:
                    best = validation['real']
                    torch.save(dict(encoder=encoder.state_dict(), adapter=text.adapter.state_dict(), step=step), output / 'selected.pt')
                temporary = snapshot.with_suffix('.tmp')
                torch.save(dict(encoder=encoder.state_dict(), adapter=text.adapter.state_dict(), optimizer=optimizer.state_dict(),
                                step=step, best=best, elapsed_seconds=entry['elapsed_seconds'], definition_ids=sorted(seen),
                                rng=rng.getstate(), torch_rng=torch.get_rng_state()), temporary)
                temporary.replace(snapshot)
    selected = torch.load(output / 'selected.pt', weights_only=True)
    encoder.load_state_dict(selected['encoder']); text.adapter.load_state_dict(selected['adapter'])
    test = evaluate(encoder, text, data['test'], english, 32, args.continuous)
    examples = []
    with torch.no_grad():
        for index in range(10):
            labels = data['test']['atokens'][index:index + 1]
            labels = labels[:, :(labels >= 0).sum(1).max()]
            values, ids = state_vectors(encoder, data['test']['afeatures'][index:index + 1], english, args.continuous)
            examples.append(dict(id=data['test']['rows'][index]['id'], english_answer=data['test']['rows'][index]['answer'],
                                 state=''.join(checkpoint['meanings'][i]['symbol'] for i in ids[0].tolist()),
                                 training_only_reconstruction=reader.tokenizer.decode(text(values, labels)[0].argmax(-1))))
    result = dict(continuous=args.continuous, answer_prefix_visible=args.teacher_prefix,
                  displayed_symbols_are_argmax_summaries=args.continuous,
                  selected_step=selected['step'], test=test, examples=examples, definition_ids_seen=len(seen),
                  elapsed_seconds=elapsed + time.monotonic() - started, runtime_exported=False, promoted=False,
                  general_chat_demonstrated=False, length_conditioned_reconstruction_is_not_generation=True)
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'examples'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='blind-v1')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--teacher-prefix', action='store_true')
    train(parser.parse_args())
