"""Train a provisional question-by-question model on direct emoji labels."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import torch

import conversation
from conversation_data import IDS
from conversation_model import SemanticAtomicDecoder, memory, pad, visible, render
from conversation_standalone_data import DIRECTORY
from emoji_lm_model import END
from run import ROOT


@torch.no_grad()
def score(encoder, decoder, reader, rows):
    counts = {group: dict(count=0, state=0, reply=0, trace=0, oracle=0, terminated=0)
              for group in ('all', 'answerable', 'abstention')}
    examples = []
    for start in range(0, len(rows), 16):
        records = rows[start:start + 16]
        meanings = encoder.encode(reader, [row['question'] for row in records])
        source, mask = memory([[] for _ in records], [visible(value) for value in meanings])
        replies = decoder.respond(reader, source, mask)
        source, mask = memory([[] for _ in records], [row['state'] for row in records])
        oracle = decoder.respond(reader, source, mask)
        for row, meaning, reply, intended in zip(records, meanings, replies, oracle):
            state_ok = END in meaning.tolist() and visible(meaning) == row['state']
            reply_ok = END in reply.tolist() and visible(reply) == row['reply']
            category = 'abstention' if row['reply'] == [IDS['❓']] else 'answerable'
            for group in ('all', category):
                value = counts[group]
                value['count'] += 1
                value['state'] += state_ok
                value['reply'] += reply_ok
                value['trace'] += state_ok and reply_ok
                value['oracle'] += END in intended.tolist() and visible(intended) == row['reply']
                value['terminated'] += END in reply.tolist()
            examples.append(dict(id=row['id'], question=row['question'], state=render(meaning), reply=render(reply),
                                 expected_state=render(row['state']), expected_reply=render(row['reply']), correct=reply_ok))
    metrics = {group: dict(count=value['count'], **{key: total / value['count'] if value['count'] else None
                                                  for key, total in value.items() if key != 'count'})
               for group, value in counts.items()}
    return dict(metrics=metrics, examples=examples, history='empty for every question',
                always_unknown_reply_accuracy=counts['abstention']['count'] / counts['all']['count'],
                targets='provisional emoji labels; not independently verified',
                features='fresh runtime text-backbone and distilled emoji features')


@torch.no_grad()
def prepare(reader, rows, output):
    paths = []
    for start in range(0, len(rows), 16):
        records = rows[start:start + 16]
        path = output / f'train-{start // 16:03d}.pt'
        if path.exists():
            if torch.load(path, weights_only=True)['rows'] != records:
                raise ValueError('Standalone feature cache differs from current labels')
        else:
            features, mask, sources = reader.text_inputs([row['question'] for row in records])
            state, state_mask = memory([[] for _ in records], [row['state'] for row in records])
            torch.save(dict(text_features=features.to(torch.float8_e4m3fn), text_mask=mask, text_ids=sources,
                            state_features=reader.emoji_inputs(state, state_mask).to(torch.float8_e4m3fn),
                            state=state, state_mask=state_mask, meaning=pad([row['state'] + [END] for row in records], -100),
                            reply=pad([row['reply'] + [END] for row in records], -100), rows=records), path)
        paths.append(path)
    return paths


def train(args):
    torch.manual_seed(503)
    rng = random.Random(503)
    rows = [json.loads(line) for line in args.dataset.read_text(encoding='utf-8').splitlines()]
    groups = {split: [row for row in rows if row['split'] == split] for split in ('train', 'validation', 'test')}
    conversation.OUTPUT = ROOT / 'runs' / 'conversation-compositional'
    encoder, decoder, reader, checkpoints = conversation.load('reliability-selected', SemanticAtomicDecoder)
    processor = checkpoints[1]['emoji_features']
    output = ROOT / 'runs' / 'conversation-standalone' / args.name
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.dataset, output / 'dataset.jsonl')
    shutil.copyfile(conversation.OUTPUT / 'reliability-selected' / processor, output / processor)
    baseline = score(encoder, decoder, reader, groups['validation'])
    (output / 'baseline-validation.json').write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding='utf-8')
    paths = prepare(reader, groups['train'], output)
    cached = [torch.load(path, weights_only=True) for path in paths]
    parameters = [value for module in (encoder, decoder) for value in module.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.0001, weight_decay=0.01)
    best = -1
    started = time.monotonic()
    with (output / 'training.jsonl').open('w', encoding='utf-8') as journal:
        for step in range(1, args.steps + 1):
            encoder.train()
            decoder.train()
            optimizer.zero_grad()
            losses = conversation.losses(encoder, decoder, rng.choice(cached))
            sum(losses).backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            if step % 200 == 0 or step == args.steps:
                encoder.eval()
                decoder.eval()
                report = score(encoder, decoder, reader, groups['validation'])
                metrics = report['metrics']
                quality = ((metrics['answerable']['reply'] or 0) * 0.5 + (metrics['abstention']['reply'] or 0) * 0.25 +
                           metrics['all']['state'] * 0.25)
                entry = dict(created=conversation.timestamp(), step=step, elapsed=time.monotonic() - started,
                             losses=[loss.item() for loss in losses], validation=metrics, selection_score=quality)
                journal.write(json.dumps(entry) + '\n')
                journal.flush()
                print(json.dumps(entry), flush=True)
                if quality > best:
                    best = quality
                    for index, (part, model) in enumerate((('encoder', encoder), ('decoder', decoder))):
                        checkpoint = dict(config=model.config, model=model.state_dict(), step=step,
                                          base_model=checkpoints[index]['base_model'], provisional=True)
                        if part == 'decoder':
                            checkpoint['emoji_features'] = processor
                        torch.save(checkpoint, output / f'{part}.pt')
                    (output / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    conversation.OUTPUT = output.parent
    encoder, decoder, reader, _ = conversation.load(args.name, SemanticAtomicDecoder)
    report = score(encoder, decoder, reader, groups['test'])
    report['created'] = conversation.timestamp()
    report['test_used_for_selection'] = False
    (output / 'test.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'experiment.json').write_text(json.dumps(dict(
        parent='conversation-compositional/reliability-selected', seed=503, steps=args.steps,
        dataset=str(args.dataset),
        dataset_sha256=hashlib.sha256((output / 'dataset.jsonl').read_bytes()).hexdigest(),
        independent_questions=True, counts={split: len(group) for split, group in groups.items()},
        selection='50% answerable exact replies, 25% abstentions, 25% exact states on validation',
        independent_teacher_review=False, general_use_ready=False), indent=2), encoding='utf-8')
    print(json.dumps(dict(test=report['metrics']), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='bootstrap')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--dataset', type=Path, default=DIRECTORY / 'provisional.jsonl')
    args = parser.parse_args()
    torch.set_num_threads(2)
    train(args)
