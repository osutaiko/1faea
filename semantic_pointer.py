"""Prepare, train, and audit the source-copying emoji prototype."""

import argparse
import itertools
import json
import sys

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from run import ROOT, BASE_MODEL
from semantic import evaluate
from semantic_data import examples
from semantic_clauses import clause_examples, challenge_examples, audit_examples
from semantic_model import PretrainedReader, SYMBOLS, initial_state, render
from semantic_pointer_model import PointerReasoner, text_inputs


OUTPUT = ROOT / 'runs' / 'semantic-pointer'


def derive_targets(row):
    return [row['entities'][:4].index(entity) for entity in row['derived']]


@torch.no_grad()
def prepare():
    reader = PretrainedReader()
    dataset = {}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    directory = ROOT / 'data' / 'semantic-pointer'
    directory.mkdir(parents=True, exist_ok=True)
    challenge = challenge_examples()
    (directory / 'challenge.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in challenge), encoding='utf-8')
    (directory / 'audit.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in audit_examples()), encoding='utf-8')
    for name, rows in clause_examples().items():
        (directory / f'{name}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        sequences, candidates, positions = [], [], []
        for start in range(0, len(rows), 16):
            features, mask, offsets, entities = text_inputs(reader, [row['text'] for row in rows[start:start + 16]])
            sequences.extend(feature[valid] for feature, valid in zip(features, mask))
            positions.append(offsets)
            candidates.append(entities)
            if (start + 16) % 256 == 0:
                print(f'{name}: encoded {start + 16}/{len(rows)}', flush=True)
        features = pad_sequence(sequences, batch_first=True)
        mask = torch.arange(features.shape[1])[None] < torch.tensor([len(sequence) for sequence in sequences])[:, None]
        state_rows = examples()[name][::6] if name == 'train' else examples()[name]
        states = initial_state(torch.tensor([row['entities'] for row in state_rows]))
        unique, inverse = torch.unique(states, dim=0, return_inverse=True)
        state_features = torch.cat([reader.state_features(batch) for batch in unique.split(32)])[inverse]
        dataset[name] = dict(features=features, mask=mask, positions=torch.cat(positions),
                             candidates=torch.cat(candidates), labels=torch.tensor([row['orientation'] for row in rows]),
                             derived=torch.tensor([derive_targets(row) for row in state_rows]), state_features=state_features)
    torch.save(dict(base_model=BASE_MODEL, dataset=dataset), OUTPUT / 'features.pt')


def loss(model, data, ids, state_ids):
    parsed = model.parser(data['features'][ids], data['mask'][ids], data['positions'][ids])
    derived = model.derive_logits(data['state_features'][state_ids])
    return (F.cross_entropy(parsed.squeeze(1), data['labels'][ids])
            + F.cross_entropy(derived.flatten(0, 1), data['derived'][state_ids].flatten()))


@torch.no_grad()
def score(model, data):
    return loss(model, data, torch.arange(len(data['labels'])), torch.arange(len(data['derived']))).item()


def train(args):
    torch.manual_seed(7)
    prepared = torch.load(OUTPUT / 'features.pt', weights_only=True)
    training, validation = prepared['dataset']['train'], prepared['dataset']['validation']
    model = PointerReasoner(training['state_features'].shape[-1])
    baseline = torch.load(ROOT / 'runs' / 'semantic' / 'model.pt', weights_only=True)
    model.answer_head.load_state_dict({key.removeprefix('answer_head.'): value
                                      for key, value in baseline['model'].items() if key.startswith('answer_head.')})
    model.answer_head.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    best_loss = float('inf')
    for step in range(1, args.steps + 1):
        ids = torch.randint(len(training['labels']), (32,))
        state_ids = torch.randint(len(training['derived']), (32,))
        optimizer.zero_grad()
        current = loss(model, training, ids, state_ids)
        current.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 200 == 0 or step == args.steps:
            model.eval()
            validation_loss = score(model, validation)
            print(f'step {step}: train={current.item():.4f}, validation={validation_loss:.4f}', flush=True)
            if validation_loss < best_loss:
                best_loss = validation_loss
                torch.save(dict(config=model.config, model=model.state_dict(), step=step,
                                validation_loss=validation_loss, base_model=prepared['base_model']), OUTPUT / 'model.pt')
            model.train()
    model, reader, checkpoint = load()
    results = dict(selected_step=checkpoint['step'], validation_loss=checkpoint['validation_loss'])
    challenge = [json.loads(line) for line in (ROOT / 'data' / 'semantic-pointer' / 'challenge.jsonl').read_text(encoding='utf-8').splitlines()]
    audit = [json.loads(line) for line in (ROOT / 'data' / 'semantic-pointer' / 'audit.jsonl').read_text(encoding='utf-8').splitlines()]
    suites = dict(validation=examples()['validation'], test=examples()['test'], challenge=challenge, audit=audit)
    for name, rows in suites.items():
        results[name] = evaluate(model, reader, rows)
        print(json.dumps({name: {key: value for key, value in results[name].items() if key != 'examples'}}, indent=2), flush=True)
    results['renaming'] = renaming(model, reader)
    (OUTPUT / 'evaluation.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')


def load():
    checkpoint = torch.load(OUTPUT / 'model.pt', weights_only=True)
    reader = PretrainedReader(model_name=checkpoint['base_model'])
    model = PointerReasoner(**checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, reader, checkpoint


@torch.no_grad()
def renaming(model, reader):
    rows = examples()['validation']
    entities = torch.tensor([row['entities'] for row in rows])
    state = initial_state(entities)
    permutations = torch.tensor(list(itertools.permutations(range(4))))
    states = state[None].expand(24, -1, -1).clone()
    states[:, :, [1, 3, 5, 7, 13, 15]] = permutations[:, entities]
    output = torch.cat([model.derive_entities(reader, batch) for batch in states.flatten(0, 1).split(32)])
    output = output.reshape(24, len(rows), 2)
    expected = permutations[:, output[0]]
    return dict(count=24 * len(rows), equivariance=(output == expected).all(-1).float().mean().item())


def generate(args):
    model, reader, _ = load()
    first, second, answer = model.generate(reader, [args.text])
    if args.trace:
        print(json.dumps(dict(initial=render(first[0]), expanded=render(second[0]), answer=SYMBOLS[answer.item()]), ensure_ascii=False))
    else:
        print(SYMBOLS[answer.item()])


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare')
    training = commands.add_parser('train')
    training.add_argument('--steps', type=int, default=2000)
    generation = commands.add_parser('generate')
    generation.add_argument('text')
    generation.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'train':
        train(args)
    else:
        generate(args)
