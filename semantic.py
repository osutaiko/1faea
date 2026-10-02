"""Train and inspect the supervised, discrete semantic prototype."""

import argparse
import json
import sys

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from run import ROOT
from semantic_data import examples, verdict
from semantic_model import PretrainedReader, SemanticReasoner, SYMBOLS, TRUE, EMPTY, initial_state, expanded_state, render


@torch.no_grad()
def cache(reader, rows):
    sequences = []
    for start in range(0, len(rows), 16):
        features, mask = reader.text_features([row['text'] for row in rows[start:start + 16]])
        sequences.extend(feature[valid] for feature, valid in zip(features, mask))
    features = pad_sequence(sequences, batch_first=True)
    mask = torch.arange(features.shape[1])[None, :] < torch.tensor([len(x) for x in sequences])[:, None]
    entities = torch.tensor([row['entities'] for row in rows])
    derived = torch.tensor([row['derived'] for row in rows])
    states = initial_state(entities)
    expanded = expanded_state(states, derived)
    def encode(ids):
        unique, inverse = torch.unique(ids, dim=0, return_inverse=True)
        return torch.cat([reader.state_features(batch) for batch in unique.split(32)])[inverse]
    isolated = expanded.clone()
    isolated[:, :8] = EMPTY
    reversed_state = isolated.clone()
    reversed_state[:, [9, 11]] = isolated[:, [11, 9]]
    answer_states = torch.cat((expanded, isolated, reversed_state))
    answer_targets = [row['answer'] - TRUE for row in rows]
    for state in torch.cat((isolated, reversed_state)).tolist():
        answer_targets.append(verdict([(state[9], state[11])], (state[13], state[15])) - TRUE)
    return features, mask, entities, derived, torch.tensor(answer_targets), encode(states), encode(answer_states)


@torch.no_grad()
def evaluate(model, reader, rows):
    counts = dict(parse=0, derive=0, answer=0, valid_proof=0, consistent_answer=0,
                  derive_given_gold_state=0, answer_given_gold_state=0, novel_proof=0, correct_trace=0)
    intervention_correct = intervention_changed = intervention_count = 0
    traces = []
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]
        first, second, answers = model.generate(reader, [row['text'] for row in batch])
        gold_first = initial_state(torch.tensor([row['entities'] for row in batch]))
        gold_derived = torch.tensor([row['derived'] for row in batch])
        predicted_derived = model.derive_entities(reader, gold_first)
        gold_second = expanded_state(gold_first, gold_derived)
        predicted_answers = model.answer_logits(reader.state_features(gold_second)).argmax(-1) + TRUE
        counts['derive_given_gold_state'] += (predicted_derived == gold_derived).all(-1).sum().item()
        counts['answer_given_gold_state'] += (predicted_answers == torch.tensor([row['answer'] for row in batch])).sum().item()
        # Remove the original facts, leaving only the derived fact and query.
        # Compare against a second state with that derived edge reversed.
        isolated = gold_second.clone()
        isolated[:, :8] = EMPTY
        reversed_state = isolated.clone()
        reversed_state[:, [9, 11]] = isolated[:, [11, 9]]
        isolated_answers = model.answer_logits(reader.state_features(isolated)).argmax(-1) + TRUE
        reversed_answers = model.answer_logits(reader.state_features(reversed_state)).argmax(-1) + TRUE
        for row, original, reversed_answer in zip(batch, isolated_answers.tolist(), reversed_answers.tolist()):
            query = tuple(row['entities'][-2:])
            edge = tuple(row['derived'])
            expected = verdict([edge], query)
            reversed_expected = verdict([edge[::-1]], query)
            if expected != reversed_expected:
                intervention_count += 1
                intervention_changed += original != reversed_answer
                intervention_correct += original == expected and reversed_answer == reversed_expected
        for row, s1, s2, answer in zip(batch, first.tolist(), second.tolist(), answers.tolist()):
            parsed_correct = [s1[i] for i in (1, 3, 5, 7, 13, 15)] == row['entities']
            derived_correct = [s2[9], s2[11]] == row['derived']
            counts['parse'] += parsed_correct
            counts['derive'] += derived_correct
            counts['answer'] += answer == row['answer']
            counts['correct_trace'] += parsed_correct and derived_correct and answer == row['answer']
            edges = [(s1[1], s1[3]), (s1[5], s1[7])]
            valid_proof = verdict(edges, (s2[9], s2[11])) == TRUE
            counts['valid_proof'] += valid_proof
            counts['novel_proof'] += valid_proof and (s2[9], s2[11]) not in edges
            counts['consistent_answer'] += verdict(edges + [(s2[9], s2[11])], (s2[13], s2[15])) == answer
            traces.append(dict(text=row['text'], initial=render(torch.tensor(s1)),
                               expanded=render(torch.tensor(s2)), answer=SYMBOLS[answer],
                               expected=SYMBOLS[row['answer']]))
    return dict(count=len(rows), **{key: value / len(rows) for key, value in counts.items()},
                intervention_count=intervention_count,
                intervention_changed=intervention_changed / intervention_count,
                intervention_correct=intervention_correct / intervention_count, examples=traces[:12])


def train(args):
    torch.manual_seed(7)
    torch.set_num_threads(4)
    reader = PretrainedReader()
    splits = examples()
    directory = ROOT / 'data' / 'semantic'
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        (directory / f'{name}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    print('Caching frozen pretrained features...', flush=True)
    features, mask, entities, derived, answers, initial_features, expanded_features = cache(reader, splits['train'])
    model = SemanticReasoner(features.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    for step in range(args.steps):
        ids = torch.randint(len(entities), (32,))
        answer_ids = torch.randint(len(answers), (32,))
        optimizer.zero_grad()
        loss = (F.cross_entropy(model.parse_logits(features[ids], mask[ids]).flatten(0, 1), entities[ids].flatten())
                + F.cross_entropy(model.derive_logits(initial_features[ids]).flatten(0, 1), derived[ids].flatten())
                + F.cross_entropy(model.answer_logits(expanded_features[answer_ids]), answers[answer_ids]))
        loss.backward()
        optimizer.step()
        if (step + 1) % 200 == 0:
            print(f'step {step + 1}: loss={loss.item():.4f}', flush=True)
    model.eval()
    output = ROOT / 'runs' / 'semantic'
    output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(config=model.config, model=model.state_dict(), base_model=reader.model_name), output / 'model.pt')
    metrics = {name: evaluate(model, reader, rows) for name, rows in splits.items()}
    (output / 'evaluation.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({name: {k: v for k, v in result.items() if k != 'examples'} for name, result in metrics.items()}, indent=2))


def generate(args):
    torch.set_num_threads(4)
    checkpoint = torch.load(ROOT / 'runs' / 'semantic' / 'model.pt', weights_only=True)
    reader = PretrainedReader(model_name=checkpoint['base_model'])
    model = SemanticReasoner(**checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    first, second, answer = model.generate(reader, [args.text])
    if args.trace:
        print(json.dumps(dict(initial=render(first[0]), expanded=render(second[0]), answer=SYMBOLS[answer.item()]), ensure_ascii=False))
    else:
        print(SYMBOLS[answer.item()])


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    training.add_argument('--steps', type=int, default=3000)
    generation = commands.add_parser('generate')
    generation.add_argument('text')
    generation.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    train(args) if args.command == 'train' else generate(args)
