"""Train an autoregressive emoji continuation model on grounded states."""

import argparse
import json
import sys

import torch
from torch.nn import functional as F

from run import ROOT
from semantic_data import examples
from semantic_clauses import challenge_examples, audit_examples
from semantic_model import FACT, LEFT, initial_state, render
from semantic_pointer import load as load_pointer
from emoji_lm_model import EmojiLM, END, START, ALPHABET, symbol_vectors, visible_tokens, render_tokens


OUTPUT = ROOT / 'runs' / 'emoji-lm'


def targets(rows):
    return torch.tensor([[FACT, row['derived'][0], LEFT, row['derived'][1], row['answer'], END] for row in rows])


@torch.no_grad()
def prepare():
    _, reader, checkpoint = load_pointer()
    dataset = {}
    directory = ROOT / 'data' / 'emoji-lm'
    directory.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in ('train', 'validation'):
        rows = examples()[name][::6] if name == 'train' else examples()[name]
        state = initial_state(torch.tensor([row['entities'] for row in rows]))
        features = torch.cat([reader.state_features(batch) for batch in state.split(32)])
        continuation = targets(rows)
        dataset[name] = dict(state=state, features=features, target=continuation)
        records = [dict(state=render(s), continuation=[ALPHABET[token] for token in t.tolist()])
                   for s, t in zip(state, continuation)]
        (directory / f'{name}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records), encoding='utf-8')
    torch.save(dict(dataset=dataset, vectors=symbol_vectors(reader), base_model=checkpoint['base_model']), OUTPUT / 'features.pt')


def loss(model, data, ids):
    target = data['target'][ids]
    prefix = torch.cat((torch.full((len(ids), 1), START), target[:, :-1]), dim=1)
    logs = model(data['state'][ids], data['features'][ids], prefix)
    return F.nll_loss(logs.flatten(0, 1), target.flatten())


@torch.no_grad()
def validation_loss(model, data):
    return sum(loss(model, data, ids).item() * len(ids)
               for ids in torch.arange(len(data['target'])).split(32)) / len(data['target'])


def train(args):
    torch.manual_seed(7)
    prepared = torch.load(OUTPUT / 'features.pt', weights_only=True)
    pointer, _, _ = load_pointer()
    model = EmojiLM(prepared['vectors'])
    model.initialize_from_pointer(pointer)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    best = float('inf')
    for step in range(1, args.steps + 1):
        data = prepared['dataset']['train']
        ids = torch.randint(len(data['target']), (32,))
        optimizer.zero_grad()
        current = loss(model, data, ids)
        current.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 200 == 0 or step == args.steps:
            model.eval()
            score = validation_loss(model, prepared['dataset']['validation'])
            print(f'step {step}: train={current.item():.4f}, validation={score:.4f}', flush=True)
            if score < best:
                best = score
                torch.save(dict(config=model.config, model=model.state_dict(), step=step,
                                validation_loss=score, base_model=prepared['base_model']), OUTPUT / 'model.pt')
            model.train()
    model, pointer, reader, checkpoint = load()
    report = dict(selected_step=checkpoint['step'], validation_loss=checkpoint['validation_loss'])
    suites = dict(validation=examples()['validation'], test=examples()['test'],
                  challenge=challenge_examples(), audit=audit_examples())
    for name, rows in suites.items():
        report[name] = evaluate(model, pointer, reader, rows)
        print(json.dumps({name: {key: value for key, value in report[name].items() if key != 'examples'}}, indent=2), flush=True)
    (OUTPUT / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


def load():
    checkpoint = torch.load(OUTPUT / 'model.pt', weights_only=True)
    pointer, reader, _ = load_pointer()
    if checkpoint['base_model'] != reader.model_name:
        raise ValueError('Language head and text parser require the same pretrained model')
    model = EmojiLM(symbol_vectors(reader), **checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, pointer, reader, checkpoint


@torch.no_grad()
def evaluate(model, pointer, reader, rows):
    full = correct_answer = ended = oracle = 0
    traces = []
    expected = targets(rows)
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]
        gold = initial_state(torch.tensor([row['entities'] for row in batch]))
        parsed = pointer.parse_state(reader, [row['text'] for row in batch])
        output = model.generate(reader, parsed)
        gold_output = model.generate(reader, gold)
        for index, (state, generated, oracle_generated) in enumerate(zip(parsed, output, gold_output)):
            target = expected[start + index].tolist()
            visible = visible_tokens(generated)
            terminated = END in generated.tolist()
            parse_correct = torch.equal(state, gold[index])
            full += parse_correct and terminated and visible == target[:-1]
            correct_answer += bool(visible) and visible[-1] == target[-2]
            ended += terminated
            oracle += END in oracle_generated.tolist() and visible_tokens(oracle_generated) == target[:-1]
            traces.append(dict(text=batch[index]['text'], state=render(state),
                               continuation=render_tokens(generated), expected=''.join(ALPHABET[t] for t in target[:-1])))
    return dict(count=len(rows), correct_trace=full / len(rows), answer=correct_answer / len(rows),
                terminated=ended / len(rows), continuation_given_gold_state=oracle / len(rows), examples=traces[:12])


def generate(args):
    model, pointer, reader, _ = load()
    state = pointer.parse_state(reader, [args.text])
    output = model.generate(reader, state)[0]
    if args.trace:
        print(json.dumps(dict(state=render(state[0]), continuation=render_tokens(output)), ensure_ascii=False))
    else:
        print(render_tokens(output))


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
