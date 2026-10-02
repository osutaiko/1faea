"""Train and use a conversational model with only emoji state between stages."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import functools
import json
import random
import sys
import time

import torch
from torch.nn import functional as F

from conversation_data import DIRECTORY, splits, write
from conversation_composition import composition_splits, write_composition
from conversation_model import AtomicDecoder, SemanticAtomicDecoder, ConversationReader, EmojiConversation, memory, pad, render, visible
from emoji_lm_model import END, START
from run import ROOT


OUTPUT = ROOT / 'runs' / 'conversation'


def timestamp():
    return datetime.now(timezone.utc).isoformat()


@torch.no_grad()
def prepare(args):
    if args.curriculum == 'social':
        write()
        dataset = splits()
    else:
        write_composition()
        dataset = composition_splits()
    reader = ConversationReader(args.base_model)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = dict(base_model=reader.model_name, created=timestamp(), splits={})
    torch.save(reader.symbol_embedding.weight.float(), OUTPUT / 'vectors.pt')
    for name, rows in dataset.items():
        if name == 'train':
            random.Random(7).shuffle(rows)
        paths = []
        directory = OUTPUT / 'features' / name
        directory.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(rows), 16):
            batch = rows[start:start + 16]
            path = directory / f'{start // 16:05d}.pt'
            if path.exists():
                cached = torch.load(path, weights_only=True)
                if cached['rows'] != batch:
                    raise ValueError('Cached rows differ; choose a new experiment directory')
                paths.append(str(path.relative_to(OUTPUT)))
                continue
            features, mask, source_ids = reader.text_inputs([row['text'] for row in batch])
            state, state_mask = memory([row['history'] for row in batch], [row['state'] for row in batch])
            state_features = reader.emoji_inputs(state, state_mask)
            data = dict(text_features=features.bfloat16(), text_mask=mask, text_ids=source_ids,
                        state=state, state_mask=state_mask, state_features=state_features.bfloat16(),
                        meaning=pad([row['state'] + [END] for row in batch], -100),
                        reply=pad([row['reply'] + [END] for row in batch], -100), rows=batch)
            data['text_features'] = data['text_features'].to(torch.float8_e4m3fn)
            data['state_features'] = data['state_features'].to(torch.float8_e4m3fn)
            temporary = path.with_suffix('.pending')
            torch.save(data, temporary)
            temporary.replace(path)
            paths.append(str(path.relative_to(OUTPUT)))
            if start % 160 == 0 or start + len(batch) == len(rows):
                print(f'{timestamp()} {name}: cached {start + len(batch)}/{len(rows)}', flush=True)
        manifest['splits'][name] = paths
    (OUTPUT / 'features.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({name: len(paths) for name, paths in manifest['splits'].items()}, indent=2), flush=True)


@functools.lru_cache(maxsize=8)
def batch(path):
    return torch.load(OUTPUT / path, weights_only=True)


def token_loss(model, features, mask, source_ids, target):
    prefix = torch.cat((torch.full((len(target), 1), START), target[:, :-1].clamp_min(0)), dim=1)
    logs = model(features, mask, source_ids, prefix)
    return F.nll_loss(logs.flatten(0, 1), target.flatten(), ignore_index=-100)


def losses(encoder, decoder, data):
    return (token_loss(encoder, data['text_features'], data['text_mask'], data['text_ids'], data['meaning']),
            token_loss(decoder, data['state_features'], data['state_mask'], data['state'], data['reply']))


@torch.no_grad()
def validation(encoder, decoder, paths):
    total = torch.zeros(2)
    count = 0
    for path in paths:
        data = batch(path)
        values = losses(encoder, decoder, data)
        total += torch.tensor([value.item() for value in values]) * len(data['state'])
        count += len(data['state'])
    return (total / count).tolist()


def train(args):
    torch.manual_seed(7)
    rng = random.Random(7)
    manifest = json.loads((OUTPUT / 'features.json').read_text(encoding='utf-8'))
    vectors = torch.load(OUTPUT / 'vectors.pt', weights_only=True)
    encoder = AtomicDecoder(vectors, width=args.width, layers=args.layers)
    decoder = AtomicDecoder(vectors, width=args.width, layers=args.layers)
    parameters = [p for model in (encoder, decoder) for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    best = [float('inf'), float('inf')]
    output = OUTPUT / args.name
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (output / 'training.jsonl').open('w', encoding='utf-8') as journal:
        for step in range(1, args.steps + 1):
            data = batch(rng.choice(manifest['splits']['train']))
            encoder.train()
            decoder.train()
            optimizer.zero_grad()
            values = losses(encoder, decoder, data)
            current = values[0] + values[1]
            current.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            if step % 200 == 0 or step == args.steps:
                encoder.eval()
                decoder.eval()
                score = validation(encoder, decoder, manifest['splits']['validation'])
                entry = dict(time=timestamp(), step=step, elapsed_seconds=time.monotonic() - start,
                             train=[value.item() for value in values], validation=score)
                journal.write(json.dumps(entry) + '\n')
                journal.flush()
                print(json.dumps(entry), flush=True)
                for index, (name, model) in enumerate([('encoder', encoder), ('decoder', decoder)]):
                    if score[index] < best[index]:
                        best[index] = score[index]
                        torch.save(dict(config=model.config, model=model.state_dict(), step=step,
                                        validation_loss=score[index], base_model=manifest['base_model']), output / f'{name}.pt')
    print(f'Training complete: {time.monotonic() - start:.1f} seconds', flush=True)


def load(name, encoder_type=AtomicDecoder):
    output = OUTPUT / name
    checkpoints = [torch.load(output / f'{part}.pt', weights_only=True) for part in ('encoder', 'decoder')]
    if checkpoints[0]['base_model'] != checkpoints[1]['base_model']:
        raise ValueError('Conversation stages require the same pretrained model')
    if not torch.equal(checkpoints[0]['model']['embedding.weight'], checkpoints[1]['model']['embedding.weight']):
        raise ValueError('Conversation stages require the same grounded emoji embeddings')
    reader = ConversationReader(checkpoints[0]['base_model'])
    models = []
    for model_type, checkpoint in zip((encoder_type, AtomicDecoder), checkpoints):
        model = model_type(checkpoint['model']['embedding.weight'], **checkpoint['config'])
        model.load_state_dict(checkpoint['model'])
        model.eval()
        models.append(model)
    reader.symbol_embedding.load_state_dict(dict(weight=models[0].embedding.weight.to(reader.symbol_embedding.weight.dtype)))
    return models[0], models[1], reader, checkpoints


@torch.no_grad()
def evaluate(args, encoder_type=AtomicDecoder):
    encoder, decoder, reader, checkpoints = load(args.name, encoder_type)
    manifest = json.loads((OUTPUT / 'features.json').read_text(encoding='utf-8'))
    rows = [row for path in manifest['splits'][args.split] for row in batch(path)['rows']]
    if args.limit:
        rows = rows[:args.limit]
    totals = defaultdict(lambda: defaultdict(int))
    examples = []
    for start in range(0, len(rows), 16):
        records = rows[start:start + 16]
        encoded = encoder.encode(reader, [row['text'] for row in records])
        selected = [visible(tokens) for tokens in encoded]
        state, mask = memory([row['history'] for row in records], selected)
        generated = decoder.respond(reader, state, mask)
        gold, gold_mask = memory([row['history'] for row in records], [row['state'] for row in records])
        oracle = decoder.respond(reader, gold, gold_mask)
        for row, meaning, output, gold_output in zip(records, encoded, generated, oracle):
            meaning_correct = END in meaning.tolist() and visible(meaning) == row['state']
            reply_correct = END in output.tolist() and visible(output) == row['reply']
            oracle_correct = END in gold_output.tolist() and visible(gold_output) == row['reply']
            for group in ('all', row['skill']):
                totals[group]['count'] += 1
                totals[group]['meaning'] += meaning_correct
                totals[group]['reply'] += reply_correct
                totals[group]['correct_trace'] += meaning_correct and reply_correct
                totals[group]['reply_given_gold_state'] += oracle_correct
                totals[group]['terminated'] += END in output.tolist()
            examples.append(dict(text=row['text'], skill=row['skill'], meaning=render(meaning),
                                 expected_meaning=render(row['state']), reply=render(output),
                                 expected_reply=render(row['reply']), correct=reply_correct))
        print(f'{timestamp()} evaluated {start + len(records)}/{len(rows)}', flush=True)
    metrics = {group: {key: value if key == 'count' else value / numbers['count']
                       for key, value in numbers.items()} for group, numbers in totals.items()}
    report = dict(created=timestamp(), split=args.split, encoder_step=checkpoints[0]['step'],
                  decoder_step=checkpoints[1]['step'], metrics=metrics, examples=examples)
    path = OUTPUT / args.name / f'evaluation-{args.split}.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2), flush=True)


def generate(args, encoder_type=AtomicDecoder):
    encoder, decoder, reader, _ = load(args.name, encoder_type)
    conversation = EmojiConversation(encoder, decoder, reader)
    for text in args.text:
        result = conversation.turn(text)
        if args.trace:
            print(json.dumps(dict(meaning=render(result['state']), reply=render(result['reply']),
                                  memory=render(conversation.history),
                                  state_terminated=result['state_terminated'], reply_terminated=result['reply_terminated'],
                                  dropped_turns=conversation.dropped_turns),
                             ensure_ascii=False))
        else:
            print(render(result['reply']))


def chat(args, encoder_type=AtomicDecoder):
    encoder, decoder, reader, _ = load(args.name, encoder_type)
    conversation = EmojiConversation(encoder, decoder, reader)
    while True:
        try:
            text = input()
        except EOFError:
            break
        if text == '/quit':
            break
        if text == '/reset':
            conversation.history.clear()
            conversation.turn_lengths.clear()
            conversation.dropped_turns = 0
            print('🔄', flush=True)
            continue
        result = conversation.turn(text)
        print(render(result['reply']), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default='conversation')
    parser.add_argument('--semantic-encoder', action='store_true')
    commands = parser.add_subparsers(dest='command', required=True)
    preparation = commands.add_parser('prepare')
    preparation.add_argument('--base-model')
    preparation.add_argument('--curriculum', choices=['social', 'compositional'], default='social')
    training = commands.add_parser('train')
    training.add_argument('--name', default='baseline')
    training.add_argument('--steps', type=int, default=4000)
    training.add_argument('--width', type=int, default=192)
    training.add_argument('--layers', type=int, default=3)
    training.add_argument('--learning-rate', type=float, default=0.0003)
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--name', default='baseline')
    evaluation.add_argument('--split', choices=['validation', 'test'], default='test')
    evaluation.add_argument('--limit', type=int)
    generation = commands.add_parser('generate')
    generation.add_argument('text', nargs='+')
    generation.add_argument('--name', default='baseline')
    generation.add_argument('--trace', action='store_true')
    chatting = commands.add_parser('chat')
    chatting.add_argument('--name', default='baseline')
    args = parser.parse_args()
    OUTPUT = ROOT / 'runs' / args.experiment
    encoder_type = SemanticAtomicDecoder if args.semantic_encoder else AtomicDecoder
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args, encoder_type)
    elif args.command == 'chat':
        chat(args, encoder_type)
    else:
        generate(args, encoder_type)
