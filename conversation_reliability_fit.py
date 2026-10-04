"""Adapt text understanding and emoji memory on mixed-attribute conversations."""

import argparse
import json
import random
import time

import torch

import conversation
from conversation_model import AtomicDecoder, SemanticAtomicDecoder, ConversationReader, memory, pad
from conversation_reliability_data import write
from emoji_lm_model import END
from run import ROOT


@torch.no_grad()
def prepare(manifest, dataset, cache_name='reliability-features'):
    reader = ConversationReader(manifest['base_model'])
    vectors = torch.load(conversation.OUTPUT / 'vectors.pt', weights_only=True)
    reader.symbol_embedding.load_state_dict(dict(weight=vectors.to(reader.symbol_embedding.weight.dtype)))
    result = {}
    directory = conversation.OUTPUT / cache_name
    directory.mkdir(exist_ok=True)
    for split in ('train', 'validation'):
        result[split] = []
        for start in range(0, len(dataset[split]), 16):
            rows = dataset[split][start:start + 16]
            path = directory / f'{split}-{start // 16:05d}.pt'
            if path.exists():
                if torch.load(path, weights_only=True)['rows'] != rows:
                    raise ValueError('Reliability cache differs from the curriculum')
            else:
                features, mask, sources = reader.text_inputs([row['text'] for row in rows])
                state, state_mask = memory([row['history'] for row in rows], [row['state'] for row in rows])
                state_features = reader.emoji_inputs(state, state_mask)
                torch.save(dict(text_features=features.to(torch.float8_e4m3fn), text_mask=mask, text_ids=sources,
                                state_features=state_features.to(torch.float8_e4m3fn), state=state,
                                state_mask=state_mask, meaning=pad([row['state'] + [END] for row in rows], -100),
                                reply=pad([row['reply'] + [END] for row in rows], -100), rows=rows), path)
            result[split].append(str(path.relative_to(conversation.OUTPUT)))
            print(f'{conversation.timestamp()} reliability features {split}: {start + len(rows)}/{len(dataset[split])}', flush=True)
    return result


def train(args, manifest, supplemental, reviewed=()):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    parent = conversation.OUTPUT / args.parent
    checkpoints = [torch.load(parent / f'{part}.pt', weights_only=True) for part in ('encoder', 'decoder')]
    models = []
    for model_type, checkpoint in zip((SemanticAtomicDecoder, AtomicDecoder), checkpoints):
        model = model_type(checkpoint['model']['embedding.weight'], **checkpoint['config'])
        model.load_state_dict(checkpoint['model'])
        models.append(model.eval())
    encoder, decoder = models
    parameters = [parameter for model in models for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    output = conversation.OUTPUT / args.name
    output.mkdir(exist_ok=True)
    validation = manifest['splits']['validation'] + supplemental['validation']
    best = conversation.validation(encoder, decoder, validation)

    def save(index, step, score):
        part = ('encoder', 'decoder')[index]
        torch.save(dict(config=models[index].config, model=models[index].state_dict(), step=step,
                        validation_loss=score, base_model=manifest['base_model'], parent=args.parent,
                        input_head='semantic' if index == 0 else 'atomic', seed=args.seed,
                        learning_rate=args.learning_rate), output / f'{part}.pt')

    for index, score in enumerate(best):
        save(index, 0, score)
    start = time.monotonic()
    with (output / 'training.jsonl').open('w', encoding='utf-8') as journal:
        for step in range(1, args.steps + 1):
            paths = manifest['splits']['train'] if step % 4 == 0 else supplemental['train']
            if reviewed and step % 8 == 1:
                paths = reviewed
            data = conversation.batch(rng.choice(paths))
            for model in models:
                model.train()
            optimizer.zero_grad()
            values = conversation.losses(encoder, decoder, data)
            (values[0] + values[1]).backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            if step % 400 == 0 or step == args.steps:
                for model in models:
                    model.eval()
                scores = conversation.validation(encoder, decoder, validation)
                entry = dict(time=conversation.timestamp(), step=step, elapsed_seconds=time.monotonic() - start,
                             train=[value.item() for value in values], validation=scores)
                journal.write(json.dumps(entry) + '\n')
                journal.flush()
                print(json.dumps(entry), flush=True)
                for index, score in enumerate(scores):
                    if score < best[index]:
                        best[index] = score
                        save(index, step, score)
    (output / 'curriculum.json').write_text(json.dumps(dict(parent=args.parent, seed=args.seed,
                                                          steps=args.steps, supplemental=supplemental, reviewed=list(reviewed),
                                                          selection='combined legacy and supplemental validation loss'),
                                                     indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='reliability')
    parser.add_argument('--parent', default='selected')
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--seed', type=int, default=431)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--reviewed', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    conversation.OUTPUT = ROOT / 'runs' / 'conversation-compositional'
    manifest = json.loads((conversation.OUTPUT / 'features.json').read_text(encoding='utf-8'))
    dataset, _ = write()
    supplemental = prepare(manifest, dataset)
    reviewed = []
    if args.reviewed:
        approved = ROOT / 'data' / 'conversation' / 'reliability' / 'approved-topics.jsonl'
        rows = [json.loads(line) for line in approved.read_text(encoding='utf-8').splitlines()]
        quantity_inputs = approved.with_name('quantity-inputs.jsonl')
        rows.extend(json.loads(line) for line in quantity_inputs.read_text(encoding='utf-8').splitlines())
        public_inputs = approved.with_name('public-scope-train.jsonl')
        rows.extend(json.loads(line) for line in public_inputs.read_text(encoding='utf-8').splitlines())
        reviewed = prepare(manifest, dict(train=rows, validation=[]), 'reviewed-topic-features')['train']
    train(args, manifest, supplemental, reviewed)
