"""Compare reviewed input augmentation and a semantic emoji output head."""

import argparse
import json
import random
import shutil
import time

import torch

import conversation
from conversation_data import DIRECTORY
from conversation_model import AtomicDecoder, SemanticAtomicDecoder, ConversationReader, pad, visible
from emoji_lm_model import END
from run import ROOT


@torch.no_grad()
def prepare(output, manifest):
    rows = [json.loads(line) for line in (DIRECTORY / 'approved-inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    directory = output / 'approved-features'
    directory.mkdir(exist_ok=True)
    reader = ConversationReader(manifest['base_model'])
    vectors = torch.load(output / 'vectors.pt', weights_only=True)
    reader.symbol_embedding.load_state_dict(dict(weight=vectors.to(reader.symbol_embedding.weight.dtype)))
    paths = []
    for start in range(0, len(rows), 16):
        selected = rows[start:start + 16]
        path = directory / f'{start // 16:05d}.pt'
        if path.exists():
            if torch.load(path, weights_only=True)['rows'] != selected:
                raise ValueError('Approved input cache differs from reviewed data')
        else:
            features, mask, ids = reader.text_inputs([row['text'] for row in selected])
            torch.save(dict(text_features=features.to(torch.float8_e4m3fn), text_mask=mask,
                            text_ids=ids, meaning=pad([row['state'] + [END] for row in selected], -100),
                            rows=selected), path)
        paths.append(str(path.relative_to(output)))
        print(f'Approved input features: {min(start + 16, len(rows))}/{len(rows)}', flush=True)
    return paths


def loss(model, data):
    return conversation.token_loss(model, data['text_features'], data['text_mask'], data['text_ids'], data['meaning'])


@torch.no_grad()
def validate(model, paths):
    total, count = 0, 0
    for path in paths:
        data = conversation.batch(path)
        size = len(data['meaning'])
        total += loss(model, data).item() * size
        count += size
    return total / count


@torch.no_grad()
def score_checkpoint(name, paths, semantic=False):
    checkpoint = torch.load(conversation.OUTPUT / name / 'encoder.pt', weights_only=True)
    model_type = SemanticAtomicDecoder if semantic else AtomicDecoder
    model = model_type(checkpoint['model']['embedding.weight'], **checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    correct, count = 0, 0
    for path in paths:
        data = conversation.batch(path)
        predictions = model.generate(lambda active: (data['text_features'][active], data['text_mask'][active],
                                                      data['text_ids'][active]), len(data['meaning']))
        for prediction, row in zip(predictions, data['rows']):
            correct += END in prediction.tolist() and visible(prediction) == row['state']
            count += 1
    report = dict(created=conversation.timestamp(), split='validation', cached_features='float8_e4m3fn',
                  count=count, exact_state_accuracy=correct / count, checkpoint_step=checkpoint['step'],
                  validation_loss=checkpoint['validation_loss'])
    (conversation.OUTPUT / name / 'encoder-validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    return report


def train(args, manifest, approved):
    torch.manual_seed(19)
    rng = random.Random(19)
    baseline = conversation.OUTPUT / 'baseline'
    checkpoint = torch.load(baseline / 'encoder.pt', weights_only=True)
    model_type = SemanticAtomicDecoder if args.semantic else AtomicDecoder
    model = model_type(checkpoint['model']['embedding.weight'], **checkpoint['config'])
    if args.semantic:
        shared = {key: value for key, value in checkpoint['model'].items() if not key.startswith('generator.')}
        missing, unexpected = model.load_state_dict(shared, strict=False)
        if set(missing) != {'output_projection.weight', 'output_bias', 'output_scale'} or unexpected:
            raise ValueError('Unexpected semantic head transfer parameters')
    else:
        model.load_state_dict(checkpoint['model'])
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.0001, weight_decay=0.01)
    output = conversation.OUTPUT / args.name
    output.mkdir(exist_ok=True)
    shutil.copyfile(baseline / 'decoder.pt', output / 'decoder.pt')
    model.eval()
    best = validate(model, manifest['splits']['validation'])
    def save(step, score):
        torch.save(dict(config=model.config, model=model.state_dict(), step=step, validation_loss=score,
                        base_model=manifest['base_model'], input_head='semantic' if args.semantic else 'atomic',
                        parent_encoder_step=checkpoint['step']), output / 'encoder.pt')
    save(0, best)
    start = time.monotonic()
    with (output / 'training.jsonl').open('w', encoding='utf-8') as journal:
        for step in range(1, args.steps + 1):
            paths = approved if step % 2 else manifest['splits']['train']
            data = conversation.batch(rng.choice(paths))
            model.train()
            optimizer.zero_grad()
            current = loss(model, data)
            current.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            if step % 200 == 0 or step == args.steps:
                model.eval()
                score = validate(model, manifest['splits']['validation'])
                entry = dict(time=conversation.timestamp(), step=step, elapsed_seconds=time.monotonic() - start,
                             train=current.item(), validation=score)
                journal.write(json.dumps(entry) + '\n')
                journal.flush()
                print(json.dumps(entry), flush=True)
                if score < best:
                    best = score
                    save(step, score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--semantic', action='store_true')
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--score-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(3)
    conversation.OUTPUT = ROOT / 'runs' / 'conversation-compositional'
    manifest = json.loads((conversation.OUTPUT / 'features.json').read_text(encoding='utf-8'))
    if not args.score_only:
        paths = prepare(conversation.OUTPUT, manifest)
        train(args, manifest, paths)
    score_checkpoint(args.name, manifest['splits']['validation'], args.semantic)
