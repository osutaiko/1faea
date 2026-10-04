"""Low-rank input adaptation with an unchanged emoji-only reply backbone."""

import argparse
import json
import random
import shutil
import time

from peft import LoraConfig, PeftModel, get_peft_model
import torch

import conversation
from conversation_model import ConversationReader, SemanticAtomicDecoder, pad
from conversation_reliability_data import DIRECTORY, dataset
from emoji_lm_model import END
from run import ROOT


class AdaptedConversationReader(ConversationReader):
    def __init__(self, model_name, adapter):
        super().__init__(model_name)
        self.backbone = PeftModel.from_pretrained(self.backbone, adapter).eval()

    @torch.no_grad()
    def emoji_inputs(self, state, mask):
        # Input adaptation must not silently alter the trained memory decoder.
        with self.backbone.disable_adapter():
            return super().emoji_inputs(state, mask)


@torch.no_grad()
def prepare(reader, rows, split):
    directory = conversation.OUTPUT / 'input-prefix-features'
    directory.mkdir(exist_ok=True)
    paths = []
    captured = {}

    def capture(module, args, kwargs):
        captured['hidden'] = args[0].detach()
        captured['kwargs'] = kwargs

    handle = reader.backbone.layers[-1].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for start in range(0, len(rows), 8):
            selected = rows[start:start + 8]
            path = directory / f'{split}-{start // 8:05d}.pt'
            if path.exists():
                if torch.load(path, weights_only=True)['rows'] != selected:
                    raise ValueError('Input-prefix cache differs from the current training rows')
            else:
                features, mask, sources = reader.text_inputs([row['text'] for row in selected])
                dimension = features.shape[-1] // 2
                torch.save(dict(hidden=captured['hidden'], kwargs=captured['kwargs'], mask=mask, sources=sources,
                                lexical=features[..., dimension:].to(torch.float8_e4m3fn),
                                targets=pad([row['state'] + [END] for row in selected], -100), rows=selected), path)
            paths.append(str(path.relative_to(conversation.OUTPUT)))
            print(f'{conversation.timestamp()} input prefix {split}: {start + len(selected)}/{len(rows)}', flush=True)
    finally:
        handle.remove()
    return paths


def features(reader, data):
    base = reader.backbone.base_model.model
    hidden = base.layers[-1](data['hidden'], **data['kwargs'])
    hidden = base.norm(hidden).float()
    return torch.cat((hidden, data['lexical'].float()), dim=-1)


def loss(reader, encoder, data):
    return conversation.token_loss(encoder, features(reader, data), data['mask'], data['sources'], data['targets'])


@torch.no_grad()
def validate(reader, encoder, paths):
    total, count = 0, 0
    for path in paths:
        data = conversation.batch(path)
        size = len(data['rows'])
        total += loss(reader, encoder, data).item() * size
        count += size
    return total / count


def train(args):
    torch.manual_seed(443)
    rng = random.Random(443)
    parent = conversation.OUTPUT / args.parent
    checkpoint = torch.load(parent / 'encoder.pt', weights_only=True)
    manifest = json.loads((conversation.OUTPUT / 'features.json').read_text(encoding='utf-8'))
    data, _ = dataset()
    approved = [json.loads(line) for line in (DIRECTORY / 'approved-topics.jsonl').read_text(encoding='utf-8').splitlines()]
    legacy = [row for path in manifest['splits']['train'] for row in conversation.batch(path)['rows']]
    training = rng.sample(data['train'], 384) + approved + rng.sample(legacy, 384)
    legacy_validation = [row for path in manifest['splits']['validation'] for row in conversation.batch(path)['rows']]
    validation = data['validation'] + rng.sample(legacy_validation, 224)
    reader = ConversationReader(checkpoint['base_model'])
    training_paths = prepare(reader, training, 'train')
    validation_paths = prepare(reader, validation, 'validation')
    if args.prepare_only:
        return
    final_layer = len(reader.backbone.layers) - 1
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=['q_proj', 'v_proj'],
                        layers_to_transform=[final_layer], layers_pattern='layers', bias='none')
    reader.backbone = get_peft_model(reader.backbone, config)
    encoder = SemanticAtomicDecoder(checkpoint['model']['embedding.weight'], **checkpoint['config'])
    encoder.load_state_dict(checkpoint['model'])
    parameters = [parameter for module in (reader.backbone, encoder) for parameter in module.parameters()
                  if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    output = conversation.OUTPUT / args.name
    output.mkdir(exist_ok=True)
    shutil.copyfile(parent / 'decoder.pt', output / 'decoder.pt')
    reader.backbone.eval()
    encoder.eval()
    best = validate(reader, encoder, validation_paths)

    def save(step, score):
        reader.backbone.save_pretrained(output / 'input-adapter', save_embedding_layers=False)
        torch.save(dict(config=encoder.config, model=encoder.state_dict(), step=step,
                        validation_loss=score, base_model=checkpoint['base_model'], input_head='semantic',
                        input_adapter='input-adapter', parent=args.parent), output / 'encoder.pt')

    save(0, best)
    start = time.monotonic()
    with (output / 'adapter-training.jsonl').open('w', encoding='utf-8') as journal:
        for step in range(1, args.steps + 1):
            reader.backbone.train()
            encoder.train()
            optimizer.zero_grad()
            current = loss(reader, encoder, conversation.batch(rng.choice(training_paths)))
            current.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            if step % 200 == 0 or step == args.steps:
                reader.backbone.eval()
                encoder.eval()
                score = validate(reader, encoder, validation_paths)
                entry = dict(time=conversation.timestamp(), step=step, elapsed_seconds=time.monotonic() - start,
                             train=current.item(), validation=score)
                journal.write(json.dumps(entry) + '\n')
                journal.flush()
                print(json.dumps(entry), flush=True)
                if score < best:
                    best = score
                    save(step, score)
    report = dict(parent=args.parent, parent_encoder_step=checkpoint['step'], steps=args.steps, rank=8, adapted_layer=final_layer,
                  adapted_modules=['q_proj', 'v_proj'], input_only=True,
                  adapter_parameters=sum(parameter.numel() for parameter in reader.backbone.parameters()
                                         if parameter.requires_grad),
                  train_rows=len(training), validation_rows=len(validation),
                  prefix_cache='training only; runtime recomputes full input features',
                  selection='validation state-token loss; no test examples used')
    (output / 'adapter-experiment.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='reliability-adapter')
    parser.add_argument('--parent', default='reliability')
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--learning-rate', type=float, default=0.00005)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    conversation.OUTPUT = ROOT / 'runs' / 'conversation-compositional'
    train(args)
