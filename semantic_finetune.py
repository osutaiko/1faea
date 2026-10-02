"""Adapt the final pretrained block without continuous state crossing stages."""

import argparse
import json
import sys

import torch
from torch import nn
from torch.nn import functional as F
from transformers.masking_utils import create_causal_mask

from run import ROOT
from semantic import cache, evaluate
from semantic_data import examples
from semantic_model import PretrainedReader, SemanticReasoner, SYMBOLS, render


class AdaptedReader:
    def __init__(self):
        self.prefix = PretrainedReader()
        backbone = self.prefix.backbone
        self.last_layer = backbone.layers[-1].float().requires_grad_(True)
        del backbone.layers[-1]
        backbone.config.num_hidden_layers -= 1
        self.norm = backbone.norm.float().requires_grad_(True)
        backbone.norm = nn.Identity()

    def finish(self, features, mask):
        positions = torch.arange(features.shape[1], device=features.device)[None, :]
        backbone = self.prefix.backbone
        attention = create_causal_mask(config=backbone.config, inputs_embeds=features,
                                      attention_mask=mask, position_ids=positions, past_key_values=None)
        rotary = backbone.rotary_emb(features, position_ids=positions)
        hidden = self.last_layer(features, attention_mask=attention,
                                 position_ids=positions, position_embeddings=rotary, use_cache=False)
        return self.norm(hidden)

    @torch.no_grad()
    def text_features(self, texts):
        features, mask = self.prefix.text_features(texts)
        return self.finish(features, mask), mask

    @torch.no_grad()
    def state_features(self, state):
        features = self.prefix.state_features(state)
        return self.finish(features, torch.ones(state.shape, dtype=torch.bool, device=state.device))


def losses(model, reader, data, ids, answer_ids):
    features, mask, entities, derived, answers, initial_features, expanded_features = data
    parsed = model.parse_logits(reader.finish(features[ids], mask[ids]), mask[ids])
    state_mask = torch.ones(len(ids), 16, dtype=torch.bool)
    inferred = model.derive_logits(reader.finish(initial_features[ids], state_mask))
    final = model.answer_logits(reader.finish(expanded_features[answer_ids], state_mask))
    return (F.cross_entropy(parsed.flatten(0, 1), entities[ids].flatten())
            + F.cross_entropy(inferred.flatten(0, 1), derived[ids].flatten())
            + F.cross_entropy(final, answers[answer_ids]))


@torch.no_grad()
def validation_loss(model, reader, data):
    total = 0
    for ids in torch.arange(len(data[2])).split(8):
        total += losses(model, reader, data, ids, ids).item() * len(ids)
    return total / len(data[2])


def train(args):
    torch.manual_seed(7)
    reader = AdaptedReader()
    splits = examples()
    print('Caching outputs of the frozen prefix...', flush=True)
    training = cache(reader.prefix, splits['train'])
    validation = cache(reader.prefix, splits['validation'])
    model = SemanticReasoner(training[0].shape[-1])
    backbone_parameters = list(reader.last_layer.parameters()) + list(reader.norm.parameters())
    optimizer = torch.optim.AdamW([
        dict(params=model.parameters(), lr=0.0005),
        dict(params=backbone_parameters, lr=0.00002),
    ])
    original = {name: value.detach().clone() for name, value in reader.last_layer.state_dict().items()}
    output = ROOT / 'runs' / 'semantic-adapted'
    output.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    for step in range(1, args.steps + 1):
        ids = torch.randint(len(training[2]), (8,))
        answer_ids = torch.randint(len(training[4]), (8,))
        optimizer.zero_grad()
        loss = losses(model, reader, training, ids, answer_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + backbone_parameters, 1.0)
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            model.eval()
            score = validation_loss(model, reader, validation)
            print(f'step {step}: train loss={loss.item():.4f}, validation loss={score:.4f}', flush=True)
            if score < best_loss:
                best_loss = score
                torch.save(dict(config=model.config, model=model.state_dict(),
                                last_layer=reader.last_layer.state_dict(), norm=reader.norm.state_dict(),
                                step=step, validation_loss=score), output / 'model.pt')
            model.train()
    checkpoint = torch.load(output / 'model.pt', weights_only=True)
    restore(model, reader, checkpoint)
    change = sum((value - original[name]).abs().sum().item()
                 for name, value in reader.last_layer.state_dict().items())
    results = dict(selected_step=checkpoint['step'], validation_loss=checkpoint['validation_loss'],
                   pretrained_parameter_change_l1=change,
                   pretrained_trainable_parameters=sum(p.numel() for p in backbone_parameters),
                   steps=args.steps, batch_size=8)
    for name in ('validation', 'test'):
        results[name] = evaluate(model, reader, splits[name])
        print(json.dumps({name: {key: value for key, value in results[name].items() if key != 'examples'}}, indent=2), flush=True)
    (output / 'evaluation.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')


def restore(model, reader, checkpoint):
    model.load_state_dict(checkpoint['model'])
    reader.last_layer.load_state_dict(checkpoint['last_layer'])
    reader.norm.load_state_dict(checkpoint['norm'])
    model.eval()
    reader.last_layer.eval()
    reader.norm.eval()


def generate(args):
    checkpoint = torch.load(ROOT / 'runs' / 'semantic-adapted' / 'model.pt', weights_only=True)
    reader = AdaptedReader()
    model = SemanticReasoner(**checkpoint['config'])
    restore(model, reader, checkpoint)
    first, second, answer = model.generate(reader, [args.text])
    if args.trace:
        print(json.dumps(dict(initial=render(first[0]), expanded=render(second[0]),
                              answer=SYMBOLS[answer.item()]), ensure_ascii=False))
    else:
        print(SYMBOLS[answer.item()])


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    training.add_argument('--steps', type=int, default=600)
    generation = commands.add_parser('generate')
    generation.add_argument('text')
    generation.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    train(args) if args.command == 'train' else generate(args)
