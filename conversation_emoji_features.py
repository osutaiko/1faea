"""Distill pretrained emoji features into a small causal emoji processor."""

import argparse
import json
import random
import shutil
import time

from peft import PeftModel
import torch
from torch import nn
from torch.nn import functional as F

import conversation
from conversation_model import AtomicDecoder, ConversationReader, MEMORY_LIMIT
from run import ROOT


class EmojiFeatureModel(nn.Module):
    def __init__(self, vectors, width=192, layers=2):
        super().__init__()
        self.config = dict(width=width, layers=layers)
        self.embedding = nn.Embedding.from_pretrained(vectors.float(), freeze=True)
        self.projection = nn.Sequential(nn.LayerNorm(vectors.shape[-1]), nn.Linear(vectors.shape[-1], width))
        self.positions = nn.Embedding(MEMORY_LIMIT, width)
        layer = nn.TransformerEncoderLayer(width, 4, width * 4, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, vectors.shape[-1]))

    def forward(self, state, mask):
        if state.dtype != torch.long:
            raise ValueError('Emoji features require integer symbol IDs')
        embedding = self.embedding(state)
        length = state.shape[1]
        hidden = self.projection(embedding) + self.positions(torch.arange(length))
        causal = torch.ones(length, length, dtype=torch.bool).triu(1)
        hidden = self.transformer(hidden, mask=causal, src_key_padding_mask=~mask)
        return torch.cat((self.output(hidden), F.layer_norm(embedding, (embedding.shape[-1],))), dim=-1)


class EmojiFeatureReader(ConversationReader):
    def __init__(self, model_name, features_path, input_adapter=None):
        super().__init__(model_name)
        if input_adapter is not None:
            self.backbone = PeftModel.from_pretrained(self.backbone, input_adapter).eval()
        checkpoint = torch.load(features_path, weights_only=True)
        self.emoji_features = EmojiFeatureModel(checkpoint['model']['embedding.weight'], **checkpoint['config'])
        self.emoji_features.load_state_dict(checkpoint['model'])
        self.emoji_features.eval().requires_grad_(False)

    @torch.no_grad()
    def emoji_inputs(self, state, mask):
        # Every reply step recomputes features. No continuous state is retained.
        return self.emoji_features(state, mask)


def losses(features, decoder, data):
    selected = features(data['state'], data['state_mask'])
    reply = conversation.token_loss(decoder, selected, data['state_mask'], data['state'], data['reply'])
    dimension = selected.shape[-1] // 2
    expected = data['state_features'][..., :dimension].float()[data['state_mask']]
    actual = selected[..., :dimension][data['state_mask']]
    alignment = F.mse_loss(actual, expected) / expected.square().mean().clamp_min(1e-6)
    return reply, alignment


@torch.no_grad()
def validate(features, decoder, paths):
    total, count = 0, 0
    for path in paths:
        data = conversation.batch(path)
        reply, alignment = losses(features, decoder, data)
        size = len(data['state'])
        total += (reply.item() + 0.05 * alignment.item()) * size
        count += size
    return total / count


def train(args):
    torch.manual_seed(449)
    rng = random.Random(449)
    parent = conversation.OUTPUT / args.parent
    checkpoint = torch.load(parent / 'decoder.pt', weights_only=True)
    features = EmojiFeatureModel(checkpoint['model']['embedding.weight'])
    decoder = AtomicDecoder(checkpoint['model']['embedding.weight'], **checkpoint['config'])
    decoder.load_state_dict(checkpoint['model'])
    manifest = json.loads((conversation.OUTPUT / 'features.json').read_text(encoding='utf-8'))
    reliability = json.loads((parent / 'curriculum.json').read_text(encoding='utf-8'))
    training = reliability['supplemental']['train'] + reliability['reviewed']
    validation = manifest['splits']['validation'] + reliability['supplemental']['validation']
    parameters = [parameter for module in (features, decoder) for parameter in module.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.0001, weight_decay=0.01)
    output = conversation.OUTPUT / args.name
    output.mkdir(exist_ok=True)
    shutil.copyfile(parent / 'encoder.pt', output / 'encoder.pt')
    encoder_checkpoint = torch.load(parent / 'encoder.pt', weights_only=True)
    if 'input_adapter' in encoder_checkpoint:
        shutil.copytree(parent / encoder_checkpoint['input_adapter'], output / encoder_checkpoint['input_adapter'], dirs_exist_ok=True)
    best = float('inf')
    start = time.monotonic()
    with (output / 'distillation-training.jsonl').open('w', encoding='utf-8') as journal:
        for step in range(1, args.steps + 1):
            paths = manifest['splits']['train'] if step % 4 == 0 else training
            data = conversation.batch(rng.choice(paths))
            features.train()
            decoder.train()
            optimizer.zero_grad()
            reply, alignment = losses(features, decoder, data)
            (reply + 0.05 * alignment).backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            if step % 400 == 0 or step == args.steps:
                features.eval()
                decoder.eval()
                score = validate(features, decoder, validation)
                entry = dict(time=conversation.timestamp(), step=step, elapsed_seconds=time.monotonic() - start,
                             reply_loss=reply.item(), feature_alignment=alignment.item(), validation=score)
                journal.write(json.dumps(entry) + '\n')
                journal.flush()
                print(json.dumps(entry), flush=True)
                if score < best:
                    best = score
                    torch.save(dict(config=features.config, model=features.state_dict(), step=step), output / 'emoji-features.pt')
                    torch.save(dict(config=decoder.config, model=decoder.state_dict(), step=step,
                                    validation_loss=score, base_model=checkpoint['base_model'], parent=args.parent,
                                    emoji_features='emoji-features.pt'), output / 'decoder.pt')
    report = dict(parent=args.parent, steps=args.steps, teacher='cached frozen SmolLM2 emoji features',
                  objective='direct emoji reply loss plus 0.05 normalized feature MSE',
                  runtime='fresh causal emoji features at every output step; only integer IDs persist',
                  input_backbone='unchanged from parent', validation_selection=True, test_used=False)
    (output / 'distillation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='reliability-distilled')
    parser.add_argument('--parent', default='reliability-reviewed')
    parser.add_argument('--steps', type=int, default=4800)
    args = parser.parse_args()
    torch.set_num_threads(2)
    conversation.OUTPUT = ROOT / 'runs' / 'conversation-compositional'
    train(args)
