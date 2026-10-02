"""Pair the validation-selected input head with the adapted emoji memory head."""

import hashlib
import json
import shutil

import torch

from conversation import timestamp
from run import ROOT


def select():
    output = ROOT / 'runs' / 'conversation-compositional'
    candidates = {name: json.loads((output / name / 'encoder-validation.json').read_text(encoding='utf-8'))
                  for name in ('baseline', 'augmented', 'semantic')}
    name = max(candidates, key=lambda name: candidates[name]['exact_state_accuracy'])
    encoder_path = output / name / 'encoder.pt'
    decoder_path = output / 'augmented' / 'decoder.pt'
    encoder = torch.load(encoder_path, weights_only=True)
    decoder = torch.load(decoder_path, weights_only=True)
    if encoder['base_model'] != decoder['base_model'] or not torch.equal(
            encoder['model']['embedding.weight'], decoder['model']['embedding.weight']):
        raise ValueError('Selected conversation stages have different pretrained grounding')
    directory = output / 'selected'
    directory.mkdir(exist_ok=True)
    for part, source in (('encoder', encoder_path), ('decoder', decoder_path)):
        shutil.copyfile(source, directory / f'{part}.pt')
    report = dict(created=timestamp(), criterion='exact_state_accuracy on cached validation inputs',
                  candidates=candidates, input_source=name, memory_source='augmented',
                  semantic_encoder=name == 'semantic', encoder_step=encoder['step'], decoder_step=decoder['step'],
                  encoder_sha256=hashlib.sha256(encoder_path.read_bytes()).hexdigest(),
                  decoder_sha256=hashlib.sha256(decoder_path.read_bytes()).hexdigest())
    (directory / 'selection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    select()
