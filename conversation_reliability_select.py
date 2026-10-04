"""Select by validation quality with explicit legacy-regression limits."""

import argparse
from collections import defaultdict
import hashlib
import json
import random
import shutil

import torch

import conversation
from conversation_model import SemanticAtomicDecoder, memory, visible
from emoji_lm_model import END
from run import ROOT


def blend_input(name, reviewed_weight):
    """Blend two input heads before validation; retain the distilled reply path."""
    root = conversation.OUTPUT
    original = torch.load(root / 'selected' / 'encoder.pt', weights_only=True)
    reviewed = torch.load(root / 'reliability-reviewed' / 'encoder.pt', weights_only=True)
    for key, value in reviewed['model'].items():
        if value.is_floating_point():
            reviewed['model'][key] = reviewed_weight * value + (1 - reviewed_weight) * original['model'][key]
    reviewed['input_head'] = f'{reviewed_weight:.0%} reviewed input head parameter blend; validation only'
    output = root / name
    output.mkdir(exist_ok=True)
    torch.save(reviewed, output / 'encoder.pt')
    for part in ('decoder.pt', 'emoji-features.pt'):
        shutil.copyfile(root / 'reliability-distilled' / part, output / part)
    (output / 'blend.json').write_text(json.dumps(dict(
        parents=['selected', 'reliability-reviewed'], weights=[1 - reviewed_weight, reviewed_weight],
        test_used=False), indent=2), encoding='utf-8')


@torch.no_grad()
def score_legacy(name):
    output = conversation.OUTPUT / name
    encoder, decoder, reader, _ = conversation.load(name, SemanticAtomicDecoder)
    manifest = json.loads((conversation.OUTPUT / 'features.json').read_text(encoding='utf-8'))
    families = defaultdict(list)
    for path in manifest['splits']['validation']:
        for row in conversation.batch(path)['rows']:
            families[row['skill']].append(row)
    rng = random.Random(455)
    rows = []
    for skill in sorted(families):
        rng.shuffle(families[skill])
        rows.extend(families[skill][:8])
    rows.sort(key=lambda row: len(row['history']))
    state_correct, oracle_correct, count = 0, 0, 0
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]
        meanings = encoder.encode(reader, [row['text'] for row in batch])
        state, mask = memory([row['history'] for row in batch], [row['state'] for row in batch])
        replies = decoder.respond(reader, state, mask)
        for row, meaning, reply in zip(batch, meanings, replies):
            count += 1
            state_correct += END in meaning.tolist() and visible(meaning) == row['state']
            oracle_correct += END in reply.tolist() and visible(reply) == row['reply']
        print(f'{conversation.timestamp()} legacy selection {name}: {count}/{len(rows)}', flush=True)
    result = dict(state_accuracy=state_correct / count, oracle_reply_accuracy=oracle_correct / count,
                  features='fresh runtime features for both stages', count=count,
                  sample='up to eight validation rows per skill, seed 455', full_validation_rows=1116)
    (output / 'legacy-selection-validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(name=name, **result)), flush=True)
    return result


def select(names):
    baseline = json.loads((conversation.OUTPUT / 'selected' / 'legacy-selection-validation.json').read_text(encoding='utf-8'))
    candidates = {}
    for name in names:
        output = conversation.OUTPUT / name
        legacy = json.loads((output / 'legacy-selection-validation.json').read_text(encoding='utf-8'))
        fresh = json.loads((output / 'reliability-validation.json').read_text(encoding='utf-8'))
        eligible = (legacy['state_accuracy'] >= baseline['state_accuracy'] - 0.03 and
                    legacy['oracle_reply_accuracy'] >= baseline['oracle_reply_accuracy'] - 0.03)
        score = (fresh['metrics']['all']['correct_trace'] * 0.5 + legacy['state_accuracy'] * 0.25 +
                 legacy['oracle_reply_accuracy'] * 0.25)
        candidates[name] = dict(eligible=eligible, score=score, legacy=legacy,
                                fresh_validation=fresh['metrics']['all'], operator_contrasts=fresh['operator_contrasts'])
    eligible = [name for name in names if candidates[name]['eligible']]
    if not eligible:
        raise ValueError('No candidate satisfies the legacy validation regression limits')
    name = max(eligible, key=lambda name: candidates[name]['score'])
    source = conversation.OUTPUT / name
    output = conversation.OUTPUT / 'reliability-selected'
    output.mkdir(exist_ok=True)
    for part in ('encoder', 'decoder'):
        shutil.copyfile(source / f'{part}.pt', output / f'{part}.pt')
    checkpoint = torch.load(source / 'encoder.pt', weights_only=True)
    if 'input_adapter' in checkpoint:
        shutil.copytree(source / checkpoint['input_adapter'], output / checkpoint['input_adapter'], dirs_exist_ok=True)
    decoder_checkpoint = torch.load(source / 'decoder.pt', weights_only=True)
    if 'emoji_features' in decoder_checkpoint:
        shutil.copyfile(source / decoder_checkpoint['emoji_features'], output / decoder_checkpoint['emoji_features'])
    report = dict(created=conversation.timestamp(), chosen=name, candidates=candidates,
                  criterion='50% fresh supplemental correct traces, 25% legacy states, 25% legacy oracle replies',
                  regression_limit='at most 3 percentage points on each legacy validation measure',
                  test_used=False, features_note='all validation measures use fresh runtime features',
                  encoder_sha256=hashlib.sha256((output / 'encoder.pt').read_bytes()).hexdigest(),
                  decoder_sha256=hashlib.sha256((output / 'decoder.pt').read_bytes()).hexdigest())
    (output / 'selection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--names', nargs='+', default=['selected', 'reliability', 'reliability-reviewed', 'reliability-adapter'])
    parser.add_argument('--score-only', action='store_true')
    parser.add_argument('--blend-only', action='store_true')
    parser.add_argument('--blend-name', default='reliability-blended')
    parser.add_argument('--reviewed-weight', type=float, choices=(0.5, 0.75), default=0.5)
    args = parser.parse_args()
    torch.set_num_threads(2)
    conversation.OUTPUT = ROOT / 'runs' / 'conversation-compositional'
    if args.blend_only:
        blend_input(args.blend_name, args.reviewed_weight)
    else:
        for name in args.names:
            score_legacy(name)
        if not args.score_only:
            select(args.names)
