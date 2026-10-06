"""Short ordered-copy ablation, not language or question-answer training."""

import copy
import hashlib
import json
import time

import torch
from torch import nn
from torch.nn import functional as F

from emoji_grounded_model import GroundedReply
from emoji_quick_checks import generate
from run import ROOT


class PositionedMemory(nn.Module):
    def __init__(self, projection, slots, width):
        super().__init__()
        self.projection = projection
        self.positions = nn.Embedding(slots, width)
        nn.init.normal_(self.positions.weight, std=.02)

    def forward(self, values):
        return self.projection(values) + self.positions(torch.arange(values.shape[1], device=values.device))


def main():
    torch.set_num_threads(2)
    torch.manual_seed(719)
    started = time.monotonic()
    base = ROOT / 'runs/emoji-general/five-hour/runtime.pt'
    checkpoint = torch.load(base, weights_only=True)
    sha = hashlib.sha256(base.read_bytes()).hexdigest()
    original = GroundedReply(checkpoint['reply']['vectors'], **checkpoint['reply_config']).eval()
    original.load_state_dict(checkpoint['reply'])
    ordered = copy.deepcopy(original)
    ordered.memory = PositionedMemory(ordered.memory, ordered.config['slots'], ordered.config['width'])
    cached = torch.load(ROOT / 'runs/emoji-pretrained/copy-v1/train-states.pt', weights_only=True)
    pairs, seen = [], set()
    for row in cached['states'].tolist():
        unique = list(dict.fromkeys(row))
        if len(unique) >= 2 and tuple(sorted(unique[:2])) not in seen:
            pairs.append(unique[:2]); seen.add(tuple(sorted(unique[:2])))
        if len(pairs) == 16:
            break
    states = torch.tensor([row for pair in pairs[:8] for row in (pair, pair[::-1])])
    held_out = torch.tensor([row for pair in pairs[8:] for row in (pair, pair[::-1])])
    prefix = torch.cat((torch.full((len(states), 1), original.end), states), 1)
    labels = torch.cat((states, torch.full((len(states), 1), original.end)), 1)
    results = {}
    for name, model in [('unordered', copy.deepcopy(original)), ('positioned', ordered)]:
        model.eval()
        values = model.vectors[states]
        with torch.no_grad():
            initial = model(values, prefix[:, :1])[:, 0]
            reversed_logits = model(values.flip(1), prefix[:, :1])[:, 0]
            initial_difference = (initial - reversed_logits).abs().max().item()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        step, training_started = 0, time.monotonic()
        while step < 200 and time.monotonic() - started < 90:
            step += 1
            model.train()
            loss = F.cross_entropy(model(values, prefix).flatten(0, 1), labels.flatten())
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        model.eval()
        generated = generate(model, values)
        held_out_replies = generate(model, model.vectors[held_out])
        expected = states.tolist()
        results[name] = dict(updates=step, training_seconds=time.monotonic() - training_started,
                             initial_reversed_logits_difference=initial_difference,
                             exact_ordered_copies=sum(a == b for a, b in zip(generated, expected)), total=len(expected),
                             held_out_exact=sum(a == b for a, b in zip(held_out_replies, held_out.tolist())),
                             held_out_total=len(held_out),
                             examples=[dict(input=''.join(checkpoint['meanings'][i]['symbol'] for i in target),
                                            reply=''.join(checkpoint['meanings'][i]['symbol'] for i in output))
                                       for target, output in zip(expected, generated)])
    report = dict(checkpoint_sha256=sha, checkpoint_unchanged=hashlib.sha256(base.read_bytes()).hexdigest() == sha,
                  diagnostic='Paired two-ID copying in both orders, derived from existing cached state IDs. No English Q&A labels.',
                  results=results, elapsed_seconds=time.monotonic() - started, promoted=False,
                  limitation='Identity/order copying and memorization only; not readable semantic roles, factual answering or general chat.')
    (ROOT / 'data/emoji-grounded/position-check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'results'}, indent=2))
    print(json.dumps({key: {k: v for k, v in value.items() if k != 'examples'} for key, value in results.items()}, indent=2))


if __name__ == '__main__':
    main()
