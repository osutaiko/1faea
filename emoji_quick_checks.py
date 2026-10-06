"""Bounded diagnostics of the active baseline; never replaces its checkpoint."""

import copy
import hashlib
import json
import time

import torch
from torch.nn import functional as F

from emoji_grounded_model import GroundedReply
from run import ROOT


@torch.no_grad()
def generate(reply, values, minimum=0):
    prefix = torch.full((len(values), 1), reply.end, dtype=torch.long)
    finished = torch.zeros(len(values), dtype=torch.bool)
    for step in range(reply.config['slots'] + 1):
        logits = reply(values, prefix)[:, -1]
        if step < minimum:
            logits[:, reply.end] = -torch.inf
        selected = logits.argmax(-1).masked_fill(finished, reply.end)
        prefix = torch.cat((prefix, selected[:, None]), 1)
        finished |= selected == reply.end
        if finished.all():
            break
    rows = []
    for ids in prefix[:, 1:].tolist():
        rows.append(ids[:ids.index(reply.end)] if reply.end in ids else ids)
    return rows


def sequences(labels, end):
    return [[index for index in row if 0 <= index < end] for row in labels.tolist()]


def main():
    torch.set_num_threads(2)
    torch.manual_seed(719)
    started = time.monotonic()
    base = ROOT / 'runs/emoji-general/five-hour'
    checkpoint = torch.load(base / 'runtime.pt', weights_only=True)
    sha = hashlib.sha256((base / 'runtime.pt').read_bytes()).hexdigest()
    cached = ROOT / 'runs/emoji-pretrained/copy-v1'
    if json.loads((cached / 'manifest.json').read_text())['encoder_checkpoint_sha256'] != sha:
        raise ValueError('Cached states do not match the active baseline')
    reply = GroundedReply(checkpoint['reply']['vectors'], **checkpoint['reply_config']).eval()
    reply.load_state_dict(checkpoint['reply'])
    test = torch.load(cached / 'test-states.pt', weights_only=True)
    states = test['states'][:30]
    values = reply.vectors[states]
    controls = dict(real=values, reversed=values.flip(1), unrelated=values.roll(15, 0), empty=torch.zeros_like(values))
    replies = {key: generate(reply, value) for key, value in controls.items()}
    longer = generate(reply, values, minimum=3)
    with torch.no_grad():
        prefix = torch.full((len(states), 1), reply.end, dtype=torch.long)
        first = reply(values, prefix)[:, 0]
        reversed_first = reply(values.flip(1), prefix)[:, 0]
        stop_probability = first.softmax(-1)[:, reply.end]
    changes = {key: sum(a != b for a, b in zip(replies['real'], rows)) for key, rows in replies.items() if key != 'real'}
    questions = [json.loads(line) for line in (base / 'test.jsonl').read_text(encoding='utf-8').splitlines()][:30]
    examples = []
    for index, row in enumerate(questions[:10]):
        render = lambda ids: ''.join(checkpoint['meanings'][i]['symbol'] for i in ids)
        examples.append(dict(question=row['question'], **{key: render(rows[index]) for key, rows in replies.items()},
                             minimum_three=render(longer[index])))
    train = torch.load(cached / 'train-states.pt', weights_only=True)
    # Distinct cached inputs avoid knowingly conflicting identical-input targets.
    indices, seen = [], set()
    for index, row in enumerate(train['states'].tolist()):
        if tuple(row) not in seen:
            indices.append(index); seen.add(tuple(row))
        if len(indices) == 16:
            break
    indices = torch.tensor(indices)
    tiny = copy.deepcopy(reply)
    optimizer = torch.optim.AdamW(tiny.parameters(), lr=.001)
    tiny_values = tiny.vectors[train['states'][indices]]
    prefix, labels = train['prefix'][indices], train['labels'][indices]
    expected = sequences(labels, tiny.end)
    with torch.no_grad():
        initial_loss = F.cross_entropy(tiny(tiny_values, prefix).flatten(0, 1), labels.flatten(), ignore_index=-100).item()
    initial_exact = sum(a == b for a, b in zip(generate(tiny, tiny_values), expected))
    train_started = time.monotonic()
    step = 0
    while step < 150 and time.monotonic() - train_started < 90:
        step += 1
        tiny.train()
        loss = F.cross_entropy(tiny(tiny_values, prefix).flatten(0, 1), labels.flatten(), ignore_index=-100)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(tiny.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    tiny.eval()
    with torch.no_grad():
        final_loss = F.cross_entropy(tiny(tiny_values, prefix).flatten(0, 1), labels.flatten(), ignore_index=-100).item()
    final_exact = sum(a == b for a, b in zip(generate(tiny, tiny_values), expected))
    audit = json.loads((ROOT / 'data/emoji-grounded/target-quality-audit.json').read_text(encoding='utf-8'))
    report = dict(checkpoint_sha256=sha, checkpoint_unchanged=hashlib.sha256((base / 'runtime.pt').read_bytes()).hexdigest() == sha,
                  test_examples=len(states), changed_replies=changes,
                  reversed_first_logits_max_difference=(first - reversed_first).abs().max().item(),
                  mean_first_stop_probability=stop_probability.mean().item(),
                  reply_lengths=[len(row) for row in replies['real']], forced_reply_lengths=[len(row) for row in longer],
                  examples=examples, tiny_set=dict(examples=16, updates=step, seconds=time.monotonic() - train_started,
                  initial_teacher_forced_loss=initial_loss, final_teacher_forced_loss=final_loss,
                  initial_exact_replies=initial_exact, final_exact_replies=final_exact,
                  limitation='Memorization of existing weak code targets; not semantic answering or held-out improvement.'),
                  target_review=audit['qualitative_review'], elapsed_seconds=time.monotonic() - started,
                  runtime_promoted=False)
    path = ROOT / 'data/emoji-grounded/quick-checks.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key not in ('examples', 'target_review')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
