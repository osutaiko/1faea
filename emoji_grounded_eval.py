"""Fixed, evaluation-only checks of literal answer concepts and relationships."""

import argparse
import json

import torch

from emoji_grounded_chat import load
from run import ROOT


# Human-authored checks, never used for training or checkpoint selection.
# Alternative symbols are explicit; these narrow checks are not general-chat accuracy.
CASES = [
    dict(question='Which animal meows?', groups=['🐈🐱'], forbidden='🐕🐶'),
    dict(question='Which animal barks?', groups=['🐕🐶'], forbidden='🐈🐱'),
    dict(question='What should I carry to stay dry in rain?', groups=['☂️☔'], forbidden=''),
    dict(question='Which fruit is yellow and curved?', groups=['🍌'], forbidden=''),
    dict(question='What do bees make?', groups=['🍯'], forbidden=''),
    dict(question='What planet do we live on?', groups=['🌍🌎🌏'], forbidden=''),
    dict(question='Show a dog chasing a cat, with the pursuer first.', groups=['🐕🐶', '🐈🐱'], forbidden='', ordered=True),
    dict(question='Show a cat chasing a dog, with the pursuer first.', groups=['🐈🐱', '🐕🐶'], forbidden='', ordered=True),
    dict(question='Show a person giving a gift to a robot, with the giver first.', groups=['🧑👤', '🎁', '🤖'], forbidden='', ordered=True),
    dict(question='Show a robot giving a gift to a person, with the giver first.', groups=['🤖', '🎁', '🧑👤'], forbidden='', ordered=True),
    dict(question='Say that smoking is forbidden.', groups=['🚭'], forbidden=''),
    dict(question='Say that bicycles are forbidden.', groups=['🚳'], forbidden=''),
]


def score(symbols, case):
    positions = [next((i for i, symbol in enumerate(symbols) if symbol in group), -1) for group in case['groups']]
    concepts = all(position >= 0 for position in positions)
    order = not case.get('ordered') or all(a < b for a, b in zip(positions, positions[1:]))
    return concepts and order and not any(symbol in case['forbidden'] for symbol in symbols)


def evaluate(name):
    model = load(name)
    rows = []
    for case in CASES:
        # Match actual catalog symbols, preserving sequence rather than Unicode codepoints.
        features, mask = model[2].text_inputs([case['question']])
        states = model[0](features, mask)[2]
        ids = model[1].generate(states)[0].tolist()
        terminated = model[1].end in ids
        ids = ids[:ids.index(model[1].end)] if model[1].end in ids else ids
        symbols = [model[3][index]['symbol'] for index in ids]
        rows.append(dict(**case, state=''.join(model[3][index]['symbol'] for index in states[0].tolist()),
                         reply=''.join(symbols), core_meanings=[model[3][index]['core_meaning'] for index in ids],
                         terminated=terminated, passed=score(symbols, case)))
    report = dict(name=name, passed=sum(row['passed'] for row in rows), total=len(rows),
                  evaluation_only=True, used_for_selection=False,
                  limitation='Narrow human-specified literal concepts and ordering; not factual or general-chat validation.', examples=rows)
    path = ROOT / 'runs' / 'emoji-grounded' / name / 'meaning-evaluation.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'examples'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='objective-v3')
    args = parser.parse_args()
    torch.set_num_threads(2)
    with torch.no_grad():
        evaluate(args.name)
