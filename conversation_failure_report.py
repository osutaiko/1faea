"""Describe frozen-checkpoint failures without using test results for training."""

from collections import defaultdict
import json

from run import ROOT


DIRECTORY = ROOT / 'runs' / 'conversation-compositional' / 'reliability-selected'


def report():
    live = json.loads((DIRECTORY / 'live-numbers-test.json').read_text(encoding='utf-8'))
    turns = defaultdict(lambda: dict(count=0, correct=0, state_correct=0, evicted=0))
    positions = defaultdict(int)
    for row in live['examples']:
        position = positions[row['episode']]
        positions[row['episode']] += 1
        group = turns[position + 1]
        group['count'] += 1
        group['correct'] += row['correct']
        group['state_correct'] += row['state'] == row['expected_state']
        group['evicted'] += row['dropped_turns'] > 0
    scope = json.loads((DIRECTORY / 'public-scope-test.json').read_text(encoding='utf-8'))
    incorrect_states = sum(row['state'] != row['expected_state'] for row in live['examples'])
    correct_state_wrong_reply = sum(row['state'] == row['expected_state'] and not row['correct']
                                    for row in live['examples'])
    result = dict(test_used_for_training=False, turns=turns, incorrect_states=incorrect_states,
                  correct_state_wrong_reply=correct_state_wrong_reply,
                  unsupported_guesses=[row for row in scope['examples']
                                       if row['skill'] == 'public_unsupported' and not row['correct']],
                  next_experiments=[
                      'Train on generated emoji histories, with separate validation entities and wording.',
                      'Train the reply decoder to distinguish user facts from incorrect assistant replies.',
                      'Expand reviewed unsupported requests and calibrate an explicit unknown state.',
                      'Evaluate a larger pretrained input reader while preserving hard emoji boundaries.',
                      'Use a fresh untouched test set for the next checkpoint; retain these tests as regression cases.'])
    (DIRECTORY / 'failure-analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 🫪 Failure analysis 🔍', '',
             'This analysis describes the frozen checkpoint. It does not select or train another model.', '',
             '| Conversation turn | Correct replies | Correct states | Turns with eviction |',
             '| --- | ---: | ---: | ---: |']
    for position, group in turns.items():
        lines.append(f"| {position} | {group['correct']}/{group['count']} | "
                     f"{group['state_correct']}/{group['count']} | {group['evicted']}/{group['count']} |")
    lines.extend(['', f'{incorrect_states} turns have an incorrect input state. '
                  f'{correct_state_wrong_reply} turns have the intended state but an incorrect reply.', '',
                  'These counts are observations, not mutually exclusive causal explanations. '
                  'Generated history differs from the supplied history used in training.', '',
                  'The stress sessions run three contrast queries sequentially. The authored training '
                  'continuation includes one query. Memory eviction occurs in only three of 384 turns.', '',
                  'On supplied-history test rows, replies given the intended state reach 94.95%. '
                  'The input state is correct on only 63.70%. Input understanding is another bottleneck.', '',
                  '## Next experiments 🧪', ''])
    lines.extend(f'{index}. {text}' for index, text in enumerate(result['next_experiments'], 1))
    (DIRECTORY / 'FAILURE_ANALYSIS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    report()
