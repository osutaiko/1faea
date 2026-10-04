"""Separate state errors, reply feedback, and full-transcript memory shift."""

import json

import torch

import conversation
from conversation_data import USER, ASSISTANT, SEPARATOR
from conversation_model import SemanticAtomicDecoder, memory, visible
from conversation_reliability_data import dataset
from emoji_lm_model import END
from run import ROOT


@torch.no_grad()
def audit():
    encoder, decoder, reader, _ = conversation.load('reliability-selected', SemanticAtomicDecoder)
    episodes = dataset()[1]['validation']
    generated_states = []
    for position in range(12):
        generated_states.append([visible(tokens) for tokens in encoder.encode(
            reader, [episode[position]['text'] for episode in episodes])])
    results = {}
    for mode in ('gold_states_gold_history', 'gold_states_reply_feedback', 'generated_states_gold_replies'):
        histories = [[] for _ in episodes]
        lengths = [[] for _ in episodes]
        count, correct, terminated, dropped = 0, 0, 0, 0
        per_turn = []
        for position in range(12):
            rows = [episode[position] for episode in episodes]
            states = generated_states[position] if mode == 'generated_states_gold_replies' else [row['state'] for row in rows]
            for history, turns, state in zip(histories, lengths, states):
                while len(history) + 1 + len(state) > 128:
                    del history[:turns.pop(0)]
                    dropped += 1
            source, mask = memory(histories, states)
            outputs = decoder.respond(reader, source, mask)
            turn_correct = 0
            for history, turns, row, state, output in zip(histories, lengths, rows, states, outputs):
                reply = visible(output)
                success = END in output.tolist() and reply == row['reply']
                count += 1
                correct += success
                turn_correct += success
                terminated += END in output.tolist()
                stored_reply = reply if mode == 'gold_states_reply_feedback' else row['reply']
                turn = [USER, *state, SEPARATOR, ASSISTANT, *stored_reply, SEPARATOR]
                history.extend(turn)
                turns.append(len(turn))
                while len(history) > 128:
                    del history[:turns.pop(0)]
                    dropped += 1
            per_turn.append(turn_correct)
        results[mode] = dict(count=count, exact_replies=correct, reply_accuracy=correct / count,
                             terminated=terminated, evicted_turns=dropped, correct_by_turn=per_turn)
        print(json.dumps(dict(mode=mode, **results[mode])), flush=True)
    report = dict(created=conversation.timestamp(), split='validation', episodes=len(episodes),
                  selection_or_training=False, note='Counterfactual diagnostic; gold values replace selected feedback paths.',
                  results=results)
    (conversation.OUTPUT / 'reliability-selected' / 'memory-audit.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    torch.set_num_threads(2)
    conversation.OUTPUT = ROOT / 'runs' / 'conversation-compositional'
    audit()
