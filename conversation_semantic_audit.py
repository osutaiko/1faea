"""Change only a query operator while holding emoji facts and entity fixed."""

import json

import torch

import conversation
from conversation_data import ids, pack_history
from conversation_model import AtomicDecoder, SemanticAtomicDecoder, memory, render, visible
from emoji_lm_model import END
from run import ROOT


@torch.no_grad()
def audit(name, encoder_type):
    _, decoder, reader, _ = conversation.load(name, encoder_type)
    examples = []
    for color, place in [('🔵', '🏠'), ('🔴', '🏞️')]:
        facts = [(ids(('📌', '🚲', '🎨', color)), ids(('👍', '🚲', color))),
                 (ids(('📌', '🚲', '📍', place)), ids(('👍', '🚲', '📍', place)))]
        for reverse in (False, True):
            history = pack_history(list(reversed(facts)) if reverse else facts)
            for operator, expected in [('🎨', ids(('🚲', color))), ('📍', ids(('🚲', '📍', place)))]:
                query = ids(('❓', '🚲', operator))
                state, mask = memory([history], [query])
                output = decoder.respond(reader, state, mask)[0]
                example = dict(history=render(history), query=render(query), reply=render(output),
                               expected=render(expected), correct=END in output.tolist() and visible(output) == expected)
                examples.append(example)
                print(json.dumps(dict(model=name, **example), ensure_ascii=False), flush=True)
    report = dict(created=conversation.timestamp(), count=len(examples),
                  exact_reply_accuracy=sum(example['correct'] for example in examples) / len(examples),
                  operator_changes_reply=sum(examples[index]['reply'] != examples[index + 1]['reply']
                                             for index in range(0, len(examples), 2)),
                  operator_pairs=len(examples) // 2,
                  examples=examples)
    (conversation.OUTPUT / name / 'attribute-audit.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    torch.set_num_threads(2)
    conversation.OUTPUT = ROOT / 'runs' / 'conversation-compositional'
    for name, encoder_type in [('baseline', AtomicDecoder), ('selected', SemanticAtomicDecoder),
                               ('dialogue', SemanticAtomicDecoder)]:
        audit(name, encoder_type)
