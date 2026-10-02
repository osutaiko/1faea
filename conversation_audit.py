"""Audit generated conversation history and counterfactual emoji memories."""

import argparse
import json

import torch

import conversation
from conversation_data import IDS, pack_history
from conversation_model import AtomicDecoder, SemanticAtomicDecoder, EmojiConversation, memory, render, visible
from emoji_lm_model import END
from run import ROOT


EPISODES = {
    'preference_correction': [
        ('Hello!', '👋', '👋😊'),
        ('I like pizza.', '💖🍕', '👍🍕'),
        ('What do I like?', '🧠💖❓', '💖🍕'),
        ('Actually, I prefer sushi instead.', '🔄💖🍣', '👌💖🍣'),
        ('What do I like now?', '🧠💖❓', '💖🍣'),
    ],
    'mood_between_memory_turns': [
        ('I like bicycles.', '💖🚲', '👍🚲'),
        ('I feel sad.', '😔', '🫂💛👂'),
        ('What do I like?', '🧠💖❓', '💖🚲'),
    ],
    'two_colors': [
        ('The 🚲 is blue.', '📌🚲🎨🔵', '👍🚲🔵'),
        ('The ☂️ is red.', '📌☂️🎨🔴', '👍☂️🔴'),
        ('What color is the ☂️?', '❓☂️🎨', '☂️🔴'),
        ('What color is the 🚲?', '❓🚲🎨', '🚲🔵'),
    ],
    'location_update': [
        ('The 🚲 is at the house.', '📌🚲📍🏠', '👍🚲📍🏠'),
        ('The 🚲 moved to the park.', '🔄🚲📍🏞️', '👌🚲📍🏞️'),
        ('Where is the 🚲?', '❓🚲📍', '🚲📍🏞️'),
    ],
    'no_memory': [('What do I like?', '🧠💖❓', '🤔❓')],
    'small_sum': [('What is 2 plus 3?', '🔢2️⃣➕3️⃣', '5️⃣')],
}


def symbol_ids(text):
    # Complete emoji sequences, longest first: joined forms stay atomic.
    result = []
    while text:
        symbol = next(symbol for symbol in sorted(IDS, key=len, reverse=True) if text.startswith(symbol))
        result.append(IDS[symbol])
        text = text[len(symbol):]
    return result


@torch.no_grad()
def audit(args):
    model_type = SemanticAtomicDecoder if args.semantic else AtomicDecoder
    encoder, decoder, reader, _ = conversation.load(args.name, model_type)
    traces = []
    for episode, turns in EPISODES.items():
        session = EmojiConversation(encoder, decoder, reader)
        for text, expected_state, expected_reply in turns:
            output = session.turn(text)
            item = dict(episode=episode, text=text, state=render(output['state']), reply=render(output['reply']),
                        expected_state=expected_state, expected_reply=expected_reply,
                        state_correct=output['state_terminated'] and output['state'] == symbol_ids(expected_state),
                        reply_correct=output['reply_terminated'] and output['reply'] == symbol_ids(expected_reply))
            traces.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
    counterfactuals = []
    # Swap the same entities' facts, query either position, and reverse fact order.
    for colors in [('🔵', '🔴'), ('🔴', '🔵')]:
        facts = [('🚲', colors[0]), ('☂️', colors[1])]
        for reverse in (False, True):
            ordered = list(reversed(facts)) if reverse else facts
            history = pack_history([(symbol_ids('📌' + entity + '🎨' + color), symbol_ids('👍' + entity + color))
                                    for entity, color in ordered])
            for entity, color in facts:
                state, mask = memory([history], [symbol_ids('❓' + entity + '🎨')])
                output = decoder.respond(reader, state, mask)[0]
                item = dict(history=render(history), query='❓' + entity + '🎨', reply=render(output),
                            expected=entity + color,
                            correct=END in output.tolist() and visible(output) == symbol_ids(entity + color))
                counterfactuals.append(item)
                print(json.dumps(item, ensure_ascii=False), flush=True)
    # Public natural prompts are an unscored qualitative audit. No assistant text
    # is passed to training or inference, and each prompt begins a fresh session.
    public = []
    source = ROOT / 'data' / 'conversation' / 'everyday-test_sft.jsonl'
    for line in source.read_text(encoding='utf-8').splitlines()[:12]:
        row = json.loads(line)
        text = next(message['content'] for message in row['messages'][1:] if message['role'] == 'user')
        output = EmojiConversation(encoder, decoder, reader).turn(text)
        item = dict(topic=row['full_topic'], text=text, state=render(output['state']), reply=render(output['reply']))
        public.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    result = dict(created=conversation.timestamp(), live_turns=len(traces),
                  live_state_accuracy=sum(row['state_correct'] for row in traces) / len(traces),
                  live_reply_accuracy=sum(row['reply_correct'] for row in traces) / len(traces),
                  counterfactual_accuracy=sum(row['correct'] for row in counterfactuals) / len(counterfactuals),
                  live=traces, counterfactuals=counterfactuals, unscored_public_prompts=public)
    (conversation.OUTPUT / args.name / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='baseline')
    parser.add_argument('--semantic', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    conversation.OUTPUT = ROOT / 'runs' / 'conversation-compositional'
    audit(args)
