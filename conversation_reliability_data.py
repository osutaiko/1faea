"""Mixed-attribute dialogues and directly authored everyday emoji replies."""

import json
import random

from conversation_composition import COLORS, PLACES, DIGITS
from conversation_data import IDS, SKILLS, pack_history, record
from conversation_model import MEMORY_LIMIT
from emoji_catalog import ALPHABET
from run import ROOT


DIRECTORY = ROOT / 'data' / 'conversation' / 'reliability'
TARGET_REVISION = 2

# These targets are authored as emojis, without an English assistant answer.
TOPICS = (
    ('cleaning', ('🎯', '🧹'), ('🗑️', '🧹', '🧽', '✨'),
     ('Help me clean my room.', 'How should I tidy up?', 'I need a cleaning routine.', 'Give me steps to clean the house.'),
     ('Suggest a way to tidy my home.', 'How can I keep my room clean?'),
     ('My room is a mess. Where do I start?', 'I want to put my home in order.')),
    ('organization', ('🎯', '🗂️'), ('📋', '🗂️', '✅'),
     ('Help me organize my tasks.', 'I need a to-do list.', 'How do I organize my work?', 'Give me a task organization plan.'),
     ('How should I sort my tasks?', 'Help me keep track of my work.'),
     ('I have too many tasks to track.', 'I need a system for my daily tasks.')),
    ('language_learning', ('🎯', '🗣️'), ('📚', '👂', '🗣️', '🔁'),
     ('Help me learn a language.', 'How can I practice a foreign language?', 'Give me language learning steps.', 'I want to learn another language.'),
     ('Suggest a language practice routine.', 'How should I study a new language?'),
     ('I want to speak a second language.', 'I need to improve my foreign language skills.')),
    ('writing', ('🎯', '✍️'), ('💡', '📝', '✍️', '🔍'),
     ('Help me write a story.', 'Give me a writing plan.', 'How do I start writing?', 'I want to write something.'),
     ('Suggest steps for writing an essay.', 'Help me draft a short story.'),
     ('I have a blank page and want to write.', 'How do I turn my idea into a draft?')),
    ('repair', ('🎯', '🔧'), ('🔍', '🔧', '🧪', '✅'),
     ('Give me a basic repair plan.', 'How should I approach fixing something?', 'I want a general repair checklist.', 'Help me plan a repair.'),
     ('Suggest a process for repairing an object.', 'How can I organize a repair job?'),
     ('What are the stages of a simple repair?', 'I need a systematic way to fix an object.')),
    ('coding', ('🎯', '💻'), ('🎯', '📝', '💻', '🧪'),
     ('Help me learn programming.', 'How should I start coding?', 'Give me a coding learning plan.', 'I want to learn to code.'),
     ('Suggest a beginner programming routine.', 'Help me start learning software development.'),
     ('I am new to programming. What steps should I take?', 'I want to build my first small program.')),
    ('debugging', ('🎯', '🐛'), ('🐛', '🔍', '🔧', '🧪', '✅'),
     ('How do I debug a program?', 'Give me debugging steps.', 'Help me find a software bug.', 'I need a debugging checklist.'),
     ('Suggest a method for finding coding errors.', 'Help me diagnose a broken program.'),
     ('My code has a bug. How should I investigate?', 'How do I locate and verify a software fix?')),
    ('teamwork', ('🎯', '🤝'), ('💬', '👂', '🤝', '🎯'),
     ('How can we work better as a team?', 'Give me teamwork advice.', 'Help me collaborate.', 'I need a team cooperation plan.'),
     ('Suggest ways to improve collaboration.', 'How can my team cooperate better?'),
     ('We need to coordinate our work as a group.', 'What helps people work together effectively?')),
    ('focus', ('🎯', '🧠'), ('📵', '🎯', '⏱️', '☕'),
     ('Help me focus.', 'How do I avoid distractions?', 'Give me a concentration plan.', 'I want to stay focused on my work.'),
     ('Suggest a routine for concentrating.', 'How can I improve my focus?'),
     ('I keep getting distracted while working.', 'I need an uninterrupted work session.')),
    ('creativity', ('🎯', '💡'), ('💡', '📝', '🧪', '🎨'),
     ('Help me brainstorm.', 'How can I generate ideas?', 'Give me a creative thinking routine.', 'I need new ideas.'),
     ('Suggest a brainstorming process.', 'How can I come up with something creative?'),
     ('I am stuck and need fresh ideas.', 'What steps help with exploring creative possibilities?')),
    ('walking', ('🎯', '🚶'), ('👟', '💧', '🚶', '🌳'),
     ('Help me plan a walk.', 'What should I take on a walk?', 'I want to go walking.', 'Give me a walking preparation plan.'),
     ('Suggest steps before going for a walk.', 'Help me prepare for a stroll.'),
     ('I want to spend time walking outdoors.', 'What basics do I need for a casual walk?')),
    ('gravity', ('❓', '🌍', '⬇️'), ('🪨', '⬇️', '🌍'),
     ('What is gravity?', 'Why do things fall down?', 'Explain gravity using emojis.', 'What pulls objects toward Earth?'),
     ('Why does a dropped object fall?', 'How does gravity affect objects?'),
     ('Why does a ball come back down after a throw?', 'What keeps us on the ground?')),
    ('computer', ('❓', '💻', '🧠'), ('⌨️', '➡️', '🧠', '➡️', '🖥️'),
     ('How does a computer work?', 'Explain a computer using emojis.', 'What does a computer do?', 'Show the basic input and output of a computer.'),
     ('Describe basic computer processing.', 'How does a computer process input?'),
     ('What happens between typing and seeing a result?', 'Show input, processing, and output with emojis.')),
    ('internet', ('❓', '🌐'), ('💻', '↔️', '🌐', '↔️', '💻'),
     ('What is the internet?', 'Explain the internet using emojis.', 'How are computers connected online?', 'Show an internet connection.'),
     ('What connects devices across the internet?', 'Describe online communication.'),
     ('How do devices share information over a network?', 'Show computers communicating around the world.')),
    ('electricity', ('❓', '⚡'), ('🔋', '➡️', '⚡', '➡️', '💡'),
     ('What is electricity?', 'Explain electricity using emojis.', 'Show electrical energy powering a light.', 'What powers an electric lamp?'),
     ('Show a battery powering a bulb.', 'Describe electrical power with emojis.'),
     ('How does battery energy reach a light?', 'Show a simple electrical energy flow.')),
    ('algorithm', ('❓', '💻', '🔢'), ('📥', '➡️', '📋', '➡️', '📤'),
     ('What is an algorithm?', 'Explain an algorithm using emojis.', 'Show a step-by-step computing process.', 'What does an algorithm do?'),
     ('Describe input, steps, and output.', 'Show how a computing procedure works.'),
     ('What turns an input into a result through ordered steps?', 'Show the structure of a computational procedure.')),
)


def topic_rows(split):
    rows = []
    for skill, state, reply, train, validation, test in TOPICS:
        phrases = dict(train=train, validation=validation, test=test)[split]
        for text in phrases:
            rows.append(record(text, state, reply, skill))
            if split == 'train':
                for variant in (text.lower(), 'Please, ' + text, text + ' Answer with emojis.'):
                    rows.append(record(variant, state, reply, skill))
    return rows


def entity_splits():
    return json.loads((DIRECTORY / 'entities.json').read_text(encoding='utf-8'))['splits']


def episodes(count, seed, split):
    rng = random.Random(seed)
    pool = entity_splits()[split]
    result = []
    for index in range(count):
        a, b, c = [ALPHABET[entity] for entity in rng.sample(pool, 3)]
        color, other_color, updated_color = [symbol for _, symbol in rng.sample(COLORS, 3)]
        place, updated_place = [symbol for _, symbol in rng.sample(PLACES, 2)]
        quantity, updated_quantity = rng.sample(DIGITS, 2)
        color_name = dict((symbol, name) for name, symbol in COLORS)[color]
        other_name = dict((symbol, name) for name, symbol in COLORS)[other_color]
        place_name = dict((symbol, name) for name, symbol in PLACES)[place]
        history, rows = [], []

        def turn(text, state, reply, skill, wrong=None):
            item = record(text, state, reply, skill, pack_history(history))
            if len(item['history']) + 1 + len(item['state']) > MEMORY_LIMIT:
                raise ValueError('Reliability episode exceeds the discrete memory budget')
            rows.append(item)
            history.append((item['state'], [IDS[symbol] for symbol in wrong] if wrong else item['reply']))

        facts = [
            (dict(train=f'The {a} is {color_name}.', validation=f'The {a} has a {color_name} color.',
                  test=f'Remember this: my {a} is colored {color_name}.')[split],
             ('📌', a, '🎨', color), ('👍', a, color), 'reliability_color_fact', ('👍', a, updated_color)),
            (dict(train=f'The {a} is at the {place_name}.', validation=f'The {a} is located at the {place_name}.',
                  test=f'I keep my {a} at the {place_name}.')[split],
             ('📌', a, '📍', place), ('👍', a, '📍', place), 'reliability_location_fact', ('👍', a, '📍', updated_place)),
            (dict(train=f'I have {quantity} {a}.', validation=f'The number of {a} is {quantity}.',
                  test=f'My collection has {quantity} {a}.')[split],
             ('📌', a, '🔢', quantity), ('👍', a, '🔢', quantity), 'quantity_fact', ('👍', a, '🔢', updated_quantity)),
            (f'The {b} is {other_name}.', ('📌', b, '🎨', other_color), ('👍', b, other_color),
             'reliability_other_entity', ('👍', a, other_color)),
        ]
        rng.shuffle(facts)
        for text, state, reply, skill, wrong in facts:
            turn(text, state, reply, skill, wrong if index % 3 == 0 else None)

        # All three queries share the same facts. Operator choice changes the target.
        snapshot = pack_history(history)
        queries = [
            (dict(train=f'What color is the {a}?', validation=f'Can you recall the color of the {a}?',
                  test=f'Tell me which color my {a} has.')[split], ('❓', a, '🎨'), (a, color)),
            (dict(train=f'Where is the {a}?', validation=f'Can you recall where the {a} is?',
                  test=f'Tell me the location of my {a}.')[split], ('❓', a, '📍'), (a, '📍', place)),
            (dict(train=f'How many {a} do I have?', validation=f'What is the number of {a}?',
                  test=f'Remind me of my {a} count.')[split], ('❓', a, '🔢'), (a, '🔢', quantity)),
        ]
        rng.shuffle(queries)
        for text, state, reply in queries:
            rows.append(record(text, state, reply, 'attribute_operator', snapshot))
        # A real continuation has one query, then a correction and a fresh query.
        text, state, reply = queries[0]
        history.append(([IDS[symbol] for symbol in state], [IDS[symbol] for symbol in reply]))
        _, social_state, social_reply, training, validation, test = rng.choice(SKILLS[:24])
        social_text = rng.choice(dict(train=training, validation=validation, test=test)[split])
        turn(social_text, social_state, social_reply, 'memory_distractor')
        operator = rng.choice(('🎨', '📍', '🔢'))
        value = dict(zip(('🎨', '📍', '🔢'), (updated_color, updated_place, updated_quantity)))[operator]
        old = dict(zip(('🎨', '📍', '🔢'), (color, place, quantity)))[operator]
        if operator == '🎨':
            name = dict((symbol, name) for name, symbol in COLORS)[value]
            text = dict(train=f'Update: the {a} is now {name}.', validation=f'Correction: the {a} is {name} now.',
                        test=f'The color of my {a} changed to {name}.')[split]
        elif operator == '📍':
            name = dict((symbol, name) for name, symbol in PLACES)[value]
            text = dict(train=f'The {a} moved to the {name}.', validation=f'Update: the {a} is now at the {name}.',
                        test=f'I moved my {a} to the {name}.')[split]
        else:
            text = dict(train=f'Update: I now have {value} {a}.', validation=f'The number of {a} is now {value}.',
                        test=f'My {a} count changed to {value}.')[split]
        acknowledgment = ('👌', a, value) if operator == '🎨' else ('👌', a, operator, value)
        wrong = ('👍', a, old) if operator == '🎨' else ('👍', a, operator, old)
        turn(text, ('🔄', a, operator, value), acknowledgment, 'attribute_update', wrong if index % 2 == 0 else None)
        for text, state, reply in queries:
            if state[-1] == operator:
                expected = (a, value) if operator == '🎨' else (a, operator, value)
                turn(text, state, expected, 'updated_attribute_recall')
                break
        turn(f'What color is the {c}?', ('❓', c, '🎨'), (c, '❓'), 'missing_entity')
        turn(f'Where is the {b}?', ('❓', b, '📍'), (b, '❓'), 'missing_attribute')
        result.append(rows)
    return result


def dataset():
    groups = {split: episodes(count, seed, split) for split, count, seed in
              [('train', 96, 409), ('validation', 16, 419), ('test', 32, 421)]}
    rows = {split: [row for episode in groups[split] for row in episode] + topic_rows(split)
            for split in groups}
    return rows, groups


def write():
    rows, groups = dataset()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    for split, records in rows.items():
        (DIRECTORY / f'{split}.jsonl').write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records), encoding='utf-8')
    (DIRECTORY / 'episodes.json').write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding='utf-8')
    metadata = dict(seeds=dict(entities=401, train=409, validation=419, test=421),
                    target_revision=TARGET_REVISION, entity_splits='frozen entities.json',
                    counts={split: len(records) for split, records in rows.items()},
                    topics=len(TOPICS), targets='directly authored emoji symbols',
                    evaluation='supplemental entity pools are disjoint; test templates are held out')
    (DIRECTORY / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata), flush=True)
    return rows, groups


if __name__ == '__main__':
    write()
