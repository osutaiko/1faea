"""Compositional dialogue supervision: facts, memory, choices, and small sums."""

import json
import random

from conversation_data import DIRECTORY, IDS, USER, ASSISTANT, SEPARATOR, pack_history, record, splits
from emoji_catalog import ALPHABET, CATALOG, ENTITIES


COLORS = (('red', '🔴'), ('blue', '🔵'), ('green', '🟢'), ('yellow', '🟡'), ('black', '⚫'), ('white', '⚪'))
PLACES = (('house', '🏠'), ('school', '🏫'), ('office', '🏢'), ('park', '🏞️'), ('beach', '🏖️'))
DIGITS = tuple(str(digit) + '\ufe0f\u20e3' for digit in range(10))
NUMBER_WORDS = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')


def label(entity, literal):
    return ALPHABET[entity] if literal else CATALOG[entity]['unicode_name']


def episode_history(records):
    return pack_history([(item['state'], item['reply']) for item in records])


def composition_splits():
    data = splits()
    # Paint/art and an unresolved color query need distinct semantic states.
    for rows in data.values():
        for row in rows:
            if row['skill'] == 'art':
                row['state'] = [IDS['❓'], IDS['🖌️']]
    reserved = {IDS[symbol] for symbol in (*DIGITS, '💖', '💔', '🔄', '🧠', '🎨', '📍', '📌', '🔢', '➕', '❓', '👍', '👌', '✅', '🚫')}
    reserved.update((USER, ASSISTANT, SEPARATOR))
    reserved.update(IDS[symbol] for _, symbol in (*COLORS, *PLACES))
    entities = [entity for entity in ENTITIES if entity not in reserved]
    for name, count, seed in [('train', 384, 211), ('validation', 48, 223), ('test', 96, 227)]:
        rng = random.Random(seed)
        for index in range(count):
            a, b, c = rng.sample(entities, 3)
            literal = index % 2 == 0
            a_name, b_name, c_name = [label(entity, literal) for entity in (a, b, c)]
            a_symbol, b_symbol, c_symbol = [ALPHABET[entity] for entity in (a, b, c)]
            like_text = dict(train=f'I like {a_name}.', validation=f'I love {a_name}.', test=f'I enjoy {a_name}.')[name]
            like = record(like_text, ('💖', a_symbol), ('👍', a_symbol), 'mixed_preferences')
            hate_text = dict(train=f'I dislike {b_name}.', validation=f'I hate {b_name}.', test=f'I do not enjoy {b_name}.')[name]
            dislike = record(hate_text, ('💔', b_symbol), ('👌', '🚫', b_symbol), 'mixed_preferences', episode_history([like]))
            data[name].extend((like, dislike))
            questions = dict(train=('What do I like?', 'What do I dislike?'),
                             validation=('What is my favorite?', 'What do I not like?'),
                             test=('Which one did I like?', 'Which one did I dislike?'))[name]
            history = episode_history([like, dislike])
            data[name].append(record(questions[0], ('🧠', '💖', '❓'), ('💖', a_symbol), 'selective_recall', history))
            data[name].append(record(questions[1], ('🧠', '💔', '❓'), ('💔', b_symbol), 'selective_recall', history))
            choose_text = dict(train=f'I like {a_name} but dislike {b_name}. Which should I choose?',
                               validation=f'I dislike {b_name} and like {a_name}. Which one should I pick?',
                               test=f'I enjoy {a_name} and do not enjoy {b_name}. Help me choose.')[name]
            data[name].append(record(choose_text, ('💖', a_symbol, '💔', b_symbol, '⚖️', '❓'),
                                     ('✅', a_symbol), 'preference_choice'))

            color_a, color_b = rng.sample(COLORS, 2)
            color_text = dict(train=f'The {a_name} is {color_a[0]}.',
                              validation=f'The color of the {a_name} is {color_a[0]}.',
                              test=f'My {a_name} has a {color_a[0]} color.')[name]
            first = record(color_text, ('📌', a_symbol, '🎨', color_a[1]),
                           ('👍', a_symbol, color_a[1]), 'color_fact')
            other = record(f'The {b_name} is {color_b[0]}.', ('📌', b_symbol, '🎨', color_b[1]),
                           ('👍', b_symbol, color_b[1]), 'color_fact', episode_history([first]))
            data[name].extend((first, other))
            question = dict(train=f'What color is the {a_name}?', validation=f'Can you tell me the color of the {a_name}?',
                            test=f'Do you remember what color my {a_name} is?')[name]
            data[name].append(record(question, ('❓', a_symbol, '🎨'), (a_symbol, color_a[1]),
                                     'color_recall', episode_history([first, other])))
            unknown_question = f'What color is the {c_name}?'
            data[name].append(record(unknown_question, ('❓', c_symbol, '🎨'), (c_symbol, '❓'),
                                     'unknown_attribute', episode_history([first, other])))
            pronoun = dict(train='What color is it?', validation='Can you recall its color?',
                           test='What was the color of that last one?')[name]
            data[name].append(record(pronoun, ('❓', '🎨'), (b_symbol, color_b[1]),
                                     'pronoun_recall', episode_history([first, other])))

            place_a, place_b = rng.sample(PLACES, 2)
            place_text = dict(train=f'The {a_name} is at the {place_a[0]}.',
                              validation=f'You can find the {a_name} at the {place_a[0]}.',
                              test=f'My {a_name} is located at the {place_a[0]}.')[name]
            placed = record(place_text, ('📌', a_symbol, '📍', place_a[1]),
                            ('👍', a_symbol, '📍', place_a[1]), 'location_fact')
            moved_text = dict(train=f'The {a_name} moved to the {place_b[0]}.',
                              validation=f'The {a_name} is now at the {place_b[0]}.',
                              test=f'Update: the {a_name} went to the {place_b[0]}.')[name]
            moved = record(moved_text, ('🔄', a_symbol, '📍', place_b[1]),
                           ('👌', a_symbol, '📍', place_b[1]), 'location_update', episode_history([placed]))
            location_question = dict(train=f'Where is the {a_name}?', validation=f'Where can I find the {a_name}?',
                                     test=f'Do you remember where my {a_name} is now?')[name]
            data[name].extend((placed, moved, record(location_question, ('❓', a_symbol, '📍'),
                                                    (a_symbol, '📍', place_b[1]), 'location_recall', episode_history([placed, moved]))))
    arithmetic = [(a, b) for a in range(6) for b in range(6)]
    random.Random(229).shuffle(arithmetic)
    pairs = dict(train=arithmetic[:28], validation=arithmetic[28:32], test=arithmetic[32:])
    for name, count, seed in [('train', 512, 233), ('validation', 64, 239), ('test', 128, 241)]:
        rng = random.Random(seed)
        for index in range(count):
            a, b = rng.choice(pairs[name])
            answer = tuple(DIGITS[int(digit)] for digit in str(a + b))
            if index % 2:
                entity = rng.choice(entities)
                entity_name, symbol = label(entity, index % 3 == 0), ALPHABET[entity]
                text = dict(train=f'I have {NUMBER_WORDS[a]} {entity_name} and get {NUMBER_WORDS[b]} more. How many altogether?',
                            validation=f'I start with {NUMBER_WORDS[a]} {entity_name}, then add {NUMBER_WORDS[b]}. What is the total?',
                            test=f'Count {NUMBER_WORDS[a]} {entity_name} plus another {NUMBER_WORDS[b]}. How many now?')[name]
                data[name].append(record(text, ('🔢', symbol, DIGITS[a], '➕', DIGITS[b]),
                                         (*answer, symbol), 'count_addition'))
            else:
                text = dict(train=f'What is {a} plus {b}?', validation=f'Add {a} and {b}.', test=f'Calculate {a} + {b}.')[name]
                data[name].append(record(text, ('🔢', DIGITS[a], '➕', DIGITS[b]), answer, 'addition'))
    for name, phrases in [('train', ('What do I like?', 'What do I dislike?', 'What is my favorite?', 'What color is it?')),
                           ('validation', ('What do I enjoy?', 'Can you remember my favorite?')),
                           ('test', ('Which thing did I say I liked?', 'Which thing did I say I disliked?', 'What was its color?'))]:
        for text in phrases:
            state = ('❓', '🎨') if 'color' in text else ('🧠', '💔' if 'dislike' in text else '💖', '❓')
            data[name].append(record(text, state, ('🤔', '❓'), 'missing_context'))
    return data


def write_composition():
    directory = DIRECTORY / 'compositional'
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in composition_splits().items():
        (directory / f'{name}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        print(f'{name}: {len(rows)}')


if __name__ == '__main__':
    write_composition()
