"""Review-selected user paraphrases and explicit supplemental input examples."""

import json
import hashlib
import random

from conversation_data import DIRECTORY, IDS
from conversation_composition import COLORS, PLACES, composition_splits
from emoji_catalog import ALPHABET, CATALOG, ENTITIES


# Indices refer to the saved Qwen candidates. Answers, changed perspectives,
# changed facts, ambiguous intents, and lost placeholders are excluded.
APPROVED_SOURCE_SHA256 = 'e5ef847240e9f27dd0a9faaa453deba390326576d5e323a0f9941e495ca3184b'
APPROVED = {
    'greeting': [0, 4, 5], 'thanks': [0, 1, 2, 3, 4, 5], 'farewell': [0, 2, 3, 4],
    'apology': [0, 1, 3, 5], 'capabilities': [0, 2, 4, 5], 'loneliness': [0, 1, 2, 4, 5],
    'anxiety': [0, 4, 5], 'anger': [2, 5], 'happiness': [0, 1, 2, 3, 4, 5],
    'tiredness': [3, 4], 'hunger': [0, 3, 4], 'thirst': [2, 4], 'boredom': [4, 5],
    'celebrate': [1, 2, 5], 'birthday': [0, 1], 'encouragement': [1, 2, 4, 5],
    'study': [0, 1, 2, 4], 'exercise': [3], 'sleep_plan': [1, 2, 4, 5],
    'packing': [0, 1, 2, 4, 5], 'cooking': [2, 3, 4, 5], 'gardening': [0, 1, 3, 4, 5],
    'rain': [0, 1, 2, 4, 5], 'sun': [1], 'reading': [2, 3, 4, 5], 'art': [0],
    'cats': [0], 'bird_flight': [1, 2], 'photosynthesis': [0, 1, 3, 4],
    'water_cycle': [3, 4], 'recycling': [0, 1, 2, 3, 4, 5],
    'recall_dislike': [3], 'color_pronoun_query': [0, 1, 2, 3, 4, 5],
    'location_query_template': [0, 3, 4],
}

EXTRA = (
    (('🧠', '💖', '❓'), ('Remind me which thing I like.', 'Can you remember the item I love?',
                         'Tell me what my favorite was.', 'Which item did I mention liking?',
                         'Please recall the preference I shared.', 'What did I say I was fond of?',
                         'Do you remember my favorite thing?', 'I forgot what I told you I liked.',
                         'Recall my earlier choice.', 'Can you remind me what I said I enjoy?')),
    (('🧠', '💔', '❓'), ('Remind me which thing I dislike.', 'Can you remember the item I hate?',
                         'Tell me which item I did not like.', 'Please recall my negative preference.',
                         'Which item did I mention disliking?', 'What did I say I could not stand?',
                         'Do you remember what I was not fond of?', 'Recall the thing I disliked earlier.')),
    (('😔',), ('I feel heartbroken today.', 'My mood has been low lately.', 'I am feeling miserable.', 'I have been feeling tearful.')),
    (('😟',), ('My thoughts are racing with worry.', 'I am unsettled and nervous.', 'This situation has me worried.', 'I am full of apprehension.')),
    (('🥱',), ('I have nothing interesting to occupy me.', 'I need an idea for something enjoyable.', 'I have too much idle time.', 'I want something fun to fill my time.')),
    (('❓', '🤖'), ('What is your identity?', 'Are you a human or an AI?', 'Describe what kind of helper you are.', 'What sort of chatbot are you?')),
)


def normalized(text):
    return ' '.join(text.lower().split()).rstrip('.!?')


def curate():
    raw = (DIRECTORY / 'generated-user-prompts.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != APPROVED_SOURCE_SHA256:
        raise ValueError('Teacher candidates changed; review them before updating the approved indices')
    source = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
    candidates = {row['skill']: row for row in source}
    dataset = composition_splits()
    reserved = {normalized(row['text']) for split in ('validation', 'test') for row in dataset[split]}
    seen = {normalized(row['text']) for row in dataset['train']}
    result = []
    rng = random.Random(317)
    for skill, indices in APPROVED.items():
        item = candidates[skill]
        for index in indices:
            template = item['candidates'][index]
            repetitions = 24 if '{item}' in template else 1
            for _ in range(repetitions):
                entity = rng.choice(ENTITIES)
                replacements = dict(item=CATALOG[entity]['unicode_name'])
                text = template.format(**replacements)
                state = [ALPHABET[entity] if symbol == '{item}' else symbol for symbol in item['state']]
                if skill == 'art':
                    state = ['❓', '🖌️']
                key = normalized(text)
                if key in reserved or key in seen:
                    continue
                seen.add(key)
                result.append(dict(text=text, state=[IDS[symbol] for symbol in state], skill=skill, source='reviewed_teacher'))
    for state, phrases in EXTRA:
        for text in phrases:
            key = normalized(text)
            if key not in reserved and key not in seen:
                seen.add(key)
                result.append(dict(text=text, state=[IDS[symbol] for symbol in state], skill='authored', source='authored'))
    patterns = (
        (('💖', '{item}'), ('I am really keen on {item}.', 'I am a fan of {item}.', 'I adore {item}.')),
        (('💔', '{item}'), ('I am not a fan of {item}.', 'I have no liking for {item}.', 'I do not enjoy {item}.')),
        (('🔄', '💖', '{item}'), ('Change my preference to {item}.', 'Forget the earlier choice; I like {item}.')),
        (('📌', '{item}', '🎨', '{color}'), ('The {item} has the color {color}.', 'I have a {color} {item}.')),
        (('❓', '{item}', '🎨'), ('Tell me what color the {item} has.', 'Which color did I mention for the {item}?')),
        (('📌', '{item}', '📍', '{place}'), ('The {item} can be found at the {place}.', 'I left the {item} at the {place}.')),
        (('🔄', '{item}', '📍', '{place}'), ('Change the location of the {item} to the {place}.', 'The {item} has relocated to the {place}.')),
        (('❓', '{item}', '📍'), ('Tell me where the {item} is.', 'Recall the location of the {item}.')),
    )
    for state, forms in patterns:
        for template in forms:
            for index in range(32):
                entity = rng.choice(ENTITIES)
                color, color_symbol = rng.choice(COLORS)
                place, place_symbol = rng.choice(PLACES)
                name = CATALOG[entity]['unicode_name'] if index % 2 else ALPHABET[entity]
                text = template.format(item=name, color=color, place=place)
                values = {'{item}': ALPHABET[entity], '{color}': color_symbol, '{place}': place_symbol}
                symbols = [values.get(symbol, symbol) for symbol in state]
                key = normalized(text)
                if key not in reserved and key not in seen:
                    seen.add(key)
                    result.append(dict(text=text, state=[IDS[symbol] for symbol in symbols], skill='authored_template', source='authored'))
    path = DIRECTORY / 'approved-inputs.jsonl'
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in result), encoding='utf-8')
    print(json.dumps(dict(accepted=len(result), teacher_groups=len(APPROVED),
                          exact_evaluation_overlaps=sum(normalized(row['text']) in reserved for row in result)), indent=2))


if __name__ == '__main__':
    curate()
