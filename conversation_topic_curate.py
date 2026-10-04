"""Keep manually reviewed user paraphrases with unchanged emoji targets."""

import hashlib
import json

from conversation_curate import normalized
from conversation_data import record
from conversation_composition import NUMBER_WORDS
from conversation_reliability_data import DIRECTORY, TOPICS, dataset
from emoji_catalog import ALPHABET


SOURCE_SHA256 = '1dcfab9f2644bf9200ff06e878782b05ac6c77e359eaa10be6e8719713bf5d6d'
APPROVED = {
    'cleaning': [0, 1, 3, 4], 'organization': [2], 'language_learning': [0, 1, 2, 3],
    'writing': [2, 3], 'repair': [1, 2], 'debugging': [1], 'teamwork': [0, 1, 2, 3, 4],
    'focus': [0, 1, 2, 3, 5], 'creativity': [2], 'walking': [2, 3],
    'gravity': [0, 1, 2, 3, 4], 'computer': [3, 5], 'electricity': [0, 2, 3, 4],
}


def curate():
    source = DIRECTORY / 'generated-topic-prompts.jsonl'
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('Topic candidates changed; review before changing approved indices')
    candidates = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
    labels = {skill: (state, reply) for skill, state, reply, *_ in TOPICS}
    data, _ = dataset()
    reserved = {normalized(row['text']) for split in ('validation', 'test') for row in data[split]}
    seen = {normalized(row['text']) for row in data['train']}
    result = []
    for group in candidates:
        state, reply = labels[group['skill']]
        for index in APPROVED.get(group['skill'], []):
            text = group['candidates'][index]
            key = normalized(text)
            if key in reserved or key in seen:
                continue
            seen.add(key)
            row = record(text, state, reply, group['skill'])
            row.update(source='reviewed_teacher', candidate_index=index)
            result.append(row)
    (DIRECTORY / 'approved-topics.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in result), encoding='utf-8')
    quantities = []
    for row in data['train']:
        if ALPHABET[row['state'][0]] not in ('📌', '🔄') or ALPHABET[row['state'][-2]] != '🔢':
            continue
        digit = ALPHABET[row['state'][-1]]
        for spelling in (digit[0], NUMBER_WORDS[int(digit[0])]):
            quantities.append(dict(row, text=row['text'].replace(digit, spelling), source='authored_numeric_wording'))
    (DIRECTORY / 'quantity-inputs.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in quantities), encoding='utf-8')
    review = dict(source_sha256=SOURCE_SHA256, approved=APPROVED, accepted=len(result),
                  candidate_count=sum(len(group['candidates']) for group in candidates),
                  rejected_reasons=['assistant answers', 'changed perspective', 'specific unsupported procedures',
                                    'changed topic', 'ambiguous requests'], authored_quantity_rows=len(quantities),
                  exact_evaluation_overlaps=0)
    (DIRECTORY / 'topic-review.json').write_text(json.dumps(review, indent=2), encoding='utf-8')
    print(json.dumps(review, indent=2), flush=True)


if __name__ == '__main__':
    curate()
