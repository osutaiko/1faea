"""Review public user inputs against the explicitly supported emoji grammar."""

import hashlib
import json

from conversation_curate import normalized
from conversation_data import DIRECTORY as SOURCE_DIRECTORY, SKILLS, record
from conversation_reliability_data import DIRECTORY, TOPICS


TRAIN_SHA256 = 'ebaad5807902139273497b010894d3cb35e045dbe3c006a42ac039079c2c9114'
TEST_SHA256 = '09dc2c58cb96b130d5e28b0d161e18081bf01ee4efb214f581b3b16bebe5d0fb'
SUPPORTED_TEST = {93: 'organization', 95: 'gardening', 101: 'sleep_plan', 114: 'teamwork', 115: 'packing'}


def user_message(row):
    return next(message['content'] for message in row['messages'][1:] if message['role'] == 'user')


def write():
    train_source = SOURCE_DIRECTORY / 'everyday-train_sft.jsonl'
    raw = train_source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TRAIN_SHA256:
        raise ValueError('Public training source changed; review the selected user inputs again')
    training = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
    test_source = SOURCE_DIRECTORY / 'everyday-test_sft.jsonl'
    test_raw = test_source.read_bytes()
    if hashlib.sha256(test_raw).hexdigest() != TEST_SHA256:
        raise ValueError('Public test source changed; review the selected user inputs again')
    testing = [json.loads(line) for line in test_raw.decode('utf-8').splitlines()]
    labels = {skill: (state, reply) for skill, state, reply, *_ in (*SKILLS, *TOPICS)}
    output = dict(train=[], test=[])
    # The arithmetic example is excluded to preserve earlier held-out sum pairs.
    for index in range(24):
        if index == 22:
            continue
        skill = {4: 'cooking', 6: 'packing'}.get(index, 'unsupported')
        state, reply = labels[skill]
        item = record(user_message(training[index]), state, reply, 'public_' + skill)
        item.update(source_index=index, source='public_user_only', supported=skill != 'unsupported')
        output['train'].append(item)
    for index in range(87, 119):
        skill = SUPPORTED_TEST.get(index, 'unsupported')
        state, reply = labels[skill]
        item = record(user_message(testing[index]), state, reply, 'public_' + skill)
        item.update(source_index=index, source='public_user_only', supported=skill != 'unsupported')
        output['test'].append(item)
    if {normalized(row['text']) for row in output['train']} & {normalized(row['text']) for row in output['test']}:
        raise ValueError('Public scope inputs overlap across splits')
    for split, rows in output.items():
        (DIRECTORY / f'public-scope-{split}.jsonl').write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    review = dict(train_source_sha256=TRAIN_SHA256, test_source_sha256=hashlib.sha256(test_raw).hexdigest(),
                  train_indices=[row['source_index'] for row in output['train']], test_indices=list(range(87, 119)),
                  supported_test=SUPPORTED_TEST, target_scope='coarse intent and documented outline only',
                  labels='authored from user requests; source assistant answers never used',
                  license='Apache-2.0', selection='test cases reserved for final assessment, not checkpoint selection')
    (DIRECTORY / 'public-scope-review.json').write_text(json.dumps(review, indent=2), encoding='utf-8')
    print(json.dumps(dict(train=len(output['train']), test=len(output['test']))), flush=True)


if __name__ == '__main__':
    write()
