"""Materialize assistant-generated emoji candidates with explicit review provenance."""

import hashlib
import json

from conversation_data import IDS
from conversation_standalone_data import DIRECTORY, validate


QUESTIONS_SHA256 = 'e203238234d0ebc6e6f0e9c6e83bbcca2e92c2008a6cd4f2f880248d90aa6158'
LABELS_SHA256 = 'bde8bb2c9dd2670e9895db5797e333e415e13e558ad3b274fbd7a889e7c49325'
# This is an audit of individual examples, not a model intent taxonomy.
AMBIGUOUS_ANSWERS = {18, 19, 21, 22, 28, 30, 31, 36, 45, 46, 50, 53, 54, 56, 59, 62,
                     63, 66, 68, 73, 76, 81, 85, 86, 89, 98, 100, 102, 108, 113, 114, 115,
                     120, 121, 125, 127, 136, 140, 150, 152, 160, 164, 170, 181, 184, 188, 190, 195}
# Same or closely related requests already present in validation/test.
QUARANTINED_TRAIN = {7, 11, 23, 52, 72, 77, 147, 188}


def build():
    questions_path = DIRECTORY / 'questions.jsonl'
    if hashlib.sha256(questions_path.read_bytes()).hexdigest() != QUESTIONS_SHA256:
        raise ValueError('Questions changed; review the indexed labels before rebuilding')
    questions = [json.loads(line) for line in questions_path.read_text(encoding='utf-8').splitlines()]
    labels_path = DIRECTORY / 'bootstrap-labels.json'
    if hashlib.sha256(labels_path.read_bytes()).hexdigest() != LABELS_SHA256:
        raise ValueError('Labels changed; repeat the semantic audit before rebuilding')
    labels = json.loads(labels_path.read_text(encoding='utf-8'))
    if sorted(row[0] for row in labels) != list(range(len(questions))):
        raise ValueError('Labels must cover every question exactly once')
    meanings = {row['symbol']: row['meaning'] for row in json.loads(
        (DIRECTORY / 'vocabulary.json').read_text(encoding='utf-8'))}
    meanings.update({'➡️': 'then / leads to', '🚫': 'not', '🔹': 'clause boundary', '❓': 'unknown'})
    candidates = []
    for index, state, answer in labels:
        # Literal readings are constructed only after the emoji labels exist.
        reading = ' | '.join(meanings[symbol] for symbol in answer)
        candidates.append(dict(id=questions[index]['id'], state=state, answer=answer,
                               answer_reading=reading, needs_context=False))
    raw_path = DIRECTORY / 'bootstrap-candidates.jsonl'
    raw_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in candidates), encoding='utf-8')
    validate(raw_path)
    review, approved = [], []
    for index, candidate in enumerate(candidates):
        question = questions[index]
        accepted = index not in AMBIGUOUS_ANSWERS
        quarantined = question['split'] == 'train' and index in QUARANTINED_TRAIN
        review.append(dict(id=candidate['id'], accepted=accepted, quarantine=quarantined,
                           reason='ambiguous or insufficient emoji answer' if not accepted else
                           'near-duplicate held-out request' if quarantined else 'provisional assistant review'))
        if accepted and not quarantined:
            approved.append(dict(**question, state=[IDS[symbol] for symbol in candidate['state']],
                                 reply=[IDS[symbol] for symbol in candidate['answer']], history=[],
                                 skill='standalone', answer_reading=candidate['answer_reading'],
                                 provisional=True))
    provenance = dict(teacher='current Codex assistant; direct emoji labels generated in this task',
                      external_api_used=False, api_probe_status=401,
                      reviewer='same assistant, separate semantic audit; not an independent model',
                      independent_verification=False, assistant_source_answers_used=False,
                      english_audit='literal symbol gloss constructed after emoji generation',
                      questions_sha256=hashlib.sha256(questions_path.read_bytes()).hexdigest(),
                      labels_sha256=hashlib.sha256(labels_path.read_bytes()).hexdigest(),
                      counts={split: sum(row['split'] == split for row in approved)
                              for split in ('train', 'validation', 'test')},
                      ambiguous_rejected=len(AMBIGUOUS_ANSWERS), quarantine_indices=sorted(QUARANTINED_TRAIN),
                      semantic_overlap_audit='known near duplicates quarantined; not exhaustive', reviews=review)
    (DIRECTORY / 'bootstrap-review.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    (DIRECTORY / 'provisional.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in approved), encoding='utf-8')
    print(json.dumps({key: value for key, value in provenance.items() if key != 'reviews'}, indent=2))


if __name__ == '__main__':
    build()
