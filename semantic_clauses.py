"""Labeled clause grammar and a fixed challenge excluded from parser training."""

import itertools

from semantic_data import examples, verdict
from semantic_model import WORDS


TRAIN_FORMS = (
    ('The {a} is left of the {b}.', 0),
    ('The {b} is right of the {a}.', 1),
    ('The {a} is to the left of the {b}.', 0),
    ('The {b} is to the right of the {a}.', 1),
    ('The {a} stands to the left of the {b}.', 0),
    ('The {b} stands to the right of the {a}.', 1),
    ('A {a} stands on the left-hand side of a {b}.', 0),
    ('A {b} stands on the right-hand side of a {a}.', 1),
    ('The {a} sits on the left of the {b}.', 0),
    ('The {b} sits on the right of the {a}.', 1),
    ('The {a} lies to the left of the {b}.', 0),
    ('The {b} lies to the right of the {a}.', 1),
    ('The {a} rests left of the {b}.', 0),
    ('The {b} rests right of the {a}.', 1),
    ('The {a} is placed to the left of the {b}.', 0),
    ('The {b} is placed to the right of the {a}.', 1),
    ('The {a} is positioned on the left of the {b}.', 0),
    ('The {b} is positioned on the right of the {a}.', 1),
    ('To the left of the {b} stands the {a}.', 1),
    ('To the right of the {a} stands the {b}.', 0),
    ('On the left of the {b} sits the {a}.', 1),
    ('On the right of the {a} sits the {b}.', 0),
    ('The {b} has the {a} to its left.', 1),
    ('The {a} has the {b} to its right.', 0),
    ('The {b} has the {a} on its left-hand side.', 1),
    ('The {a} has the {b} on its right-hand side.', 0),
    ("The {b}'s left neighbor is the {a}.", 1),
    ("The {a}'s right neighbor is the {b}.", 0),
    ('Is the {a} left of the {b}?', 0),
    ('Is the {b} right of the {a}?', 1),
    ('Is the {a} to the left of the {b}?', 0),
    ('Is the {b} to the right of the {a}?', 1),
    ('Is the {a} located to the left of the {b}?', 0),
    ('Is the {b} located to the right of the {a}?', 1),
    ('Is the {a} positioned left of the {b}?', 0),
    ('Is the {b} positioned right of the {a}?', 1),
    ('Is the {a} on the left of the {b}?', 0),
    ('Is the {b} on the right of the {a}?', 1),
    ('Is it true that the {a} is left of the {b}?', 0),
    ('Is it true that the {b} is right of the {a}?', 1),
)
VALIDATION_FORMS = (
    ('The {a} is located on the left side of the {b}.', 0),
    ('The {b} is located on the right side of the {a}.', 1),
    ('Is the {a} standing left of the {b}?', 0),
    ('Is the {b} standing right of the {a}?', 1),
)
CHALLENGE_FACTS = (
    'The {a} appears to the left of the {b}.',
    'The {b} is situated on the right-hand side of the {a}.',
    'The {b} keeps the {a} on its left-hand side.',
    'The {a} occupies a position to the left of the {b}.',
)
CHALLENGE_QUERIES = (
    'Would you say the {a} lies left of the {b}?',
    'Would you say the {b} is situated to the right of the {a}?',
)
AUDIT_FACTS = (
    'Situated to the left of the {b} is the {a}.',
    'The {a} can be found immediately to the left of the {b}.',
    'The {b} has the {a} as its left neighbor.',
    'The {b} is positioned farther right than the {a}.',
)
AUDIT_QUERIES = (
    'Can you tell whether the {a} is to the left of the {b}?',
    'Can you tell whether the {b} is to the right of the {a}?',
)


def clause_examples():
    splits = {}
    for name, forms in (('train', TRAIN_FORMS), ('validation', VALIDATION_FORMS)):
        rows = []
        for a, b in itertools.permutations(range(4), 2):
            for form, orientation in forms:
                rows.append(dict(text=form.format(a=WORDS[a], b=WORDS[b]),
                                 orientation=orientation, left=a, right=b, form=form))
        splits[name] = rows
    return splits


def paragraphs(fact_forms, query_forms):
    rows = []
    for index, original in enumerate(examples()['test']):
        u, v, w, z, x, y = original['entities']
        first = fact_forms[index % len(fact_forms)].format(a=WORDS[u], b=WORDS[v])
        second = fact_forms[(index + 1) % len(fact_forms)].format(a=WORDS[w], b=WORDS[z])
        query = query_forms[index % len(query_forms)].format(a=WORDS[x], b=WORDS[y])
        rows.append(dict(text=f'{first} {second} {query}', entities=original['entities'],
                         derived=original['derived'], answer=verdict([(u, v), (w, z)], (x, y))))
    return rows


def challenge_examples():
    return paragraphs(CHALLENGE_FACTS, CHALLENGE_QUERIES)


def audit_examples():
    return paragraphs(AUDIT_FACTS, AUDIT_QUERIES)
