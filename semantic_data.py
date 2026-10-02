"""Automatically verified semantic traces; the checker is not used for inference."""

import itertools
import random

from semantic_model import WORDS, TRUE, FALSE, UNKNOWN


def verdict(edges, query):
    closure = set(edges)
    changed = True
    while changed:
        expanded = closure | {(a, d) for a, b in closure for c, d in closure if b == c}
        changed = expanded != closure
        closure = expanded
    if any(a == b for a, b in closure):
        return None  # Contradictory strict-left-of facts.
    if query in closure:
        return TRUE
    if query[::-1] in closure:
        return FALSE
    return UNKNOWN


def examples():
    chains = list(itertools.permutations(range(4), 3))
    random.Random(7).shuffle(chains)
    splits = {"train": [], "validation": [], "test": []}
    for index, (a, b, c) in enumerate(chains):
        split = "train" if index < 16 else "validation" if index < 20 else "test"
        templates = range(6) if split == "train" else (6,) if split == "validation" else (7,)
        for facts in (((a, b), (b, c)), ((b, c), (a, b))):
            for x, y in itertools.permutations(range(4), 2):
                for template in templates:
                    first, second = facts
                    u, v = WORDS[first[0]], WORDS[first[1]]
                    w, z = WORDS[second[0]], WORDS[second[1]]
                    subject, obj = WORDS[x], WORDS[y]
                    if template == 0:
                        text = f"The {u} is left of the {v}. The {w} is left of the {z}. Is the {subject} left of the {obj}?"
                    elif template == 1:
                        text = f"The {v} is to the right of the {u}. The {z} is to the right of the {w}. Is the {subject} to the left of the {obj}?"
                    elif template == 2:
                        text = f"To the left of the {v} stands the {u}. To the left of the {z} stands the {w}. Is the {subject} positioned left of the {obj}?"
                    elif template == 3:
                        text = f"A {u} stands on the left-hand side of a {v}. A {w} stands on the left-hand side of a {z}. Is the {subject} located to the left of the {obj}?"
                    elif template == 4:
                        text = f"The {u} is left of the {v}. To the left of the {z} stands the {w}. Is the {subject} positioned left of the {obj}?"
                    elif template == 5:
                        text = f"To the left of the {v} stands the {u}. The {z} is to the right of the {w}. Is the {subject} located to the left of the {obj}?"
                    elif template == 6:
                        text = f"The {u} stands to the left of the {v}. The {w} stands to the left of the {z}. Is the {subject} on the left of the {obj}?"
                    else:
                        text = f"The {v} has the {u} to its left. The {z} has the {w} to its left. Is the {subject} situated to the left of the {obj}?"
                    splits[split].append(dict(text=text, entities=[*first, *second, x, y],
                                              derived=[a, c], answer=verdict(facts, (x, y)),
                                              chain=[a, b, c], template=template))
    return splits
