"""Atomic Unicode emoji sequences and their published human-readable names."""

import re

from run import ROOT
from semantic_model import SYMBOLS, WORDS


SOURCE = ROOT / 'data' / 'unicode' / 'emoji-test.txt'


def catalog():
    records = [dict(symbol=symbol, name=word, unicode_name=None, group='reasoning', subgroup='operators')
               for symbol, word in zip((*SYMBOLS, '🔚', '▶️'), (*WORDS, 'end', 'start'))]
    known = {record['symbol']: record for record in records}
    group = subgroup = None
    for line in SOURCE.read_text(encoding='utf-8').splitlines():
        if line.startswith('# group: '):
            group = line.removeprefix('# group: ')
        elif line.startswith('# subgroup: '):
            subgroup = line.removeprefix('# subgroup: ')
        elif line and not line.startswith('#'):
            codes, rest = line.split(';', 1)
            status, description = rest.split('#', 1)
            if status.strip() not in ('fully-qualified', 'component'):
                continue
            symbol = ''.join(chr(int(code, 16)) for code in codes.split())
            name = re.split(r' E\d+\.\d+ ', description.strip(), maxsplit=1)[1]
            if symbol in known:
                known[symbol]['unicode_name'] = name
            else:
                record = dict(symbol=symbol, name=name, unicode_name=name, group=group, subgroup=subgroup)
                records.append(record)
                known[symbol] = record
    return records


CATALOG = catalog()
ALPHABET = tuple(record['symbol'] for record in CATALOG)
# Operators retain explicit project meanings; every other emoji can name an entity.
ENTITIES = (0, 1, 2, 3, *range(13, len(CATALOG)))
