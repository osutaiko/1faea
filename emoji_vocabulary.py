"""Unicode emoji meanings and strict output-catalog validation."""

import json
import re

from run import ROOT


def meaning_map():
    rows = json.loads((ROOT / 'data/emoji-meanings/map.json').read_text(encoding='utf-8'))
    known = {row['symbol'] for row in rows}
    official = set()
    for line in (ROOT / 'data/unicode/emoji-test-18.txt').read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#'):
            continue
        codes, rest = line.split(';', 1)
        status, description = rest.split('#', 1)
        if status.strip() not in ('fully-qualified', 'component'):
            continue
        symbol = ''.join(chr(int(code, 16)) for code in codes.split())
        official.add(symbol)
        if symbol not in known:
            name = re.split(r' E\d+\.\d+ ', description.strip(), maxsplit=1)[1]
            rows.append(dict(symbol=symbol, core_meaning=name))
            known.add(symbol)
    if official != known:
        raise ValueError('Meaning map does not match the official Unicode emoji set')
    return {row['symbol']: row['core_meaning'] for row in rows}


def catalog_symbols(values, meanings):
    symbols = sorted(meanings, key=len, reverse=True)
    result = []
    invalid = []
    for value in values:
        position = 0
        while position < len(value):
            symbol = next((item for item in symbols if value.startswith(item, position)), None)
            if symbol is None:
                invalid.append(value[position])
                position += 1
            else:
                result.append(symbol)
                position += len(symbol)
    if invalid:
        raise ValueError(f'Model returned symbols outside the project catalog: {invalid}')
    return result
