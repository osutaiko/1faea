"""Run stateless hosted emoji chat with direct or two-call emoji output."""

import argparse
import json
import os
from pathlib import Path
import re
import urllib.request


ROOT = Path(__file__).resolve().parent
MODEL = 'gpt-6-luna'
API = 'https://api.openai.com/v1/responses'
SCHEMA = dict(type='object', properties={
    'emojis': dict(type='array', items=dict(type='string'), minItems=1, maxItems=8),
}, required=['emojis'], additionalProperties=False)
ENCODE = ('Solve the user request and encode only its concise answer meaning as an emoji state. '
          'This is a fresh, independent request. Output no ordinary words. Answer factual choices correctly. '
          'Use only Unicode emoji characters, not plain symbols or pictographs such as ⏻. '
          'For advice, show short actions in sequence. Preserve negation and important relations. '
          'If you cannot answer, use 🤷 and ❓. Use at most 8 emojis.')
DECODE = ('You receive only an emoji state and literal meanings for the symbols. Express its answer directly, '
          'using emojis only. Preserve order and negation. Add no new concepts. If one emoji already expresses it, '
          'return that emoji. Use only Unicode emoji characters, not plain symbols or pictographs. Output no words.')
DIRECT = ("Answer the user's request directly. Keep the full question in mind while deciding what to say; do not "
          'translate it word by word or create an intermediate summary. Express only the final answer using '
          'Unicode emojis such as 🔌 and 📵, never plain symbols such as ⏻. Preserve important facts, negation, '
          'and relationships. For advice, show concise actions '
          'in order. If the answer cannot be expressed clearly, use 🤷 and ❓. Use at most 8 emojis and no words.')


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


def structured_emojis(data, instructions):
    if not os.environ.get('OPENAI_API_KEY'):
        raise RuntimeError('Set OPENAI_API_KEY before starting emoji_api_chat.py')
    body = dict(model=MODEL, instructions=instructions, input=data, store=False, max_output_tokens=180,
                reasoning=dict(effort='low'),
                text=dict(format=dict(type='json_schema', name='emoji_sequence', strict=True, schema=SCHEMA)))
    request = urllib.request.Request(API, data=json.dumps(body, ensure_ascii=False).encode('utf-8'),
        headers={'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY'],
                 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if payload.get('status') != 'completed':
        details = payload.get('incomplete_details') or payload.get('error')
        raise ValueError(f"Emoji response did not complete: {payload.get('status')} ({details})")
    outputs = [part['text'] for item in payload['output'] if item['type'] == 'message'
               for part in item['content'] if part['type'] == 'output_text']
    if not outputs:
        raise ValueError('Model returned no emoji response')
    emojis = json.loads(''.join(outputs)).get('emojis')
    if not isinstance(emojis, list) or not 1 <= len(emojis) <= 8:
        raise ValueError('Model returned an invalid emoji sequence')
    return emojis


class EmojiAPIChat:
    def __init__(self):
        self.meanings = meaning_map()

    def answer(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Enter a non-empty question')
        state = catalog_symbols(structured_emojis(question.strip(), ENCODE), self.meanings)
        definitions = [{'symbol': symbol, 'meaning': self.meanings[symbol]} for symbol in state]
        decoder_input = json.dumps(dict(emoji_state=state, symbol_meanings=definitions), ensure_ascii=False)
        reply = catalog_symbols(structured_emojis(decoder_input, DECODE), self.meanings)
        return dict(state=state, symbols=reply, reply=''.join(reply))


class EmojiDirectAPIChat:
    def __init__(self):
        self.meanings = meaning_map()

    def answer(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Enter a non-empty question')
        symbols = catalog_symbols(structured_emojis(question.strip(), DIRECT), self.meanings)
        return dict(symbols=symbols, reply=''.join(symbols))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question')
    parser.add_argument('--trace', action='store_true', help='Show JSON details when available')
    parser.add_argument('--direct', action='store_true', help='Answer directly in emojis in one model call')
    args = parser.parse_args()
    chat = EmojiDirectAPIChat() if args.direct else EmojiAPIChat()
    if args.question:
        result = chat.answer(args.question)
        print(json.dumps(result, ensure_ascii=False) if args.trace else result['reply'])
    else:
        print('Stateless emoji replies. Each question starts fresh. Type /quit to exit.')
        while True:
            try:
                question = input('> ').strip()
            except EOFError:
                break
            if question == '/quit':
                break
            if question:
                try:
                    result = chat.answer(question)
                except ValueError as error:
                    print(f'⚠️ {error}', flush=True)
                    continue
                print(json.dumps(result, ensure_ascii=False) if args.trace else result['reply'], flush=True)
