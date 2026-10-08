"""Generate emoji-only answers through a local llama.cpp server."""

import json
import os
from urllib.request import Request, urlopen

from emoji_vocabulary import catalog_symbols, meaning_map


MODEL_ID = 'unsloth/Qwen3.5-0.8B-GGUF:UD-IQ2_XXS'
SERVER_URL = os.environ.get('LLAMA_CPP_URL', 'http://127.0.0.1:8080/v1')
MAX_EMOJIS = 8
MAX_HISTORY_TURNS = 6
MAX_HISTORY_MESSAGES = MAX_HISTORY_TURNS * 2
MAX_GENERATION_TOKENS = 128
INSTRUCTIONS = (
    "Understand and answer the user's latest message in the context of this conversation. "
    'Think silently, then reply with a concise sequence of relevant emoji concepts. Use distinct '
    'symbols in a natural order to preserve important facts, negation, quantities, comparisons, '
    'and relationships from recent turns. Resolve references using conversation context. Match '
    'the intent: answer questions directly, acknowledge feelings, or give a useful next step. '
    'Prefer specific answers over generic reactions. Never repeat symbols as filler. Avoid '
    'unsupported claims; when uncertain, express uncertainty.'
)
EXAMPLES = [
    ('Which planet do we live on?', '🌍'),
    ('Is mixing bleach and ammonia safe?', '🧪☠️🚫'),
    ('I feel nervous about starting a new job. Any advice?', '🫂🌬️🌱'),
    ('Thanks, that helped!', '😊✨'),
]


def emoji_grammar(meanings):
    """Return GBNF that permits one to eight catalog emoji and nothing else."""
    symbols = sorted(meanings, key=len, reverse=True)
    alternatives = ' | '.join(f'"{symbol}"' for symbol in symbols)
    optional_emojis = ' '.join('emoji?' for _ in range(MAX_EMOJIS - 1))
    return f'root ::= emoji {optional_emojis}\nemoji ::= {alternatives}'


class EmojiLocalChat:
    def __init__(self):
        self.meanings = meaning_map()
        self.grammar = emoji_grammar(self.meanings)

    def answer(self, question, history=()):
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Enter a non-empty message')

        messages = [{'role': 'system', 'content': INSTRUCTIONS}]
        for prompt, reply in EXAMPLES:
            messages.extend([{'role': 'user', 'content': prompt},
                             {'role': 'assistant', 'content': reply}])
        messages.extend(history[-MAX_HISTORY_MESSAGES:])
        messages.append({'role': 'user', 'content': question.strip()})
        body = json.dumps({
            'model': MODEL_ID,
            'messages': messages,
            'temperature': 0,
            'max_tokens': MAX_GENERATION_TOKENS,
            'grammar': self.grammar,
        }, ensure_ascii=False).encode('utf-8')
        request = Request(f'{SERVER_URL}/chat/completions', data=body,
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=300) as response:
            payload = json.loads(response.read())

        text = payload['choices'][0]['message']['content']
        result = catalog_symbols([text], self.meanings)
        if not result or len(result) > MAX_EMOJIS:
            raise ValueError('Model returned an empty or overlong emoji reply')
        return ''.join(result)


def remember_turn(history, question, reply):
    history.extend([{'role': 'user', 'content': question},
                    {'role': 'assistant', 'content': reply}])
    del history[:-MAX_HISTORY_MESSAGES]
