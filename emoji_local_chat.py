"""Generate emoji-only answers through a local llama.cpp server."""

import json
import os
import shutil
import subprocess
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from emoji_vocabulary import catalog_symbols, meaning_map


MODEL_ID = 'unsloth/Qwen3.5-0.8B-GGUF:UD-IQ2_XXS'
SERVER_URL = os.environ.get('LLAMA_CPP_URL', 'http://127.0.0.1:8080/v1')
MAX_EMOJIS = 8
MAX_HISTORY_TURNS = 6
MAX_HISTORY_MESSAGES = MAX_HISTORY_TURNS * 2
MAX_GENERATION_TOKENS = 128
MODEL_STARTUP_TIMEOUT = 1800
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
    rules = []
    nodes = []
    for start in range(0, len(symbols), 16):
        name = f'chunk{start // 16}'
        alternatives = ' | '.join(f'"{symbol}"' for symbol in symbols[start:start + 16])
        rules.append(f'{name} ::= {alternatives}')
        nodes.append(name)

    index = 0
    while len(nodes) > 1:
        parents = []
        for start in range(0, len(nodes), 2):
            if start + 1 == len(nodes):
                parents.append(nodes[start])
                continue
            name = f'branch{index}'
            rules.append(f'{name} ::= {nodes[start]} | {nodes[start + 1]}')
            parents.append(name)
            index += 1
        nodes = parents

    rules.insert(0, f'emoji ::= {nodes[0]}')
    optional_emojis = ' '.join('emoji?' for _ in range(MAX_EMOJIS - 1))
    return f'root ::= emoji {optional_emojis}\n' + '\n'.join(rules)


class EmojiLocalChat:
    def __init__(self):
        self.meanings = meaning_map()
        self.grammar = emoji_grammar(self.meanings)
        self.server_process = None

    def start_server(self):
        """Start llama.cpp locally if the configured API endpoint is not up."""
        if self._server_is_ready():
            return

        endpoint = urlparse(SERVER_URL)
        if endpoint.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise RuntimeError(f'Cannot reach the configured model server at {SERVER_URL}')

        llama = shutil.which('llama')
        if llama is None:
            raise RuntimeError(
                'llama.cpp is not installed. Install it once, then start the bot again.'
            )

        port = endpoint.port or (443 if endpoint.scheme == 'https' else 80)
        self.server_process = subprocess.Popen([
            llama, 'serve', '-hf', MODEL_ID, '-c', '1024', '-np', '1',
            '--host', '127.0.0.1', '--port', str(port),
        ])
        deadline = time.monotonic() + MODEL_STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self._server_is_ready():
                return
            if self.server_process.poll() is not None:
                raise RuntimeError(
                    f'llama.cpp model server exited with code {self.server_process.returncode}'
                )
            time.sleep(1)

        self.close_server()
        raise RuntimeError('Timed out waiting for llama.cpp to load the model')

    def _server_is_ready(self):
        try:
            with urlopen(f'{SERVER_URL}/models', timeout=2):
                return True
        except OSError:
            return False

    def close_server(self):
        if self.server_process is not None and self.server_process.poll() is None:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()

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
