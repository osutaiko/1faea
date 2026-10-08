"""Generate emoji-only answers through a local llama.cpp server."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from emoji_vocabulary import catalog_symbols, meaning_map


MODEL_ID = 'unsloth/Qwen3.5-0.8B-GGUF:UD-IQ2_XXS'
SERVER_URL = os.environ.get('LLAMA_CPP_URL', 'http://127.0.0.1:8080/v1')
MAX_EMOJIS = 6
MAX_HISTORY_TURNS = 6
MAX_HISTORY_MESSAGES = MAX_HISTORY_TURNS * 2
MAX_GENERATION_TOKENS = 16
MODEL_STARTUP_TIMEOUT = 1800
INSTRUCTIONS = (
    "Answer the latest message using recent context. Reply only with up to six distinct, "
    'relevant emojis in meaningful order. Preserve key details and relationships. Avoid filler.'
)
def emoji_grammar(meanings):
    """Return a compact GBNF for up to six single-codepoint emojis."""
    codepoints = set()
    source = Path(__file__).resolve().parent / 'data' / 'unicode' / 'emoji-test-18.txt'
    for line in source.read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#'):
            continue
        codes, description = line.split(';', 1)
        if description.split('#', 1)[0].strip() != 'fully-qualified':
            continue
        code = codes.split()
        if len(code) == 1:
            symbol = chr(int(code[0], 16))
            if symbol in meanings:
                codepoints.add(ord(symbol))

    ranges = []
    for codepoint in sorted(codepoints):
        if ranges and codepoint == ranges[-1][1] + 1:
            ranges[-1][1] = codepoint
        else:
            ranges.append([codepoint, codepoint])
    chars = ''.join(
        f'\\U{start:08X}' if start == end else f'\\U{start:08X}-\\U{end:08X}'
        for start, end in ranges
    )
    optional_emojis = ' '.join('emoji?' for _ in range(MAX_EMOJIS - 1))
    return f'root ::= emoji {optional_emojis}\nemoji ::= [{chars}]'


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
            installed_llama = Path.home() / '.llama-app' / 'llama'
            app_data = os.environ.get('LOCALAPPDATA')
            if os.name == 'nt' and app_data:
                installed_llama = Path(app_data) / 'Programs' / 'llama.cpp' / 'llama.exe'
            if installed_llama.is_file():
                llama = str(installed_llama)
        llama_server = shutil.which('llama-server')
        if llama is None and llama_server is None:
            raise RuntimeError(
                'llama.cpp is not installed or is not on PATH. Install it with '
                '`curl -LsSf https://llama.app/install.sh | sh`, then start the bot again.'
            )

        port = endpoint.port or (443 if endpoint.scheme == 'https' else 80)
        if llama is not None:
            command = [llama, 'serve']
        else:
            command = [llama_server]
        self.server_process = subprocess.Popen(command + [
            '-hf', MODEL_ID, '-c', '1024', '-np', '1', '--host', '127.0.0.1',
            '--port', str(port),
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
        messages.extend(history[-MAX_HISTORY_MESSAGES:])
        messages.append({'role': 'user', 'content': question.strip()})
        body = json.dumps({
            'model': MODEL_ID,
            'messages': messages,
            'temperature': 0,
            'max_tokens': MAX_GENERATION_TOKENS,
            'chat_template_kwargs': {'enable_thinking': False},
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
