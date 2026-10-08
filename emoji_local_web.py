"""Serve a localhost chat page backed by the Discord bot's local model."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

from emoji_local_chat import EmojiLocalChat, remember_turn


PAGE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>🫪 Emoji bot test</title>
<style>
  body { max-width: 720px; margin: 3rem auto; padding: 0 1rem; font: 18px system-ui; background: #10151c; color: #edf1f7; }
  form { display: flex; gap: .5rem; }
  input, button { font: inherit; padding: .7rem; border-radius: .5rem; }
  input { flex: 1; min-width: 0; }
  #messages { min-height: 12rem; }
  .reply { font-size: 2rem; }
</style>
<h1>🫪 Emoji bot test</h1>
<p>The last six turns are remembered in this browser tab. History stays in memory and clears when the server restarts.</p>
<div id="messages" aria-live="polite"></div>
<form><input id="text" maxlength="4000" placeholder="Ask something…" required><button>Send</button></form>
<button id="clear" type="button">Clear memory</button>
<p id="status">Ready</p>
<script>
const form = document.querySelector('form');
const input = document.querySelector('#text');
const messages = document.querySelector('#messages');
const status = document.querySelector('#status');
const sessionId = sessionStorage.getItem('emoji-session-id') || crypto.randomUUID();
sessionStorage.setItem('emoji-session-id', sessionId);
form.addEventListener('submit', async event => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  const user = document.createElement('p');
  user.textContent = question;
  messages.append(user);
  input.value = '';
  input.disabled = true;
  status.textContent = 'Thinking…';
  try {
    const response = await fetch('/chat', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text: question, session_id: sessionId}) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    const reply = document.createElement('p');
    reply.className = 'reply';
    reply.textContent = data.reply;
    messages.append(reply);
    status.textContent = 'Ready';
  } catch (error) {
    status.textContent = error.message;
  } finally {
    input.disabled = false;
    input.focus();
  }
});
document.querySelector('#clear').addEventListener('click', async () => {
  await fetch('/clear', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({session_id: sessionId}) });
  messages.replaceChildren();
  status.textContent = 'Memory cleared';
  input.focus();
});
</script>
"""


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, value, content_type):
        content = value.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if self.path != '/':
            self.send_error(404)
            return
        self.respond(200, PAGE, 'text/html; charset=utf-8')

    def do_POST(self):
        if self.path not in ('/chat', '/clear'):
            self.send_error(404)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 32768:
                raise ValueError('Enter a message up to 4000 characters.')
            payload = json.loads(self.rfile.read(length))
            session_id = payload.get('session_id') if isinstance(payload, dict) else None
            if not isinstance(session_id, str) or len(session_id) > 64:
                raise ValueError('Invalid browser session.')
            if self.path == '/clear':
                self.server.histories.pop(session_id, None)
                self.respond(200, json.dumps({'ok': True}), 'application/json; charset=utf-8')
                return
            text = payload.get('text') if isinstance(payload, dict) else None
            if not isinstance(text, str) or not text.strip() or len(text) > 4000:
                raise ValueError('Enter a message up to 4000 characters.')
            history = self.server.histories.setdefault(session_id, [])
            reply = self.server.chat.answer(text.strip(), history.copy())
            remember_turn(history, text.strip(), reply)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            self.respond(400, json.dumps({'error': str(error)}), 'application/json; charset=utf-8')
            return
        self.respond(200, json.dumps({'reply': reply}, ensure_ascii=False),
                     'application/json; charset=utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7861)
    args = parser.parse_args()
    print('Loading the local emoji chat model…', flush=True)
    chat = EmojiLocalChat()
    server = HTTPServer(('127.0.0.1', args.port), Handler)
    server.chat = chat
    server.histories = {}
    print(f'Open http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
