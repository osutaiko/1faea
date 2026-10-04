"""Serve a local, single-session playground for direct emoji conversations."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

import torch

import conversation
from conversation_model import SemanticAtomicDecoder, EmojiConversation, render
from run import ROOT


class Handler(BaseHTTPRequestHandler):
    def send(self, status, content, content_type):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(content)

    def json(self, status, value):
        self.send(status, json.dumps(value, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8')

    def do_GET(self):
        if self.path != '/':
            self.send_error(404)
            return
        page = (ROOT / 'conversation.html').read_text(encoding='utf-8')
        if self.server.independent:
            page = page.replace('Read text. Remember in emojis. Reply in emojis.',
                                'Ask a question. Get an emoji answer. Each question is independent.')
            page = page.replace('I like pizza. / What do I like?', 'What do plants need to grow?')
            page = page.replace('One local session.', 'Each question starts with empty model memory.')
        self.send(200, page.encode('utf-8'), 'text/html; charset=utf-8')

    def do_POST(self):
        session = self.server.session
        if self.path == '/reset':
            session.history.clear()
            session.turn_lengths.clear()
            session.dropped_turns = 0
            self.json(200, dict(reply='🔄', state='', memory='', dropped_turns=0))
            return
        if self.path != '/turn':
            self.send_error(404)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 32768:
                raise ValueError('Send a short JSON message.')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or not isinstance(payload.get('text'), str) or not payload['text'].strip():
                raise ValueError('Enter a nonempty text message.')
            if self.server.independent:
                session.history.clear()
                session.turn_lengths.clear()
                session.dropped_turns = 0
            result = session.turn(payload['text'].strip())
        except ValueError as error:
            self.json(400, dict(error=str(error)))
            return
        self.json(200, dict(reply=render(result['reply']), state=render(result['state']),
                            memory=render(session.history), dropped_turns=session.dropped_turns,
                            state_terminated=result['state_terminated'], reply_terminated=result['reply_terminated']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='reliability-selected')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--experiment', default='conversation-compositional')
    parser.add_argument('--independent', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    conversation.OUTPUT = ROOT / 'runs' / args.experiment
    encoder, decoder, reader, _ = conversation.load(args.name, SemanticAtomicDecoder)
    server = HTTPServer(('127.0.0.1', args.port), Handler)
    server.session = EmojiConversation(encoder, decoder, reader)
    server.independent = args.independent
    print(f'Emoji playground: http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()
