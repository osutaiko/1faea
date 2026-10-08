import json
from http.server import HTTPServer
import threading
import unittest
from urllib.request import Request, urlopen

from emoji_local_web import Handler


class FakeChat:
    def __init__(self):
        self.calls = []

    def answer(self, question, history):
        self.calls.append((question, history))
        return '🐈'


class EmojiLocalWebTests(unittest.TestCase):
    def setUp(self):
        self.server = HTTPServer(('127.0.0.1', 0), Handler)
        self.server.chat = FakeChat()
        self.server.histories = {}
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def post(self, path, payload):
        request = Request(self.base_url + path, data=json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request) as response:
            return json.loads(response.read())

    def test_remembers_turns_per_browser_session_and_clear_resets_them(self):
        session = 'browser-tab-a'
        self.post('/chat', {'session_id': session, 'text': 'I have a cat.'})
        self.post('/chat', {'session_id': session, 'text': 'What pet do I have?'})
        self.post('/chat', {'session_id': 'browser-tab-b', 'text': 'What pet do I have?'})
        self.post('/clear', {'session_id': session})
        self.post('/chat', {'session_id': session, 'text': 'What pet do I have now?'})

        self.assertEqual(self.server.chat.calls[1][1], [
            {'role': 'user', 'content': 'I have a cat.'},
            {'role': 'assistant', 'content': '🐈'},
        ])
        self.assertEqual(self.server.chat.calls[2][1], [])
        self.assertEqual(self.server.chat.calls[3][1], [])


if __name__ == '__main__':
    unittest.main()
