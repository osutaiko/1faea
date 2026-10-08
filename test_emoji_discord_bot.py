import asyncio
import io
import json
import unittest
from urllib.error import URLError
from types import SimpleNamespace
from unittest.mock import Mock, patch

from emoji_discord_bot import MAX_MESSAGE_LENGTH, create_client, message_text
from emoji_local_chat import (INSTRUCTIONS, MAX_HISTORY_MESSAGES, EmojiLocalChat,
                              emoji_grammar, remember_turn)


class FakeClient:
    def __init__(self, intents):
        self.intents = intents
        self.events = {}

    def event(self, function):
        self.events[function.__name__] = function
        return function


class FakeTyping:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class FakeMessage:
    def __init__(self, content, bot=False, channel_name='🫪', author_id=1, channel_id=10):
        self.author = SimpleNamespace(bot=bot, id=author_id)
        self.content = content
        self.channel = SimpleNamespace(typing=FakeTyping, name=channel_name)
        self.channel.id = channel_id
        self.guild = SimpleNamespace(id=100)
        self.sent = []

    async def reply(self, content, **kwargs):
        self.sent.append(content)


class FakeChat:
    def __init__(self):
        self.questions = []

    def answer(self, question, history=()):
        self.questions.append((question, list(history)))
        return '🐈'


class FakeDiscord:
    Client = FakeClient

    class Intents:
        @staticmethod
        def default():
            return SimpleNamespace(message_content=False)

    class AllowedMentions:
        @staticmethod
        def none():
            return None


class DiscordBotTests(unittest.TestCase):
    def test_prompt_guides_answers_and_gbnf_only_allows_catalog_emojis(self):
        self.assertIn('using recent context', INSTRUCTIONS)
        self.assertIn('up to six distinct', INSTRUCTIONS)
        self.assertIn('Avoid filler', INSTRUCTIONS)
        grammar = emoji_grammar({'🐈': 'cat', '🐈‍⬛': 'black cat', '🐶': 'dog'})
        self.assertIn('root ::= emoji emoji?', grammar)
        self.assertIn(r'\U0001F408', grammar)
        self.assertIn(r'\U0001F436', grammar)
        self.assertNotIn('\u200d', grammar)
        self.assertNotIn('branch', grammar)

    def test_answer_sends_emoji_grammar_and_validates_server_output(self):
        chat = EmojiLocalChat()
        response = io.BytesIO(json.dumps({
            'choices': [{'message': {'content': '🌍'}}],
        }).encode())
        with patch('emoji_local_chat.urlopen', return_value=response) as request_mock:
            self.assertEqual(chat.answer('Where do we live?'), '🌍')

        sent = json.loads(request_mock.call_args.args[0].data)
        self.assertIn('grammar', sent)
        self.assertEqual(sent['chat_template_kwargs'], {'enable_thinking': False})
        self.assertIn(r'\U0001F300-\U0001F320', sent['grammar'])

    def test_starts_local_model_server_when_endpoint_is_down(self):
        chat = EmojiLocalChat()
        process = SimpleNamespace(returncode=None, poll=lambda: None,
                                  terminate=Mock(), wait=Mock())
        with patch('emoji_local_chat.urlopen', side_effect=[URLError('refused'), io.BytesIO(b'{}')]), \
                patch('emoji_local_chat.shutil.which', return_value='/usr/bin/llama'), \
                patch('emoji_local_chat.subprocess.Popen', return_value=process) as popen:
            chat.start_server()

        self.assertEqual(popen.call_args.args[0][:4], [
            '/usr/bin/llama', 'serve', '-hf', 'unsloth/Qwen3.5-0.8B-GGUF:UD-IQ2_XXS',
        ])
        self.assertEqual(popen.call_args.args[0][-4:], [
            '--host', '127.0.0.1', '--port', '8080',
        ])
        chat.close_server()
        process.terminate.assert_called_once()

    def test_finds_llama_in_official_linux_install_location(self):
        chat = EmojiLocalChat()
        process = SimpleNamespace(returncode=None, poll=lambda: None,
                                  terminate=Mock(), wait=Mock())

        class FakePath:
            def __init__(self, value):
                self.value = value

            def __truediv__(self, part):
                return FakePath(f'{self.value}/{part}')

            def is_file(self):
                return True

            def __str__(self):
                return self.value

        with (
            patch('emoji_local_chat.urlopen', side_effect=[URLError('refused'), io.BytesIO(b'{}')]),
            patch('emoji_local_chat.shutil.which', return_value=None),
            patch('emoji_local_chat.os.name', 'posix'),
            patch('emoji_local_chat.Path.home', return_value=FakePath('/home/opc')),
            patch('emoji_local_chat.subprocess.Popen', return_value=process) as popen,
        ):
            chat.start_server()

        self.assertEqual(popen.call_args.args[0][0], '/home/opc/.llama-app/llama')
        process.terminate.assert_not_called()

    def test_uses_server_executable_when_llama_command_is_not_on_path(self):
        chat = EmojiLocalChat()
        process = SimpleNamespace(returncode=None, poll=lambda: None,
                                  terminate=Mock(), wait=Mock())
        with patch('emoji_local_chat.urlopen', side_effect=[URLError('refused'), io.BytesIO(b'{}')]), \
                patch('emoji_local_chat.shutil.which', side_effect=[None, '/usr/bin/llama-server']), \
                patch('emoji_local_chat.Path.is_file', return_value=False), \
                patch('emoji_local_chat.subprocess.Popen', return_value=process) as popen:
            chat.start_server()

        self.assertEqual(popen.call_args.args[0][:2], ['/usr/bin/llama-server', '-hf'])

    def test_reuses_an_already_running_model_server(self):
        chat = EmojiLocalChat()
        with patch('emoji_local_chat.urlopen', return_value=io.BytesIO(b'{}')), \
                patch('emoji_local_chat.subprocess.Popen') as popen:
            chat.start_server()
        popen.assert_not_called()

    def test_first_message_replies_without_prefix_and_starts_with_empty_history(self):
        chat = FakeChat()
        client = create_client(chat, FakeDiscord)
        message = FakeMessage('  Which animal meows?  ')
        asyncio.run(client.events['on_message'](message))
        self.assertTrue(client.intents.message_content)
        self.assertEqual(chat.questions, [('Which animal meows?', [])])
        self.assertEqual(message.sent, ['🐈'])

    def test_keeps_recent_history_separate_per_user(self):
        chat = FakeChat()
        client = create_client(chat, FakeDiscord)
        first = FakeMessage('My favorite animal is a cat.')
        followup = FakeMessage('What is my favorite animal?')
        other_user = FakeMessage('What is my favorite animal?', author_id=2)
        for message in (first, followup, other_user):
            asyncio.run(client.events['on_message'](message))

        self.assertEqual(chat.questions[1][1], [
            {'role': 'user', 'content': 'My favorite animal is a cat.'},
            {'role': 'assistant', 'content': '🐈'},
        ])
        self.assertEqual(chat.questions[2][1], [])

    def test_limits_memory_to_six_turns(self):
        history = []
        for turn in range(7):
            remember_turn(history, str(turn), '🐈')
        self.assertEqual(len(history), MAX_HISTORY_MESSAGES)
        self.assertEqual(history[0], {'role': 'user', 'content': '1'})

    def test_ignores_bot_messages_without_calling_the_model(self):
        chat = FakeChat()
        client = create_client(chat, FakeDiscord)
        message = FakeMessage('hello', bot=True)
        asyncio.run(client.events['on_message'](message))
        self.assertEqual(chat.questions, [])
        self.assertEqual(message.sent, [])

    def test_ignores_messages_outside_the_allowlisted_channel(self):
        chat = FakeChat()
        client = create_client(chat, FakeDiscord)
        message = FakeMessage('hello', channel_name='general')
        asyncio.run(client.events['on_message'](message))
        self.assertEqual(chat.questions, [])
        self.assertEqual(message.sent, [])

    def test_rejects_oversized_messages_without_calling_the_model(self):
        chat = FakeChat()
        client = create_client(chat, FakeDiscord)
        message = FakeMessage('x' * (MAX_MESSAGE_LENGTH + 1))
        asyncio.run(client.events['on_message'](message))
        self.assertEqual(chat.questions, [])
        self.assertEqual(message.sent, ['🙅📏'])

    def test_returns_plain_user_message_without_command_prefix(self):
        message = SimpleNamespace(author=SimpleNamespace(bot=False), content='  Hello there  ',
                                  channel=SimpleNamespace(name='🫪'))
        self.assertEqual(message_text(message), 'Hello there')

    def test_ignores_bot_messages(self):
        message = SimpleNamespace(author=SimpleNamespace(bot=True), content='hello')
        self.assertIsNone(message_text(message))

    def test_ignores_empty_messages(self):
        message = SimpleNamespace(author=SimpleNamespace(bot=False), content='  ',
                                  channel=SimpleNamespace(name='🫪'))
        self.assertIsNone(message_text(message))

    def test_message_limit_is_finite(self):
        self.assertEqual(MAX_MESSAGE_LENGTH, 4000)


if __name__ == '__main__':
    unittest.main()
