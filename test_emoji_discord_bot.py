import asyncio
import unittest
from types import SimpleNamespace

from emoji_discord_bot import MAX_MESSAGE_LENGTH, create_client, message_text
from emoji_local_chat import INSTRUCTIONS, EmojiTokenGrammar


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
    def __init__(self, content, bot=False):
        self.author = SimpleNamespace(bot=bot)
        self.content = content
        self.channel = SimpleNamespace(typing=FakeTyping)
        self.sent = []

    async def reply(self, content, **kwargs):
        self.sent.append(content)


class FakeChat:
    def __init__(self):
        self.questions = []

    def answer(self, question):
        self.questions.append(question)
        return {'reply': '🐈'}


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
    def test_prompt_guides_answer_quality_while_decoder_enforces_emoji_tokens(self):
        self.assertIn('as a standalone message', INSTRUCTIONS)
        self.assertIn('important facts, negation, quantities, comparisons', INSTRUCTIONS)
        self.assertIn('Never repeat symbols as filler', INSTRUCTIONS)
        grammar = EmojiTokenGrammar({'🐈': (10,)}, eos_token_id=99)
        self.assertEqual(grammar.allowed(grammar.start()), [10])

    def test_token_grammar_allows_only_emoji_paths_and_eos_at_boundaries(self):
        grammar = EmojiTokenGrammar({'🐈': (10,), '🐈‍⬛': (10, 12), '🐶': (11, 13)},
                                    eos_token_id=99, max_emojis=2)
        states = grammar.start()
        self.assertEqual(set(grammar.allowed(states)), {10, 11})

        states = grammar.advance(states, 10)
        self.assertEqual(set(grammar.allowed(states)), {10, 11, 12, 99})

        states = grammar.advance(states, 11)
        self.assertEqual(set(grammar.allowed(states)), {13})
        states = grammar.advance(states, 13)
        self.assertEqual(grammar.allowed(states), [99])

    def test_token_grammar_rejects_tokens_outside_emoji_paths(self):
        grammar = EmojiTokenGrammar({'🐈': (10,)}, eos_token_id=99)
        with self.assertRaisesRegex(ValueError, 'left the emoji vocabulary'):
            grammar.advance(grammar.start(), 42)

    def test_replies_without_prefix_using_only_the_current_message(self):
        chat = FakeChat()
        client = create_client(chat, FakeDiscord)
        message = FakeMessage('  Which animal meows?  ')
        asyncio.run(client.events['on_message'](message))
        self.assertTrue(client.intents.message_content)
        self.assertEqual(chat.questions, ['Which animal meows?'])
        self.assertEqual(message.sent, ['🐈'])

    def test_ignores_bot_messages_without_calling_the_model(self):
        chat = FakeChat()
        client = create_client(chat, FakeDiscord)
        message = FakeMessage('hello', bot=True)
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
        message = SimpleNamespace(author=SimpleNamespace(bot=False), content='  Hello there  ')
        self.assertEqual(message_text(message), 'Hello there')

    def test_ignores_bot_messages(self):
        message = SimpleNamespace(author=SimpleNamespace(bot=True), content='hello')
        self.assertIsNone(message_text(message))

    def test_ignores_empty_messages(self):
        message = SimpleNamespace(author=SimpleNamespace(bot=False), content='  ')
        self.assertIsNone(message_text(message))

    def test_message_limit_is_finite(self):
        self.assertEqual(MAX_MESSAGE_LENGTH, 4000)


if __name__ == '__main__':
    unittest.main()
