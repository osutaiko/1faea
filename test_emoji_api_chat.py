import json
import unittest
from unittest.mock import patch

from emoji_api_chat import DIRECT, EmojiAPIChat, EmojiDirectAPIChat, catalog_symbols


class EmojiAPIChatTests(unittest.TestCase):
    def test_direct_answer_uses_the_full_question_in_one_call(self):
        chat = object.__new__(EmojiDirectAPIChat)
        chat.meanings = {'🐈': 'cat', '🐶': 'dog'}
        with patch('emoji_api_chat.structured_emojis', return_value=['🐈']) as call:
            result = chat.answer('Which animal meows?')
        self.assertEqual(result, {'symbols': ['🐈'], 'reply': '🐈'})
        self.assertEqual(call.call_args.args, ('Which animal meows?', DIRECT))
        self.assertEqual(call.call_count, 1)

    def test_splits_multiple_catalog_symbols_returned_as_one_item(self):
        self.assertEqual(catalog_symbols(['🚫💧'], {'🚫': 'no', '💧': 'water'}), ['🚫', '💧'])

    def test_rejects_symbols_outside_the_catalog(self):
        with self.assertRaisesRegex(ValueError, 'outside the project catalog'):
            catalog_symbols(['⏻'], {'🔌': 'plug'})

    def test_each_request_starts_fresh_and_decoder_gets_only_emoji_state(self):
        chat = object.__new__(EmojiAPIChat)
        chat.meanings = {'🐈': 'cat', '🐶': 'dog', '❓': 'question'}
        with patch('emoji_api_chat.structured_emojis', side_effect=[['🐈'], ['🐈'], ['🐶'], ['🐶']]) as call:
            first = chat.answer('Which animal meows?')
            second = chat.answer('Which animal barks?')

        self.assertEqual(first['reply'], '🐈')
        self.assertEqual(second['reply'], '🐶')
        self.assertEqual(call.call_args_list[0].args[0], 'Which animal meows?')
        decoder_input = json.loads(call.call_args_list[1].args[0])
        self.assertEqual(decoder_input, {'emoji_state': ['🐈'],
                                         'symbol_meanings': [{'symbol': '🐈', 'meaning': 'cat'}]})
        self.assertNotIn('Which animal meows?', call.call_args_list[1].args[0])
        self.assertEqual(call.call_args_list[2].args[0], 'Which animal barks?')

    def test_rejects_symbols_outside_the_catalog(self):
        chat = object.__new__(EmojiAPIChat)
        chat.meanings = {'🐈': 'cat'}
        with patch('emoji_api_chat.structured_emojis', return_value=['cat']):
            with self.assertRaisesRegex(ValueError, 'outside the project catalog'):
                chat.answer('Which animal meows?')

    def test_rejects_empty_requests_without_an_api_call(self):
        chat = object.__new__(EmojiAPIChat)
        chat.meanings = {}
        with patch('emoji_api_chat.structured_emojis') as call:
            with self.assertRaisesRegex(ValueError, 'non-empty'):
                chat.answer('   ')
        call.assert_not_called()


if __name__ == '__main__':
    unittest.main()
