import contextlib
import io
import unittest
from unittest.mock import Mock, patch

import emoji_local_cli


class LocalCliTests(unittest.TestCase):
    def test_prints_replies_and_keeps_recent_turns(self):
        chat = Mock()
        chat.answer.side_effect = ['🐈', '🐶']
        output = io.StringIO()
        with patch.object(emoji_local_cli, 'EmojiLocalChat', return_value=chat), \
                patch('builtins.input', side_effect=['I like cats', 'What do I like?', '/quit']), \
                contextlib.redirect_stdout(output):
            emoji_local_cli.main()

        self.assertEqual(chat.answer.call_args_list[1].args, ('What do I like?', [
            {'role': 'user', 'content': 'I like cats'},
            {'role': 'assistant', 'content': '🐈'},
        ]))
        self.assertIn('🐈', output.getvalue())
        self.assertIn('🐶', output.getvalue())
        chat.start_server.assert_called_once()
        chat.close_server.assert_called_once()


if __name__ == '__main__':
    unittest.main()
