import json
import unittest
from jsonschema import ValidationError, validate

from emoji_two_pass import ANSWER, emoji_schema, second_messages


class TwoPassTests(unittest.TestCase):
    def test_transport_preserves_multicodepoint_emojis_and_rejects_english(self):
        symbols = ['🐈', '👨‍👩‍👧‍👦', '1️⃣']
        schema = emoji_schema(symbols)
        raw = json.dumps(symbols, ensure_ascii=False)
        validate(json.loads(raw), schema)
        self.assertEqual(json.loads(raw), symbols)
        with self.assertRaises(ValidationError):
            validate(['cat'], schema)
        with self.assertRaises(ValidationError):
            validate(symbols * 2, schema)

    def test_second_call_has_only_fixed_instruction_and_supplied_emoji_state(self):
        state = ['🐕', '🐈', '❓']
        messages = second_messages(state)
        self.assertEqual(messages, [dict(role='system', content=ANSWER),
                                    dict(role='user', content=json.dumps(state, ensure_ascii=False))])
        self.assertEqual(json.loads(messages[1]['content']), state)


if __name__ == '__main__':
    unittest.main()
