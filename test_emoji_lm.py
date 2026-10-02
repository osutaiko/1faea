import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from emoji_lm import generate
from emoji_lm_model import EmojiLM, END, START, visible_tokens
from semantic_model import initial_state, FACT, LEFT, TRUE


class EmojiLMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = EmojiLM(torch.randn(13, 16), width=16).eval()
        self.state = initial_state(torch.tensor([[0, 1, 1, 2, 0, 2]]))
        self.features = torch.randn(1, 16, 16)

    def test_next_symbol_distribution_is_normalized(self):
        prefix = torch.tensor([[START, FACT, 0]])
        logs = self.model(self.state, self.features, prefix)
        self.assertTrue(torch.isfinite(logs).all())
        torch.testing.assert_close(logs.exp().sum(-1), torch.ones(1, 3))

    def test_teacher_forcing_does_not_expose_future_symbols(self):
        first = torch.tensor([[START, FACT, 0, LEFT, 2]])
        second = torch.tensor([[START, FACT, 1, LEFT, 3]])
        original = self.model(self.state, self.features, first)
        changed = self.model(self.state, self.features, second)
        torch.testing.assert_close(original[:, :2], changed[:, :2])

    def test_each_generated_symbol_rebuilds_from_integer_states(self):
        class Reader:
            def __init__(self):
                self.states = []
            def state_features(self, state):
                self.states.append(state.clone())
                return torch.zeros(1, 16, 16)
        continuation = [FACT, 0, LEFT, 2, TRUE, END]
        prefixes = []
        def forward(state, features, prefix):
            prefixes.append(prefix.clone())
            scores = torch.full((1, prefix.shape[1], END + 1), -10.0)
            scores[:, -1, continuation[prefix.shape[1] - 1]] = 0
            return scores
        reader = Reader()
        with patch.object(self.model, 'forward', side_effect=forward):
            output = self.model.generate(reader, self.state)
        self.assertEqual(output.tolist(), [continuation])
        self.assertEqual(visible_tokens(output[0]), continuation[:-1])
        self.assertEqual(len(reader.states), len(continuation))
        for index, (state, prefix) in enumerate(zip(reader.states, prefixes)):
            self.assertEqual(state.dtype, torch.long)
            torch.testing.assert_close(state, self.state)
            self.assertEqual(prefix.tolist(), [[START, *continuation[:index]]])

    def test_language_generation_does_not_use_the_classifiers_answer(self):
        class Pointer:
            def parse_state(pointer, reader, texts):
                return self.state
            def derive_entities(pointer, *args):
                raise AssertionError('Language head must generate its own deduction')
            def answer_logits(pointer, *args):
                raise AssertionError('Language head must generate its own answer')
        model = SimpleNamespace(generate=lambda reader, state: torch.tensor([[FACT, 0, LEFT, 2, TRUE, END]]))
        output = io.StringIO()
        with patch('emoji_lm.load', return_value=(model, Pointer(), None, None)), contextlib.redirect_stdout(output):
            generate(SimpleNamespace(text='input', trace=False))
        self.assertEqual(output.getvalue().strip(), '📌🐱⬅️🐦✅')


if __name__ == '__main__':
    unittest.main()
