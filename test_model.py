"""Checks for the enforced bottleneck, gradient path, and causal decoder."""

import unittest

import torch
from torch.nn import functional as F

from model import BOS, EMOJIS, EOS, EmojiLanguageModel, decode_emojis, encode_emojis


class BottleneckTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = EmojiLanguageModel(input_dim=12, width=16)
        self.features = torch.randn(3, 12)

    def test_composed_emojis_are_atomic_and_text_is_rejected(self):
        text = "❄️2️⃣❤️🐱"
        tokens = encode_emojis(text)
        self.assertEqual(len(tokens), 4)
        self.assertEqual(decode_emojis(tokens), text)
        with self.assertRaises(ValueError):
            encode_emojis("🐱 cat")

    def test_states_are_one_hot_in_training_and_inference(self):
        for training in (True, False):
            self.model.train(training)
            state, trace = self.model.reason(self.features)
            self.assertTrue(torch.equal(state, F.one_hot(state.argmax(-1), len(EMOJIS)).float()))
            self.assertEqual(tuple(trace.shape), (3, 3, 4))
            self.assertTrue((trace < EOS).all())

    def test_each_round_receives_only_reembedded_discrete_state(self):
        self.model.eval()
        received = []
        handle = self.model.transition.register_forward_pre_hook(
            lambda module, args: received.append(args[0].detach().clone())
        )
        _, trace = self.model.reason(self.features)
        handle.remove()
        for index, actual in enumerate(received):
            expected = self.model.emoji_embedding(trace[:, index]) + self.model.slot_position
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_final_loss_trains_the_input_and_intermediate_quantizers(self):
        prefix = torch.full((3, 1), BOS)
        logits, _ = self.model(self.features, prefix)
        F.cross_entropy(logits[:, 0], torch.tensor([1, 2, 3])).backward()
        for layer in (self.model.input_projection[1], self.model.state_head):
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)

    def test_decoder_cannot_see_future_answer_tokens(self):
        self.model.eval()
        state, _ = self.model.reason(self.features)
        prefix = torch.tensor([[BOS, 1, 2], [BOS, 2, 3], [BOS, 3, 4]])
        changed = prefix.clone()
        changed[:, 2] = 5
        before = self.model.answer_logits(state, prefix)
        after = self.model.answer_logits(state, changed)
        torch.testing.assert_close(before[:, :2], after[:, :2], rtol=0, atol=0)

    def test_generation_is_emoji_only_and_deterministic(self):
        self.model.eval()
        first, trace = self.model.generate(self.features)
        second, second_trace = self.model.generate(self.features)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(trace, second_trace))
        for row in first:
            text = decode_emojis(row.tolist())
            self.assertTrue(text)
            encode_emojis(text)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
