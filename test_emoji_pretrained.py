from types import SimpleNamespace
import unittest

import torch
from torch import nn

from emoji_pretrained_model import PretrainedReply


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 8)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, use_cache):
        if use_cache:
            raise AssertionError('Runtime must rebuild from hard IDs')
        return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(1))


class PretrainedTests(unittest.TestCase):
    def test_decoder_rejects_continuous_states_and_predicts_only_emoji_classes(self):
        reply = PretrainedReply(TinyBackbone(), torch.randn(5, 8), torch.randn(8)).eval()
        with self.assertRaisesRegex(ValueError, 'integer'):
            reply(torch.randn(1, 2, 8), torch.zeros(1, 1, dtype=torch.long))
        logits = reply(torch.tensor([[1, 2]]), torch.tensor([[5, 3]]))
        self.assertEqual(logits.shape, (1, 2, 6))
        self.assertFalse(reply.vectors.requires_grad)

    def test_generation_stops_on_end_without_a_text_head(self):
        reply = PretrainedReply(TinyBackbone(), torch.randn(5, 8), torch.randn(8)).eval()
        with torch.no_grad():
            reply.output.weight.zero_()
            reply.output.bias.zero_()
            reply.output.bias[reply.end] = 10
            reply.copy_gate.bias.fill_(20)
        self.assertEqual(reply.generate(torch.tensor([[1, 2]])).tolist(), [[reply.end]])

    def test_pointer_preserves_supplied_identity_and_normalizes_probabilities(self):
        reply = PretrainedReply(TinyBackbone(), torch.randn(5, 8), torch.randn(8)).eval()
        states = torch.arange(5)[:, None].expand(-1, 3)
        prefix = torch.full((5, 1), reply.end, dtype=torch.long)
        probabilities = reply(states, prefix).exp()
        self.assertTrue(torch.allclose(probabilities.sum(-1), torch.ones(5, 1), atol=1e-6))
        self.assertEqual(probabilities[:, 0].argmax(-1).tolist(), list(range(5)))


if __name__ == '__main__':
    unittest.main()
