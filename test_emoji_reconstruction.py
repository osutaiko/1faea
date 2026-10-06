from types import SimpleNamespace
import unittest

import torch
from torch import nn

from emoji_reconstruction_fit import BlindReconstruction, state_vectors


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=True)
        self.embedding = nn.Embedding(7, 4)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, use_cache):
        self.values = inputs_embeds.detach()
        return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(1))


class ReconstructionTests(unittest.TestCase):
    def test_shifted_prefix_control_exposes_only_previous_answer_tokens(self):
        backbone = Backbone()
        decoder = BlindReconstruction(backbone, 6, teacher_prefix=True)
        states = torch.randn(1, 3, 4)
        first = decoder(states, torch.tensor([[1, 2, 3]]))
        second = decoder(states, torch.tensor([[4, 5, 0]]))
        self.assertTrue(torch.equal(first[:, 0], second[:, 0]))
        self.assertFalse(torch.equal(first[:, 1:], second[:, 1:]))
        self.assertTrue(torch.equal(backbone.values[:, 3:], backbone.embedding(torch.tensor([[6, 4, 5]]))))

    def test_continuous_control_removes_only_hard_quantization(self):
        logits = torch.tensor([[[2., 1., 0.]]], requires_grad=True)
        encoder = lambda features, mask: (logits, None, logits.argmax(-1))
        features = torch.zeros(1, 1, 6)
        book = torch.eye(3)
        hard, ids = state_vectors(encoder, features, book)
        soft, soft_ids = state_vectors(encoder, features, book, continuous=True)
        self.assertTrue(torch.equal(hard, book[ids]))
        self.assertTrue(torch.equal(ids, soft_ids))
        self.assertFalse(torch.equal(hard, soft))
        soft.square().sum().backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_target_words_never_enter_decoder_and_gradients_reach_states(self):
        backbone = Backbone()
        decoder = BlindReconstruction(backbone, 6)
        states = torch.randn(1, 3, 4, requires_grad=True)
        first = decoder(states, torch.tensor([[1, 2, 3]]))
        second = decoder(states, torch.tensor([[4, 5, 0]]))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(backbone.values[:, 3:], backbone.embedding(torch.full((1, 3), 6))))
        second.square().sum().backward()
        self.assertGreater(states.grad.abs().sum().item(), 0)
        self.assertIsNone(backbone.embedding.weight.grad)


if __name__ == '__main__':
    unittest.main()
