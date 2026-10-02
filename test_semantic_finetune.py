import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import LlamaConfig, LlamaModel

from semantic_finetune import AdaptedReader
from semantic_model import PretrainedReader


class FineTuneTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        config = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, vocab_size=16)
        self.prefix = PretrainedReader.__new__(PretrainedReader)
        self.prefix.backbone = LlamaModel(config).eval().requires_grad_(False)
        self.prefix.symbol_embedding = nn.Embedding.from_pretrained(torch.randn(11, 32), freeze=True)
        self.original = copy.deepcopy(self.prefix.backbone)
        with patch('semantic_finetune.PretrainedReader', return_value=self.prefix):
            self.reader = AdaptedReader()

    def test_split_pretrained_pass_matches_full_pass_with_padding(self):
        ids = torch.tensor([[0, 4, 1, 5], [0, 2, 7, 7]])
        embeddings = self.prefix.symbol_embedding(ids)
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
        with torch.no_grad():
            expected = self.original(inputs_embeds=embeddings, attention_mask=mask, use_cache=False).last_hidden_state
            frozen = self.prefix.backbone(inputs_embeds=embeddings, attention_mask=mask, use_cache=False).last_hidden_state
            actual = self.reader.finish(frozen, mask)
        torch.testing.assert_close(actual[mask], expected[mask])

    def test_pretrained_block_changes_without_updating_prefix(self):
        state = torch.randint(11, (2, 16))
        frozen = self.prefix.state_features(state)
        parameter = self.reader.last_layer.self_attn.q_proj.weight
        before = parameter.detach().clone()
        optimizer = torch.optim.SGD(self.reader.last_layer.parameters(), lr=0.1)
        actual = self.reader.finish(frozen, torch.ones_like(state, dtype=torch.bool))
        actual[:, -1, 0].sum().backward()
        self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in self.prefix.backbone.parameters()))
        self.assertFalse(frozen.requires_grad)
        optimizer.step()
        self.assertGreater((parameter - before).abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
