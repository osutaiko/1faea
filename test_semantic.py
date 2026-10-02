import unittest
from types import SimpleNamespace

import torch
from torch import nn

from semantic_data import examples, verdict
from semantic_model import PretrainedReader, SemanticReasoner, TRUE, FALSE, UNKNOWN
from semantic_model import initial_state, expanded_state
from semantic import evaluate


class SemanticTests(unittest.TestCase):
    def audit_trace(self, parsed, derived):
        class Reader:
            def state_features(self, state):
                return torch.zeros(len(state), 16, 16)
        class Model:
            def generate(self, reader, texts):
                first = initial_state(torch.tensor([parsed]))
                return first, expanded_state(first, torch.tensor([derived])), torch.tensor([TRUE])
            def derive_entities(self, reader, state):
                return torch.tensor([[0, 2]])
            def answer_logits(self, features):
                return torch.tensor([[10.0, 0.0, 0.0]])
        row = dict(text='audit', entities=[0, 1, 1, 2, 0, 2], derived=[0, 2], answer=TRUE)
        return evaluate(Model(), Reader(), [row])

    def test_correct_answer_cannot_hide_an_incorrect_parse(self):
        result = self.audit_trace([1, 0, 2, 1, 0, 2], [2, 0])
        self.assertEqual(result['answer'], 1)
        self.assertEqual(result['parse'], 0)
        self.assertEqual(result['correct_trace'], 0)

    def test_repeating_an_input_fact_is_not_a_new_deduction(self):
        result = self.audit_trace([0, 1, 1, 2, 0, 2], [0, 1])
        self.assertEqual(result['valid_proof'], 1)
        self.assertEqual(result['novel_proof'], 0)
        self.assertEqual(result['correct_trace'], 0)

    def test_reference_semantics(self):
        edges = [(0, 1), (1, 2)]
        self.assertEqual(verdict(edges, (0, 2)), TRUE)
        self.assertEqual(verdict(edges, (2, 0)), FALSE)
        self.assertEqual(verdict(edges, (0, 3)), UNKNOWN)
        self.assertIsNone(verdict([(0, 1), (1, 0)], (0, 1)))

    def test_compositional_split(self):
        splits = examples()
        chains = {name: {tuple(row['chain']) for row in rows} for name, rows in splits.items()}
        self.assertFalse(chains['train'] & chains['validation'])
        self.assertFalse(chains['train'] & chains['test'])
        self.assertFalse(chains['validation'] & chains['test'])
        templates = {name: {row['template'] for row in rows} for name, rows in splits.items()}
        self.assertFalse(templates['train'] & templates['validation'])
        self.assertFalse(templates['train'] & templates['test'])
        prompts = [row['text'] for rows in splits.values() for row in rows]
        self.assertEqual(len(prompts), len(set(prompts)))
        for rows in splits.values():
            for row in rows:
                a, b, c, d, x, y = row['entities']
                self.assertEqual(row['answer'], verdict([(a, b), (c, d)], (x, y)))
                self.assertEqual(verdict([(a, b), (c, d)], tuple(row['derived'])), TRUE)

    def test_deduction_adapter_learns_from_supervision(self):
        model = SemanticReasoner(16, width=16)
        logits = model.derive_logits(torch.randn(2, 16, 16))
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), torch.tensor([0, 2, 1, 3]))
        loss.backward()
        for parameter in (model.derive_queries, model.state_positions, model.derive_head.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_fresh_backbone_receives_only_symbol_embeddings(self):
        class Backbone(nn.Module):
            def forward(self, **kwargs):
                self.arguments = kwargs
                return SimpleNamespace(last_hidden_state=kwargs['inputs_embeds'])
        reader = PretrainedReader.__new__(PretrainedReader)
        reader.backbone = Backbone()
        reader.symbol_embedding = nn.Embedding(11, 16)
        state = torch.randint(11, (2, 16))
        reader.state_features(state)
        self.assertEqual(set(reader.backbone.arguments), {'inputs_embeds', 'use_cache'})
        self.assertFalse(reader.backbone.arguments['use_cache'])
        torch.testing.assert_close(reader.backbone.arguments['inputs_embeds'], reader.symbol_embedding(state))

    def test_generation_crosses_two_discrete_boundaries(self):
        class Reader:
            def __init__(self):
                self.states = []
            def text_features(self, texts):
                return torch.randn(len(texts), 8, 16), torch.ones(len(texts), 8, dtype=torch.bool)
            def state_features(self, state):
                self.states.append(state.clone())
                self.assert_integer(state)
                return torch.nn.functional.one_hot(state, 16).float()
            @staticmethod
            def assert_integer(state):
                assert state.dtype == torch.long
        reader = Reader()
        model = SemanticReasoner(16, width=16).eval()
        first, second, answer = model.generate(reader, ['input'])
        self.assertEqual(len(reader.states), 2)
        torch.testing.assert_close(first, reader.states[0])
        torch.testing.assert_close(second, reader.states[1])
        self.assertIn(answer.item(), (TRUE, FALSE, UNKNOWN))
        self.assertEqual(second[0, 8].item(), 4)
        self.assertEqual(second[0, 10].item(), 5)


if __name__ == '__main__':
    unittest.main()
