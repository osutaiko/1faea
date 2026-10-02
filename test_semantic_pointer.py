import unittest
from unittest.mock import patch

import torch

from semantic_data import examples
from semantic_clauses import clause_examples, challenge_examples, audit_examples
from semantic_pointer import derive_targets
from semantic_pointer_model import PointerHead, PointerReasoner, split_clauses, mentions


class PointerTests(unittest.TestCase):
    def test_gold_pointers_match_semantic_labels(self):
        for rows in clause_examples().values():
            for row in rows:
                text = row['text']
                _, candidates = mentions(text, [(i, i + 1) for i in range(len(text))])
                orientation = row['orientation']
                self.assertEqual([candidates[orientation], candidates[1 - orientation]], [row['left'], row['right']])
        for rows in examples().values():
            for row in rows:
                self.assertEqual([row['entities'][i] for i in derive_targets(row)], row['derived'])

    def test_unsupported_lexical_inputs_are_explicit(self):
        with self.assertRaisesRegex(ValueError, 'two explicit entity mentions'):
            mentions('The kitten is left of the dog.', [(0, 3)])

    def test_clauses_are_separate_pretrained_inputs(self):
        text = 'The cat is left of the dog. The dog is left of the bird. Is the cat left of the bird?'
        clauses = split_clauses(text)
        self.assertEqual(clauses, ['The cat is left of the dog.', 'The dog is left of the bird.', 'Is the cat left of the bird?'])
        with self.assertRaisesRegex(ValueError, 'two separate fact sentences'):
            split_clauses('cat dog bird fish cat dog')

    def test_fresh_challenge_clauses_are_excluded_from_parser_training(self):
        splits = clause_examples()
        training = {row['text'] for row in splits['train']}
        validation = {row['text'] for row in splits['validation']}
        self.assertFalse(training & validation)
        challenge = challenge_examples()
        audit = audit_examples()
        challenge_clauses = {clause for row in challenge for clause in split_clauses(row['text'])}
        audit_clauses = {clause for row in audit for clause in split_clauses(row['text'])}
        self.assertFalse(challenge_clauses & audit_clauses)
        for row in challenge + audit:
            for clause in split_clauses(row['text']):
                self.assertNotIn(clause, training | validation)
                mentions(clause, [(i, i + 1) for i in range(len(clause))])

    def test_pointer_scores_ignore_padding_and_feature_sequence_offset(self):
        head = PointerHead(16, 16, 1, 2, 1).eval()
        features = torch.randn(2, 8, 16)
        positions = torch.tensor([[2, 6], [1, 5]])
        original = head(features, torch.ones(2, 8, dtype=torch.bool), positions)
        padded = torch.cat((torch.zeros(2, 3, 16), features, torch.zeros(2, 2, 16)), dim=1)
        mask = torch.cat((torch.zeros(2, 3, dtype=torch.bool), torch.ones(2, 8, dtype=torch.bool),
                          torch.zeros(2, 2, dtype=torch.bool)), dim=1)
        torch.testing.assert_close(head(padded, mask, positions + 3), original)

    def test_relative_position_weights_receive_supervision(self):
        head = PointerHead(16, 16, 1, 2, 1)
        logits = head(torch.randn(2, 8, 16), torch.ones(2, 8, dtype=torch.bool), torch.tensor([[2, 6], [1, 5]]))
        torch.nn.functional.cross_entropy(logits.squeeze(1), torch.tensor([0, 1])).backward()
        self.assertGreater(head.relative_positions.weight.grad.abs().sum().item(), 0)

    def test_generation_copies_sources_and_crosses_discrete_boundaries(self):
        class Reader:
            def __init__(self):
                self.states = []
            def state_features(self, state):
                self.states.append(state.clone())
                return torch.nn.functional.one_hot(state, 16).float()
        model = PointerReasoner(16, width=16).eval()
        reader = Reader()
        text = (torch.randn(3, 4, 32), torch.ones(3, 4, dtype=torch.bool),
                torch.tensor([[0, 1], [0, 1], [0, 1]]), torch.tensor([[1, 0], [2, 1], [0, 2]]))
        parsed = torch.nn.functional.one_hot(torch.tensor([[1], [1], [0]]), 2).float() * 10
        derived = torch.nn.functional.one_hot(torch.tensor([[0, 3]]), 4).float() * 10
        with patch('semantic_pointer_model.text_inputs', return_value=text) as encoding, \
                patch.object(model.parser, 'forward', return_value=parsed), \
                patch.object(model, 'derive_logits', return_value=derived):
            first, second, _ = model.generate(reader, ['The dog is right of the cat. The bird is right of the dog. Is the cat left of the bird?'])
        self.assertEqual(encoding.call_args.args[1], ['The dog is right of the cat.', 'The bird is right of the dog.', 'Is the cat left of the bird?'])
        self.assertEqual(first[0, [1, 3, 5, 7, 13, 15]].tolist(), [0, 1, 1, 2, 0, 2])
        self.assertEqual(second[0, [9, 11]].tolist(), [0, 2])
        self.assertEqual(len(reader.states), 2)
        torch.testing.assert_close(reader.states[0], first)
        torch.testing.assert_close(reader.states[1], second)


if __name__ == '__main__':
    unittest.main()
