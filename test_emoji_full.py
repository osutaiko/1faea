import unittest

import torch

from emoji_catalog import ALPHABET, CATALOG, ENTITIES
from emoji_full import FullEmojiLM, rows
from emoji_lm_model import EmojiLM, START
from semantic_model import FACT, LEFT, SYMBOLS, initial_state


class FullEmojiTests(unittest.TestCase):
    def test_catalog_keeps_atomic_sequences_and_fixed_operator_ids(self):
        self.assertEqual(ALPHABET[:len(SYMBOLS)], SYMBOLS)
        self.assertEqual(len(ALPHABET), len(set(ALPHABET)))
        self.assertGreater(len(ALPHABET), 3900)
        self.assertIn('👩🏽‍🚀', ALPHABET)
        self.assertIn('🇰🇷', ALPHABET)
        self.assertTrue(all(row['name'] for row in CATALOG))
        self.assertTrue(all(row['unicode_name'] for row in CATALOG))

    def test_training_covers_every_entity_and_test_combinations_are_disjoint(self):
        train = rows(len(ENTITIES) * 2, 17, cover=True)
        test = rows(512, 31)
        used = {entity for row in train for entity in row['entities']}
        self.assertEqual(used, set(ENTITIES))
        self.assertFalse({tuple(row['entities']) for row in train} &
                         {tuple(row['entities']) for row in test})

    def test_language_model_can_copy_full_catalog_ids(self):
        model = EmojiLM(torch.randn(len(ALPHABET), 16), width=16).eval()
        entity = ALPHABET.index('👩🏽‍🚀')
        state = initial_state(torch.tensor([[entity, 1, 1, 2, entity, 2]]))
        logs = model(state, torch.randn(1, 16, 16), torch.tensor([[START, FACT, entity, LEFT]]))
        self.assertEqual(logs.shape, (1, 4, len(ALPHABET)))
        torch.testing.assert_close(logs.exp().sum(-1), torch.ones(1, 4))
        self.assertTrue((logs[..., entity].exp() > 0).all())
        self.assertTrue((logs[..., START].exp() < 1e-8).all())

    def test_discrete_identity_features_ignore_entity_renaming(self):
        model = FullEmojiLM(torch.randn(len(ALPHABET), 16), width=16).eval()
        first = initial_state(torch.tensor([[0, 1, 1, 2, 0, 2]]))
        second = initial_state(torch.tensor([[100, 200, 200, 300, 100, 300]]))
        features = torch.randn(1, 16, 16)
        torch.testing.assert_close(model.memory_features(first, features), model.memory_features(second, features))


if __name__ == '__main__':
    unittest.main()
