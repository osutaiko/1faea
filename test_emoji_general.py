import unittest

import torch

from emoji_general_model import GeneralEncoder, compact
from emoji_general_run import full_meanings, targets


class GeneralTests(unittest.TestCase):
    def test_catalog_includes_every_official_sequence_and_control_looking_emoji(self):
        rows, report = full_meanings()
        self.assertEqual(len(rows), report['official_symbols'])
        self.assertEqual(len({row['symbol'] for row in rows}), len(rows))
        self.assertEqual([row['id'] for row in rows], list(range(len(rows))))
        self.assertTrue({'🔚', '▶️', '🫪'} <= {row['symbol'] for row in rows})

    def test_compaction_excludes_padding_and_encoder_commits_exact_vectors(self):
        vectors = torch.eye(16)
        features = torch.cat((vectors[:3], vectors[:3]), dim=-1).unsqueeze(0)
        features = torch.cat((features, torch.full((1, 3, 32), 1e6)), dim=1)
        mask = torch.tensor([[True, True, True, False, False, False]])
        values = compact(features, mask, length=3)
        self.assertLess(values.abs().max().item(), 2)
        encoder = GeneralEncoder(vectors, slots=3, width=16)
        logits, selected, ids = encoder(values, torch.ones(1, 3, dtype=torch.bool))
        self.assertTrue(torch.equal(selected, vectors[ids]))
        selected.square().sum().backward()
        self.assertTrue(torch.isfinite(encoder.output.weight.grad).all())

    def test_variable_targets_use_literal_anchors_and_terminate(self):
        ids = torch.tensor([[1, 1, 2, 2], [3, 3, 3, 3]])
        anchors = torch.tensor([[4, -100, -100, -100], [-100, -100, -100, -100]])
        prefix, labels = targets(ids, anchors, 5)
        self.assertEqual(labels.tolist(), [[4, 1, 2, 5], [3, 5, -100, -100]])
        self.assertEqual(prefix.tolist(), [[5, 4, 1, 2], [5, 3, 5, 5]])


if __name__ == '__main__':
    unittest.main()
