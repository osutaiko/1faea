import unittest

import torch

from emoji_grounded_model import GroundedEncoder, GroundedReply, MeaningLexicon, discrete_vectors, ordered_content_loss
from emoji_meanings import base_name, validate_associations


class GroundedTests(unittest.TestCase):
    def test_shared_embedding_offset_does_not_collapse_literal_initialization(self):
        vectors = torch.eye(16) + 100
        lexical = torch.nn.functional.layer_norm(vectors[:3], (16,)).unsqueeze(0)
        features = torch.cat((torch.zeros_like(lexical), lexical), dim=-1)
        encoder = GroundedEncoder(vectors, slots=3, width=16).eval()
        ids = encoder(features, torch.ones(1, 3, dtype=torch.bool))[2]
        self.assertEqual(ids.tolist(), [[0, 1, 2]])
        with torch.no_grad():
            encoder.output.weight.fill_(1000)
            encoder.output.bias.fill_(1000)
        ids = encoder(features, torch.ones(1, 3, dtype=torch.bool))[2]
        self.assertEqual(ids.tolist(), [[0, 1, 2]])

    def test_forward_vectors_are_exact_codes_and_gradients_reach_logits(self):
        torch.manual_seed(2)
        vectors = torch.randn(5, 8)
        logits = torch.randn(2, 3, 5, requires_grad=True)
        selected, ids = discrete_vectors(logits, vectors)
        self.assertTrue(torch.equal(selected, vectors[ids]))
        selected.square().sum().backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_map_variants_keep_core_identity(self):
        self.assertEqual(base_name('thumbs up: medium-dark skin tone'), 'thumbs up')
        self.assertEqual(base_name('people holding hands: light skin tone, dark skin tone'), 'people holding hands')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_associations([dict(id='cat', associations=['pet', 'pet'])], ['cat'])

    def test_anchors_use_whole_words_and_literal_core_priority(self):
        rows = [dict(core_meaning='cat', associations=['pet']), dict(core_meaning='dog', associations=['pet'])]
        lexicon = MeaningLexicon(rows)
        self.assertEqual(lexicon.anchors('The cat is a pet.', 8), [0])
        self.assertEqual(lexicon.anchors('concatenate', 8), [])
        self.assertEqual(lexicon.anchors('a pet', 8), [])

    def test_ordered_content_penalizes_reversed_roles(self):
        values = torch.eye(3).unsqueeze(0)
        features = torch.cat((values, values), dim=-1)
        mask = torch.ones(1, 3, dtype=torch.bool)
        projection = torch.nn.Identity()
        correct = ordered_content_loss(values, features, mask, projection)
        wrong = ordered_content_loss(values.flip(1), features, mask, projection)
        self.assertLess(correct.item(), wrong.item())

    def test_meaning_evaluation_rejects_reversed_roles_and_forbidden_concepts(self):
        from emoji_grounded_eval import score
        case = dict(groups=['🐕🐶', '🐈🐱'], forbidden='', ordered=True)
        self.assertTrue(score(['🐕', '➡️', '🐈'], case))
        self.assertFalse(score(['🐈', '➡️', '🐕'], case))
        self.assertFalse(score(['🐈', '🐕'], dict(groups=['🐈🐱'], forbidden='🐕🐶')))

    def test_only_integer_state_reaches_direct_runtime(self):
        torch.manual_seed(3)
        vectors = torch.randn(5, 16)
        encoder = GroundedEncoder(vectors, slots=3, width=16).eval()
        reply = GroundedReply(vectors, slots=3, width=16).eval()
        logits, selected, ids = encoder(torch.randn(2, 4, 32), torch.ones(2, 4, dtype=torch.bool))
        self.assertTrue(torch.equal(selected, vectors[ids]))
        with self.assertRaisesRegex(ValueError, 'integer'):
            reply.generate(selected)
        generated = reply.generate(ids)
        self.assertTrue(((generated >= 0) & (generated <= reply.end)).all())
        self.assertFalse(encoder.vectors.requires_grad)
        self.assertFalse(reply.vectors.requires_grad)
        self.assertFalse(reply.embedding.weight.requires_grad)


if __name__ == '__main__':
    unittest.main()
