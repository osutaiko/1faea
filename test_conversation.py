import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from conversation_data import USER, ASSISTANT, SEPARATOR, IDS, SKILLS, splits
from conversation_model import AtomicDecoder, SemanticAtomicDecoder, EmojiConversation, MEMORY_LIMIT, memory, visible
from conversation_composition import composition_splits
from conversation_dialogue_fit import examples as dialogue_examples
from emoji_catalog import ALPHABET
from emoji_lm_model import END, START


class ConversationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = AtomicDecoder(torch.randn(len(ALPHABET), 16), width=16, layers=1).eval()

    def test_distribution_is_normalized_with_and_without_copy_candidates(self):
        features = torch.randn(2, 5, 32)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        sources = torch.tensor([[-1, -1, -1, -1, -1], [USER, IDS['💖'], IDS['🍕'], SEPARATOR, ASSISTANT]])
        logs = self.model(features, mask, sources, torch.tensor([[START, IDS['👋']], [START, IDS['👍']]]))
        self.assertTrue(torch.isfinite(logs).all())
        torch.testing.assert_close(logs.exp().sum(-1), torch.ones(2, 2))
        self.assertTrue((logs[..., START].exp() < 1e-8).all())

    def test_teacher_forcing_is_causal(self):
        features = torch.randn(1, 4, 32)
        mask = torch.ones(1, 4, dtype=torch.bool)
        source = torch.tensor([[USER, IDS['💖'], IDS['🍕'], SEPARATOR]])
        first = torch.tensor([[START, IDS['👍'], IDS['🍕']]])
        second = torch.tensor([[START, IDS['👍'], IDS['🍔']]])
        original = self.model(features, mask, source, first)
        changed = self.model(features, mask, source, second)
        torch.testing.assert_close(original[:, :2], changed[:, :2])

    def test_semantic_head_accepts_quantized_cache_and_normalizes_outputs(self):
        model = SemanticAtomicDecoder(torch.randn(len(ALPHABET), 16), width=16, layers=1).eval()
        features = torch.randn(1, 3, 32).to(torch.float8_e4m3fn)
        mask = torch.ones(1, 3, dtype=torch.bool)
        sources = torch.full((1, 3), -1)
        logs = model(features, mask, sources, torch.tensor([[START]]))
        self.assertTrue(torch.isfinite(logs).all())
        torch.testing.assert_close(logs.exp().sum(-1), torch.ones(1, 1))
        self.assertLess(logs[..., START].exp().item(), 1e-8)

    def test_reply_rebuilds_features_from_hard_memory_at_every_symbol(self):
        state, mask = memory([[]], [[IDS['👋']]])
        calls = []
        def inputs(source, source_mask):
            calls.append((source.clone(), source_mask.clone()))
            return torch.randn(1, 2, 32)
        reader = SimpleNamespace(emoji_inputs=inputs)
        with patch.object(self.model, 'select', side_effect=[torch.tensor([IDS['👋']]), torch.tensor([IDS['😊']]), torch.tensor([END])]):
            result = self.model.respond(reader, state, mask)
        self.assertEqual(visible(result[0]), [IDS['👋'], IDS['😊']])
        self.assertEqual(len(calls), 3)
        for source, source_mask in calls:
            self.assertEqual(source.dtype, torch.long)
            torch.testing.assert_close(source, state)
            torch.testing.assert_close(source_mask, mask)

    def test_session_keeps_only_emoji_ids_and_passes_no_text_features_to_reply(self):
        seen = []
        encoder = SimpleNamespace(encode=lambda reader, texts: torch.tensor([[IDS['💖'], IDS['🍕'], END]]))
        def respond(reader, state, mask):
            seen.append(state.tolist())
            return torch.tensor([[IDS['👍'], IDS['🍕'], END]])
        session = EmojiConversation(encoder, SimpleNamespace(respond=respond), None)
        result = session.turn('I like pizza.')
        self.assertEqual(seen, [[[USER, IDS['💖'], IDS['🍕']]]])
        self.assertEqual(result['state'], [IDS['💖'], IDS['🍕']])
        self.assertEqual(session.history, [USER, IDS['💖'], IDS['🍕'], SEPARATOR, ASSISTANT, IDS['👍'], IDS['🍕'], SEPARATOR])
        self.assertTrue(all(isinstance(token, int) for token in session.history))

    def test_memory_padding_is_masked_and_context_overflow_is_explicit(self):
        state, mask = memory([[], [USER, IDS['👋']]], [[IDS['🙏']], [IDS['💖'], IDS['🍕']]])
        self.assertEqual(mask.sum(1).tolist(), [2, 5])
        self.assertEqual(state.shape, (2, 5))
        with self.assertRaisesRegex(ValueError, 'memory limit'):
            memory([[USER] * MEMORY_LIMIT], [[IDS['👋']]])

    def test_session_evicts_whole_turns_without_retaining_text(self):
        encoder = SimpleNamespace(encode=lambda reader, texts: torch.tensor([[IDS['👋'], END]]))
        decoder = SimpleNamespace(respond=lambda reader, state, mask: torch.tensor([[IDS['👋'], IDS['😊'], END]]))
        session = EmojiConversation(encoder, decoder, None)
        for _ in range(30):
            session.turn('hello')
            self.assertLessEqual(len(session.history), MEMORY_LIMIT)
        self.assertGreater(session.dropped_turns, 0)
        self.assertEqual(sum(session.turn_lengths), len(session.history))
        self.assertEqual(session.history[0], USER)
        self.assertTrue(all(isinstance(token, int) for token in session.history))

    def test_skills_have_separate_training_validation_and_test_phrases(self):
        for _, _, _, train, validation, test in SKILLS:
            self.assertFalse(set(train) & set(validation))
            self.assertFalse(set(train) & set(test))
            self.assertFalse(set(validation) & set(test))
        data = splits()
        self.assertTrue(any(row['history'] for row in data['test']))
        self.assertTrue(any(row['skill'] == 'updated_recall' for row in data['test']))

    def test_semantic_memory_does_not_have_conflicting_reply_targets(self):
        meanings = {}
        for rows in composition_splits().values():
            for row in rows:
                key = (tuple(row['history']), tuple(row['state']))
                target = tuple(row['reply'])
                if key in meanings:
                    self.assertEqual(meanings[key], target)
                meanings[key] = target

    def test_long_dialogue_targets_follow_user_updates_despite_wrong_assistant_history(self):
        rows = dialogue_examples(4, 337)
        updated = next(row for row in rows if row['skill'] == 'dialogue_updated_recall')
        segments = []
        current = []
        for token in updated['history']:
            if token == SEPARATOR:
                segments.append(current)
                current = []
            else:
                current.append(token)
        user = next(segment[1:] for segment in reversed(segments) if segment[0] == USER)
        assistant = next(segment[1:] for segment in reversed(segments) if segment[0] == ASSISTANT)
        self.assertEqual(user[:2], [IDS['🔄'], IDS['💖']])
        self.assertEqual(updated['reply'], [IDS['💖'], user[2]])
        self.assertNotEqual(updated['reply'][1], assistant[1])
        self.assertGreater(max(len(row['history']) for row in rows), 48)
        for row in rows:
            self.assertLessEqual(len(row['history']) + 1 + len(row['state']), MEMORY_LIMIT)


if __name__ == '__main__':
    unittest.main()
