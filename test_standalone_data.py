import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import conversation_standalone_data as data
import conversation_standalone_teacher as teacher
import conversation_standalone_review as review
import conversation_standalone_compile as compile_data


class StandaloneDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.patch = patch.object(data, 'DIRECTORY', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        (self.root / 'questions.jsonl').write_text(json.dumps(dict(
            id='question', question='Example input', split='test', source_index=25)) + '\n', encoding='utf-8')
        self.row = dict(id='question', state=['🐱'], answer=['🐱'], answer_reading='A cat.', needs_context=False)

    def validate(self, rows):
        path = self.root / 'teacher.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        data.validate(path)

    def test_preserves_reserved_split_and_requires_semantic_review(self):
        self.validate([self.row])
        candidate = json.loads((self.root / 'candidates.jsonl').read_text(encoding='utf-8'))
        self.assertEqual(candidate['split'], 'test')
        self.assertEqual(candidate['semantic_review'], 'pending')

    def test_rejects_text_and_multiple_symbols_in_one_atomic_item(self):
        for answer in (['cat'], ['🐱🐶']):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                self.validate([dict(self.row, answer=answer)])

    def test_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            self.validate([self.row, self.row])

    def test_context_dependent_request_must_abstain(self):
        with self.assertRaises(ValueError):
            self.validate([dict(self.row, needs_context=True)])
        self.validate([dict(self.row, needs_context=True, answer=['❓'])])


class TeacherRequestTests(unittest.TestCase):
    def test_authentication_failure_is_reported_without_credentials(self):
        from io import BytesIO
        error = urllib.error.HTTPError('https://api.openai.com', 401, 'Unauthorized', None,
                                       BytesIO(b'{"error":{"code":"invalid_api_key"}}'))
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-secret'}), patch.object(
                teacher.urllib.request, 'urlopen', side_effect=error):
            with self.assertRaisesRegex(RuntimeError, 'HTTP 401') as caught:
                teacher.response('test-model', 'instructions', [], {})
        self.assertNotIn('test-secret', str(caught.exception))

    def test_refusal_is_not_treated_as_a_dataset_candidate(self):
        from unittest.mock import MagicMock
        result = MagicMock()
        result.__enter__.return_value.read.return_value = json.dumps(dict(
            status='completed', output=[dict(type='message', content=[dict(type='refusal', refusal='No')])])).encode()
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-secret'}), patch.object(
                teacher.urllib.request, 'urlopen', return_value=result):
            with self.assertRaisesRegex(ValueError, 'no structured output'):
                teacher.response('test-model', 'instructions', [], {})


class BlindReviewTests(unittest.TestCase):
    def test_literal_readings_reuse_paid_calls_without_reusing_changed_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'api').mkdir()
            questions = root / 'questions.jsonl'
            questions.write_text(json.dumps(dict(id='example', question='Name a fruit.')) + '\n', encoding='utf-8')
            symbols = {symbol for row in review.CONTROLS for field in ('state', 'answer') for symbol in row[field]} | {'🍎'}
            (root / 'vocabulary.json').write_text(json.dumps([dict(symbol=symbol, meaning='literal-' + symbol) for symbol in symbols]), encoding='utf-8')
            (root / 'api' / 'candidates.jsonl').write_text(json.dumps(dict(id='example', state=['🍎'], answer=['🍎'])) + '\n', encoding='utf-8')
            seen = []

            def respond(model, instructions, rows, schema, reasoning_effort=None):
                seen.append(rows)
                self.assertTrue(all('question' in row and 'answer' not in row for row in rows))
                if rows[0]['id'] == 'example':
                    self.assertEqual(rows[0]['answer_reading'], 'literal-🍎')
                result = [dict(id=row['id'], state_faithful=True, answer_faithful=True, reason='test') for row in rows]
                return dict(rows=result), dict(id=str(len(seen)), model=model, usage={})

            with patch.object(review, 'DIRECTORY', root), patch.object(review, 'response', side_effect=respond):
                review.run('test-model')
                self.assertEqual(len(seen), 2)
                review.run('test-model')
                self.assertEqual(len(seen), 2)
                questions.write_text(json.dumps(dict(id='example', question='Changed question.')) + '\n', encoding='utf-8')
                review.run('test-model')
                self.assertEqual(len(seen), 3)


class ReviewedCompileTests(unittest.TestCase):
    def test_quarantines_overlap_and_conflicting_training_states(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'api'
            report_dir = source / 'review-literal-gpt-5.4-mini'
            report_dir.mkdir(parents=True)
            specs = [('good', 'train', '🍎', '🍎'), ('overlap', 'train', '🍌', '🍌'),
                     ('conflict-a', 'train', '🐈', '🐈'), ('conflict-b', 'train', '🐈', '🐕'),
                     ('validation', 'validation', '🐕', '🐕'), ('test', 'test', '🐟', '🐟')]
            questions = [dict(id=id_, question=id_, split=split) for id_, split, _, _ in specs]
            candidates = [dict(id=id_, state=[state], answer=[answer], answer_reading='literal', needs_context=False) for id_, _, state, answer in specs]
            (root / 'questions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in questions), encoding='utf-8')
            (source / 'candidates.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in candidates), encoding='utf-8')
            (report_dir / 'blind-review-report.json').write_text(json.dumps(dict(controls_passed=True, candidates=6,
                verdicts=[dict(id=row['id'], approved=True, answer_reading='literal') for row in candidates])), encoding='utf-8')
            result = dict(rows=[dict(id=row['id'], overlaps=row['id'] == 'overlap', reason='test') for row in questions if row['split'] == 'train'])
            with patch.object(compile_data, 'DIRECTORY', root), patch.object(compile_data, 'response', return_value=(result, {})):
                compile_data.build()
            compiled = [json.loads(line) for line in (source / 'provisional.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual({row['id'] for row in compiled}, {'good', 'validation', 'test'})
            summary = json.loads((source / 'dataset-report.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['conflicting_train_states'], 1)
            self.assertEqual(summary['quarantined_overlap'], 1)


if __name__ == '__main__':
    unittest.main()
