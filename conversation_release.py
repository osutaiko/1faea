"""Write a model card and package the validation-selected local checkpoint."""

from datetime import datetime
import hashlib
import json
import zipfile

import torch

from conversation import timestamp
from run import ROOT


STARTED = '2026-10-02T02:30:41+00:00'
DIRECTORY = ROOT / 'runs' / 'conversation-compositional' / 'reliability-selected'


def release():
    selection = json.loads((DIRECTORY / 'selection.json').read_text(encoding='utf-8'))
    reports = {name: json.loads((DIRECTORY / f'{name}.json').read_text(encoding='utf-8')) for name in
               ('reliability-test', 'reliability-numbers-test', 'live-numbers-test', 'legacy-test', 'public-scope-test')}
    finished = timestamp()
    elapsed = (datetime.fromisoformat(finished) - datetime.fromisoformat(STARTED)).total_seconds()
    metadata = dict(started=STARTED, finished=finished, elapsed_seconds=elapsed, selected=selection['chosen'],
                    tests=60, reports={name: report['metrics'] for name, report in reports.items()},
                    selection=selection, general_use_ready=False,
                    constraints=dict(text_tokens=128, emoji_memory=128, output_symbols=32, quantity_range=[0, 9]))
    (DIRECTORY / 'development.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    summary = [
        '# 🫪 Conversation checkpoint', '',
        f"Selected variant: `{selection['chosen']}`. Selection used validation only. 🧠✅", '',
        '| Evaluation | Rows/turns | Exact replies | Correct states and replies |',
        '| --- | ---: | ---: | ---: |',
    ]
    for title, key in [('New combinations', 'reliability-test'), ('Ordinary number wording', 'reliability-numbers-test'),
                       ('Own-history stress sessions', 'live-numbers-test'), ('Earlier curriculum', 'legacy-test')]:
        metrics = reports[key]['metrics']['all']
        summary.append(f"| {title} | {metrics['count']} | {metrics['reply']:.1%} | {metrics['correct_trace']:.1%} |")
    scope = reports['public-scope-test']['metrics']
    summary.extend(['', f"Public scope audit: {scope['supported']['reply']:.1%} correct replies on "
                    f"{scope['supported']['count']} supported requests; {scope['unsupported']['reply']:.1%} correct "
                    f"abstentions on {scope['unsupported']['count']} unsupported requests. 🔍", '',
                    'These scores measure a small authored semantic language. They do not establish general conversation quality.', '',
                    'The 32 twelve-turn stress sessions run all three contrast queries in sequence. Training snapshots',
                    'share history across these queries, and the authored continuation includes only one query.',
                    'The stress evaluation therefore measures a harder transcript distribution. It never inserts gold history.', '',
                    'The earlier checkpoint scores 7.7% on the same number-wording test and 4.7% on the same own-history stress sessions.',
                    'The new checkpoint improves these measures but still fails most complete stress conversations.', '',
                    'Validation counterfactual audit: even with correct prior states and replies, the full transcript',
                    'produces only 44.8% exact replies on 192 turns. Eviction occurs in zero turns of that audit.',
                    'On the test stress sessions, only three of 384 turns have any memory eviction.', '',
                    'The model reads text and selects atomic emoji states. Replies come directly from emoji memory.',
                    'Only integer emoji IDs persist between stages and turns. Numerical computation within stages remains continuous.',
                    'No English assistant answer is generated and translated into emojis. 🫪💬', '',
                    'The additional curriculum defines 16 everyday topics, mixed attributes, updates, and quantities.',
                    'It contains 1,408 training, 224 validation, and 416 test rows. Supplemental entity pools are frozen and disjoint.',
                    'Training also uses 38 reviewed teacher user paraphrases, 256 authored number variants, and 23 reviewed public inputs.',
                    'Public assistant answers are never used as targets. 📚', '',
                    'Limits: 128 input tokens, 128 emoji memory symbols, 32 generated symbols, and stored quantities from zero through nine.',
                    'Older whole turns are evicted. Entities require emoji glyphs or recognizable Unicode names.',
                    'Unknown names, plurals, complex requests, and long conversations remain unreliable.',
                    'Replies are short outlines, not detailed instructions. This model is not ready for general use. 🧪', '',
                    'Pretrained input model: HuggingFaceTB/SmolLM2-360M-Instruct, pinned in `data/conversation/sources.json`.',
                    'Teacher user paraphrases: Qwen/Qwen2.5-0.5B-Instruct, pinned in `data/conversation/teacher.json`.',
                    'Public user inputs: HuggingFaceTB/everyday-conversations-llama3.1-2k, Apache-2.0.',
                    'The checkpoint archive excludes the pretrained model cache. Run from the repository with its dependencies and cached model.', '',
                    'A twelve-query validation benchmark measures the emoji reply stage at 0.021 seconds median,',
                    'versus 1.04 seconds for the pretrained-feature parent. Both answer eleven queries correctly.',
                    'This excludes text encoding and loading. Sixteen audited inputs produce identical single and batched outputs.', '',
                    '```powershell',
                    r'.venv\Scripts\python.exe -X utf8 conversation_web.py',
                    r'.venv\Scripts\python.exe -X utf8 conversation.py --experiment conversation-compositional --semantic-encoder chat --name reliability-selected',
                    '```', '', f'Development time: {elapsed / 3600:.2f} hours. All 60 implementation tests pass. ✅'])
    (DIRECTORY / 'MODEL_CARD.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')
    encoder = torch.load(DIRECTORY / 'encoder.pt', weights_only=True)
    decoder = torch.load(DIRECTORY / 'decoder.pt', weights_only=True)
    files = [DIRECTORY / 'encoder.pt', DIRECTORY / 'decoder.pt', DIRECTORY / 'selection.json',
             DIRECTORY / 'MODEL_CARD.md', DIRECTORY / 'development.json', DIRECTORY / 'FAILURE_ANALYSIS.md']
    files.extend(DIRECTORY / f'{name}.json' for name in reports)
    files.extend(DIRECTORY / f'{name}.json' for name in
                 ('runtime-benchmark', 'memory-audit', 'batch-audit', 'restore-test', 'failure-analysis', 'baseline-comparison'))
    if 'input_adapter' in encoder:
        files.extend(path for path in (DIRECTORY / encoder['input_adapter']).rglob('*') if path.is_file())
    if 'emoji_features' in decoder:
        features = DIRECTORY / decoder['emoji_features']
        feature_checkpoint = torch.load(features, weights_only=True)
        if not torch.equal(feature_checkpoint['model']['embedding.weight'], encoder['model']['embedding.weight']):
            raise ValueError('Packaged emoji processor and input model have different symbol embeddings')
        files.append(features)
    archive = DIRECTORY / 'checkpoint.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as package:
        for path in files:
            package.write(path, path.relative_to(DIRECTORY).as_posix())
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None:
            raise ValueError('Checkpoint archive failed its checksum test')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (DIRECTORY / 'checkpoint.sha256').write_text(digest + '  checkpoint.zip\n', encoding='utf-8')
    print(json.dumps(dict(selected=selection['chosen'], elapsed_seconds=elapsed, archive=str(archive), sha256=digest)), flush=True)


if __name__ == '__main__':
    release()
