"""Download bounded, premade UltraChat slices without generating new answers."""

import argparse
import hashlib
import json
import urllib.parse
import urllib.request

from run import ROOT


DIRECTORY = ROOT / 'data' / 'emoji-grounded'
DATASET = 'HuggingFaceH4/ultrachat_200k'


def prepare(pages):
    DIRECTORY.mkdir(exist_ok=True)
    raw = DIRECTORY / 'pages'
    raw.mkdir(exist_ok=True)
    seen = set()
    groups = {name: [] for name in ('train', 'validation', 'test')}
    rejected = dict(truncated=0, length=0, duplicate=0)
    for split, count in (('train_sft', pages), ('test_sft', 2)):
        for page in range(count):
            path = raw / f'{split}-{page:04d}.json'
            if not path.exists():
                query = urllib.parse.urlencode(dict(dataset=DATASET, config='default', split=split, offset=page * 100, length=100))
                with urllib.request.urlopen('https://datasets-server.huggingface.co/rows?' + query, timeout=120) as result:
                    payload = json.load(result)
                path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
            payload = json.loads(path.read_text(encoding='utf-8'))
            for item in payload['rows']:
                if item['truncated_cells']:
                    rejected['truncated'] += 1
                    continue
                row = item['row']
                messages = row['messages']
                if len(messages) < 2 or messages[0]['role'] != 'user' or messages[1]['role'] != 'assistant':
                    raise ValueError('Expected an initial user/assistant pair')
                question, answer = (messages[index]['content'].strip() for index in (0, 1))
                # Reject long pairs rather than silently truncating the reference answer.
                if not question or not answer or len(question) > 500 or len(answer) > 650:
                    rejected['length'] += 1
                    continue
                key = hashlib.sha256(' '.join(question.lower().split()).encode()).hexdigest()
                if key in seen:
                    rejected['duplicate'] += 1
                    continue
                seen.add(key)
                name = 'train' if split == 'train_sft' else ('validation' if int(key[:8], 16) % 2 else 'test')
                groups[name].append(dict(id=row['prompt_id'], question=question, answer=answer,
                                         source_split=split, source_index=item['row_idx']))
            print(json.dumps(dict(source_split=split, page=page, counts={key: len(value) for key, value in groups.items()})), flush=True)
    for name, rows in groups.items():
        if not rows:
            raise ValueError('Each dataset split needs at least one short pair')
        (DIRECTORY / f'{name}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    report = dict(dataset=DATASET, source='https://huggingface.co/datasets/' + DATASET,
                  license='MIT', training_pages=pages, counts={key: len(value) for key, value in groups.items()},
                  generated_qa_labels=0, first_turn_only=True, rejected=rejected,
                  split_policy='source train_sft for training; source test_sft hash partition for validation/test; exact prompt deduplication only',
                  files_sha256={name: hashlib.sha256((DIRECTORY / f'{name}.jsonl').read_bytes()).hexdigest() for name in groups})
    (DIRECTORY / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pages', type=int, default=20)
    prepare(parser.parse_args().pages)
