"""Download pinned public resources for the conversation experiments."""

import json
import urllib.parse
import urllib.request

from huggingface_hub import HfApi, snapshot_download

from run import ROOT


MODEL = 'HuggingFaceTB/SmolLM2-360M-Instruct'
MODEL_REVISION = 'a10cc1512eabd3dde888204e902eca88bddb4951'
TEACHER = 'Qwen/Qwen2.5-0.5B-Instruct'
TEACHER_REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
DATASET = 'HuggingFaceTB/everyday-conversations-llama3.1-2k'
DIRECTORY = ROOT / 'data' / 'conversation'


def setup():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    model_revision = MODEL_REVISION
    dataset_revision = api.dataset_info(DATASET).sha
    snapshot_download(MODEL, revision=model_revision, cache_dir=ROOT / '.hf-cache',
                      allow_patterns=['*.json', '*.safetensors', 'README.md', 'LICENSE'])
    snapshot_download(TEACHER, revision=TEACHER_REVISION, cache_dir=ROOT / '.hf-cache',
                      allow_patterns=['*.json', '*.safetensors', 'README.md', 'LICENSE'])
    (DIRECTORY / 'teacher.json').write_text(json.dumps(dict(model=TEACHER, revision=TEACHER_REVISION,
                                                          license='Apache-2.0'), indent=2), encoding='utf-8')
    counts = {}
    for split in ('train_sft', 'test_sft'):
        rows = []
        offset = 0
        while True:
            query = urllib.parse.urlencode(dict(dataset=DATASET, config='default', split=split,
                                                offset=offset, length=100))
            with urllib.request.urlopen('https://datasets-server.huggingface.co/rows?' + query, timeout=60) as response:
                page = json.load(response)
            rows.extend(item['row'] for item in page['rows'])
            offset += len(page['rows'])
            print(f'{split}: {offset}/{page["num_rows_total"]}', flush=True)
            if offset >= page['num_rows_total']:
                break
        (DIRECTORY / f'everyday-{split}.jsonl').write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        counts[split] = len(rows)
    metadata = dict(model=MODEL, model_revision=model_revision, dataset=DATASET,
                    dataset_revision=dataset_revision, counts=counts,
                    model_license='Apache-2.0', dataset_license='Apache-2.0')
    (DIRECTORY / 'sources.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    setup()
