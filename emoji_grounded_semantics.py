"""Frozen sentence-semantic features for the question and emoji descriptions."""

import json
import urllib.request

import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from emoji_grounded_data import DIRECTORY
from run import ROOT


MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


class SemanticReader:
    def __init__(self, model_name, revision, local=True):
        self.model_name, self.revision = model_name, revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision,
                                                       cache_dir=ROOT / '.hf-cache', local_files_only=local)
        self.backbone = AutoModel.from_pretrained(model_name, revision=revision,
                                                  cache_dir=ROOT / '.hf-cache', local_files_only=local,
                                                  use_safetensors=True).eval().requires_grad_(False)

    @torch.no_grad()
    def embeddings(self, texts):
        inputs = self.tokenizer(texts, padding=True, return_tensors='pt')
        if inputs.input_ids.shape[1] > 256:
            raise ValueError('Semantic input exceeds 256 word pieces')
        hidden = self.backbone(**inputs).last_hidden_state
        mask = inputs.attention_mask.unsqueeze(-1)
        return F.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1)

    @torch.no_grad()
    def text_inputs(self, texts):
        inputs = self.tokenizer(texts, padding=True, return_tensors='pt')
        if inputs.input_ids.shape[1] > 256:
            raise ValueError('Semantic input exceeds 256 word pieces')
        hidden = self.backbone(**inputs).last_hidden_state
        return torch.cat((hidden, F.layer_norm(hidden, (hidden.shape[-1],))), dim=-1), inputs.attention_mask.bool()

    @torch.no_grad()
    def meaning_vectors(self, rows):
        values = []
        for start in range(0, len(rows), 32):
            batch = rows[start:start + 32]
            core = self.embeddings([row['core_meaning'] for row in batch])
            related = self.embeddings([row['core_meaning'] + '; ' + ', '.join(row['associations']) for row in batch])
            values.append(F.normalize(.8 * core + .2 * related, dim=-1))
        return torch.cat(values)


def download():
    path = DIRECTORY / 'semantic-reader.json'
    if path.exists():
        source = json.loads(path.read_text(encoding='utf-8'))
    else:
        with urllib.request.urlopen('https://huggingface.co/api/models/' + MODEL, timeout=60) as result:
            metadata = json.load(result)
        source = dict(model=MODEL, revision=metadata['sha'], license='Apache-2.0',
                      source='https://huggingface.co/' + MODEL, dimensions=384)
    SemanticReader(source['model'], source['revision'], local=False)
    path.write_text(json.dumps(source, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(source), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    download()
