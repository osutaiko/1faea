"""Short answer-side semantic retrieval control; no decoder or training."""

import json
import time

import torch
from torch.nn import functional as F

from emoji_general_model import GeneralEncoder
from run import ROOT


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    started = time.monotonic()
    base = ROOT / 'runs/emoji-general/five-hour'
    checkpoint = torch.load(base / 'runtime.pt', weights_only=True)
    encoder = GeneralEncoder(checkpoint['encoder']['vectors'], **checkpoint['encoder_config']).eval()
    encoder.load_state_dict(checkpoint['encoder'])
    results = {}
    examples = []
    for split in ('validation', 'test'):
        data = torch.load(base / (split + '-features.pt'), weights_only=True)
        features = data['afeatures']
        logits, hard, _ = encoder(features, torch.ones(features.shape[:2], dtype=torch.bool))
        original = F.normalize(features[..., :encoder.vectors.shape[1]].float().mean(1), dim=-1)
        controls = {'hard': hard.mean(1)}
        # Temperatures are reported separately; none is selected on test results.
        for temperature in (1., 3., 10.):
            controls[f'soft_temperature_{temperature:g}'] = ((logits / temperature).softmax(-1) @ encoder.vectors).mean(1)
        for count in (2, 4, 8):
            weights, indices = logits.topk(count, dim=-1)
            controls[f'sparse_top_{count}'] = (weights.softmax(-1).unsqueeze(-1) * encoder.vectors[indices]).sum(2).mean(1)
        controls['unrelated_hard'] = hard.mean(1).roll(len(features) // 2, 0)
        metrics = {}
        for name, values in controls.items():
            similarities = F.normalize(values, dim=-1) @ original.T
            order = similarities.argsort(dim=1, descending=True)
            expected = torch.arange(len(features))[:, None]
            ranks = (order == expected).nonzero()[:, 1] + 1
            metrics[name] = dict(top1=int((ranks == 1).sum()), top5=int((ranks <= 5).sum()),
                                 mean_rank=ranks.float().mean().item(),
                                 mean_matched_cosine=similarities.diag().mean().item())
            if split == 'test' and name in ('hard', 'soft_temperature_1'):
                for index in range(5):
                    chosen = order[index, 0].item()
                    examples.append(dict(mode=name, answer=data['rows'][index]['answer'],
                                         retrieved_answer=data['rows'][chosen]['answer'], correct=chosen == index))
        results[split] = dict(count=len(features), metrics=metrics)
    report = dict(results=results, examples=examples, elapsed_seconds=time.monotonic() - started,
                  training_updates=0, answer_side_oracle=True, semantic_reference='cached MiniLM answer features',
                  limitations=['Same semantic space defines both emoji meanings and retrieval references.',
                               'Mean pooling does not measure order, relations, reasoning, or question answering.',
                               'Soft weights can carry information beyond the displayed top emoji.'],
                  runtime_promoted=False)
    path = ROOT / 'data/emoji-grounded/continuous-quick-check.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'examples'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
