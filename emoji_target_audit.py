"""Audit existing premade answer targets without generating training labels."""

from collections import Counter, defaultdict
import hashlib
import json
import random

import torch

from emoji_general_model import GeneralEncoder
from emoji_general_run import targets
from emoji_grounded_model import MeaningLexicon
from run import ROOT


@torch.no_grad()
def audit():
    torch.set_num_threads(2)
    base = ROOT / 'runs/emoji-general/five-hour'
    checkpoint = torch.load(base / 'runtime.pt', weights_only=True)
    meanings = checkpoint['meanings']
    lexicon = MeaningLexicon(meanings)
    encoder = GeneralEncoder(checkpoint['encoder']['vectors'], **checkpoint['encoder_config']).eval()
    encoder.load_state_dict(checkpoint['encoder'])
    report = dict(checkpoint_sha256=hashlib.sha256((base / 'runtime.pt').read_bytes()).hexdigest(),
                  selected_step=checkpoint['selected_step'], evaluation_only=True, generated_qa_labels=0,
                  limitation='Literal matches and target collisions are structural diagnostics, not semantic accuracy. No automated judge or new training labels.',
                  splits={})
    for split in ('train', 'validation', 'test'):
        path = base / (split + '-features.pt')
        data = torch.load(path, weights_only=True)
        answer_ids = []
        for start in range(0, len(data['rows']), 32):
            values = data['afeatures'][start:start + 32]
            answer_ids.append(encoder(values, torch.ones(values.shape[:2], dtype=torch.bool))[2])
        ids = torch.cat(answer_ids)
        _, labels = targets(ids, data['aanchors'], len(meanings))
        lengths = Counter()
        collisions = defaultdict(list)
        matched = retained = truncated = association_only = literal_first = no_literal = 0
        records = []
        for row, sequence, anchors in zip(data['rows'], labels.tolist(), data['aanchors'].tolist()):
            codes = [index for index in sequence if 0 <= index < len(meanings)]
            literal = lexicon.anchors(row['answer'], len(meanings), literal_only=True)
            matched += len(literal)
            retained += sum(index in codes for index in literal)
            truncated += len(literal) > 6
            no_literal += not literal
            anchored = [index for index in anchors if index >= 0]
            association_only += sum(index not in literal for index in anchored)
            literal_first += bool(codes and codes[0] in literal)
            lengths[len(codes)] += 1
            collisions[tuple(codes)].append(row)
            records.append(dict(id=row['id'], question=row['question'], english_answer=row['answer'],
                                target=''.join(meanings[index]['symbol'] for index in codes),
                                target_core_meanings=[meanings[index]['core_meaning'] for index in codes],
                                literal_symbols=[meanings[index]['symbol'] for index in literal],
                                omitted_literal_symbols=[meanings[index]['symbol'] for index in literal if index not in codes]))
        shared = [dict(target=''.join(meanings[index]['symbol'] for index in codes), count=len(rows),
                       examples=[dict(id=row['id'], answer=row['answer']) for row in rows[:3]])
                  for codes, rows in collisions.items() if len({row['answer'].casefold().strip() for row in rows}) > 1]
        shared.sort(key=lambda row: row['count'], reverse=True)
        examples = random.Random(719).sample(records, min(20, len(records)))
        report['splits'][split] = dict(rows=len(records), target_lengths=dict(sorted(lengths.items())),
                                      literal_ids_detected=matched, literal_ids_retained=retained,
                                      answers_with_more_than_six_literal_ids=truncated, answers_without_exact_literal_matches=no_literal,
                                      association_derived_anchor_ids=association_only, literal_first_symbol_rows=literal_first,
                                      distinct_targets=len(collisions), shared_targets_for_distinct_answers=len(shared),
                                      most_common_shared_targets=shared[:10], seeded_examples=examples,
                                      dataset_features_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        print(json.dumps(dict(split=split, rows=len(records), literal_ids_detected=matched, literal_ids_retained=retained,
                              association_derived_anchor_ids=association_only, shared_targets=len(shared))), flush=True)
    output = ROOT / 'data/emoji-grounded/target-quality-audit.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    audit()
