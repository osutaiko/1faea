"""Small pretrained-decoder trial on existing premade Q&A and emoji states."""

import argparse
import hashlib
import json
import random
import time

from peft import get_peft_model_state_dict, set_peft_model_state_dict
import torch
from torch.nn import functional as F

from emoji_general_model import GeneralEncoder, anchored_states
from emoji_general_run import targets
from emoji_grounded_model import MeaningLexicon
from emoji_pretrained_model import PretrainedReply, backbone
from run import ROOT


@torch.no_grad()
def evaluate(reply, data):
    reply.eval()
    total, count = 0., 0
    for start in range(0, len(data['states']), 4):
        labels = data['labels'][start:start + 4]
        logits = reply(data['states'][start:start + 4], data['prefix'][start:start + 4])
        total += F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction='sum').item()
        count += (labels >= 0).sum().item()
    return total / count


def train(args):
    torch.set_num_threads(2)
    torch.manual_seed(719)
    rng = random.Random(719)
    base = ROOT / 'runs/emoji-general/five-hour'
    output = ROOT / 'runs/emoji-pretrained' / args.name
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads((ROOT / 'data/conversation/sources.json').read_text())
    checkpoint = torch.load(base / 'runtime.pt', weights_only=True)
    initialization = torch.load(base / 'initialization.pt', weights_only=True)
    encoder = GeneralEncoder(checkpoint['encoder']['vectors'], **checkpoint['encoder_config']).eval()
    encoder.load_state_dict(checkpoint['encoder'])
    model = backbone(source)
    end_vector = model.get_input_embeddings().weight[model.config.eos_token_id].detach().float()
    reply = PretrainedReply(model, initialization['english'], end_vector)
    groups = {}
    for split in ('train', 'validation', 'test'):
        path = output / (split + '-states.pt')
        if path.exists():
            groups[split] = torch.load(path, weights_only=True)
            continue
        data = torch.load(base / (split + '-features.pt'), weights_only=True)
        states, answers = [], []
        with torch.no_grad():
            for start in range(0, len(data['rows']), 32):
                for key, destination in (('qfeatures', states), ('afeatures', answers)):
                    values = data[key][start:start + 32]
                    destination.append(encoder(values, torch.ones(values.shape[:2], dtype=torch.bool))[2])
        prefix, labels = targets(torch.cat(answers), data['aanchors'], reply.end)
        question_states = torch.cat(states)
        if args.anchor_input:
            lexicon = MeaningLexicon(checkpoint['meanings'])
            literal = [lexicon.anchors(row['question'], question_states.shape[1], literal_only=True) for row in data['rows']]
            anchors = torch.tensor([ids + [-100] * (question_states.shape[1] - len(ids)) for ids in literal])
            question_states = anchored_states(question_states, anchors)
        groups[split] = dict(states=question_states, prefix=prefix, labels=labels)
        torch.save(groups[split], path)
    manifest = dict(decoder='pretrained-pointer-v1', source=source, encoder_checkpoint_sha256=hashlib.sha256((base / 'runtime.pt').read_bytes()).hexdigest(),
                    dataset=json.loads((base / 'dataset-report.json').read_text()),
                    steps=args.steps, warmup_steps=args.warmup_steps, anchor_input=args.anchor_input,
                    seed=719, runtime_english_answer_generation=False, generated_qa_labels=0)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    parameters = [parameter for parameter in reply.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=.0002)
    best = float('inf')
    started = time.monotonic()
    with (output / 'training.jsonl').open('w') as journal:
        for step in range(1, args.warmup_steps + args.steps + 1):
            reply.train()
            if step <= args.warmup_steps or step % 10 == 0:
                indices = torch.tensor([((step - 1) * 4 + index) % reply.end for index in range(4)])
                states = indices[:, None].expand(-1, 8)
                prefix = torch.stack((torch.full_like(indices, reply.end), indices), 1)
                labels = torch.stack((indices, torch.full_like(indices, reply.end)), 1)
            else:
                group = groups['train']
                indices = torch.tensor(rng.sample(range(len(group['states'])), 4))
                states, prefix, labels = (group[key][indices] for key in ('states', 'prefix', 'labels'))
            if step > args.warmup_steps:
                with torch.no_grad():
                    predicted = reply(states, prefix).argmax(-1)
                    prefix = prefix.clone()
                    replace = torch.rand(prefix[:, 1:].shape) < .5
                    prefix[:, 1:] = torch.where(replace, predicted[:, :-1], prefix[:, 1:])
            logits = reply(states, prefix)
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            if step % 100 == 0 or step == args.warmup_steps + args.steps:
                validation = evaluate(reply, groups['validation'])
                entry = dict(step=step, phase='meaning' if step <= args.warmup_steps else 'answer', loss=loss.item(), validation_loss=validation, elapsed_seconds=time.monotonic() - started)
                journal.write(json.dumps(entry) + '\n'); journal.flush()
                print(json.dumps(entry), flush=True)
                if step > args.warmup_steps and validation < best:
                    best = validation
                    heads = {key: value for key, value in reply.state_dict().items() if not key.startswith('backbone.')}
                    torch.save(dict(heads=heads, lora=get_peft_model_state_dict(model), source=source,
                                    encoder=checkpoint['encoder'], encoder_config=checkpoint['encoder_config'],
                                    semantic_model=checkpoint['semantic_model'], meanings=checkpoint['meanings'],
                                    selected_step=step, manifest=manifest), output / 'runtime.pt')
    selected = torch.load(output / 'runtime.pt', weights_only=True)
    reply.load_state_dict(selected['heads'], strict=False)
    set_peft_model_state_dict(model, selected['lora'])
    test = evaluate(reply, groups['test'])
    if args.anchor_input:
        from emoji_anchored_chat import answer
    else:
        from emoji_pretrained_chat import answer
    from emoji_grounded_semantics import SemanticReader
    from emoji_grounded_eval import CASES, score
    semantic = SemanticReader(checkpoint['semantic_model']['model'], checkpoint['semantic_model']['revision'])
    checks = []
    for case in CASES:
        result = answer(encoder, reply.eval(), semantic, checkpoint['meanings'], case['question'])
        checks.append(dict(**case, **result, passed=score(result['symbols'], case)))
    result = dict(selected_step=selected['selected_step'], elapsed_seconds=time.monotonic() - started, test_loss=test,
                  fixed_checks_passed=sum(row['passed'] for row in checks), fixed_checks_total=len(checks), examples=checks,
                  trainable_parameters=sum(parameter.numel() for parameter in parameters),
                  runtime_english_answer_generation=False, generated_qa_labels=0, promoted=False, general_use_ready=False)
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'examples'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='copy-v1')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--warmup-steps', type=int, default=0)
    parser.add_argument('--anchor-input', action='store_true')
    train(parser.parse_args())
