"""Train emoji-only memory on varied fact order and intervening dialogue."""

import argparse
import json
import random
import time

import torch

import conversation
from conversation_composition import COLORS
from conversation_data import IDS, SKILLS, USER, ASSISTANT, SEPARATOR, pack_history, record
from conversation_model import AtomicDecoder, ConversationReader, memory, pad
from emoji_catalog import ALPHABET, ENTITIES
from emoji_lm_model import END
from run import ROOT


def examples(count, seed):
    rng = random.Random(seed)
    reserved = {IDS[symbol] for _, symbol in COLORS}
    reserved.update((USER, ASSISTANT, SEPARATOR))
    reserved.update(IDS[symbol] for symbol in ('💖', '💔', '🔄', '🧠', '🎨', '📍', '🔢', '➕', '👌'))
    reserved.update(token for _, state, reply, *_ in SKILLS for token in [IDS[symbol] for symbol in (*state, *reply)])
    entities = [entity for entity in ENTITIES if entity not in reserved]
    rows = []
    for _ in range(count):
        a, b = [ALPHABET[entity] for entity in rng.sample(entities, 2)]
        color_a, color_b = [color for _, color in rng.sample(COLORS, 2)]
        turns = [(tuple(IDS[symbol] for symbol in ('📌', a, '🎨', color_a)),
                  tuple(IDS[symbol] for symbol in ('👍', a, color_a))),
                 (tuple(IDS[symbol] for symbol in ('📌', b, '🎨', color_b)),
                  tuple(IDS[symbol] for symbol in ('👍', b, color_b)))]
        rng.shuffle(turns)
        history = pack_history(turns)
        for entity, color in ((a, color_a), (b, color_b)):
            rows.append(record('', ('❓', entity, '🎨'), (entity, color), 'varied_color_query', history))
        # A social reply remains appropriate after stored facts, and must not
        # replace those facts when the following user asks about them.
        _, state, reply, *_ = rng.choice(SKILLS[:24])
        rows.append(record('', state, reply, 'social_after_facts', history))
        extended = pack_history([*turns, ([IDS[symbol] for symbol in state], [IDS[symbol] for symbol in reply])])
        rows.append(record('', ('❓', a, '🎨'), (a, color_a), 'color_after_social', extended))
    return rows


@torch.no_grad()
def prepare(output, base_model, dataset, cache_name):
    directory = output / cache_name
    directory.mkdir(exist_ok=True)
    reader = ConversationReader(base_model)
    vectors = torch.load(output / 'vectors.pt', weights_only=True)
    reader.symbol_embedding.load_state_dict(dict(weight=vectors.to(reader.symbol_embedding.weight.dtype)))
    paths = {}
    for split, rows in dataset.items():
        paths[split] = []
        for start in range(0, len(rows), 16):
            selected = rows[start:start + 16]
            path = directory / f'{split}-{start // 16:05d}.pt'
            if path.exists():
                if torch.load(path, weights_only=True)['rows'] != selected:
                    raise ValueError('Memory cache differs from the current curriculum')
            else:
                state, mask = memory([row['history'] for row in selected], [row['state'] for row in selected])
                features = reader.emoji_inputs(state, mask)
                torch.save(dict(state_features=features.to(torch.float8_e4m3fn), state=state, state_mask=mask,
                                reply=pad([row['reply'] + [END] for row in selected], -100), rows=selected), path)
            paths[split].append(str(path.relative_to(output)))
            print(f'Memory features {split}: {min(start + 16, len(rows))}/{len(rows)}', flush=True)
    return paths


def loss(model, data):
    return conversation.token_loss(model, data['state_features'], data['state_mask'], data['state'], data['reply'])


@torch.no_grad()
def validate(model, paths):
    total, count = 0, 0
    for path in paths:
        data = conversation.batch(path)
        total += loss(model, data).item() * len(data['state'])
        count += len(data['state'])
    return total / count


def train(args, manifest, supplemental):
    torch.manual_seed(23)
    rng = random.Random(23)
    output = conversation.OUTPUT / args.name
    checkpoint = torch.load(output / 'decoder.pt', weights_only=True)
    model = AtomicDecoder(checkpoint['model']['embedding.weight'], **checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.0001, weight_decay=0.01)
    validation = manifest['splits']['validation'] + supplemental['validation']
    model.eval()
    best = validate(model, validation)
    start = time.monotonic()
    with (output / 'memory-training.jsonl').open('w', encoding='utf-8') as journal:
        for step in range(1, args.steps + 1):
            paths = supplemental['train'] if step % 2 else manifest['splits']['train']
            model.train()
            optimizer.zero_grad()
            current = loss(model, conversation.batch(rng.choice(paths)))
            current.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            if step % 200 == 0 or step == args.steps:
                model.eval()
                score = validate(model, validation)
                entry = dict(time=conversation.timestamp(), step=step, elapsed_seconds=time.monotonic() - start,
                             train=current.item(), validation=score)
                journal.write(json.dumps(entry) + '\n')
                journal.flush()
                print(json.dumps(entry), flush=True)
                if score < best:
                    best = score
                    torch.save(dict(config=model.config, model=model.state_dict(), step=step,
                                    validation_loss=score, base_model=manifest['base_model'],
                                    parent_decoder_step=checkpoint['step']), output / 'decoder.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--steps', type=int, default=3000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    conversation.OUTPUT = ROOT / 'runs' / 'conversation-compositional'
    manifest = json.loads((conversation.OUTPUT / 'features.json').read_text(encoding='utf-8'))
    dataset = dict(train=examples(192, 311), validation=examples(32, 313))
    supplemental = prepare(conversation.OUTPUT, manifest['base_model'], dataset, 'memory-features')
    train(args, manifest, supplemental)
