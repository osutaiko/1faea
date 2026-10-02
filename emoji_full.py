"""Train and evaluate emoji-only continuations over the Unicode repertoire."""

import argparse
import json
import random
import re
import sys

import torch
from torch import nn
from torch.nn import functional as F

from run import ROOT
from semantic_model import FACT, LEFT, TRUE, FALSE, UNKNOWN, initial_state
from semantic_pointer import load as load_pointer
from semantic_pointer_model import split_clauses
from semantic_clauses import TRAIN_FORMS, VALIDATION_FORMS
from emoji_lm_model import EmojiLM, END, START, visible_tokens
from emoji_catalog import ALPHABET, CATALOG, ENTITIES


OUTPUT = ROOT / 'runs' / 'emoji-full'


class FullEmojiLM(EmojiLM):
    def __init__(self, vectors, width=128):
        super().__init__(vectors, width)
        self.identity_projection = nn.Linear(16, width, bias=False)

    def memory_features(self, state, state_features):
        # Exact equality preserves discrete identity when names have similar
        # embeddings. Relation inference and verdicts remain learned outputs.
        identity = (state[:, :, None] == state[:, None, :]).float()
        return super().memory_features(state, state_features) + self.identity_projection(identity)


def rows(count, seed, cover=False):
    rng = random.Random(seed)
    result = []
    for index in range(count):
        a = ENTITIES[index % len(ENTITIES)] if cover else rng.choice(ENTITIES)
        b, c, d = rng.sample([entity for entity in ENTITIES if entity != a], 3)
        order = [(a, b), (b, c)]
        rng.shuffle(order)
        query, answer = rng.choice([((a, c), TRUE), ((c, a), FALSE), ((d, a), UNKNOWN)])
        result.append(dict(entities=[*order[0], *order[1], *query], derived=[a, c], answer=answer))
    return result


def vectors(reader):
    result = []
    embedding = reader.backbone.get_input_embeddings()
    with torch.no_grad():
        for record in CATALOG:
            ids = reader.tokenizer.encode(record['name'], add_special_tokens=False)
            result.append(embedding(torch.tensor(ids)).mean(0))
    return torch.stack(result).float()


def install_vectors(reader, values):
    dtype = reader.backbone.get_input_embeddings().weight.dtype
    reader.symbol_embedding = nn.Embedding.from_pretrained(values.to(dtype=dtype), freeze=True)


def render(tokens):
    return ''.join(ALPHABET[token] for token in tokens)


@torch.no_grad()
def clause_inputs(reader, clauses):
    names = {CATALOG[index]['name'].lower(): index for index in ENTITIES}
    names.update({CATALOG[index]['unicode_name'].lower(): index for index in ENTITIES})
    names.update({ALPHABET[index]: index for index in ENTITIES})
    pattern = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(name) for name in sorted(names, key=len, reverse=True)) + r')(?!\w)', re.IGNORECASE)
    inputs = reader.tokenizer(clauses, padding=True, return_offsets_mapping=True, return_tensors='pt')
    offsets = inputs.pop('offset_mapping').tolist()
    positions, candidates = [], []
    for clause, spans in zip(clauses, offsets):
        matches = list(pattern.finditer(clause))
        if len(matches) != 2:
            raise ValueError('Each clause requires two exact catalog names or emoji entities')
        positions.append([max(index for index, (start, end) in enumerate(spans)
                              if start < match.end() and end > match.start()) for match in matches])
        candidates.append([names[match.group().lower()] for match in matches])
    inputs = inputs.to(reader.device)
    features = reader.backbone(**inputs, use_cache=False).last_hidden_state.float()
    lexical = reader.backbone.get_input_embeddings()(inputs.input_ids).float()
    features = torch.cat((features, F.layer_norm(lexical, (lexical.shape[-1],))), dim=-1)
    return features, inputs.attention_mask.bool(), torch.tensor(positions), torch.tensor(candidates)


@torch.no_grad()
def parse(pointer, reader, texts):
    clauses = [clause for text in texts for clause in split_clauses(text)]
    features, mask, positions, candidates = clause_inputs(reader, clauses)
    choice = pointer.parser(features, mask, positions).argmax(-1)
    order = torch.stack((choice, 1 - choice), dim=-1).flatten(1)
    return initial_state(candidates.gather(1, order).reshape(len(texts), 6))


@torch.no_grad()
def prepare():
    _, reader, checkpoint = load_pointer()
    values = vectors(reader)
    install_vectors(reader, values)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    directory = ROOT / 'data' / 'emoji-full'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'catalog.json').write_text(json.dumps(CATALOG, ensure_ascii=False, indent=2), encoding='utf-8')
    dataset = {}
    for name, examples in [('train', rows(len(ENTITIES) * 2, 17, cover=True)),
                           ('validation', rows(256, 23)), ('test', rows(512, 31))]:
        state = initial_state(torch.tensor([row['entities'] for row in examples]))
        target = torch.tensor([[FACT, row['derived'][0], LEFT, row['derived'][1], row['answer'], END] for row in examples])
        features = []
        for start in range(0, len(state), 32):
            features.append(reader.state_features(state[start:start + 32]))
            if start % 1024 == 0:
                print(f'{name}: cached {start}/{len(state)}', flush=True)
        dataset[name] = dict(state=state, target=target, features=torch.cat(features))
        records = [dict(state=render(s.tolist()), continuation=render(t.tolist())) for s, t in zip(state, target)]
        (directory / f'{name}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records), encoding='utf-8')
    torch.save(dict(dataset=dataset, vectors=values, base_model=checkpoint['base_model'], alphabet=ALPHABET), OUTPUT / 'features.pt')
    print(f'{len(ALPHABET)} atomic symbols; {len(ENTITIES)} entity symbols', flush=True)


def loss(model, data, ids):
    target = data['target'][ids]
    prefix = torch.cat((torch.full((len(ids), 1), START), target[:, :-1]), dim=1)
    return F.nll_loss(model(data['state'][ids], data['features'][ids], prefix).flatten(0, 1), target.flatten())


@torch.no_grad()
def evaluate(model, reader, data):
    exact = verdict = ended = 0
    for ids in torch.arange(len(data['state'])).split(32):
        output = model.generate(reader, data['state'][ids])
        for generated, expected in zip(output, data['target'][ids]):
            actual = visible_tokens(generated)
            target = expected.tolist()[:-1]
            ended += END in generated.tolist()
            exact += END in generated.tolist() and actual == target
            verdict += bool(actual) and actual[-1] == target[-1]
    return dict(count=len(data['state']), exact=exact / len(data['state']),
                answer=verdict / len(data['state']), terminated=ended / len(data['state']))


@torch.no_grad()
def evaluate_text(model, pointer, reader, data, named):
    states, targets, texts = [], [], []
    for state, target in zip(data['state'][:96], data['target'][:96]):
        entities = state[[1, 3, 5, 7, 13, 15]].tolist()
        names = [CATALOG[index]['unicode_name'] if named else ALPHABET[index] for index in entities]
        if any(re.search(r'[.?]', name) for name in names):
            continue
        a, b, c, d, x, y = names
        texts.append(f'The {a} is left of the {b}. The {c} is left of the {d}. Is the {x} left of the {y}?')
        states.append(state)
        targets.append(target)
    parsed = torch.cat([parse(pointer, reader, texts[start:start + 16]) for start in range(0, len(texts), 16)])
    report = evaluate(model, reader, dict(state=parsed, target=torch.stack(targets)))
    report['parsed_state'] = (parsed == torch.stack(states)).all(1).float().mean().item()
    return report


def train(args):
    torch.manual_seed(7)
    prepared = torch.load(OUTPUT / 'features.pt', weights_only=True)
    pointer, reader, _ = load_pointer()
    install_vectors(reader, prepared['vectors'])
    model = FullEmojiLM(prepared['vectors'])
    model.initialize_from_pointer(pointer)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    best = float('inf')
    for step in range(1, args.steps + 1):
        data = prepared['dataset']['train']
        ids = torch.randint(len(data['state']), (32,))
        optimizer.zero_grad()
        current = loss(model, data, ids)
        current.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 200 == 0 or step == args.steps:
            model.eval()
            validation = prepared['dataset']['validation']
            with torch.no_grad():
                score = sum(loss(model, validation, ids).item() * len(ids)
                            for ids in torch.arange(len(validation['state'])).split(32)) / len(validation['state'])
            print(f'step {step}: train={current.item():.4f}, validation={score:.4f}', flush=True)
            if score < best:
                best = score
                torch.save(dict(model=model.state_dict(), config=model.config, alphabet=ALPHABET,
                                step=step, validation_loss=score, base_model=prepared['base_model']), OUTPUT / 'model.pt')
            model.train()
    checkpoint = torch.load(OUTPUT / 'model.pt', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    report = dict(symbols=len(ALPHABET), entity_symbols=len(ENTITIES), selected_step=checkpoint['step'])
    for name in ('validation', 'test'):
        report[name] = evaluate(model, reader, prepared['dataset'][name])
        print(json.dumps({name: report[name]}), flush=True)
    for name, named in [('text_names', True), ('text_glyphs', False)]:
        report[name] = evaluate_text(model, pointer, reader, prepared['dataset']['test'], named)
        print(json.dumps({name: report[name]}), flush=True)
    (OUTPUT / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


def generate(args):
    checkpoint = torch.load(OUTPUT / 'model.pt', weights_only=True)
    if tuple(checkpoint['alphabet']) != ALPHABET:
        raise ValueError('Checkpoint and Unicode catalog differ')
    pointer, reader, _ = load_pointer()
    pointer.parser.load_state_dict(torch.load(OUTPUT / 'parser.pt', weights_only=True)['model'])
    model = FullEmojiLM(checkpoint['model']['embedding.weight'], **checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    install_vectors(reader, model.embedding.weight)
    state = parse(pointer, reader, [args.text])
    output = model.generate(reader, state)[0]
    print(render(visible_tokens(output)))


def adapt_parser(args):
    torch.manual_seed(7)
    pointer, reader, _ = load_pointer()
    dataset = {}
    for name, forms, count, seed in [('train', TRAIN_FORMS, 2048, 41),
                                     ('validation', VALIDATION_FORMS, 256, 43)]:
        rng = random.Random(seed)
        texts, labels = [], []
        while len(texts) < count:
            entities = rng.sample(ENTITIES, 2)
            names = [CATALOG[index]['unicode_name'] if len(texts) % 2 else ALPHABET[index] for index in entities]
            if any(re.search(r'[.?]', name) for name in names):
                continue
            form, orientation = rng.choice(forms)
            texts.append(form.format(a=names[0], b=names[1]))
            labels.append(orientation)
        batches = [clause_inputs(reader, texts[start:start + 32]) for start in range(0, count, 32)]
        # Batch-local padding avoids allocating a corpus-wide longest sequence.
        dataset[name] = [(features, mask, positions, torch.tensor(labels[start:start + len(features)]))
                         for start, (features, mask, positions, _) in zip(range(0, count, 32), batches)]
        print(f'parser {name}: cached {count} clauses', flush=True)
    optimizer = torch.optim.AdamW(pointer.parser.parameters(), lr=0.0003)
    best = float('inf')
    rng = random.Random(7)
    for step in range(1, args.steps + 1):
        features, mask, positions, labels = rng.choice(dataset['train'])
        pointer.parser.train()
        optimizer.zero_grad()
        current = F.cross_entropy(pointer.parser(features, mask, positions)[:, 0], labels)
        current.backward()
        nn.utils.clip_grad_norm_(pointer.parser.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            pointer.parser.eval()
            with torch.no_grad():
                score = sum(F.cross_entropy(pointer.parser(f, m, p)[:, 0], y).item() for f, m, p, y in dataset['validation']) / len(dataset['validation'])
            print(f'parser step {step}: train={current.item():.4f}, validation={score:.4f}', flush=True)
            if score < best:
                best = score
                torch.save(dict(model=pointer.parser.state_dict(), step=step, validation_loss=score), OUTPUT / 'parser.pt')
    pointer.parser.load_state_dict(torch.load(OUTPUT / 'parser.pt', weights_only=True)['model'])
    pointer.parser.eval()
    checkpoint = torch.load(OUTPUT / 'model.pt', weights_only=True)
    model = FullEmojiLM(checkpoint['model']['embedding.weight'], **checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    install_vectors(reader, model.embedding.weight)
    prepared = torch.load(OUTPUT / 'features.pt', weights_only=True)
    report = json.loads((OUTPUT / 'evaluation.json').read_text(encoding='utf-8'))
    for name, named in [('text_names', True), ('text_glyphs', False)]:
        report[name] = evaluate_text(model, pointer, reader, prepared['dataset']['test'], named)
        print(json.dumps({name: report[name]}), flush=True)
    report['parser_selected_step'] = torch.load(OUTPUT / 'parser.pt', weights_only=True)['step']
    (OUTPUT / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (ROOT / 'data' / 'emoji-full' / 'catalog.json').write_text(json.dumps(CATALOG, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare')
    training = commands.add_parser('train')
    training.add_argument('--steps', type=int, default=2000)
    adaptation = commands.add_parser('adapt-parser')
    adaptation.add_argument('--steps', type=int, default=1000)
    generation = commands.add_parser('generate')
    generation.add_argument('text')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'train':
        train(args)
    elif args.command == 'adapt-parser':
        adapt_parser(args)
    else:
        generate(args)
