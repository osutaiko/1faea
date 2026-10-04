"""Evaluate fresh features, operator contrasts, and model-generated histories."""

import argparse
from collections import defaultdict
import json

import torch

import conversation
from conversation_model import SemanticAtomicDecoder, EmojiConversation, conversation_turns, memory, render, visible
from conversation_reliability_data import DIRECTORY, TARGET_REVISION, dataset
from conversation_data import IDS
from emoji_catalog import ALPHABET
from emoji_lm_model import END
from run import ROOT


def summarize(totals):
    return {group: {key: value if key == 'count' else value / numbers['count']
                    for key, value in numbers.items()} for group, numbers in totals.items()}


@torch.no_grad()
def evaluate(args):
    encoder, decoder, reader, checkpoints = conversation.load(args.name, SemanticAtomicDecoder)
    rows, episodes = dataset()
    if args.live_only:
        live(encoder, decoder, reader, args, episodes[args.split])
        return
    selected = rows[args.split]
    if args.numbers:
        def numeric(row):
            if row['state'][0] in (IDS['📌'], IDS['🔄']) and row['state'][-2] == IDS['🔢']:
                digit = ALPHABET[row['state'][-1]]
                return dict(row, text=row['text'].replace(digit, digit[0]))
            return row
        selected = [numeric(row) for row in selected]
        episodes[args.split] = [[numeric(row) for row in episode] for episode in episodes[args.split]]
    if args.public:
        selected = [json.loads(line) for line in (DIRECTORY / f'public-scope-{args.split}.jsonl').read_text(encoding='utf-8').splitlines()]
    if args.legacy:
        manifest = json.loads((conversation.OUTPUT / 'features.json').read_text(encoding='utf-8'))
        selected = [row for path in manifest['splits'][args.split] for row in conversation.batch(path)['rows']]
    if args.limit:
        selected = selected[:args.limit]
    selected.sort(key=lambda row: len(row['history']))
    totals = defaultdict(lambda: defaultdict(int))
    examples = []
    for start in range(0, len(selected), 16):
        records = selected[start:start + 16]
        meanings = encoder.encode(reader, [row['text'] for row in records])
        states, mask = memory([row['history'] for row in records], [visible(tokens) for tokens in meanings])
        replies = decoder.respond(reader, states, mask)
        gold, gold_mask = memory([row['history'] for row in records], [row['state'] for row in records])
        oracle = decoder.respond(reader, gold, gold_mask)
        for row, meaning, reply, oracle_reply in zip(records, meanings, replies, oracle):
            state_correct = END in meaning.tolist() and visible(meaning) == row['state']
            reply_correct = END in reply.tolist() and visible(reply) == row['reply']
            oracle_correct = END in oracle_reply.tolist() and visible(oracle_reply) == row['reply']
            for group in ('all', row['skill']):
                totals[group]['count'] += 1
                totals[group]['meaning'] += state_correct
                totals[group]['reply'] += reply_correct
                totals[group]['correct_trace'] += state_correct and reply_correct
                totals[group]['reply_given_gold_state'] += oracle_correct
                totals[group]['terminated'] += END in reply.tolist()
            if args.public:
                group = 'supported' if row['supported'] else 'unsupported'
                totals[group]['count'] += 1
                totals[group]['reply'] += reply_correct
                totals[group]['meaning'] += state_correct
            examples.append(dict(text=row['text'], skill=row['skill'], state=render(meaning),
                                 expected_state=render(row['state']), reply=render(reply),
                                 expected_reply=render(row['reply']), oracle_reply=render(oracle_reply),
                                 correct=reply_correct, history=render(row['history'])))
        print(f'{conversation.timestamp()} {args.name} {args.split}: {start + len(records)}/{len(selected)}', flush=True)
    operator_groups = defaultdict(list)
    for item in examples:
        if item['skill'] == 'attribute_operator':
            operator_groups[item['history']].append(item)
    contrasts = dict(groups=len(operator_groups),
                     all_queries_correct=sum(all(item['correct'] for item in items) for items in operator_groups.values()),
                     distinct_oracle_replies=sum(len({item['oracle_reply'] for item in items}) == 3
                                                for items in operator_groups.values()))
    report = dict(created=conversation.timestamp(), split=args.split, legacy=args.legacy, target_revision=TARGET_REVISION,
                  encoder_step=checkpoints[0]['step'], decoder_step=checkpoints[1]['step'],
                  features=('fresh runtime text-backbone and distilled emoji features' if 'emoji_features' in checkpoints[1]
                            else 'fresh runtime backbone features'), metrics=summarize(totals),
                  operator_contrasts=contrasts, examples=examples)
    prefix = 'public-scope' if args.public else 'legacy' if args.legacy else 'reliability'
    if args.numbers:
        prefix += '-numbers'
    (conversation.OUTPUT / args.name / f'{prefix}-{args.split}.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(metrics=report['metrics'], operator_contrasts=contrasts), indent=2), flush=True)
    if args.live and not args.legacy and not args.public:
        live(encoder, decoder, reader, args, episodes[args.split])


@torch.no_grad()
def live(encoder, decoder, reader, args, episodes):
    examples = []
    totals = defaultdict(lambda: defaultdict(int))
    for start in range(0, len(episodes), 16):
        group = episodes[start:start + 16]
        sessions = [EmojiConversation(encoder, decoder, reader) for _ in group]
        for position in range(max(len(episode) for episode in group)):
            active = [(index, session, episode[position]) for index, (session, episode) in enumerate(zip(sessions, group))
                      if position < len(episode)]
            outputs = conversation_turns([session for _, session, _ in active], [row['text'] for _, _, row in active])
            for (index, session, row), output in zip(active, outputs):
                add_live(totals, examples, start + index, session, row, output)
        print(f'{conversation.timestamp()} {args.name} live {args.split}: {start + len(group)}/{len(episodes)}', flush=True)
    report = dict(created=conversation.timestamp(), split=args.split, episodes=len(episodes), batch_size=16,
                  history='model-generated states and replies; never replaced with gold history',
                  metrics=summarize(totals), examples=examples)
    prefix = 'live-numbers' if args.numbers else 'live'
    (conversation.OUTPUT / args.name / f'{prefix}-{args.split}.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['metrics'], indent=2), flush=True)


def add_live(totals, examples, index, session, row, output):
    state_correct = output['state_terminated'] and output['state'] == row['state']
    reply_correct = output['reply_terminated'] and output['reply'] == row['reply']
    for group in ('all', row['skill']):
        totals[group]['count'] += 1
        totals[group]['meaning'] += state_correct
        totals[group]['reply'] += reply_correct
        totals[group]['correct_trace'] += state_correct and reply_correct
        totals[group]['terminated'] += output['reply_terminated']
    examples.append(dict(episode=index, text=row['text'], skill=row['skill'],
                         state=render(output['state']), reply=render(output['reply']),
                         expected_state=render(row['state']), expected_reply=render(row['reply']),
                         correct=reply_correct, dropped_turns=session.dropped_turns))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='reliability')
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--legacy', action='store_true')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--live-only', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--numbers', action='store_true')
    parser.add_argument('--public', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    conversation.OUTPUT = ROOT / 'runs' / 'conversation-compositional'
    evaluate(args)
