"""Compare fresh emoji reply latency on identical validation memories."""

import argparse
import json
import statistics
import time

import torch

import conversation
from conversation_model import SemanticAtomicDecoder, memory, visible
from conversation_reliability_data import dataset
from emoji_lm_model import END
from run import ROOT


@torch.no_grad()
def benchmark(names):
    rows = dataset()[0]['validation']
    queries = [row for row in rows if row['skill'] == 'attribute_operator'][:12]
    results = {}
    for name in names:
        _, decoder, reader, _ = conversation.load(name, SemanticAtomicDecoder)
        elapsed, correct, lengths = [], 0, []
        state, mask = memory([queries[0]['history']], [queries[0]['state']])
        decoder.respond(reader, state, mask)
        for row in queries:
            state, mask = memory([row['history']], [row['state']])
            started = time.perf_counter()
            reply = decoder.respond(reader, state, mask)[0]
            elapsed.append(time.perf_counter() - started)
            correct += END in reply.tolist() and visible(reply) == row['reply']
            lengths.append(len(visible(reply)))
        results[name] = dict(queries=len(queries), median_seconds=statistics.median(elapsed),
                             total_seconds=sum(elapsed), exact_replies=correct,
                             output_symbols=lengths, seconds=elapsed)
        print(json.dumps(dict(name=name, **results[name])), flush=True)
        del decoder, reader
    report = dict(created=conversation.timestamp(), threads=torch.get_num_threads(),
                  scope='single-query emoji reply stage; excludes text encoding and model loading',
                  memories='first twelve attribute-operator validation rows; gold states and history', results=results)
    (conversation.OUTPUT / 'reliability-selected' / 'runtime-benchmark.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--names', nargs='+', default=['reliability-reviewed', 'reliability-selected'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    conversation.OUTPUT = ROOT / 'runs' / 'conversation-compositional'
    benchmark(args.names)
