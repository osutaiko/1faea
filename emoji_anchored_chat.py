"""Direct emoji replies with ordered literal evidence from the existing map."""

import argparse
import json

import torch

from emoji_general_model import anchored_states, compact
from emoji_grounded_model import MeaningLexicon
from emoji_pretrained_chat import load
from run import ROOT


@torch.no_grad()
def answer(encoder, reply, reader, meanings, question):
    features, mask = reader.text_inputs([question])
    features = compact(features, mask)
    states = encoder(features, torch.ones(features.shape[:2], dtype=torch.bool))[2]
    anchors = MeaningLexicon(meanings).anchors(question, states.shape[1], literal_only=True)
    states = anchored_states(states, torch.tensor([anchors], dtype=torch.long))
    generated = reply.generate(states)[0].tolist()
    ids = generated[:generated.index(reply.end)] if reply.end in generated else generated
    symbols = [meanings[index]['symbol'] for index in ids]
    return dict(state=''.join(meanings[index]['symbol'] for index in states[0].tolist()), reply=''.join(symbols),
                symbols=symbols, core_meanings=[meanings[index]['core_meaning'] for index in ids], terminated=reply.end in generated)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='anchored-v1')
    parser.add_argument('--question')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'runs/emoji-pretrained' / args.name / 'manifest.json').read_text())
    if not manifest['anchor_input']:
        raise ValueError('This chat requires an anchored-input experiment')
    torch.set_num_threads(2)
    model = load(args.name)
    if args.question:
        print(json.dumps(answer(*model, args.question), ensure_ascii=False))
    else:
        while True:
            question = input('> ').strip()
            if question:
                print(json.dumps(answer(*model, question), ensure_ascii=False))
