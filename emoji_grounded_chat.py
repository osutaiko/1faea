"""Run standalone questions through hard emoji states and direct emoji replies."""

import argparse
import json

import torch

from emoji_grounded_model import GroundedEncoder, GroundedReply
from emoji_grounded_semantics import SemanticReader
from run import ROOT


def load(name):
    checkpoint = torch.load(ROOT / 'runs' / 'emoji-grounded' / name / 'runtime.pt', weights_only=True)
    encoder = GroundedEncoder(checkpoint['encoder']['vectors'], **checkpoint['encoder_config'])
    encoder.load_state_dict(checkpoint['encoder'])
    reply = GroundedReply(checkpoint['reply']['vectors'], **checkpoint['reply_config'])
    reply.load_state_dict(checkpoint['reply'])
    reader = SemanticReader(checkpoint['semantic_model']['model'], checkpoint['semantic_model']['revision'])
    return encoder.eval(), reply.eval(), reader, checkpoint['meanings']


@torch.no_grad()
def answer(encoder, reply, reader, meanings, question):
    features, mask = reader.text_inputs([question])
    states = encoder(features, mask)[2]
    generated = reply.generate(states)[0].tolist()
    ids = generated[:generated.index(reply.end)] if reply.end in generated else generated
    return dict(state=''.join(meanings[index]['symbol'] for index in states[0].tolist()),
                reply=''.join(meanings[index]['symbol'] for index in ids),
                core_meanings=[meanings[index]['core_meaning'] for index in ids],
                terminated=reply.end in generated)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='semantic-v1')
    parser.add_argument('--question')
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = load(args.name)
    if args.question:
        print(json.dumps(answer(*model, args.question), ensure_ascii=False))
    else:
        print('Experimental standalone emoji model. Enter a question; Ctrl+C exits.')
        while True:
            question = input('> ').strip()
            if question:
                print(json.dumps(answer(*model, question), ensure_ascii=False))
