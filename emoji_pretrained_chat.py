"""Chat with a pretrained transformer whose answer vocabulary is emojis only."""

import argparse
import json

from peft import set_peft_model_state_dict
import torch

from emoji_general_model import GeneralEncoder, compact
from emoji_grounded_semantics import SemanticReader
from emoji_pretrained_model import PretrainedReply, backbone
from run import ROOT


def load(name):
    checkpoint = torch.load(ROOT / 'runs/emoji-pretrained' / name / 'runtime.pt', weights_only=True)
    encoder = GeneralEncoder(checkpoint['encoder']['vectors'], **checkpoint['encoder_config'])
    encoder.load_state_dict(checkpoint['encoder'])
    model = backbone(checkpoint['source'])
    vectors = checkpoint['heads']['vectors']
    reply = PretrainedReply(model, vectors[:-1], vectors[-1])
    missing = reply.load_state_dict(checkpoint['heads'], strict=False).missing_keys
    if any(not key.startswith('backbone.') for key in missing):
        raise ValueError('Incomplete emoji decoder checkpoint')
    set_peft_model_state_dict(model, checkpoint['lora'])
    source = checkpoint['semantic_model']
    reader = SemanticReader(source['model'], source['revision'])
    return encoder.eval(), reply.eval(), reader, checkpoint['meanings']


@torch.no_grad()
def answer(encoder, reply, reader, meanings, question):
    features, mask = reader.text_inputs([question])
    features = compact(features, mask)
    states = encoder(features, torch.ones(features.shape[:2], dtype=torch.bool))[2]
    generated = reply.generate(states)[0].tolist()
    ids = generated[:generated.index(reply.end)] if reply.end in generated else generated
    symbols = [meanings[index]['symbol'] for index in ids]
    return dict(state=''.join(meanings[index]['symbol'] for index in states[0].tolist()), reply=''.join(symbols),
                symbols=symbols, core_meanings=[meanings[index]['core_meaning'] for index in ids], terminated=reply.end in generated)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='pilot')
    parser.add_argument('--question')
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = load(args.name)
    if args.question:
        print(json.dumps(answer(*model, args.question), ensure_ascii=False))
    else:
        while True:
            question = input('> ').strip()
            if question:
                print(json.dumps(answer(*model, question), ensure_ascii=False))
