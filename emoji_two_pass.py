"""Pretrained direct emoji generation with a fresh emoji-only second call."""

import argparse
import json
from pathlib import Path
import time

from huggingface_hub import hf_hub_download
import outlines
from outlines.inputs import Chat
from outlines.types import JsonSchema
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run import ROOT


ENCODE = ('Encode the user question and the information needed to answer it as emojis. '
          'Preserve question intent, entities, actions and negation. Output a JSON array of emojis only. '
          'Never output English reasoning or an English answer.')
ANSWER = ('The user message is a question encoded entirely as emojis. Answer that question using emojis. '
          'Output a JSON array of emojis only. Never output English reasoning or an English answer.')


def second_messages(state):
    return [dict(role='system', content=ANSWER), dict(role='user', content=json.dumps(state, ensure_ascii=False))]


def emoji_schema(symbols):
    return dict(type='array', items=dict(type='string', enum=sorted(symbols)), minItems=1, maxItems=4)


class EmojiModel:
    def __init__(self):
        self.source = json.loads((ROOT / 'data/conversation/sources.json').read_text())
        config = hf_hub_download(self.source['model'], 'config.json', revision=self.source['model_revision'],
                                 cache_dir=ROOT / '.hf-cache', local_files_only=True)
        directory = Path(config).parent
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True, dtype=torch.bfloat16).eval()
        meanings = json.loads((ROOT / 'runs/emoji-general/five-hour/meanings.json').read_text(encoding='utf-8'))
        self.symbols = {row['symbol'] for row in meanings}
        self.constrained = outlines.from_transformers(self.model, self.tokenizer)
        self.generator = outlines.Generator(self.constrained, JsonSchema(emoji_schema(self.symbols)), backend='llguidance')

    @torch.no_grad()
    def generate(self, messages):
        # The generator starts a fresh model cache for each prompt.
        with torch.compiler.set_stance('force_eager'):
            raw = self.generator(Chat(messages), max_new_tokens=512, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        symbols = json.loads(raw)
        if not isinstance(symbols, list) or not 1 <= len(symbols) <= 4 or any(symbol not in self.symbols for symbol in symbols):
            raise ValueError('Incomplete or invalid constrained emoji output')
        return symbols

    def answer(self, question):
        started = time.monotonic()
        state = self.generate([dict(role='system', content=ENCODE), dict(role='user', content=question)])
        reply = self.generate(second_messages(state))
        return dict(state=''.join(state), reply=''.join(reply), state_symbols=state, symbols=reply,
                    elapsed_seconds=time.monotonic() - started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = EmojiModel()
    if args.evaluate:
        from emoji_grounded_eval import CASES, score
        examples = []
        for case in CASES[:6]:
            row = dict(question=case['question'], **model.answer(case['question']))
            row['passed'] = score(row['symbols'], case)
            examples.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        report = dict(model=model.source['model'], revision=model.source['model_revision'], vocabulary=len(model.symbols),
                      constraint_backend='Outlines with LLGuidance', maximum_symbols_per_stage=4,
                      trained=False, original_question_visible_to_second_call=False, cross_call_hidden_cache=False,
                      runtime_english_answer_generation=False, json_is_transport_only=True,
                      fixed_checks_passed=sum(row['passed'] for row in examples), fixed_checks_total=len(examples),
                      general_chat_demonstrated=False, examples=examples)
        (ROOT / 'data/emoji-grounded/two-pass-result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    elif args.question:
        print(json.dumps(model.answer(args.question), ensure_ascii=False))
    else:
        while True:
            question = input('> ').strip()
            if question:
                result = model.answer(question)
                print(result['state'])
                print(result['reply'])
