"""Generate offline English user prompts; emoji states/replies remain authored."""

import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from conversation_data import DIRECTORY, SKILLS
from run import ROOT


TEMPLATES = (
    ('recall_like', ('🧠', '💖', '❓'), ('What do I like?', 'Tell me my favorite.', 'Can you remember what I enjoy?')),
    ('recall_dislike', ('🧠', '💔', '❓'), ('What do I dislike?', 'Tell me what I hate.', 'Can you remember what I do not like?')),
    ('preference_like_template', ('💖', '{item}'), ('I like {item}.', 'I love {item}.', 'My favorite is {item}.')),
    ('preference_dislike_template', ('💔', '{item}'), ('I dislike {item}.', 'I hate {item}.', 'I do not like {item}.')),
    ('correction_template', ('🔄', '💖', '{item}'), ('Actually, I prefer {item} instead.', 'I changed my mind, I like {item} now.')),
    ('color_fact_template', ('📌', '{item}', '🎨', '{color}'), ('The {item} is {color}.', 'My {item} has a {color} color.')),
    ('color_query_template', ('❓', '{item}', '🎨'), ('What color is the {item}?', 'Tell me the color of the {item}.')),
    ('color_pronoun_query', ('❓', '🎨'), ('What color is it?', 'Can you remember its color?')),
    ('location_fact_template', ('📌', '{item}', '📍', '{place}'), ('The {item} is at the {place}.', 'My {item} is located at the {place}.')),
    ('location_move_template', ('🔄', '{item}', '📍', '{place}'), ('The {item} moved to the {place}.', 'The {item} is now at the {place}.')),
    ('location_query_template', ('❓', '{item}', '📍'), ('Where is the {item}?', 'Where can I find the {item}?')),
    ('addition_template', ('🔢', '{a}', '➕', '{b}'), ('What is {a} plus {b}?', 'Add {a} and {b}.')),
    ('count_addition_template', ('🔢', '{item}', '{a}', '➕', '{b}'), ('I have {a} {item} and get {b} more. How many altogether?',)),
)


def generate():
    torch.set_num_threads(2)
    torch.manual_seed(313)
    sources = json.loads((DIRECTORY / 'teacher.json').read_text(encoding='utf-8'))
    options = dict(revision=sources['revision'], cache_dir=ROOT / '.hf-cache', local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(sources['model'], **options)
    model = AutoModelForCausalLM.from_pretrained(sources['model'], dtype=torch.bfloat16, **options).eval()
    output = DIRECTORY / 'generated-user-prompts.jsonl'
    with output.open('w', encoding='utf-8') as journal:
        specs = [(skill, state, reply, training) for skill, state, reply, training, *_ in SKILLS]
        specs.extend((skill, state, (), training) for skill, state, training in TEMPLATES)
        for skill, state, reply, training in specs:
            prompt = ('Write six different short messages a USER could send with the same intent as these examples: '
                      + ' / '.join(training[:3])
                      + '\nDo not answer the messages. Do not write advice or explanations. '
                      'Return only a JSON array of six strings. Use varied natural wording. '
                      'Keep every brace-enclosed placeholder exactly as written.')
            messages = [dict(role='system', content='You generate faithful paraphrases of user messages, never answers.'),
                        dict(role='user', content=prompt)]
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                    tokenize=True, return_dict=True, return_tensors='pt')
            with torch.no_grad():
                tokens = model.generate(**inputs, max_new_tokens=240, do_sample=True, temperature=0.7,
                                        top_p=0.95, repetition_penalty=1.05, use_cache=True,
                                        pad_token_id=tokenizer.eos_token_id)
            raw = tokenizer.decode(tokens[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            match = re.search(r'\[.*?\]', raw, re.DOTALL)
            candidates = []
            if match:
                try:
                    values = json.loads(match.group())
                except json.JSONDecodeError:
                    values = []
                candidates = [value.strip() for value in values if isinstance(value, str) and 3 <= len(value.strip()) <= 160]
            row = dict(skill=skill, state=state, reply=reply, candidates=candidates, raw=raw)
            journal.write(json.dumps(row, ensure_ascii=False) + '\n')
            journal.flush()
            print(json.dumps(dict(skill=skill, candidates=candidates), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    generate()
