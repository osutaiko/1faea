"""Generate direct emoji-only answers with a locally loaded pretrained LLM."""

import torch
from pathlib import Path
from huggingface_hub import snapshot_download, try_to_load_from_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

from emoji_vocabulary import catalog_symbols, meaning_map
from run import ROOT


MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
MODEL_REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
MAX_EMOJIS = 8
INSTRUCTIONS = (
    "Understand and answer the user's current request as a standalone message. Work out the answer "
    'before choosing a concise sequence of relevant emoji concepts. Use distinct symbols in a natural '
    'order to preserve important facts, negation, quantities, comparisons, and relationships. Match the '
    'intent: answer questions directly, acknowledge feelings, or give a useful next step. Prefer specific '
    'answers over generic reactions. Never repeat symbols as filler. Avoid unsupported claims; when '
    'uncertain, express uncertainty.'
)
EXAMPLES = [
    ('Which planet do we live on?', '🌍'),
    ('Is mixing bleach and ammonia safe?', '🧪☠️🚫'),
    ('I feel nervous about starting a new job. Any advice?', '🫂🌬️🌱'),
    ('Thanks, that helped!', '😊✨'),
]


class EmojiTokenGrammar:
    """Allow only complete catalog emoji token sequences, followed by EOS."""

    def __init__(self, encodings, eos_token_id, max_emojis=MAX_EMOJIS):
        self.transitions = [{}]
        self.terminal = [False]
        self.eos_token_id = eos_token_id
        self.max_emojis = max_emojis
        self.max_symbol_tokens = max(map(len, encodings.values()))

        for tokens in encodings.values():
            node = 0
            for token in tokens:
                if token not in self.transitions[node]:
                    self.transitions[node][token] = len(self.transitions)
                    self.transitions.append({})
                    self.terminal.append(False)
                node = self.transitions[node][token]
            self.terminal[node] = True

    @property
    def max_new_tokens(self):
        return self.max_symbol_tokens * self.max_emojis + 1

    def start(self):
        return frozenset({(0, 0)})

    def advance(self, states, token):
        next_states = set()
        for node, completed in states:
            child = self.transitions[node].get(token)
            if child is not None:
                next_states.add((child, completed))
            if self.terminal[node] and completed + 1 < self.max_emojis:
                child = self.transitions[0].get(token)
                if child is not None:
                    next_states.add((child, completed + 1))
        if not next_states:
            raise ValueError('Generated token sequence left the emoji vocabulary')
        return frozenset(next_states)

    def allowed(self, states):
        tokens = set()
        can_end = False
        for node, completed in states:
            tokens.update(self.transitions[node])
            if self.terminal[node]:
                can_end = True
                if completed + 1 < self.max_emojis:
                    tokens.update(self.transitions[0])
        if can_end:
            tokens.add(self.eos_token_id)
        return sorted(tokens)


class EmojiLocalChat:
    def __init__(self):
        self.meanings = meaning_map()
        cache_dir = ROOT / '.hf-cache'
        cached_config = try_to_load_from_cache(
            MODEL_ID, 'config.json', revision=MODEL_REVISION, cache_dir=cache_dir)
        cached_weights = try_to_load_from_cache(
            MODEL_ID, 'model.safetensors', revision=MODEL_REVISION, cache_dir=cache_dir)
        model_path = (Path(cached_config).parent if isinstance(cached_config, str) and
                      isinstance(cached_weights, str) else snapshot_download(
                          MODEL_ID, revision=MODEL_REVISION, cache_dir=cache_dir,
                          allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model']))
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True)
        encodings = {symbol: tuple(self.tokenizer.encode(symbol, add_special_tokens=False))
                     for symbol in self.meanings}
        if any(not tokens or self.tokenizer.decode(tokens, clean_up_tokenization_spaces=False) != symbol
               for symbol, tokens in encodings.items()):
            raise ValueError('Tokenizer cannot represent every catalog emoji exactly')
        if len(set(encodings.values())) != len(encodings):
            raise ValueError('Tokenizer maps distinct catalog emojis to the same token sequence')
        self.grammar = EmojiTokenGrammar(encodings, self.tokenizer.eos_token_id)

        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, dtype=torch.float32)
        self.model.to(self.device).eval()

    def answer(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Enter a non-empty message')
        messages = [{'role': 'system', 'content': INSTRUCTIONS}]
        for prompt, reply in EXAMPLES:
            messages.extend([{'role': 'user', 'content': prompt},
                             {'role': 'assistant', 'content': reply}])
        messages.append({'role': 'user', 'content': question.strip()})
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        prompt_length = input_ids.shape[1]

        def allowed_tokens(_, sequence):
            states = self.grammar.start()
            for token in sequence[prompt_length:].tolist():
                states = self.grammar.advance(states, token)
            return self.grammar.allowed(states)

        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                do_sample=False, max_new_tokens=self.grammar.max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=allowed_tokens)
        symbols = self.tokenizer.decode(
            output[0, prompt_length:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        result = catalog_symbols([symbols], self.meanings)
        if not result or len(result) > MAX_EMOJIS:
            raise ValueError('Model returned an empty or overlong emoji reply')
        return ''.join(result)
