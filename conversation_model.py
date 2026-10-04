"""Text understanding, discrete emoji memory, and direct emoji reply generation."""

import json
from pathlib import Path
import re

from huggingface_hub import hf_hub_download
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from emoji_catalog import ALPHABET, CATALOG
from emoji_lm_model import END, START
from conversation_data import USER, ASSISTANT, SEPARATOR
from run import ROOT


MEMORY_LIMIT = 128
PREFIX_LIMIT = 32

EMBEDDING_LABELS = {'👤': 'user', '🤖': 'assistant', '🔹': 'separator', '💖': 'like',
                    '💔': 'dislike', '🔄': 'update', '🧠': 'remember', '🎯': 'plan',
                    '🎨': 'color', '📍': 'location', '🔢': 'number', '➕': 'plus', '⚪': 'white'}
EMBEDDING_LABELS.update({str(digit) + '\ufe0f\u20e3': word for digit, word in
                         enumerate(('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'))})


class ConversationReader:
    def __init__(self, model_name=None):
        sources = json.loads((ROOT / 'data' / 'conversation' / 'sources.json').read_text(encoding='utf-8'))
        self.model_name = model_name or sources['model']
        revision = sources['model_revision'] if self.model_name == sources['model'] else None
        config = hf_hub_download(self.model_name, 'config.json', revision=revision, cache_dir=ROOT / '.hf-cache',
                                 local_files_only=True)
        directory = Path(config).parent
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        self.backbone = AutoModel.from_pretrained(directory, local_files_only=True,
                                                  dtype=torch.bfloat16).eval().requires_grad_(False)
        embedding = self.backbone.get_input_embeddings()
        with torch.no_grad():
            values = [embedding(torch.tensor(self.tokenizer.encode(EMBEDDING_LABELS.get(row['symbol'], row['name']),
                                                                  add_special_tokens=False))).mean(0)
                      for row in CATALOG]
        self.symbol_embedding = nn.Embedding.from_pretrained(torch.stack(values), freeze=True)
        names = {row['unicode_name'].lower(): index for index, row in enumerate(CATALOG)}
        names.update({row['symbol']: index for index, row in enumerate(CATALOG)})
        self.names = names
        self.pattern = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(name) for name in
                                                        sorted(names, key=len, reverse=True)) + r')(?!\w)', re.IGNORECASE)

    @torch.no_grad()
    def text_inputs(self, texts):
        formatted = [self.tokenizer.apply_chat_template([dict(role='user', content=text)], tokenize=False,
                                                        add_generation_prompt=False) for text in texts]
        inputs = self.tokenizer(formatted, padding=True, return_offsets_mapping=True, return_tensors='pt')
        offsets = inputs.pop('offset_mapping').tolist()
        if inputs.input_ids.shape[1] > MEMORY_LIMIT:
            raise ValueError(f'Input exceeds the {MEMORY_LIMIT}-token text limit')
        source_ids = torch.full_like(inputs.input_ids, -1)
        for row, (text, spans) in enumerate(zip(formatted, offsets)):
            for match in self.pattern.finditer(text):
                index = max(index for index, (start, end) in enumerate(spans)
                            if start < match.end() and end > match.start())
                source_ids[row, index] = self.names[match.group().lower()]
        hidden = self.backbone(**inputs, use_cache=False).last_hidden_state.float()
        lexical = self.backbone.get_input_embeddings()(inputs.input_ids).float()
        features = torch.cat((hidden, F.layer_norm(lexical, (lexical.shape[-1],))), dim=-1)
        return features, inputs.attention_mask.bool(), source_ids

    @torch.no_grad()
    def emoji_inputs(self, state, mask):
        embeddings = self.symbol_embedding(state)
        hidden = self.backbone(inputs_embeds=embeddings, attention_mask=mask,
                               use_cache=False).last_hidden_state.float()
        return torch.cat((hidden, F.layer_norm(embeddings.float(), (embeddings.shape[-1],))), dim=-1)


class AtomicDecoder(nn.Module):
    def __init__(self, vectors, width=192, layers=3):
        super().__init__()
        self.config = dict(width=width, layers=layers)
        dimension = vectors.shape[-1]
        self.embedding = nn.Embedding.from_pretrained(vectors.float(), freeze=True)
        self.memory_projection = nn.Sequential(nn.LayerNorm(dimension * 2), nn.Linear(dimension * 2, width))
        self.prefix_projection = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, width))
        self.memory_positions = nn.Embedding(MEMORY_LIMIT, width)
        self.prefix_positions = nn.Embedding(PREFIX_LIMIT, width)
        self.identity_projection = nn.Linear(MEMORY_LIMIT, width, bias=False)
        layer = nn.TransformerDecoderLayer(width, 4, width * 4, dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.keys = nn.Linear(width, width, bias=False)
        self.generator = nn.Linear(width, len(vectors) - 1)
        self.copy_gate = nn.Linear(width, 1)

    def vocabulary_logits(self, hidden):
        return self.generator(hidden)

    def forward(self, features, mask, source_ids, prefix):
        length = features.shape[1]
        memory = self.memory_projection(features.float())
        memory = memory + self.memory_positions(torch.arange(length))
        present = mask & (source_ids >= 0)
        identity = (source_ids[:, :, None] == source_ids[:, None, :]) & present[:, :, None] & present[:, None, :]
        identity = F.pad(identity.float(), (0, MEMORY_LIMIT - length))
        memory = memory + self.identity_projection(identity)
        target = self.prefix_projection(self.embedding(prefix))
        target = target + self.prefix_positions(torch.arange(prefix.shape[1]))
        causal = torch.ones(prefix.shape[1], prefix.shape[1], dtype=torch.bool).triu(1)
        hidden = self.decoder(target, memory, tgt_mask=causal, memory_key_padding_mask=~mask)
        scores = hidden @ self.keys(memory).transpose(1, 2) / hidden.shape[-1] ** 0.5
        copy_mask = present & (source_ids != END) & (source_ids != START)
        copy_mask &= (source_ids != USER) & (source_ids != ASSISTANT) & (source_ids != SEPARATOR)
        # Rows without grounded mentions use the vocabulary distribution only.
        scores = scores.masked_fill(~copy_mask[:, None], -1e4)
        attention = F.softmax(scores, dim=-1) * copy_mask[:, None]
        copied = torch.zeros(*hidden.shape[:2], len(ALPHABET))
        copied.scatter_add_(2, source_ids.clamp_min(0)[:, None].expand(-1, prefix.shape[1], -1), attention)
        probabilities = F.softmax(self.vocabulary_logits(hidden), dim=-1)
        generated = torch.cat((probabilities[..., :START], torch.zeros_like(probabilities[..., :1]),
                               probabilities[..., START:]), dim=-1)
        gate = torch.sigmoid(self.copy_gate(hidden))
        gate = torch.where(copy_mask.any(1)[:, None, None], gate, torch.ones_like(gate))
        return (gate * generated + (1 - gate) * copied).clamp_min(1e-9).log()

    @torch.no_grad()
    def select(self, features, mask, source_ids, prefix):
        return self(features, mask, source_ids, prefix)[:, -1].argmax(-1)

    @torch.no_grad()
    def encode(self, reader, texts):
        features, mask, source_ids = reader.text_inputs(texts)
        return self.generate(lambda active: (features[active], mask[active], source_ids[active]), len(texts))

    @torch.no_grad()
    def respond(self, reader, state, mask):
        # Fresh pretrained features at every symbol. Only integer IDs persist.
        return self.generate(lambda active: (reader.emoji_inputs(state[active], mask[active]),
                                             mask[active], state[active]), len(state))

    @torch.no_grad()
    def generate(self, inputs, batch_size):
        if self.training:
            raise RuntimeError('Call eval() before generation')
        prefix = torch.full((batch_size, 1), START)
        finished = torch.zeros(batch_size, dtype=torch.bool)
        for _ in range(PREFIX_LIMIT):
            active = (~finished).nonzero().flatten()
            features, mask, source_ids = inputs(active)
            token = torch.full((batch_size,), END)
            token[active] = self.select(features, mask, source_ids, prefix[active])
            prefix = torch.cat((prefix, token[:, None]), dim=1)
            finished |= token == END
            if finished.all():
                break
        return prefix[:, 1:]


class SemanticAtomicDecoder(AtomicDecoder):
    """Predict atomic emojis through their frozen named semantic embeddings."""
    def __init__(self, vectors, width=192, layers=3):
        super().__init__(vectors, width, layers)
        del self.generator
        self.output_projection = nn.Linear(width, vectors.shape[-1], bias=False)
        self.output_bias = nn.Parameter(torch.zeros(len(vectors) - 1))
        self.output_scale = nn.Parameter(torch.tensor(3.0))

    def vocabulary_logits(self, hidden):
        vocabulary = torch.cat((self.embedding.weight[:START], self.embedding.weight[START + 1:]))
        keys = F.normalize(vocabulary, dim=-1)
        query = F.normalize(self.output_projection(hidden), dim=-1)
        return query @ keys.T * self.output_scale.exp().clamp_max(100) + self.output_bias


def visible(tokens):
    values = tokens.tolist() if isinstance(tokens, torch.Tensor) else list(tokens)
    return values[:values.index(END)] if END in values else values


def render(tokens):
    return ''.join(ALPHABET[token] for token in visible(tokens))


def pad(sequences, value):
    return nn.utils.rnn.pad_sequence([torch.tensor(sequence, dtype=torch.long) for sequence in sequences],
                                    batch_first=True, padding_value=value)


def memory(histories, states):
    sequences = [[*history, USER, *state] for history, state in zip(histories, states)]
    if any(len(sequence) > MEMORY_LIMIT for sequence in sequences):
        raise ValueError(f'Conversation exceeds the {MEMORY_LIMIT}-emoji memory limit')
    state = pad(sequences, 7)
    mask = torch.arange(state.shape[1])[None] < torch.tensor([len(sequence) for sequence in sequences])[:, None]
    return state, mask


class EmojiConversation:
    def __init__(self, encoder, decoder, reader):
        self.encoder, self.decoder, self.reader = encoder, decoder, reader
        self.history = []
        self.turn_lengths = []
        self.dropped_turns = 0

    @torch.no_grad()
    def turn(self, text):
        return conversation_turns([self], [text])[0]


@torch.no_grad()
def conversation_turns(sessions, texts):
    """Batch independent sessions while retaining only each session's emoji IDs."""
    if not sessions or len(sessions) != len(texts):
        raise ValueError('Provide one message per nonempty session batch')
    first = sessions[0]
    if any((session.encoder is not first.encoder or session.decoder is not first.decoder or
            session.reader is not first.reader) for session in sessions):
        raise ValueError('Batched sessions must share their model and reader')
    if len({id(session) for session in sessions}) != len(sessions):
        raise ValueError('Each session may appear only once in a batch')
    selected = first.encoder.encode(first.reader, texts)
    states = [visible(tokens) for tokens in selected]
    for session, state in zip(sessions, states):
        while len(session.history) + 1 + len(state) > MEMORY_LIMIT:
            count = session.turn_lengths.pop(0)
            del session.history[:count]
            session.dropped_turns += 1
    source, mask = memory([session.history for session in sessions], states)
    generated = first.decoder.respond(first.reader, source, mask)
    results = []
    for session, state, meaning, output in zip(sessions, states, selected, generated):
        reply = visible(output)
        turn = [USER, *state, SEPARATOR, ASSISTANT, *reply, SEPARATOR]
        session.history.extend(turn)
        session.turn_lengths.append(len(turn))
        while len(session.history) > MEMORY_LIMIT:
            count = session.turn_lengths.pop(0)
            del session.history[:count]
            session.dropped_turns += 1
        results.append(dict(state=state, reply=reply, state_terminated=END in meaning.tolist(),
                            reply_terminated=END in output.tolist()))
    return results
