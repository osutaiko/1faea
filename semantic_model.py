"""Direct semantic emoji selections with a fresh pretrained pass per stage."""

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from run import BASE_MODEL, ROOT


SYMBOLS = ("🐱", "🐶", "🐦", "🐟", "📌", "⬅️", "🔍", "⚪", "✅", "❌", "❓")
WORDS = ("cat", "dog", "bird", "fish", "fact", "left", "question", "empty", "true", "false", "unknown")
ENTITY_SLOTS = (1, 3, 5, 7, 13, 15)
FACT, LEFT, QUERY, EMPTY = 4, 5, 6, 7
TRUE, FALSE, UNKNOWN = 8, 9, 10


def initial_state(entities):
    state = torch.tensor([FACT, 0, LEFT, 0, FACT, 0, LEFT, 0,
                          EMPTY, EMPTY, EMPTY, EMPTY, QUERY, 0, LEFT, 0],
                         device=entities.device).expand(entities.shape[0], -1).clone()
    state[:, ENTITY_SLOTS] = entities
    return state


def expanded_state(state, derived_entities):
    result = state.clone()
    result[:, 8] = FACT
    result[:, 9] = derived_entities[:, 0]
    result[:, 10] = LEFT
    result[:, 11] = derived_entities[:, 1]
    return result


def render(state):
    tokens = state.tolist()
    return " ".join("".join(SYMBOLS[token] for token in tokens[start:start + 4])
                    for start in range(0, len(tokens), 4))


class PretrainedReader:
    def __init__(self, device="cpu", model_name=BASE_MODEL):
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=ROOT / ".hf-cache", local_files_only=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.backbone = AutoModel.from_pretrained(
            model_name, cache_dir=ROOT / ".hf-cache", local_files_only=True
        ).to(device).eval()
        self.backbone.requires_grad_(False)
        # Static initialization from existing word embeddings grounds the new
        # atomic symbols. Runtime emoji passes never tokenize or generate words.
        with torch.no_grad():
            embedding = self.backbone.get_input_embeddings()
            vectors = [embedding(torch.tensor(self.tokenizer.encode(word, add_special_tokens=False),
                                              device=device)).mean(0) for word in WORDS]
            self.symbol_embedding = nn.Embedding.from_pretrained(torch.stack(vectors), freeze=True)

    @torch.no_grad()
    def text_features(self, texts):
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        if inputs.input_ids.shape[1] > self.backbone.config.max_position_embeddings:
            raise ValueError("Text exceeds the pretrained context limit")
        features = self.backbone(**inputs, use_cache=False).last_hidden_state.float()
        return features, inputs.attention_mask.bool()

    @torch.no_grad()
    def state_features(self, state):
        # Input consists exclusively of atomic selected symbols. No input-text
        # features, past key/value cache, or earlier hidden state is accepted.
        embeddings = self.symbol_embedding(state)
        return self.backbone(inputs_embeds=embeddings, use_cache=False).last_hidden_state.float()


class SemanticReasoner(nn.Module):
    def __init__(self, input_dim, width=128):
        super().__init__()
        self.config = dict(input_dim=input_dim, width=width)
        self.input_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))
        self.entity_queries = nn.Parameter(torch.randn(6, width))
        layer = nn.TransformerDecoderLayer(width, 4, width * 4, dropout=0, batch_first=True)
        self.parser = nn.TransformerDecoder(layer, 1)
        self.entity_head = nn.Linear(width, 4)
        self.state_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))
        self.state_positions = nn.Parameter(torch.randn(16, width))
        self.derive_queries = nn.Parameter(torch.randn(2, width))
        self.derive_decoder = nn.TransformerDecoder(layer, 2)
        self.derive_head = nn.Linear(width, 4)
        self.answer_head = nn.Sequential(nn.LayerNorm(16 * input_dim),
                                         nn.Linear(16 * input_dim, width), nn.GELU(),
                                         nn.Linear(width, 3))

    def parse_logits(self, features, attention_mask):
        memory = self.input_projection(features)
        queries = self.entity_queries.expand(features.shape[0], -1, -1)
        hidden = self.parser(queries, memory, memory_key_padding_mask=~attention_mask)
        return self.entity_head(hidden)

    def derive_logits(self, state_features):
        memory = self.state_projection(state_features) + self.state_positions
        queries = self.derive_queries.expand(state_features.shape[0], -1, -1)
        return self.derive_head(self.derive_decoder(queries, memory))

    def answer_logits(self, state_features):
        return self.answer_head(state_features.flatten(1))

    def derive_entities(self, reader, state):
        return self.derive_logits(reader.state_features(state)).argmax(-1)

    @torch.no_grad()
    def generate(self, reader, texts):
        if self.training:
            raise RuntimeError("Call eval() before generation")
        features, mask = reader.text_features(texts)
        entities = self.parse_logits(features, mask).argmax(-1)
        first = initial_state(entities)
        # Only integer symbol IDs cross either of these boundaries.
        derived = self.derive_entities(reader, first)
        second = expanded_state(first, derived)
        answer = self.answer_logits(reader.state_features(second)).argmax(-1) + TRUE
        return first, second, answer
