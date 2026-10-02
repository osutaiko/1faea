"""Learn source-token pointers instead of memorizing entity output classes."""

import re

import torch
from torch import nn
from torch.nn import functional as F

from semantic_model import WORDS, TRUE, initial_state, expanded_state


ENTITY_PATTERN = re.compile(r'\b(cat|dog|bird|fish)\b', re.IGNORECASE)


def mentions(text, offsets):
    matches = list(ENTITY_PATTERN.finditer(text))
    if len(matches) != 2:
        raise ValueError('Each clause requires two explicit entity mentions')
    positions = []
    for match in matches:
        tokens = [index for index, (start, end) in enumerate(offsets)
                  if start < match.end() and end > match.start()]
        positions.append(tokens[-1])
    return positions, [WORDS.index(match.group().lower()) for match in matches]


def split_clauses(text):
    clauses = re.findall(r'[^.?]+[.?]', text)
    if len(clauses) != 3 or len(''.join(clauses)) != len(text.rstrip()):
        raise ValueError('This prototype requires two separate fact sentences and one query')
    return [clause.strip() for clause in clauses]


@torch.no_grad()
def text_inputs(reader, texts):
    inputs = reader.tokenizer(texts, padding=True, return_offsets_mapping=True, return_tensors='pt')
    offsets = inputs.pop('offset_mapping').tolist()
    inputs = inputs.to(reader.device)
    if inputs.input_ids.shape[1] > reader.backbone.config.max_position_embeddings:
        raise ValueError('Text exceeds the pretrained context limit')
    positions, entities = zip(*(mentions(text, offset) for text, offset in zip(texts, offsets)))
    features = reader.backbone(**inputs, use_cache=False).last_hidden_state.float()
    lexical = reader.backbone.get_input_embeddings()(inputs.input_ids).float()
    lexical = F.layer_norm(lexical, (lexical.shape[-1],))
    features = torch.cat((features, lexical), dim=-1)
    return (features, inputs.attention_mask.bool(), torch.tensor(positions, device=reader.device),
            torch.tensor(entities, device=reader.device))


class PointerHead(nn.Module):
    def __init__(self, input_dim, width, outputs, candidates, layers):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))
        self.queries = nn.Parameter(torch.randn(outputs, width))
        self.positions = nn.Parameter(torch.randn(candidates, width))
        self.relative_positions = nn.Embedding(candidates * 65, width)
        layer = nn.TransformerDecoderLayer(width, 4, width * 4, dropout=0, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.keys = nn.Linear(width, width, bias=False)

    def forward(self, features, mask, positions):
        memory = self.projection(features)
        distances = torch.arange(features.shape[1], device=features.device)[None, :, None] - positions[:, None, :]
        roles = torch.arange(positions.shape[1], device=features.device)[None, None, :] * 65
        memory = memory + self.relative_positions(distances.clamp(-32, 32) + 32 + roles).sum(2)
        candidates = memory.gather(1, positions[:, :, None].expand(-1, -1, memory.shape[-1]))
        queries = self.queries.expand(features.shape[0], -1, -1)
        hidden = self.decoder(queries, memory, memory_key_padding_mask=~mask)
        keys = self.keys(candidates + self.positions)
        return hidden @ keys.transpose(1, 2) / memory.shape[-1] ** 0.5


class PointerReasoner(nn.Module):
    def __init__(self, input_dim, width=128):
        super().__init__()
        self.config = dict(input_dim=input_dim, width=width)
        self.parser = PointerHead(2 * input_dim, width, 1, 2, 1)
        self.deduction = PointerHead(input_dim, width, 2, 4, 2)
        self.answer_head = nn.Sequential(nn.LayerNorm(16 * input_dim), nn.Linear(16 * input_dim, width),
                                         nn.GELU(), nn.Linear(width, 3))

    def derive_logits(self, features):
        positions = torch.tensor([1, 3, 5, 7], device=features.device).expand(features.shape[0], -1)
        return self.deduction(features, torch.ones(features.shape[:2], dtype=torch.bool, device=features.device), positions)

    def derive_entities(self, reader, state):
        pointers = self.derive_logits(reader.state_features(state)).argmax(-1)
        return state[:, [1, 3, 5, 7]].gather(1, pointers)

    def answer_logits(self, features):
        return self.answer_head(features.flatten(1))

    @torch.no_grad()
    def parse_state(self, reader, texts):
        clauses = [clause for text in texts for clause in split_clauses(text)]
        features, mask, positions, candidates = text_inputs(reader, clauses)
        orientation = self.parser(features, mask, positions).argmax(-1)
        pairs = torch.stack((orientation, 1 - orientation), dim=-1)
        return initial_state(candidates.gather(1, pairs.flatten(1)).reshape(len(texts), 6))

    @torch.no_grad()
    def generate(self, reader, texts):
        if self.training:
            raise RuntimeError('Call eval() before generation')
        first = self.parse_state(reader, texts)
        second = expanded_state(first, self.derive_entities(reader, first))
        answer = self.answer_logits(reader.state_features(second)).argmax(-1) + TRUE
        return first, second, answer
