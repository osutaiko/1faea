"""Autoregressive next-emoji probabilities over a grounded symbolic alphabet."""

import torch
from torch import nn
from torch.nn import functional as F

from semantic_model import SYMBOLS


END, START = len(SYMBOLS), len(SYMBOLS) + 1
ALPHABET = (*SYMBOLS, '🔚', '▶️')


def symbol_vectors(reader):
    with torch.no_grad():
        embedding = reader.backbone.get_input_embeddings()
        extra = [embedding(torch.tensor(reader.tokenizer.encode(word, add_special_tokens=False))).mean(0)
                 for word in ('end', 'start')]
    return torch.cat((reader.symbol_embedding.weight, torch.stack(extra))).float()


class EmojiLM(nn.Module):
    def __init__(self, vectors, width=128):
        super().__init__()
        input_dim = vectors.shape[-1]
        self.config = dict(width=width)
        self.embedding = nn.Embedding.from_pretrained(vectors, freeze=True)
        self.prefix_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))
        self.prefix_positions = nn.Embedding(8, width)
        self.memory_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))
        self.relative_positions = nn.Embedding(4 * 65, width)
        self.copy_positions = nn.Parameter(torch.randn(4, width))
        layer = nn.TransformerDecoderLayer(width, 4, width * 4, dropout=0, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, 2)
        self.keys = nn.Linear(width, width, bias=False)
        self.generator = nn.Linear(width, len(vectors) - 1)
        self.copy_gate = nn.Linear(width, 1)

    def initialize_from_pointer(self, pointer):
        head = pointer.deduction
        self.memory_projection.load_state_dict(head.projection.state_dict())
        self.relative_positions.load_state_dict(head.relative_positions.state_dict())
        self.decoder.load_state_dict(head.decoder.state_dict())
        self.keys.load_state_dict(head.keys.state_dict())
        with torch.no_grad():
            self.copy_positions.copy_(head.positions)
            self.prefix_positions.weight[1].copy_(head.queries[0])
            self.prefix_positions.weight[3].copy_(head.queries[1])

    def memory_features(self, state, state_features):
        return self.memory_projection(state_features)

    def forward(self, state, state_features, prefix):
        memory = self.memory_features(state, state_features)
        positions = torch.tensor([1, 3, 5, 7], device=state.device)
        distances = torch.arange(16, device=state.device)[:, None] - positions[None]
        roles = torch.arange(4, device=state.device)[None] * 65
        memory = memory + self.relative_positions(distances.clamp(-32, 32) + 32 + roles).sum(1)
        target = self.prefix_projection(self.embedding(prefix))
        target = target + self.prefix_positions(torch.arange(prefix.shape[1], device=state.device))
        causal = torch.ones(prefix.shape[1], prefix.shape[1], dtype=torch.bool, device=state.device).triu(1)
        hidden = self.decoder(target, memory, tgt_mask=causal)
        keys = self.keys(memory[:, positions] + self.copy_positions)
        attention = F.softmax(hidden @ keys.transpose(1, 2) / hidden.shape[-1] ** 0.5, dim=-1)
        copied = torch.zeros(*hidden.shape[:2], self.embedding.num_embeddings, device=state.device)
        source_ids = state[:, positions][:, None].expand(-1, prefix.shape[1], -1)
        copied.scatter_add_(2, source_ids, attention)
        probabilities = F.softmax(self.generator(hidden), dim=-1)
        generated = torch.cat((probabilities[..., :START], torch.zeros_like(probabilities[..., :1]),
                               probabilities[..., START:]), dim=-1)
        gate = torch.sigmoid(self.copy_gate(hidden))
        return (gate * generated + (1 - gate) * copied).clamp_min(1e-9).log()

    @torch.no_grad()
    def generate(self, reader, state):
        if self.training:
            raise RuntimeError('Call eval() before generation')
        prefix = torch.full((len(state), 1), START, device=state.device)
        finished = torch.zeros(len(state), dtype=torch.bool, device=state.device)
        for _ in range(8):
            # Every next-symbol computation reconstructs features from selected
            # IDs. No past key/value cache or previous hidden states are retained.
            features = reader.state_features(state)
            token = self(state, features, prefix)[:, -1].argmax(-1)
            token = torch.where(finished, END, token)
            prefix = torch.cat((prefix, token[:, None]), dim=1)
            finished |= token == END
            if finished.all():
                break
        return prefix[:, 1:]


def visible_tokens(tokens):
    values = tokens.tolist()
    return values[:values.index(END)] if END in values else values


def render_tokens(tokens):
    return ''.join(ALPHABET[token] for token in visible_tokens(tokens))
