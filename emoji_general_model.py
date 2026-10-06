"""Meaning-space encoder for standalone English questions and hard emoji states."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from emoji_grounded_model import discrete_vectors


def compact(features, mask, length=16):
    values = features.float() * mask.unsqueeze(-1)
    values = F.adaptive_avg_pool1d(values.transpose(1, 2), length).transpose(1, 2)
    weights = F.adaptive_avg_pool1d(mask.float().unsqueeze(1), length).transpose(1, 2)
    return (values / weights.clamp_min(1e-6)).half()


def anchored_states(states, anchors):
    """Retain ordered literal evidence before the learned discrete states."""
    rows = []
    for learned, literal in zip(states.tolist(), anchors.tolist()):
        ordered = list(dict.fromkeys(index for index in literal if index >= 0))
        rows.append((ordered + learned)[:states.shape[1]])
    return torch.tensor(rows, dtype=torch.long, device=states.device)


class GeneralEncoder(nn.Module):
    def __init__(self, vectors, slots=8, width=128):
        super().__init__()
        self.config = dict(slots=slots, width=width)
        self.register_buffer('vectors', vectors.float())
        dimension = vectors.shape[1]
        self.input = nn.Sequential(nn.LayerNorm(dimension * 2), nn.Linear(dimension * 2, width))
        self.queries = nn.Parameter(torch.randn(slots, width) / math.sqrt(width))
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.output = nn.Linear(width, dimension)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, mask):
        source = self.input(features.float())
        queries = self.queries.unsqueeze(0).expand(len(features), -1, -1)
        hidden, _ = self.attention(queries, source, source, key_padding_mask=~mask, need_weights=False)
        dimension = self.vectors.shape[1]
        values = features[..., :dimension].float() * mask.unsqueeze(-1)
        pooled = F.adaptive_avg_pool1d(values.transpose(1, 2), self.config['slots']).transpose(1, 2)
        weights = F.adaptive_avg_pool1d(mask.float().unsqueeze(1), self.config['slots']).transpose(1, 2)
        pooled = pooled / weights.clamp_min(1e-6)
        global_mean = values.sum(1, keepdim=True) / mask.sum(1, keepdim=True).unsqueeze(-1)
        prior = F.normalize(.7 * pooled + .3 * global_mean, dim=-1)
        adjustment = self.output(hidden + queries).tanh() / math.sqrt(dimension)
        logits = F.normalize(prior + .5 * adjustment, dim=-1) @ F.normalize(self.vectors, dim=-1).T * 30
        selected, ids = discrete_vectors(logits, self.vectors)
        return logits, selected, ids
