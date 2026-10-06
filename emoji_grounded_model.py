"""Meaning-anchored emoji bottlenecks and direct autoregressive emoji replies."""

import math
import re

import torch
from torch import nn
from torch.nn import functional as F


def discrete_vectors(logits, vectors):
    """Hard symbol vectors in forward; straight-through gradients in training."""
    soft = logits.softmax(-1)
    ids = logits.argmax(-1)
    hard = F.one_hot(ids, len(vectors)).to(soft.dtype)
    selected = hard + (soft - soft.detach())
    return selected @ vectors, ids


def semantic_keys(vectors):
    values = F.layer_norm(vectors, (vectors.shape[-1],))
    return F.normalize(values - values.mean(0), dim=-1)


class MeaningLexicon:
    def __init__(self, rows):
        self.rows = rows
        terms = {}
        for index, row in enumerate(rows):
            for term in [row['core_meaning'], *row['associations']]:
                key = term.casefold().strip()
                terms.setdefault(key, set()).add(index)
        # Broad words shared by many symbols are weak evidence, not precise labels.
        self.terms = {term: sorted(ids) for term, ids in terms.items() if len(term) >= 3 and len(ids) <= 32}
        self.pattern = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(term) for term in sorted(self.terms, key=len, reverse=True)) + r')(?!\w)', re.IGNORECASE)

    def anchors(self, text, slots, literal_only=False):
        found = []
        for match in self.pattern.finditer(text):
            # Variants with the same associations share evidence; prefer a literal core match.
            ids = self.terms[match.group().casefold()]
            literal = [index for index in ids if self.rows[index]['core_meaning'].casefold() == match.group().casefold()]
            if literal_only and not literal:
                continue
            candidates = literal or ids
            cores = {self.rows[index]['core_meaning'] for index in candidates}
            if len(cores) != 1:
                continue
            index = min(candidates)
            if index not in found:
                found.append(index)
        return found[:slots]


class GroundedEncoder(nn.Module):
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
        lexical = features[..., self.vectors.shape[-1]:].float() * mask.unsqueeze(-1)
        pooled = F.adaptive_avg_pool1d(lexical.transpose(1, 2), self.config['slots']).transpose(1, 2)
        weights = F.adaptive_avg_pool1d(mask.float().unsqueeze(1), self.config['slots']).transpose(1, 2)
        pooled = pooled / weights.clamp_min(1e-6)
        center = F.layer_norm(self.vectors, (self.vectors.shape[-1],)).mean(0)
        lexical_query = pooled - center
        residual = self.output(hidden + queries).tanh() / math.sqrt(self.vectors.shape[-1])
        query = lexical_query + .1 * lexical_query.norm(dim=-1, keepdim=True) * residual
        logits = F.normalize(query, dim=-1) @ semantic_keys(self.vectors).T * 20
        selected, ids = discrete_vectors(logits, self.vectors)
        return logits, selected, ids


class GroundedReply(nn.Module):
    def __init__(self, vectors, slots=8, width=128):
        super().__init__()
        self.config = dict(slots=slots, width=width)
        self.register_buffer('vectors', vectors.float())
        self.end = len(vectors)
        dimension = vectors.shape[1]
        self.memory = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, width))
        self.embedding = nn.Embedding.from_pretrained(torch.cat((vectors.float(), torch.zeros(1, dimension))), freeze=True)
        self.input = nn.Linear(dimension, width)
        self.positions = nn.Embedding(slots + 2, width)
        layer = nn.TransformerDecoderLayer(width, 4, width * 4, dropout=.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, 2)
        self.output = nn.Linear(width, dimension)
        self.end_head = nn.Linear(width, 1)

    def forward(self, state_vectors, prefix):
        length = prefix.shape[1]
        target = self.input(self.embedding(prefix)) + self.positions(torch.arange(length, device=prefix.device))
        mask = torch.ones(length, length, dtype=torch.bool, device=prefix.device).triu(1)
        hidden = self.decoder(target, self.memory(state_vectors), tgt_mask=mask)
        logits = F.normalize(self.output(hidden), dim=-1) @ semantic_keys(self.vectors).T * 20
        return torch.cat((logits, self.end_head(hidden)), dim=-1)

    @torch.no_grad()
    def generate(self, state_ids):
        if self.training or state_ids.dtype != torch.long:
            raise ValueError('Generation requires eval mode and integer emoji states')
        prefix = torch.full((len(state_ids), 1), self.end, dtype=torch.long, device=state_ids.device)
        finished = torch.zeros(len(state_ids), dtype=torch.bool, device=state_ids.device)
        for _ in range(self.config['slots'] + 1):
            # Rebuild each step from hard IDs; no English decoder or latent cache is called.
            logits = self(self.vectors[state_ids], prefix)[:, -1]
            selected = logits.argmax(-1)
            selected = torch.where(finished, self.end, selected)
            prefix = torch.cat((prefix, selected[:, None]), dim=1)
            finished |= selected == self.end
            if finished.all():
                break
        return prefix[:, 1:]


class TrainingTextDecoder(nn.Module):
    """Frozen pretrained English decoder, never included in the runtime checkpoint."""
    def __init__(self, backbone, end_token):
        super().__init__()
        if not backbone.config.tie_word_embeddings:
            raise ValueError('This auxiliary decoder requires a pretrained tied language head')
        self.backbone = backbone.eval().requires_grad_(False)
        self.end_token = end_token
        dimension = backbone.get_input_embeddings().weight.shape[1]
        self.adapter = nn.Linear(dimension, dimension)
        nn.init.eye_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)

    def forward(self, state_vectors, targets, prefix_mask=None):
        embedding = self.backbone.get_input_embeddings()
        labels = targets.clone()
        prefix = torch.cat((torch.full_like(targets[:, :1], self.end_token), targets[:, :-1]), dim=1)
        prefix = prefix.masked_fill(prefix < 0, self.end_token)
        if prefix_mask is not None:
            prefix = prefix.masked_fill(prefix_mask, self.end_token)
        values = torch.cat((self.adapter(state_vectors).to(embedding.weight.dtype), embedding(prefix)), dim=1)
        mask = torch.cat((torch.ones(state_vectors.shape[:2], dtype=torch.bool), labels >= 0), dim=1)
        hidden = self.backbone(inputs_embeds=values, attention_mask=mask, use_cache=False).last_hidden_state[:, state_vectors.shape[1]:]
        logits = F.linear(hidden, embedding.weight).float()
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


def ordered_content_loss(selected, features, mask, projection):
    """Training-only ordered contextual reconstruction, including an order contrast."""
    dimension = selected.shape[-1]
    content = features[..., dimension:].float() * mask.unsqueeze(-1)
    target = F.adaptive_avg_pool1d(content.transpose(1, 2), selected.shape[1]).transpose(1, 2)
    weights = F.adaptive_avg_pool1d(mask.float().unsqueeze(1), selected.shape[1]).transpose(1, 2)
    target = F.normalize(target / weights.clamp_min(1e-6), dim=-1)
    prediction = F.normalize(projection(selected), dim=-1)
    correct = (prediction - target).square().sum(-1).mean()
    reversed_loss = (prediction.flip(1) - target).square().sum(-1).mean()
    return correct + F.relu(.1 + correct - reversed_loss)
