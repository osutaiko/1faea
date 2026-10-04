"""Adapt a pretrained transformer to predict only emoji IDs from emoji IDs."""

from pathlib import Path

from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from run import ROOT


def backbone(source):
    config = hf_hub_download(source['model'], 'config.json', revision=source['model_revision'],
                             cache_dir=ROOT / '.hf-cache', local_files_only=True)
    model = AutoModel.from_pretrained(Path(config).parent, local_files_only=True, dtype=torch.bfloat16)
    return get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj', 'v_proj'], lora_dropout=.05))


class PretrainedReply(nn.Module):
    def __init__(self, model, vectors, end_vector, slots=8):
        super().__init__()
        self.backbone = model
        self.slots = slots
        self.end = len(vectors)
        self.register_buffer('vectors', torch.cat((vectors.float(), end_vector.float().unsqueeze(0))))
        dimension = vectors.shape[1]
        self.input = nn.Linear(dimension, dimension)
        nn.init.eye_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        self.norm = nn.LayerNorm(dimension)
        self.output = nn.Linear(dimension, self.end + 1)
        with torch.no_grad():
            self.output.weight.copy_(F.normalize(self.vectors, dim=-1))
            self.output.bias.zero_()

    def forward(self, states, prefix):
        if states.dtype != torch.long or prefix.dtype != torch.long:
            raise ValueError('Pretrained reply accepts only integer emoji states and prefixes')
        ids = torch.cat((states, prefix), dim=1)
        values = self.input(self.vectors[ids]).to(self.backbone.get_input_embeddings().weight.dtype)
        hidden = self.backbone(inputs_embeds=values, use_cache=False).last_hidden_state[:, states.shape[1]:].float()
        return self.output(self.norm(hidden))

    @torch.no_grad()
    def generate(self, states):
        if self.training or states.dtype != torch.long:
            raise ValueError('Generation requires eval mode and integer emoji states')
        prefix = torch.full((len(states), 1), self.end, dtype=torch.long)
        finished = torch.zeros(len(states), dtype=torch.bool)
        for _ in range(self.slots + 1):
            # Recompute from hard IDs each time; no English tokens or hidden-state cache.
            selected = self(states, prefix)[:, -1].argmax(-1)
            selected = selected.masked_fill(finished, self.end)
            prefix = torch.cat((prefix, selected[:, None]), 1)
            finished |= selected == self.end
            if finished.all():
                break
        return prefix[:, 1:]
