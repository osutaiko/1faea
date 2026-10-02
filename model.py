"""Text input -> discrete emoji states -> autoregressive emoji answer."""

import torch
from torch import nn
from torch.nn import functional as F


EMOJIS = tuple(
    "🌍 🔥 ❄️ ☀️ 🌙 🐱 🐶 🐟 🐦 🌱 💧 🍎 🍌 🍊 ☕ 🍕 🚗 🚲 ✈️ 🚢 "
    "🏠 🏫 📚 🎓 🧠 💡 🔍 🧩 🔗 📦 🔑 ⚖️ ⏳ ➕ ➖ ✖️ ➗ 🟰 ✅ ❌ "
    "⬆️ ⬇️ ⬅️ ➡️ 🔴 🔵 🟢 🟡 ⚫ ⚪ 😀 😢 😠 😨 😴 ❤️ 🤝 🎉 🛑 "
    "0️⃣ 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣".split()
)
EOS = len(EMOJIS)
BOS = EOS + 1


def encode_emojis(text):
    """Each complete allowlisted emoji is one token, including composed emojis."""
    text = "".join(text.split())
    tokens = []
    candidates = sorted(enumerate(EMOJIS), key=lambda item: -len(item[1]))
    while text:
        for token, emoji in candidates:
            if text.startswith(emoji):
                tokens.append(token)
                text = text[len(emoji):]
                break
        else:
            raise ValueError(f"Unsupported emoji or non-emoji text: {text!r}")
    if not tokens:
        raise ValueError("An answer must contain at least one emoji")
    return tokens


def decode_emojis(tokens):
    result = []
    for token in tokens:
        if token == EOS:
            break
        result.append(EMOJIS[token])
    return "".join(result)


class EmojiLanguageModel(nn.Module):
    def __init__(self, input_dim, width=64, slots=4, rounds=2, max_answer=16):
        super().__init__()
        self.config = dict(input_dim=input_dim, width=width, slots=slots,
                           rounds=rounds, max_answer=max_answer)
        self.slots = slots
        self.rounds = rounds
        self.max_answer = max_answer
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_scale", torch.ones(input_dim))
        self.emoji_embedding = nn.Embedding(BOS + 1, width)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, slots * len(EMOJIS))
        )
        self.slot_position = nn.Parameter(torch.randn(slots, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            width, 4, width * 4, dropout=0, batch_first=True
        )
        self.transition = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.state_head = nn.Linear(width, len(EMOJIS))
        with torch.no_grad():
            self.state_head.weight.copy_(self.emoji_embedding.weight[:EOS] / width ** 0.5)
            self.state_head.bias.zero_()
        answer_layer = nn.TransformerEncoderLayer(
            width, 4, width * 4, dropout=0, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(answer_layer, 1, enable_nested_tensor=False)
        self.answer_position = nn.Parameter(
            torch.randn(slots + max_answer, width) * 0.02
        )
        self.answer_head = nn.Linear(width, EOS + 1)

    def discretize(self, logits, temperature):
        hard = F.one_hot(logits.argmax(-1), len(EMOJIS)).to(logits.dtype)
        if self.training:
            # Exactly one-hot in the forward pass; soft surrogate in backward.
            soft = F.softmax(logits / temperature, dim=-1)
            return hard - soft.detach() + soft
        return hard

    def reason(self, features, temperature=1.0):
        features = (features - self.feature_mean) / self.feature_scale
        logits = self.input_projection(features).view(-1, self.slots, len(EMOJIS))
        state = self.discretize(logits, temperature)
        trace = [state.argmax(-1)]
        for _ in range(self.rounds):
            # Reconstruct embeddings from discrete choices. No feature/prompt
            # access, hidden-state recurrence, or attention cache across rounds.
            embeddings = state @ self.emoji_embedding.weight[:EOS]
            hidden = self.transition(embeddings + self.slot_position)
            state = self.discretize(self.state_head(hidden), temperature)
            trace.append(state.argmax(-1))
        return state, torch.stack(trace, dim=1)

    def answer_logits(self, state, prefix):
        memory = state @ self.emoji_embedding.weight[:EOS]
        inputs = torch.cat((memory, self.emoji_embedding(prefix)), dim=1)
        if inputs.shape[1] > self.answer_position.shape[0]:
            raise ValueError("Answer exceeds the configured token limit")
        inputs = inputs + self.answer_position[:inputs.shape[1]]
        causal_mask = torch.triu(
            torch.ones(inputs.shape[1], inputs.shape[1], dtype=torch.bool,
                       device=inputs.device), diagonal=1
        )
        hidden = self.decoder(inputs, mask=causal_mask)
        return self.answer_head(hidden[:, self.slots:])

    def forward(self, features, prefix, temperature=1.0):
        state, trace = self.reason(features, temperature)
        return self.answer_logits(state, prefix), trace

    @torch.no_grad()
    def generate(self, features):
        if self.training:
            raise RuntimeError("Call eval() before generation")
        state, trace = self.reason(features)
        prefix = torch.full((features.shape[0], 1), BOS,
                            dtype=torch.long, device=features.device)
        finished = torch.zeros(features.shape[0], dtype=torch.bool,
                               device=features.device)
        emitted = []
        for step in range(self.max_answer):
            logits = self.answer_logits(state, prefix)[:, -1]
            if step == 0:
                logits[:, EOS] = -torch.inf
            token = logits.argmax(-1)
            token = torch.where(finished, EOS, token)
            emitted.append(token)
            finished |= token == EOS
            if finished.all():
                break
            # Only emitted token IDs persist between decoder calls.
            prefix = torch.cat((prefix, token[:, None]), dim=1)
        return torch.stack(emitted, dim=1), trace
