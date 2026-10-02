"""Self-supervised text reconstruction through discrete emoji bottlenecks."""

import argparse
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from vector_quantize_pytorch import FSQ

from model import EMOJIS


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/reconstruction"
OUTPUT = ROOT / "runs/reconstruction-positional"
EOS, BOS, PAD = 256, 257, 258


def passages(paragraph, max_bytes):
    text = " ".join(paragraph.split())
    while text:
        size = cut = 0
        for character in text:
            size += len(character.encode("utf-8"))
            if size > max_bytes:
                break
            cut += 1
        if cut < len(text):
            boundary = text.rfind(" ", 0, cut + 1)
            if boundary > 0:
                cut = boundary
        passage, text = text[:cut].strip(), text[cut:].strip()
        if len(passage.encode("utf-8")) >= 12:
            yield passage


def prepare(corpus, max_bytes=48):
    raw = Path(corpus).read_bytes()
    text = raw.decode("utf-8").split("*** END OF THE PROJECT GUTENBERG EBOOK", 1)[0]
    chapters = re.split(r"(?m)^CHAPTER [IVX]+\.\s*\n", text)[1:]
    if len(chapters) != 12:
        raise ValueError("Expected the twelve chapter headings in Gutenberg ebook 11")
    splits = {"train": [], "validation": [], "test": []}
    seen = set()
    for number, chapter in enumerate(chapters, 1):
        split = "train" if number <= 9 else "validation" if number == 10 else "test"
        for paragraph in re.split(r"\n\s*\n", chapter):
            for passage in passages(paragraph, max_bytes):
                if passage not in seen:
                    splits[split].append(passage)
                    seen.add(passage)
    DATA.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, limit in (("train", 2048), ("validation", 256), ("test", 256)):
        sampled = random.Random(7).sample(splits[split], min(limit, len(splits[split])))
        counts[split] = len(sampled)
        (DATA / f"{split}.jsonl").write_text(
            "\n".join(json.dumps({"text": passage}, ensure_ascii=False) for passage in sampled)
            + "\n", encoding="utf-8"
        )
    metadata = dict(source="https://www.gutenberg.org/ebooks/11",
                    source_sha256=hashlib.sha256(raw).hexdigest(), max_bytes=max_bytes,
                    chapters=dict(train="1-9", validation="10", test="11-12"),
                    counts=counts, seed=7)
    (DATA / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def read_split(name, max_bytes):
    texts = [json.loads(line)["text"] for line in
             (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    if not texts or any(not text or len(text.encode("utf-8")) > max_bytes for text in texts):
        raise ValueError("Passages must be nonempty and fit the configured byte limit")
    return texts


def batch_text(texts, device):
    encoded = [list(text.encode("utf-8")) for text in texts]
    length = max(map(len, encoded))
    source = torch.full((len(texts), length), PAD, dtype=torch.long, device=device)
    prefix = torch.full((len(texts), length + 1), PAD, dtype=torch.long, device=device)
    targets = torch.full_like(prefix, -100)
    prefix[:, 0] = BOS
    for index, tokens in enumerate(encoded):
        row = torch.tensor(tokens, device=device)
        source[index, :len(tokens)] = row
        prefix[index, 1:len(tokens) + 1] = row
        targets[index, :len(tokens)] = row
        targets[index, len(tokens)] = EOS
    return source, prefix, targets


class TextAutoencoder(nn.Module):
    def __init__(self, slots, discrete=True, width=64, max_bytes=48, quantization="categorical"):
        super().__init__()
        if quantization not in ("categorical", "scalar"):
            raise ValueError("Unsupported quantization method")
        if quantization == "scalar" and not discrete:
            raise ValueError("Scalar quantization is a discrete experiment")
        self.config = dict(slots=slots, discrete=discrete, width=width,
                           max_bytes=max_bytes, quantization=quantization)
        self.quantization = quantization
        self.discrete = discrete
        self.max_bytes = max_bytes
        self.text_embedding = nn.Embedding(PAD + 1, width, padding_idx=PAD)
        self.text_position = nn.Parameter(torch.randn(max_bytes + 1, width))
        encoder_layer = nn.TransformerEncoderLayer(
            width, 4, width * 4, dropout=0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, 1, enable_nested_tensor=False)
        self.pool = nn.AdaptiveAvgPool1d(slots)
        if quantization == "scalar":
            self.code_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2))
            self.scalar_quantizer = FSQ(levels=[8, 8])
            self.codebook = nn.Sequential(
                nn.Linear(2, width), nn.GELU(), nn.Linear(width, width), nn.LayerNorm(width)
            )
        else:
            self.code_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, len(EMOJIS)))
            self.codebook = nn.Embedding(len(EMOJIS), width)
        self.memory_position = nn.Parameter(torch.randn(slots, width))
        decoder_layer = nn.TransformerDecoderLayer(
            width, 4, width * 4, dropout=0, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, 1)
        self.output_head = nn.Linear(width, EOS + 1)

    def encode(self, source, temperature=1.0):
        if source.shape[1] > self.max_bytes:
            raise ValueError("Source exceeds the configured byte limit")
        source = F.pad(source, (0, self.max_bytes - source.shape[1]), value=PAD)
        padding = source == PAD
        hidden = self.encoder(self.text_embedding(source) + self.text_position[:source.shape[1]],
                              src_key_padding_mask=padding)
        hidden = hidden.masked_fill(padding.unsqueeze(-1), 0)
        pooled = self.pool(hidden.transpose(1, 2)).transpose(1, 2)
        logits = self.code_head(pooled)
        if self.quantization == "scalar":
            choices, _ = self.scalar_quantizer(logits)
            return choices
        soft = F.softmax(logits / temperature, dim=-1)
        if self.discrete:
            hard = F.one_hot(logits.argmax(-1), len(EMOJIS)).to(logits.dtype)
            choices = hard - soft.detach() + soft if self.training else hard
        else:
            choices = soft
        return choices

    def state_indices(self, choices):
        if self.quantization == "scalar":
            return self.scalar_quantizer.codes_to_indices(choices).long()
        return choices.argmax(-1)

    def decode(self, choices, prefix):
        # Discrete models pass only one-hot choices or fixed grid points here,
        # never source features, source lengths, or soft mixtures.
        memory = self.codebook(choices) if self.quantization == "scalar" else choices @ self.codebook.weight
        memory = memory + self.memory_position
        hidden = self.text_embedding(prefix) + self.text_position[:prefix.shape[1]]
        mask = torch.triu(torch.ones(prefix.shape[1], prefix.shape[1], dtype=torch.bool,
                                    device=prefix.device), diagonal=1)
        decoded = self.decoder(hidden, memory, tgt_mask=mask,
                               tgt_key_padding_mask=prefix == PAD)
        return self.output_head(decoded)

    def forward(self, source, prefix, temperature=1.0):
        return self.decode(self.encode(source, temperature), prefix)

    @torch.no_grad()
    def generate(self, choices):
        if self.training:
            raise RuntimeError("Call eval() before generation")
        prefix = torch.full((choices.shape[0], 1), BOS, dtype=torch.long, device=choices.device)
        finished = torch.zeros(choices.shape[0], dtype=torch.bool, device=choices.device)
        outputs = [[] for _ in range(choices.shape[0])]
        for _ in range(self.max_bytes + 1):
            tokens = self.decode(choices, prefix)[:, -1].argmax(-1)
            for index, token in enumerate(tokens.tolist()):
                if not finished[index] and token != EOS:
                    outputs[index].append(token)
            finished |= tokens == EOS
            if finished.all():
                break
            tokens = torch.where(finished, EOS, tokens)
            prefix = torch.cat((prefix, tokens[:, None]), dim=1)
        return [bytes(tokens) for tokens in outputs]


@torch.no_grad()
def evaluate(model, texts, device, sample_count=16):
    model.eval()
    loss_sum = swapped_sum = token_count = correct = byte_count = 0
    used = []
    for start in range(0, len(texts), 32):
        source, prefix, target = batch_text(texts[start:start + 32], device)
        choices = model.encode(source)
        logits = model.decode(choices, prefix)
        swapped = model.decode(choices.roll(1, dims=0), prefix)
        loss_sum += F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="sum").item()
        swapped_sum += F.cross_entropy(swapped.flatten(0, 1), target.flatten(), reduction="sum").item()
        token_count += (target != -100).sum().item()
        byte_mask = (target >= 0) & (target < EOS)
        correct += ((logits.argmax(-1) == target) & byte_mask).sum().item()
        byte_count += byte_mask.sum().item()
        used.append(model.state_indices(choices).cpu())
    sample = texts[:sample_count]
    source, _, _ = batch_text(sample, device)
    choices = model.encode(source)
    generated = model.generate(choices)
    expected = [text.encode("utf-8") for text in sample]
    report = dict(loss=loss_sum / token_count, bits_per_token=loss_sum / token_count / math.log(2),
                  shuffled_state_loss=swapped_sum / token_count,
                  teacher_forced_byte_accuracy=correct / byte_count,
                  generation_samples=len(sample),
                  generated_exact_match=sum(a == b for a, b in zip(generated, expected)) / len(sample),
                  generated_byte_accuracy=sum(sum(x == y for x, y in zip(a, b))
                                              for a, b in zip(generated, expected))
                                          / sum(max(len(a), len(b)) for a, b in zip(generated, expected)),
                  examples=[dict(original=text, reconstruction=raw.decode("utf-8", errors="replace"),
                                 emoji_state="".join(EMOJIS[token] for token in codes.tolist())
                                 if model.discrete else None)
                            for text, raw, codes in zip(sample, generated, model.state_indices(choices))])
    if model.discrete:
        codes = torch.cat(used)
        report["unique_state_sequences"] = len(torch.unique(codes, dim=0))
        report["unique_emojis"] = len(torch.unique(codes))
        report["mean_distinct_emojis_per_passage"] = sum(len(torch.unique(row)) for row in codes) / len(codes)
        report["all_slots_identical_fraction"] = (codes == codes[:, :1]).all(1).float().mean().item()
    return report


def compare(args):
    splits = {name: read_split(name, 48) for name in ("train", "validation", "test")}
    if any(set(splits[a]) & set(splits[b]) for a, b in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Corpus splits must be disjoint")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for slots in args.slots:
        for discrete in ((True,) if args.quantization == "scalar" else (True, False)):
            torch.manual_seed(7)
            rng = random.Random(7)
            model = TextAutoencoder(slots, discrete, quantization=args.quantization).to(args.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.steps, eta_min=0.00005
            )
            kind = "fsq" if args.quantization == "scalar" else ("emoji" if discrete else "continuous")
            name = f"{kind}-{slots}"
            print(f"Training {name}", flush=True)
            for step in range(args.steps):
                model.train()
                batch = rng.choices(splits["train"], k=args.batch_size)
                source, prefix, target = batch_text(batch, args.device)
                logits = model(source, prefix)
                loss = F.cross_entropy(logits.flatten(0, 1), target.flatten())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                if (step + 1) % 100 == 0:
                    print(f"  {step + 1}/{args.steps}: loss={loss.item():.3f}", flush=True)
            result = dict(name=name, slots=slots, discrete=discrete,
                          parameter_count=sum(p.numel() for p in model.parameters()),
                          max_discrete_bits=slots * math.log2(64 if args.quantization == "scalar" else len(EMOJIS))
                                            if discrete else None,
                          validation=evaluate(model, splits["validation"], args.device))
            if not args.validation_only:
                result["test"] = evaluate(model, splits["test"], args.device)
            torch.save(dict(config=model.config, weights=model.state_dict()), output / f"{name}.pt")
            results.append(result)
            report = dict(architecture="fixed-position pooling, matched token/position initialization scales",
                          steps=args.steps, batch_size=args.batch_size, seed=7,
                          quantization=args.quantization, validation_only=args.validation_only,
                          corpus=json.loads((DATA / "metadata.json").read_text(encoding="utf-8")),
                          results=results)
            (output / "comparison.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            split = "validation" if args.validation_only else "test"
            metric = result[split]
            print(f"  {split} loss={metric['loss']:.3f}; "
                  f"shuffled={metric['shuffled_state_loss']:.3f}; "
                  f"free exact={metric['generated_exact_match']:.1%}", flush=True)


def reconstruct(args):
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model = TextAutoencoder(**saved["config"]).to(args.device).eval()
    model.load_state_dict(saved["weights"])
    if not args.text or len(args.text.encode("utf-8")) > model.max_bytes:
        raise ValueError(f"Text must contain 1-{model.max_bytes} UTF-8 bytes")
    source, _, _ = batch_text([args.text], args.device)
    with torch.no_grad():
        state = model.encode(source)
        output = model.generate(state)[0].decode("utf-8", errors="replace")
    print(json.dumps(dict(original=args.text, reconstruction=output,
                         emoji_state="".join(EMOJIS[i] for i in model.state_indices(state)[0].tolist())
                         if model.discrete else None), ensure_ascii=False))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    commands = parser.add_subparsers(dest="command", required=True)
    preparing = commands.add_parser("prepare")
    preparing.add_argument("--corpus", default=str(ROOT / "data/alice.txt"))
    training = commands.add_parser("compare")
    training.add_argument("--steps", type=int, default=2000)
    training.add_argument("--slots", nargs="+", type=int, choices=(4, 16, 32, 48), default=(4, 16, 32, 48))
    training.add_argument("--batch-size", type=int, default=24)
    training.add_argument("--output", default=str(OUTPUT))
    training.add_argument("--quantization", choices=("categorical", "scalar"), default="categorical")
    training.add_argument("--validation-only", action="store_true")
    inference = commands.add_parser("reconstruct")
    inference.add_argument("text")
    inference.add_argument("--checkpoint", default=str(OUTPUT / "emoji-48.pt"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.corpus)
    elif args.command == "compare":
        if args.steps < 1 or args.batch_size < 1:
            parser.error("steps and batch-size must be positive")
        compare(args)
    else:
        reconstruct(args)


if __name__ == "__main__":
    main()
