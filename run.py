"""Train or run the discrete-emoji language experiment."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from model import BOS, EOS, EmojiLanguageModel, decode_emojis, encode_emojis


ROOT = Path(__file__).resolve().parent
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"


def read_data(path, max_answer):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError("The dataset is empty")
    for row in rows:
        if not isinstance(row["prompt"], str) or not row["prompt"].strip():
            raise ValueError("Each prompt must be nonempty text")
        tokens = encode_emojis(row["answer"])
        if len(tokens) >= max_answer:
            raise ValueError("Answers must leave room for the end token")
        row["tokens"] = tokens
    return rows


@torch.no_grad()
def encode_text(prompts, model_name, device, batch_size=8, local_files_only=False):
    cache = ROOT / ".hf-cache"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache,
                                              local_files_only=local_files_only)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoder = AutoModel.from_pretrained(model_name, cache_dir=cache,
                                        local_files_only=local_files_only).to(device).eval()
    features = []
    for start in range(0, len(prompts), batch_size):
        text = [f"Question: {prompt}\nAnswer:" for prompt in prompts[start:start + batch_size]]
        inputs = tokenizer(text, padding=True, return_tensors="pt").to(device)
        if inputs.input_ids.shape[1] > encoder.config.max_position_embeddings:
            raise ValueError("Prompt exceeds the pretrained model's context limit")
        output = encoder(**inputs, use_cache=False).last_hidden_state
        last = inputs.attention_mask.sum(-1) - 1
        features.append(output[torch.arange(len(text), device=device), last].float().cpu())
    return torch.cat(features).to(device)


def batch_targets(rows, device):
    length = max(len(row["tokens"]) + 1 for row in rows)
    targets = torch.full((len(rows), length), -100, dtype=torch.long, device=device)
    prefix = torch.full((len(rows), length), EOS, dtype=torch.long, device=device)
    prefix[:, 0] = BOS
    for index, row in enumerate(rows):
        tokens = torch.tensor(row["tokens"], device=device)
        targets[index, :len(tokens)] = tokens
        targets[index, len(tokens)] = EOS
        prefix[index, 1:len(tokens) + 1] = tokens
    return prefix, targets


@torch.no_grad()
def evaluate(model, features, rows):
    model.eval()
    prefix, targets = batch_targets(rows, features.device)
    logits, _ = model(features, prefix)
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten()).item()
    tokens, trace = model.generate(features)
    answers = [decode_emojis(sequence.tolist()) for sequence in tokens]
    exact = sum(answer == decode_emojis(row["tokens"])
                for answer, row in zip(answers, rows)) / len(rows)
    # Intervention: substitute another example's final state. A loss increase
    # suggests the decoder uses its state; it does not prove useful reasoning.
    state, _ = model.reason(features)
    swapped = model.answer_logits(state.roll(1, dims=0), prefix)
    swapped_loss = F.cross_entropy(swapped.flatten(0, 1), targets.flatten()).item()
    return {
        "loss": loss, "exact_match": exact, "swapped_state_loss": swapped_loss,
        "examples": [
            dict(prompt=row["prompt"], expected=row["answer"], answer=answer,
                 states=[decode_emojis(step.tolist()) for step in steps])
            for row, answer, steps in zip(rows, answers, trace)
        ],
    }


def train(args):
    torch.manual_seed(7)
    rows = read_data(args.data, 16)
    validation = read_data(args.validation, 16)
    if {row["prompt"] for row in rows} & {row["prompt"] for row in validation}:
        raise ValueError("Training and validation prompts must be disjoint")
    features = encode_text([row["prompt"] for row in rows + validation],
                           BASE_MODEL, args.device)
    train_features = features[:len(rows)]
    validation_features = features[len(rows):]
    model = EmojiLanguageModel(features.shape[-1]).to(args.device)
    model.feature_mean.copy_(train_features.mean(0))
    model.feature_scale.copy_(train_features.std(0, unbiased=False).clamp_min(0.1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    initial = evaluate(model, validation_features, validation)
    print(f"Initial validation: loss={initial['loss']:.3f}, "
          f"exact={initial['exact_match']:.1%}", flush=True)
    for step in range(args.steps):
        model.train()
        indices = torch.randint(len(rows), (args.batch_size,), device=args.device)
        batch = [rows[index] for index in indices.tolist()]
        prefix, targets = batch_targets(batch, args.device)
        temperature = max(0.5, 1.0 - step / args.steps)
        logits, _ = model(train_features[indices], prefix, temperature)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{args.steps}: loss={loss.item():.3f}", flush=True)
    report = {
        "training": evaluate(model, train_features, rows),
        "validation": evaluate(model, validation_features, validation),
        "initial_validation": initial,
        "steps": args.steps, "seed": 7,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(config=model.config, weights=model.state_dict(),
                    base_model=BASE_MODEL), output / "model.pt")
    (output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Train exact: {report['training']['exact_match']:.1%}; "
          f"validation exact: {report['validation']['exact_match']:.1%}")
    print(f"Saved model and evaluation to {output.resolve()}")


def generate(args):
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model = EmojiLanguageModel(**saved["config"]).to(args.device).eval()
    model.load_state_dict(saved["weights"])
    features = encode_text([args.prompt], saved["base_model"], args.device,
                           local_files_only=True)
    tokens, trace = model.generate(features)
    answer = decode_emojis(tokens[0].tolist())
    if args.trace:
        print(json.dumps(dict(answer=answer, states=[decode_emojis(step.tolist())
                         for step in trace[0]]), ensure_ascii=False))
    else:
        print(answer)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--data", default=str(ROOT / "data/train.jsonl"))
    training.add_argument("--validation", default=str(ROOT / "data/validation.jsonl"))
    training.add_argument("--steps", type=int, default=1000)
    training.add_argument("--batch-size", type=int, default=16)
    training.add_argument("--output", default=str(ROOT / "runs/demo"))
    training.set_defaults(action=train)
    inference = commands.add_parser("generate")
    inference.add_argument("prompt")
    inference.add_argument("--checkpoint", default=str(ROOT / "runs/demo/model.pt"))
    inference.add_argument("--trace", action="store_true")
    inference.set_defaults(action=generate)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "train" and (args.steps < 1 or args.batch_size < 1):
        parser.error("steps and batch-size must be positive")
    args.action(args)


if __name__ == "__main__":
    main()
