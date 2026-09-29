"""Stage 2: fine-tune a small causal LM to write the text an edit inserts.

QLoRA on Qwen2.5-0.5B: a 4-bit NF4 base with LoRA (r=16) on every
attention and MLP projection, trained on this machine's RTX 3070 (8GB),
per PLAN.md §2. Each example is a prompt (`src.stage2.examples`) plus the
edit's inserted text, and the loss covers the inserted text only.

Two variants train identically and differ only in the prompt:
- `context`: page, date, section and the surrounding text;
- `context+triggers`: the same, plus the Stage 1 signals as text, including
  the titles of linked pages that were bursting the day before.

Both are scored on the test split (Dec 2025-Jun 2026, after the base
model's training data) by the mean negative log-likelihood of the
inserted-text tokens:
- per example, compared pairwise;
- overall, and on the examples whose prompt names a bursting linked page.

The untuned base model is scored as a floor, and a few generations are
saved for inspection.

Memory on 8GB:
- Sequences are left-padded so every target sits at the end, and logits
  are computed only for the last positions. With Qwen's 152K vocabulary,
  full-sequence logits would be by far the biggest tensor.
- The model must be in `train()` mode during training, or gradient
  checkpointing silently doesn't apply.

Usage:
    python scripts/train_stage2.py [--max-train N] [--examples DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# torch before pyarrow: on Windows, loading pyarrow's native libraries first
# can make torch's c10.dll fail to initialize (WinError 1114).
import torch  # isort: skip
import torch.nn.functional as F  # isort: skip
import numpy as np
import pyarrow.parquet as pq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.stage2.examples import build_prompt, trigger_text

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
EXAMPLES_DIR = Path("data/processed/enwiki/stage2/examples")
OUT_DIR = Path("data/processed/enwiki/stage2")
VARIANTS = ("context", "context+triggers")
MAX_PROMPT_TOKENS = 384
TARGET_TOKENS = 128
MICRO_BATCH, ACCUMULATION, EVAL_BATCH = 8, 2, 16
LEARNING_RATE, WARMUP_STEPS = 2e-4, 50
N_GENERATIONS = 12
SEED = 1234


def prompt_for(example: dict, variant: str) -> str:
    triggers = trigger_text(example, example["bursting_neighbors"]) if variant == "context+triggers" else None
    return build_prompt(example["page_title"], example["date"].isoformat(), example["section"], example["context"], triggers)


def encode(tokenizer, example: dict, variant: str) -> tuple[list[int], list[int]]:
    """(prompt ids, target ids). An over-long prompt loses the start of its
    context rather than its header lines."""
    target = tokenizer(example["added_text"], add_special_tokens=False).input_ids[: TARGET_TOKENS - 1]
    target.append(tokenizer.eos_token_id)
    context = example["context"]
    while True:
        prompt = tokenizer(prompt_for(example | {"context": context}, variant), add_special_tokens=False).input_ids
        if len(prompt) <= MAX_PROMPT_TOKENS or len(context) < 50:
            return prompt[-MAX_PROMPT_TOKENS:], target
        context = context[len(context) // 5 :]


def collate(pairs: list[tuple[list[int], list[int]]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Left-padded input ids, attention mask, and labels (-100 except on
    target tokens), plus K: how many trailing positions need logits."""
    length = max(len(p) + len(t) for p, t in pairs)
    ids = torch.full((len(pairs), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(pairs), length), dtype=torch.long)
    labels = torch.full((len(pairs), length), -100, dtype=torch.long)
    for row, (prompt, target) in enumerate(pairs):
        seq = prompt + target
        ids[row, length - len(seq) :] = torch.tensor(seq)
        mask[row, length - len(seq) :] = 1
        labels[row, length - len(target) :] = torch.tensor(target)
    return ids, mask, labels, max(len(t) for _, t in pairs) + 1


def target_nll(model, ids, mask, labels, keep: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-example summed NLL of target tokens, and their counts."""
    logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=keep).logits[:, :-1]
    tail = labels[:, -(keep - 1) :]
    token_nll = F.cross_entropy(logits.float().transpose(1, 2), tail, ignore_index=-100, reduction="none")
    counts = (tail != -100).sum(dim=1)
    return token_nll.sum(dim=1), counts


def load_model(adapters: bool):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, dtype=torch.bfloat16,
                                                 device_map={"": 0})
    model.config.use_cache = False
    if not adapters:
        return model
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    return get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))


def train(model, pairs: list, pad_id: int, validation: list) -> list[dict]:
    rng = random.Random(SEED)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    batches = [order[i : i + MICRO_BATCH] for i in range(0, len(order), MICRO_BATCH)]
    steps = math.ceil(len(batches) / ACCUMULATION)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / WARMUP_STEPS)
                                                 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    model.train()
    log, running, start = [], [], time.monotonic()
    for i, batch in enumerate(batches):
        ids, mask, labels, keep = (x.cuda() if torch.is_tensor(x) else x for x in collate([pairs[j] for j in batch], pad_id))
        nll, counts = target_nll(model, ids, mask, labels, keep)
        loss = nll.sum() / counts.sum()
        (loss / ACCUMULATION).backward()
        running.append(loss.item())
        if (i + 1) % ACCUMULATION == 0 or i + 1 == len(batches):
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); schedule.step(); optimizer.zero_grad(set_to_none=True)
            step = (i + 1) // ACCUMULATION
            if step % 50 == 0 or i + 1 == len(batches):
                entry = {"step": step, "train_loss": float(np.mean(running)), "seconds": time.monotonic() - start}
                running = []
                if i + 1 == len(batches) or step % 250 == 0:
                    entry["validation_loss"] = token_mean(score(model, validation, pad_id))
                    model.train()
                log.append(entry)
                print(f"    step {step}/{steps}: " + ", ".join(f"{k} {v:.3f}" for k, v in entry.items() if k != "step"), flush=True)
    return log


@torch.no_grad()
def score(model, pairs: list, pad_id: int) -> list[tuple[float, int]]:
    """(summed target NLL, target tokens) per example, in order."""
    model.eval()
    out = []
    for i in range(0, len(pairs), EVAL_BATCH):
        ids, mask, labels, keep = (x.cuda() if torch.is_tensor(x) else x for x in collate(pairs[i : i + EVAL_BATCH], pad_id))
        nll, counts = target_nll(model, ids, mask, labels, keep)
        out += list(zip(nll.tolist(), counts.tolist()))
    return out


def token_mean(scores: list[tuple[float, int]]) -> float:
    return sum(n for n, _ in scores) / sum(c for _, c in scores)


def paired(a: list[tuple[float, int]], b: list[tuple[float, int]], subset: list[bool] | None = None) -> dict:
    """Per-example mean-token-NLL difference b - a: mean, standard error, share improved."""
    diffs = np.array([nb / cb - na / ca for (na, ca), (nb, cb) in zip(a, b)])
    if subset is not None:
        diffs = diffs[np.array(subset)]
    return {"mean": float(diffs.mean()), "se": float(diffs.std(ddof=1) / math.sqrt(len(diffs))),
            "share_improved": float((diffs < 0).mean()), "examples": int(len(diffs))}


@torch.no_grad()
def generate(model, tokenizer, examples: list[dict], variant: str) -> list[str]:
    model.eval()
    model.config.use_cache = True
    outputs = []
    for example in examples:
        prompt, _ = encode(tokenizer, example, variant)
        ids = torch.tensor([prompt]).cuda()
        new = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=64, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        outputs.append(tokenizer.decode(new[0, ids.shape[1] :], skip_special_tokens=True))
    model.config.use_cache = False
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-train", type=int, default=None, help="cap training examples (smoke runs)")
    parser.add_argument("--examples", type=Path, default=EXAMPLES_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    torch.manual_seed(SEED)
    rows = pq.read_table(args.examples).to_pylist()
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "validation", "test")}
    if args.max_train:
        by_split = {s: v[: args.max_train if s == "train" else max(args.max_train // 4, 8)] for s, v in by_split.items()}
    print({s: len(v) for s, v in by_split.items()})
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    encoded = {v: {s: [encode(tokenizer, r, v) for r in rs] for s, rs in by_split.items()} for v in VARIANTS}
    test = by_split["test"]
    has_neighbors = [bool(r["bursting_neighbors"]) for r in test]
    rng = random.Random(SEED)
    showcase = rng.sample([r for r in test if r["bursting_neighbors"]], min(N_GENERATIONS // 2, sum(has_neighbors)))
    showcase += rng.sample([r for r in test if not r["bursting_neighbors"]], min(N_GENERATIONS // 2, len(test) - sum(has_neighbors)))

    results: dict = {"base_model": BASE_MODEL, "examples": {s: len(v) for s, v in by_split.items()},
                     "test_with_bursting_neighbors": sum(has_neighbors), "scores": {}, "training": {}}
    base = load_model(adapters=False)
    for variant in VARIANTS:
        results["scores"][f"base / {variant}"] = score(base, encoded[variant]["test"], pad_id)
    del base
    torch.cuda.empty_cache()

    generations = {v: [] for v in VARIANTS}
    for variant in VARIANTS:
        print(f"Training {variant}:")
        model = load_model(adapters=True)
        results["training"][variant] = train(model, encoded[variant]["train"], pad_id, encoded[variant]["validation"])
        results["scores"][f"tuned / {variant}"] = score(model, encoded[variant]["test"], pad_id)
        generations[variant] = generate(model, tokenizer, showcase, variant)
        model.save_pretrained(str(args.out / "adapters" / variant.replace("+", "_")))
        del model
        torch.cuda.empty_cache()

    scores = results["scores"]
    results["test_token_nll"] = {name: token_mean(s) for name, s in scores.items()}
    results["comparisons"] = {
        "tuned: triggers vs context": paired(scores["tuned / context"], scores["tuned / context+triggers"]),
        "tuned: triggers vs context, bursting neighbors": paired(scores["tuned / context"], scores["tuned / context+triggers"], has_neighbors),
        "tuned: triggers vs context, no bursting neighbors": paired(scores["tuned / context"], scores["tuned / context+triggers"], [not h for h in has_neighbors]),
        "context: tuned vs base": paired(scores["base / context"], scores["tuned / context"]),
    }
    results["generations"] = [
        {"page": r["page_title"], "date": r["date"].isoformat(), "bursting_neighbors": r["bursting_neighbors"],
         "actual": r["added_text"][:300], **{v: generations[v][i] for v in VARIANTS}}
        for i, r in enumerate(showcase)
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print("Test NLL per inserted-text token (perplexity):")
    for name, nll in results["test_token_nll"].items():
        print(f"  {name}: {nll:.3f} ({math.exp(nll):.1f})")
    for name, d in results["comparisons"].items():
        print(f"  {name}: {d['mean']:+.4f} ± {d['se']:.4f} per token (improved on {d['share_improved']:.1%} of {d['examples']:,})")
    print(f"Wrote {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
