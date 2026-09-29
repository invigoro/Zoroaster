"""Stage 2: fine-tune a small causal LM to write the text an edit inserts.

QLoRA on Qwen2.5-0.5B: a 4-bit NF4 base with LoRA (r=16) on every
attention and MLP projection, trained on this machine's RTX 3070 (8GB),
per PLAN.md §2. Each example is a prompt (`src.stage2.examples`) plus the
edit's inserted text, and the loss covers the inserted text only.

Variants train identically and differ only in the prompt's trigger text:
- `context`: page, date, section and the surrounding text;
- `context+triggers`: plus the Stage 1 signals as text, including the titles
  of linked pages that were bursting the day before;
- `context+triggers+changes`: plus what changed on those pages that day
  (`build_stage2_neighbor_changes.py`).

Each variant trains once per seed, so run-to-run noise can be told apart
from real differences. All are scored on the test split (Dec 2025-Jun
2026, after the base model's training data) by the mean negative
log-likelihood of the inserted-text tokens:
- per example, compared pairwise, per seed and pooled over seeds;
- overall, and on the examples whose prompt names bursting linked pages,
  or shows what changed on them.

The untuned base model is scored as a floor, and a few generations are
saved for inspection.

Pages are named as they were titled then (`build_stage2_titles.py`), not
as of the dump snapshot, which would leak later renames.

The prompt budget (MAX_PROMPT_TOKENS) is large enough that trigger text
doesn't crowd out context. Truncation is counted and reported: at 384
tokens, the first run trimmed the triggers variant's context more often.
A shortened prompt loses the far end of its context, which biases a
comparison against the longer prompt. So comparisons leave out test
examples shortened in either prompt; `comparisons_including_shortened`
keeps them.

Memory on 8GB:
- Sequences are left-padded so every target sits at the end, and logits
  are computed only for the last positions. With Qwen's 152K vocabulary,
  full-sequence logits would be by far the biggest tensor.
- The model must be in `train()` mode during training, or gradient
  checkpointing silently doesn't apply.

Usage:
    python scripts/train_stage2.py [--seeds 1234 2345] [--variants ...] [--max-train N]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# torch before pyarrow: with pyarrow 18.x on Windows, loading pyarrow first
# made torch's c10.dll fail to initialize (WinError 1114).
import torch  # isort: skip
import torch.nn.functional as F  # isort: skip
import numpy as np
import pyarrow.parquet as pq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.stage2.examples import build_prompt, trigger_text

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
EXAMPLES_DIR = Path("data/processed/enwiki/stage2/examples")
CHANGES_DIR = Path("data/processed/enwiki/stage2/neighbor_changes")
TITLES_PATH = Path("data/processed/enwiki/stage2/titles.parquet")
OUT_DIR = Path("data/processed/enwiki/stage2")
VARIANTS = ("context", "context+triggers", "context+triggers+changes")
SEEDS = (1234, 2345)
MAX_PROMPT_TOKENS = 512
TARGET_TOKENS = 128
MAX_CHANGES = 3
MICRO_BATCH, ACCUMULATION, EVAL_BATCH = 8, 2, 16
LEARNING_RATE, WARMUP_STEPS = 2e-4, 50
N_GENERATIONS = 12


def prompt_for(example: dict, variant: str) -> str:
    triggers = None
    if variant != "context":
        changes = example.get("neighbor_changes") if variant == "context+triggers+changes" else None
        triggers = trigger_text(example, example["bursting_neighbors"], changes=changes, max_changes=MAX_CHANGES)
    return build_prompt(example["page_title"], example["date"].isoformat(), example["section"], example["context"], triggers)


def encode(tokenizer, example: dict, variant: str) -> tuple[list[int], list[int], bool]:
    """(prompt ids, target ids, whether the context had to be shortened). An
    over-long prompt loses the start of its context, not its header lines."""
    target = tokenizer(example["added_text"], add_special_tokens=False).input_ids[: TARGET_TOKENS - 1]
    target.append(tokenizer.eos_token_id)
    context, truncated = example["context"], False
    while True:
        prompt = tokenizer(prompt_for(example | {"context": context}, variant), add_special_tokens=False).input_ids
        if len(prompt) <= MAX_PROMPT_TOKENS or len(context) < 50:
            return prompt[-MAX_PROMPT_TOKENS:], target, truncated
        context, truncated = context[len(context) // 5 :], True


def attach_changes(rows: list[dict], changes_dir: Path) -> None:
    """Give each row `neighbor_changes`: {neighbor title: snippet or None} for the day before."""
    changes: dict = {}
    if changes_dir.exists():
        table = pq.read_table(changes_dir, columns=["title", "date", "snippet"])
        changes = {(t, d): s for t, d, s in zip(*(table[c].to_pylist() for c in ("title", "date", "snippet")))}
    for row in rows:
        prev = row["date"] - timedelta(days=1)
        row["neighbor_changes"] = {t: changes.get((t, prev)) for t in row["bursting_neighbors"]}


def point_in_time_titles(rows: list[dict], titles_path: Path) -> dict[str, int]:
    """Rename pages to their titles at the time (`build_stage2_titles.py`):
    each example's page as of its edit, each bursting neighbor as of the day
    before. Runs after `attach_changes`, whose keys it renames too. Returns
    how many titles changed."""
    renamed = {"pages": 0, "neighbors": 0}
    if not titles_path.exists():
        print(f"No {titles_path}: prompts use the snapshot's titles")
        return renamed
    table = pq.read_table(titles_path).to_pylist()
    page = {r["revision_id"]: r["title_then"] for r in table if r["kind"] == "page" and r["title_then"]}
    neighbor = {(r["title"], r["date"]): r["title_then"] for r in table if r["kind"] == "neighbor" and r["title_then"]}
    for row in rows:
        then = page.get(row["revision_id"], row["page_title"])
        renamed["pages"] += then != row["page_title"]
        row["page_title"] = then
        prev = row["date"] - timedelta(days=1)
        names = {t: neighbor.get((t, prev), t) for t in row["bursting_neighbors"]}
        renamed["neighbors"] += sum(t != n for t, n in names.items())
        row["bursting_neighbors"] = [names[t] for t in row["bursting_neighbors"]]
        row["neighbor_changes"] = {names[t]: s for t, s in row["neighbor_changes"].items()}
    return renamed


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


def _batch(pairs: list, pad_id: int):
    ids, mask, labels, keep = collate(pairs, pad_id)
    return ids.cuda(), mask.cuda(), labels.cuda(), keep


def train(model, pairs: list, pad_id: int, validation: list, seed: int) -> list[dict]:
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    batches = [order[i : i + MICRO_BATCH] for i in range(0, len(order), MICRO_BATCH)]
    steps = math.ceil(len(batches) / ACCUMULATION)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / WARMUP_STEPS)
                                                 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    model.train()
    log, running, start = [], [], time.monotonic()
    for i, batch in enumerate(batches):
        nll, counts = target_nll(model, *_batch([pairs[j] for j in batch], pad_id))
        loss = nll.sum() / counts.sum()
        (loss / ACCUMULATION).backward()
        running.append(loss.item())
        if (i + 1) % ACCUMULATION == 0 or i + 1 == len(batches):
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); schedule.step(); optimizer.zero_grad(set_to_none=True)
            step = (i + 1) // ACCUMULATION
            if step % 100 == 0 or i + 1 == len(batches):
                entry = {"step": step, "train_loss": float(np.mean(running)), "seconds": time.monotonic() - start}
                running = []
                if i + 1 == len(batches):
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
        nll, counts = target_nll(model, *_batch(pairs[i : i + EVAL_BATCH], pad_id))
        out += list(zip(nll.tolist(), counts.tolist()))
    return out


def token_mean(scores: list[tuple[float, int]]) -> float:
    return sum(n for n, _ in scores) / sum(c for _, c in scores)


def pooled(runs: list[list[tuple[float, int]]]) -> list[tuple[float, int]]:
    """Per-example NLL averaged over runs (the targets, so the counts, are the same)."""
    return [(float(np.mean([run[i][0] for run in runs])), runs[0][i][1]) for i in range(len(runs[0]))]


def unshortened(mask: list[bool] | None, shortened_a: list[bool], shortened_b: list[bool]) -> list[bool]:
    """`mask` (None: every example) without the examples either prompt had to shorten."""
    keep = [not (x or y) for x, y in zip(shortened_a, shortened_b)]
    return keep if mask is None else [m and k for m, k in zip(mask, keep)]


def paired(a: list[tuple[float, int]], b: list[tuple[float, int]], subset: list[bool] | None = None) -> dict:
    """Per-example mean-token-NLL difference b - a: mean, standard error, share improved."""
    diffs = np.array([nb / cb - na / ca for (na, ca), (nb, cb) in zip(a, b)])
    if subset is not None:
        diffs = diffs[np.array(subset, dtype=bool)]
    n = len(diffs)
    nan = float("nan")
    return {"mean": float(diffs.mean()) if n else nan,
            "se": float(diffs.std(ddof=1) / math.sqrt(n)) if n > 1 else nan,
            "share_improved": float((diffs < 0).mean()) if n else nan, "examples": n}


@torch.no_grad()
def generate(model, tokenizer, examples: list[dict], variant: str) -> list[str]:
    model.eval()
    model.config.use_cache = True
    outputs = []
    for example in examples:
        prompt, _, _ = encode(tokenizer, example, variant)
        ids = torch.tensor([prompt]).cuda()
        new = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=64, do_sample=False,
                             repetition_penalty=1.2, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        outputs.append(tokenizer.decode(new[0, ids.shape[1] :], skip_special_tokens=True))
    model.config.use_cache = False
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--max-train", type=int, default=None, help="cap training examples (smoke runs)")
    parser.add_argument("--examples", type=Path, default=EXAMPLES_DIR)
    parser.add_argument("--changes", type=Path, default=CHANGES_DIR)
    parser.add_argument("--titles", type=Path, default=TITLES_PATH)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    rows = pq.read_table(args.examples).to_pylist()
    attach_changes(rows, args.changes)
    renamed = point_in_time_titles(rows, args.titles)
    print("titles that differed from the snapshot's:", renamed)
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "validation", "test")}
    if args.max_train:
        by_split = {s: v[: args.max_train if s == "train" else max(args.max_train // 4, 8)] for s, v in by_split.items()}
    print({s: len(v) for s, v in by_split.items()})
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    encoded, truncation, shortened = {}, {}, {}
    for v in args.variants:
        encoded[v] = {s: [encode(tokenizer, r, v) for r in rs] for s, rs in by_split.items()}
        truncation[v] = {s: sum(t for _, _, t in e) for s, e in encoded[v].items()}
        shortened[v] = [t for _, _, t in encoded[v]["test"]]
        encoded[v] = {s: [(p, t) for p, t, _ in e] for s, e in encoded[v].items()}
    print("prompts truncated (context shortened):", truncation)

    test = by_split["test"]
    subsets = {
        "all": None,
        "bursting neighbors": [bool(r["bursting_neighbors"]) for r in test],
        "changes shown": [any(r["neighbor_changes"].get(t) for t in r["bursting_neighbors"]) for r in test],
        "no bursting neighbors": [not r["bursting_neighbors"] for r in test],
    }
    rng = random.Random(args.seeds[0])
    shown = [r for r, has in zip(test, subsets["changes shown"]) if has]
    showcase = rng.sample(shown, min(N_GENERATIONS // 2, len(shown)))
    quiet = [r for r in test if not r["bursting_neighbors"]]
    showcase += rng.sample(quiet, min(N_GENERATIONS // 2, len(quiet)))

    results: dict = {"base_model": BASE_MODEL, "seeds": args.seeds, "examples": {s: len(v) for s, v in by_split.items()},
                     "subset_sizes": {k: (sum(v) if v else len(test)) for k, v in subsets.items()},
                     "truncation": truncation, "renamed": renamed, "scores": {}, "training": {}, "peak_gpu_gb": {}}
    base = load_model(adapters=False)
    for v in args.variants:
        results["scores"][f"base / {v}"] = score(base, encoded[v]["test"], pad_id)
    del base
    torch.cuda.empty_cache()

    generations: dict[str, list[str]] = {}
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for v in args.variants:
            name = f"{v} / seed {seed}"
            print(f"Training {name}:")
            torch.manual_seed(seed)
            torch.cuda.reset_peak_memory_stats()
            model = load_model(adapters=True)
            results["training"][name] = train(model, encoded[v]["train"], pad_id, encoded[v]["validation"], seed)
            results["scores"][f"tuned / {name}"] = score(model, encoded[v]["test"], pad_id)
            results["peak_gpu_gb"][name] = torch.cuda.max_memory_allocated() / 1e9
            if seed == args.seeds[0]:
                generations[v] = generate(model, tokenizer, showcase, v)
            model.save_pretrained(str(args.out / "adapters" / f"{v.replace('+', '_')}_seed{seed}"))
            (args.out / "results.partial.json").write_text(json.dumps(results, default=str))  # in case of a crash
            del model
            torch.cuda.empty_cache()

    scores = results["scores"]
    for v in args.variants:
        scores[f"tuned / {v} / pooled"] = pooled([scores[f"tuned / {v} / seed {s}"] for s in args.seeds])
    results["test_token_nll"] = {name: token_mean(s) for name, s in scores.items()}
    pairs_to_compare = [(a, b) for a, b in [("context", "context+triggers"), ("context+triggers", "context+triggers+changes"),
                                            ("context", "context+triggers+changes")] if a in args.variants and b in args.variants]
    runs = [f"seed {s}" for s in args.seeds] + ["pooled"]
    for key, restrict in [("comparisons", True), ("comparisons_including_shortened", False)]:
        results[key] = {
            f"{b} vs {a} / {run} / {subset}": paired(
                scores[f"tuned / {a} / {run}"], scores[f"tuned / {b} / {run}"],
                unshortened(mask, shortened[a], shortened[b]) if restrict else mask,
            )
            for a, b in pairs_to_compare for run in runs for subset, mask in subsets.items()
        }
    if len(args.seeds) > 1:  # run-to-run noise: the same variant, two seeds
        results["seed_noise"] = {v: paired(scores[f"tuned / {v} / seed {args.seeds[0]}"], scores[f"tuned / {v} / seed {args.seeds[1]}"])
                                 for v in args.variants}
    results["generations"] = [
        {"page": r["page_title"], "date": r["date"].isoformat(), "bursting_neighbors": r["bursting_neighbors"][:MAX_CHANGES],
         "changes": {t: s for t, s in r["neighbor_changes"].items() if s} , "actual": r["added_text"][:300],
         **{v: generations[v][i] for v in generations}}
        for i, r in enumerate(showcase)
    ]
    (args.out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print("Test NLL per inserted-text token (perplexity):")
    for name, nll in results["test_token_nll"].items():
        print(f"  {name}: {nll:.4f} ({math.exp(nll):.2f})")
    print("Paired per-example differences (per token):")
    for name, d in results["comparisons"].items():
        if "/ pooled /" in name or d["examples"] < 30:
            print(f"  {name}: {d['mean']:+.4f} ± {d['se']:.4f} (improved on {d['share_improved']:.1%} of {d['examples']:,})")
    for v, d in results.get("seed_noise", {}).items():
        print(f"  seed noise, {v}: {d['mean']:+.4f} ± {d['se']:.4f}")
    print(f"Wrote {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
