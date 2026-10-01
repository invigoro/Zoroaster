"""Version 2, phase 3: fine-tune a small causal LM to forecast a page's day of edits.

QLoRA on Qwen2.5-1.5B (`--model`), set up as in Stage 2 (`train_stage2.py`):
a 4-bit NF4 base, LoRA (r=16) on every projection, one epoch, and loss on
the target only. Each example pairs a prompt with a target
(`src/forecast/prompts.py`):
- the prompt describes the page as it stood at the end of D-1;
- the target is the structured forecast that the day's actual change
  makes (its main sections and kinds of change), then the day's new prose.

Variants differ only in the prompt (`src.forecast.prompts.VARIANTS`):
- `page` is the control;
- `yesterday` adds yesterday's change;
- `full` adds yesterday's change, the Stage 1 signals and the bursting
  linked pages.

Each run is scored on the test split in two ways:
- **Forecasts**, which is what the site would show.
  - Headers are greedy-decoded, parsed and scored (`src/forecast/metrics.py`).
  - They're compared per example against the "yesterday again" and "most
    common" baselines (`v2_baselines.py`), and between variants.
  - Groups: all pages; the burst model's top pages, which the site shows;
    random pages; and pages about living people.
- **Likelihood**: the mean NLL per token of the header and, on days with
  new prose, of the text.

The generated headers are checked too: did they parse, and do the sections
they name exist on the page? For living people, invented section names are
listed, because phase 4 has to filter out sensitive ones.

Writes `data/processed/enwiki/v2/qwen2.5-1.5b/results.json`, the test
generations (`generations.json`) and each run's adapters.

Usage:
    python scripts/train_v2.py [--variants full page] [--seeds 1234] [--micro-batch 4]
        [--max-train N] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# torch before pyarrow (see train_stage2.py).
import torch  # isort: skip
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from scripts.build_v2_targets import OUT_DIR
from scripts.fetch_v2_examples import EXAMPLES_DIR
from scripts.train_stage2 import collate, load_model, paired, token_mean, token_nll, train
from scripts.v2_baselines import common_kinds, forecasts
from src.forecast.changes import LEAD
from src.forecast.metrics import main_sections, score, summarize
from src.forecast.prompts import TEXT_MARK, VARIANTS, build_prompt, header, new_text, parse_forecast, point_in_time_titles

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
TITLES_PATH = OUT_DIR / "titles.parquet"
RUN_DIR = OUT_DIR / "qwen2.5-1.5b"
SEEDS = (1234,)
MICRO_BATCH = 4
MAX_PROMPT_TOKENS = 1024
LEAD_CHARS = (600, 300, 0)
TARGET_TOKENS = 256
HEADER_TOKENS = 96
GENERATION_BATCH, GENERATION_TOKENS = 16, 10_000
SCORE_CHUNK = 2
METRICS = ("section_precision", "main_section_hit", "kinds_jaccard")
BASELINES = ("yesterday again", "most common")
UNUSED_COLUMNS = {"page_text", "spans", "blocks"}


def encode(tokenizer, example: dict, variant: str) -> tuple[list[int], list[int], int, bool]:
    """(prompt ids, target ids, how many target ids are the header, whether the lead was shortened).

    The header and the text are tokenized apart so the header's tokens are
    known exactly. Qwen's tokenizer splits at the newline between them
    anyway. A prompt over MAX_PROMPT_TOKENS gets a shorter lead."""
    head = tokenizer(header(example), add_special_tokens=False).input_ids
    text = tokenizer(new_text(example), add_special_tokens=False).input_ids
    target = (head + text)[: TARGET_TOKENS - 1] + [tokenizer.eos_token_id]
    for lead_chars in LEAD_CHARS:
        prompt = tokenizer(build_prompt(example, variant, lead_chars), add_special_tokens=False).input_ids
        if len(prompt) <= MAX_PROMPT_TOKENS:
            break
    return prompt, target, min(len(head), len(target) - 1), lead_chars != LEAD_CHARS[0]


def split_nll(nll: torch.Tensor, target_lengths: list[int], header_lengths: list[int]) -> list[tuple[float, int, float, int]]:
    """Each row's per-token NLL (its target at the end) summed over its
    header and over the rest: (header NLL, header tokens, text NLL, text tokens)."""
    width = nll.shape[1]
    out = []
    for row, n, h in zip(nll, target_lengths, header_lengths):
        target = row[width - n :]
        out.append((target[:h].sum().item(), h, target[h:].sum().item(), n - h))
    return out


@torch.no_grad()
def score_parts(model, encoded: list[tuple], pad_id: int, batch: int) -> list[tuple[float, int, float, int]]:
    """`split_nll` for every example, in order."""
    model.eval()
    torch.cuda.empty_cache()  # training's cached blocks (see train_stage2.py)
    out = []
    for i in range(0, len(encoded), batch):
        chunk = encoded[i : i + batch]
        ids, mask, labels, keep = collate([(p, t) for p, t, _, _ in chunk], pad_id)
        nll, _ = token_nll(model, ids.cuda(), mask.cuda(), labels.cuda(), keep, chunk=SCORE_CHUNK)
        out += split_nll(nll.cpu(), [len(t) for _, t, _, _ in chunk], [h for _, _, h, _ in chunk])
    return out


def left_padded(prompts: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    length = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), length), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, prompt in enumerate(prompts):
        ids[row, length - len(prompt) :] = torch.tensor(prompt)
        mask[row, length - len(prompt) :] = 1
    return ids, mask


def length_batches(lengths: list[int], max_rows: int, max_tokens: int) -> list[list[int]]:
    """Indices grouped shortest first, each group at most `max_rows` long and
    at most `max_tokens` once padded to its longest."""
    batches: list[list[int]] = []
    for i in sorted(range(len(lengths)), key=lambda i: lengths[i]):
        if batches and len(batches[-1]) < max_rows and (len(batches[-1]) + 1) * lengths[i] <= max_tokens:
            batches[-1].append(i)
        else:
            batches.append([i])
    return batches


@torch.no_grad()
def forecast_headers(model, tokenizer, prompts: list[list[int]], pad_id: int) -> list[str]:
    """Greedy-decoded headers for each prompt, in order: up to the text line, which isn't generated.

    Prompts are batched by length (`length_batches`). On the longest
    prompts, a full batch of GENERATION_BATCH peaked at 6.1GB in a probe."""
    model.eval()
    model.config.use_cache = True
    torch.cuda.empty_cache()
    out = [""] * len(prompts)
    for batch in length_batches([len(p) for p in prompts], GENERATION_BATCH, GENERATION_TOKENS):
        ids, mask = left_padded([prompts[i] for i in batch], pad_id)
        new = model.generate(input_ids=ids.cuda(), attention_mask=mask.cuda(), max_new_tokens=HEADER_TOKENS,
                             do_sample=False, stop_strings=[TEXT_MARK], tokenizer=tokenizer, pad_token_id=pad_id)
        for i, text in zip(batch, tokenizer.batch_decode(new[:, ids.shape[1] :], skip_special_tokens=True)):
            out[i] = text
    model.config.use_cache = False
    return out


def groups(rows: list[dict]) -> dict[str, list[bool]]:
    return {"all": [True] * len(rows), "top": [r["selection"] == "top" for r in rows],
            "random": [r["selection"] == "random" for r in rows], "living": [bool(r["living"]) for r in rows]}


def paired_metrics(a: list[dict], b: list[dict], subset: list[bool]) -> dict:
    """Per metric, b - a over the subset's examples where both are defined: mean, standard error, count."""
    out = {}
    for name in METRICS:
        diffs = np.array([y[name] - x[name] for x, y, keep in zip(a, b, subset)
                          if keep and not (math.isnan(x[name]) or math.isnan(y[name]))])
        n = len(diffs)
        out[name] = {"mean": float(diffs.mean()) if n else math.nan,
                     "se": float(diffs.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan, "examples": n}
    return out


def validity(texts: list[str], predicted: list[dict], rows: list[dict]) -> dict:
    """Whether the generated headers parse, and how often they name sections not on the page at the end of D-1.

    Some sections are new on D, so naming one isn't wrong in itself:
    `actual_new_share` is how often the actual main sections were new."""
    named = [(s, r) for f, r in zip(predicted, rows) for s in f["sections"]]
    invented = [(s, r) for s, r in named if s != LEAD and s not in r["heading_titles"]]
    actual = [(s, r) for r in rows for s in main_sections(r["sections"], r["section_chars"])]
    return {
        "parsed": float(np.mean([("Sections:" in t and "Kinds:" in t) for t in texts])),
        "sections_named": len(named) / len(rows),
        "kinds_named": float(np.mean([len(f["kinds"]) for f in predicted])),
        "invented_share": len(invented) / len(named) if named else math.nan,
        "actual_new_share": (sum(s != LEAD and s not in r["heading_titles"] for s, r in actual) / len(actual)
                             if actual else math.nan),
        "invented_for_living_people": Counter(s for s, r in invented if r["living"]).most_common(40),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--micro-batch", type=int, default=MICRO_BATCH, choices=(1, 2, 4, 8, 16))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--variants", nargs="+", default=["full", "page"], choices=VARIANTS)
    parser.add_argument("--max-train", type=int, default=None, help="cap training examples (smoke runs)")
    parser.add_argument("--examples", type=Path, default=EXAMPLES_DIR)
    parser.add_argument("--titles", type=Path, default=TITLES_PATH)
    parser.add_argument("--out", type=Path, default=RUN_DIR)
    args = parser.parse_args(argv)
    start = time.monotonic()
    columns = [c for c in pq.read_schema(next(args.examples.glob("*.parquet"))).names if c not in UNUSED_COLUMNS]
    rows = pq.read_table(args.examples, columns=columns).to_pylist()
    renamed = point_in_time_titles(rows, pq.read_table(args.titles).to_pylist()) if args.titles.exists() else None
    print("titles that differed from the snapshot's:", renamed or f"(no {args.titles})")
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "validation", "test")}
    if args.max_train:
        by_split = {s: v[: args.max_train if s == "train" else max(args.max_train // 4, 8)] for s, v in by_split.items()}
    print({s: len(v) for s, v in by_split.items()})
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    encoded = {v: {s: [encode(tokenizer, r, v) for r in rs] for s, rs in by_split.items()} for v in args.variants}
    shortened = {v: {s: sum(e[3] for e in es) for s, es in encoded[v].items()} for v in args.variants}
    longest = {v: max(len(p) + len(t) for es in encoded[v].values() for p, t, _, _ in es) for v in args.variants}
    print(f"prompts with a shortened lead: {shortened}; longest sequences: {longest}")

    test = by_split["test"]
    subsets = groups(test)
    has_text = [bool(r["prose"]) for r in test]
    common = common_kinds(by_split["train"])
    scores = {b: [score(forecasts(r, common)[b], r) for r in test] for b in BASELINES}
    results: dict = {
        "base_model": args.model, "micro_batch": args.micro_batch, "seeds": args.seeds, "variants": args.variants,
        "examples": {s: len(v) for s, v in by_split.items()}, "group_sizes": {g: sum(m) for g, m in subsets.items()},
        "renamed": renamed, "shortened": shortened, "longest": longest, "common_kinds": common,
        "training": {}, "seconds": {}, "peak_gpu_gb": {}, "nll": {}, "validity": {},
    }
    nll: dict[str, list] = {}
    texts: dict[str, list[str]] = {}
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for v in args.variants:
            name = f"{v} / seed {seed}"
            print(f"Training {name}:", flush=True)
            began = time.monotonic()
            torch.manual_seed(seed)
            torch.cuda.reset_peak_memory_stats()
            model = load_model(args.model, adapters=True)
            pairs = {s: [(p, t) for p, t, _, _ in es] for s, es in encoded[v].items()}
            results["training"][name] = train(model, pairs["train"], pad_id, pairs["validation"], seed, args.micro_batch)
            nll[name] = score_parts(model, encoded[v]["test"], pad_id, 2 * args.micro_batch)
            texts[name] = forecast_headers(model, tokenizer, [p for p, _, _, _ in encoded[v]["test"]], pad_id)
            predicted = [parse_forecast(t) for t in texts[name]]
            scores[name] = [score(f, r) for f, r in zip(predicted, test)]
            results["validity"][name] = validity(texts[name], predicted, test)
            with_text = [(t, tc) for (_, _, t, tc), keep in zip(nll[name], has_text) if keep]
            results["nll"][name] = {"header": token_mean([(h, hc) for h, hc, _, _ in nll[name]]),
                                    "text": token_mean(with_text) if with_text else math.nan}
            results["peak_gpu_gb"][name] = torch.cuda.max_memory_allocated() / 1e9
            results["seconds"][name] = time.monotonic() - began
            model.save_pretrained(str(args.out / "adapters" / f"{v}_seed{seed}"))
            (args.out / "results.partial.json").write_text(json.dumps(results, default=str))  # in case of a crash
            print(f"  {name}: header NLL {results['nll'][name]['header']:.4f}, text NLL {results['nll'][name]['text']:.4f}, "
                  f"{results['seconds'][name]:,.0f}s, peak {results['peak_gpu_gb'][name]:.1f} GB", flush=True)
            del model
            torch.cuda.empty_cache()

    runs = [n for n in scores if n not in BASELINES]
    results["forecasts"] = {g: {"examples": sum(m)} | {n: summarize([s for s, keep in zip(scores[n], m) if keep])
                                                       for n in scores}
                            for g, m in subsets.items()}
    results["vs_baselines"] = {f"{n} vs {b}": {g: paired_metrics(scores[b], scores[n], m) for g, m in subsets.items()}
                               for n in runs for b in BASELINES}
    variant_pairs = [(a, b) for a, b in [("page", "yesterday"), ("yesterday", "full"), ("page", "full")]
                     if a in args.variants and b in args.variants]
    def compare(a: str, b: str, subset: list[bool]) -> dict:
        header_nll = paired([x[:2] for x in nll[a]], [x[:2] for x in nll[b]], subset)
        text_nll = paired([x[2:] for x in nll[a]], [x[2:] for x in nll[b]], [k and t for k, t in zip(subset, has_text)])
        return paired_metrics(scores[a], scores[b], subset) | {"header_nll": header_nll, "text_nll": text_nll}

    results["between_variants"] = {f"{b} vs {a} / seed {seed}": {g: compare(f"{a} / seed {seed}", f"{b} / seed {seed}", m)
                                                                 for g, m in subsets.items()}
                                   for a, b in variant_pairs for seed in args.seeds}
    (args.out / "results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    keys = [{"page_id": r["page_id"], "date": r["date"].isoformat(), "page_title": r["page_title"],
             "selection": r["selection"], "living": r["living"]} for r in test]
    (args.out / "generations.json").write_text(json.dumps({"examples": keys, "runs": texts}, indent=0), encoding="utf-8")

    print("\nTest forecasts (all pages / the burst model's top pages):")
    for n in scores:
        a, t = results["forecasts"]["all"][n], results["forecasts"]["top"][n]
        print(f"  {n:22} " + ", ".join(f"{m} {a[m]:.3f}/{t[m]:.3f}" for m in METRICS))
    print("Paired differences (b - a, mean ± se), all pages / top pages:")
    for name, by_group in list(results["vs_baselines"].items()) + list(results["between_variants"].items()):
        print(f"  {name}: " + "; ".join(
            f"{m} {by_group['all'][m]['mean']:+.3f} ± {by_group['all'][m]['se']:.3f} / "
            f"{by_group['top'][m]['mean']:+.3f} ± {by_group['top'][m]['se']:.3f}" for m in METRICS))
    for n, check in results["validity"].items():
        print(f"  {n}: parsed {check['parsed']:.1%}, {check['sections_named']:.2f} sections and "
              f"{check['kinds_named']:.2f} kinds named, {check['invented_share']:.1%} of named sections not on the page "
              f"(actual: {check['actual_new_share']:.1%})")
    print(f"Wrote {args.out / 'results.json'} ({time.monotonic() - start:,.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
