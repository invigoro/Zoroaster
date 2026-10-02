"""Version 2: rank forecasts by a trained model's probabilities, instead of greedy decoding.

For each validation and test page-day, this reads two things from a
`train_v2.py` adapter (`src/forecast/ranking.py`):
- each section candidate's probability of being named first. The forecast
  names the top MAX_SECTIONS.
- each kind's exact probability of being listed, after the model's own
  greedy sections line. The forecast lists the kinds whose probability
  passes a per-kind threshold, chosen on validation to maximize the mean
  kinds Jaccard.

Test greedy headers come from the run's `generations.json`. Validation's
are generated here.

One forward pass per example scores everything (`tree_logprobs`):
- every section continuation and all 512 kinds lines go into one token
  trie after the prompt;
- a custom attention mask lets each trie token see only the prompt and its
  own ancestors.
That's about 1,900 tokens a pass. Scoring the continuations as separate
rows cost 17 passes per example, and each pass has a fixed cost of about
0.14 s on this GPU.

Test forecasts are scored against "yesterday again" and the greedy
forecast, overall and by group, per example and paired. Two ranked
forecasts are scored:
- the main one, with the validation-chosen thresholds;
- a reference with every threshold at 0.5, which needs no tuning.

Writes `ranked.json` into the run directory, which `v2_report.py` adds to
the report.

Usage:
    python scripts/rank_v2.py [--run-dir DIR] [--variant full] [--seed 1234] [--max-examples N] [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# torch before pyarrow (see train_stage2.py).
import torch  # isort: skip
import numpy as np
import pyarrow.parquet as pq
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from scripts.train_stage2 import collate, token_nll
from scripts.train_v2 import (BASE_MODEL, EXAMPLES_DIR, METRICS, RUN_DIR, TITLES_PATH, UNUSED_COLUMNS, encode,
                              forecast_headers, groups, paired_metrics)
from scripts.v2_baselines import common_kinds, forecasts
from src.forecast.changes import KINDS
from src.forecast.metrics import main_sections, score, summarize
from src.forecast.prompts import NONE, parse_forecast, point_in_time_titles
from src.forecast.ranking import (build_trie, choose_thresholds, first_section_probs, kind_marginals,
                                  kinds_continuation, kinds_forecast, kinds_lines, rank_sections, section_candidates,
                                  section_continuations)

LOGSUMEXP_ROWS = 256
TOP_SHOWN = 10


def tree_mask(prompt_len: int, parents: list[int]) -> torch.Tensor:
    """Boolean attention for a prompt followed by a trie's nodes. The prompt
    attends causally, and each node attends to the whole prompt, its
    ancestors and itself."""
    n = len(parents)
    mask = torch.zeros((prompt_len + n, prompt_len + n), dtype=torch.bool)
    mask[:prompt_len, :prompt_len] = torch.ones((prompt_len, prompt_len), dtype=torch.bool).tril()
    mask[prompt_len:, :prompt_len] = True
    for j, parent in enumerate(parents):
        row = prompt_len + j
        if parent >= 0:
            mask[row, prompt_len:] = mask[prompt_len + parent, prompt_len:]
        mask[row, row] = True
    return mask


@torch.no_grad()
def tree_logprobs(model, base: list[int], continuations: list[list[int]]) -> list[float]:
    """log P(continuation | base) for each continuation, in one forward pass.

    The continuations become a token trie (`build_trie`) after the base. With
    `tree_mask`, and positions counted from the base's end, each one is scored
    as if it followed the base alone."""
    if model.config._attn_implementation != "sdpa":
        raise ValueError("tree scoring needs SDPA attention, which reads a boolean mask as 'may attend'")
    tokens, parents, depths, paths = build_trie(continuations)
    device, length = model.device, len(base)
    input_ids = torch.tensor([base + tokens], device=device)
    positions = torch.tensor([list(range(length)) + [length + d for d in depths]], device=device)
    mask = tree_mask(length, parents)[None, None].to(device)
    logits = model(input_ids=input_ids, attention_mask=mask, position_ids=positions, use_cache=False,
                   logits_to_keep=len(tokens) + 1).logits[0]
    # Kept row 0 is the base's last token, which predicts the trie's first tokens; row k + 1 is node k.
    predictor = torch.tensor([0 if p < 0 else p + 1 for p in parents], device=device)
    normalizer = torch.cat([torch.logsumexp(logits[i : i + LOGSUMEXP_ROWS].float(), dim=-1)
                            for i in range(0, len(logits), LOGSUMEXP_ROWS)])
    edges = (logits[predictor, torch.tensor(tokens, device=device)].float() - normalizer[predictor]).tolist()
    return [sum(edges[node] for node in path) for path in paths]


def load_trained(model_name: str, adapter: Path):
    """The base model prepared as for training (`train_stage2.load_model`), with a trained adapter."""
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, dtype=torch.bfloat16,
                                                device_map={"": 0})
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(base, str(adapter))
    model.eval()
    return model


def ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def rank_example(model, tokenizer, prompt: list[int], row: dict, greedy: dict, line_ids: list[list[int]],
                 lines: list[tuple]) -> dict:
    """Section probabilities and kind marginals for one page-day, from one forward pass.

    The kinds lines follow the greedy sections line (`chain`). Its own
    probability is divided out, so the kinds' marginals are conditional
    on it."""
    candidates = section_candidates(row)
    section_ids = [ids(tokenizer, c) for c in section_continuations(candidates)]
    chain = ids(tokenizer, f" {'; '.join(greedy['sections']) or NONE}\nKinds:")
    logprobs = tree_logprobs(model, prompt + ids(tokenizer, "Sections:"),
                             section_ids + [chain] + [chain + line for line in line_ids])
    n = len(section_ids)
    probs = first_section_probs(logprobs[:n], candidates)
    marginals, valid = kind_marginals(lines, [lp - logprobs[n] for lp in logprobs[n + 1 :]])
    order = sorted(range(len(candidates)), key=lambda i: -probs[i])[:TOP_SHOWN]
    return {"sections": rank_sections(candidates, probs), "section_probs": [[candidates[i], probs[i]] for i in order],
            "section_mass": sum(probs), "kinds": marginals, "kinds_mass": valid}


def tokenization_mismatches(tokenizer, rows: list[dict], lines: list[tuple]) -> int:
    """How many scored sequences tokenize differently in pieces than written
    out whole, which is how training tokenized the header."""
    head, bad = ids(tokenizer, "Sections:"), 0
    for row in rows:
        for c in section_continuations(section_candidates(row)):
            bad += head + ids(tokenizer, c) != ids(tokenizer, "Sections:" + c)
        chain = f" {'; '.join(main_sections(row['sections'], row['section_chars'])) or NONE}\nKinds:"
        for kinds in (lines[0], lines[1], lines[-1]):
            whole = ids(tokenizer, "Sections:" + chain + kinds_continuation(kinds))
            bad += head + ids(tokenizer, chain) + ids(tokenizer, kinds_continuation(kinds)) != whole
    chain = " (lead)\nKinds:"
    bad += sum(head + ids(tokenizer, chain) + ids(tokenizer, kinds_continuation(k))
               != ids(tokenizer, "Sections:" + chain + kinds_continuation(k)) for k in lines)
    return bad


def check_against_full_forward(model, prompt: list[int], continuations: list[list[int]], pad_id: int) -> dict:
    """How far tree log-probabilities are from scoring each sequence alone,
    next to how far those move when the same sequences are batched together:
    the rounding noise of bf16 on this GPU."""
    def uncached(pairs):
        input_ids, mask, labels, keep = collate(pairs, pad_id)
        nll, _ = token_nll(model, input_ids.to(model.device), mask.to(model.device), labels.to(model.device), keep)
        return (-nll.sum(dim=1)).tolist()

    tree = tree_logprobs(model, prompt, continuations)
    alone = [uncached([(prompt, c)])[0] for c in continuations]
    together = uncached([(prompt, c) for c in continuations])
    return {"tree vs alone": max(abs(a - b) for a, b in zip(tree, alone)),
            "batched vs alone": max(abs(a - b) for a, b in zip(together, alone)),
            "tokens": [len(c) for c in continuations]}


def evaluate(test: list[dict], named: dict[str, list[dict]]) -> tuple[dict, dict]:
    subsets = groups(test)
    scores = {n: [score(f, r) for f, r in zip(fs, test)] for n, fs in named.items()}
    summary = {g: {"examples": sum(m)} | {n: summarize([s for s, keep in zip(scores[n], m) if keep]) for n in scores}
               for g, m in subsets.items()}
    ranked = [n for n in named if "ranked" in n]
    others = [n for n in named if "ranked" not in n]
    paired = {f"{n} vs {b}": {g: paired_metrics(scores[b], scores[n], m) for g, m in subsets.items()}
              for n in ranked for b in others}
    return summary, paired


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--variant", default="full")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-examples", type=int, default=None, help="cap each split (smoke runs)")
    parser.add_argument("--check", action="store_true", help="compare tree and one-at-a-time scores on one example")
    args = parser.parse_args(argv)
    start = time.monotonic()
    run = f"{args.variant} / seed {args.seed}"
    label = f"model ({args.variant} prompt"
    columns = [c for c in pq.read_schema(next(EXAMPLES_DIR.glob("*.parquet"))).names if c not in UNUSED_COLUMNS]
    rows = pq.read_table(EXAMPLES_DIR, columns=columns).to_pylist()
    point_in_time_titles(rows, pq.read_table(TITLES_PATH).to_pylist())
    common = common_kinds([r for r in rows if r["split"] == "train"])
    by_split = {s: [r for r in rows if r["split"] == s][: args.max_examples] for s in ("validation", "test")}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    lines = kinds_lines()
    line_ids = [ids(tokenizer, kinds_continuation(k)) for k in lines]
    mismatches = tokenization_mismatches(tokenizer, by_split["validation"][:200], lines)
    print(f"{ {s: len(v) for s, v in by_split.items()} }; sequences tokenized differently in pieces: {mismatches}",
          flush=True)
    prompts = {s: [encode(tokenizer, r, args.variant)[0] for r in rs] for s, rs in by_split.items()}
    model = load_trained(args.model, args.run_dir / "adapters" / f"{args.variant}_seed{args.seed}")
    if args.check:
        sample = [ids(tokenizer, c) for c in section_continuations(section_candidates(by_split["test"][0]))[:6]]
        sample += line_ids[:3] + line_ids[-3:]
        gaps = check_against_full_forward(model, prompts["test"][0] + ids(tokenizer, "Sections:"), sample, pad_id)
        print("largest log-probability gaps:", gaps, flush=True)

    generations = json.loads((args.run_dir / "generations.json").read_text(encoding="utf-8"))
    aligned = [(k["page_id"], k["date"]) for k in generations["examples"][: len(by_split["test"])]] == [
        (r["page_id"], r["date"].isoformat()) for r in by_split["test"]]
    greedy_texts = {"validation": None, "test": generations["runs"][run][: len(by_split["test"])] if aligned else None}
    details: dict[str, list] = {}
    for split, rs in by_split.items():
        if greedy_texts[split] is None:
            began = time.monotonic()
            greedy_texts[split] = forecast_headers(model, tokenizer, prompts[split], pad_id)
            print(f"{split}: generated {len(rs):,} greedy headers ({time.monotonic() - began:,.0f}s)", flush=True)
        greedy = [parse_forecast(t) for t in greedy_texts[split]]
        out: list = [None] * len(rs)
        began = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for n, i in enumerate(sorted(range(len(rs)), key=lambda i: -len(prompts[split][i]))):  # longest first
            out[i] = rank_example(model, tokenizer, prompts[split][i], rs[i], greedy[i], line_ids, lines)
            torch.cuda.empty_cache()  # sizes change with every prompt (see train_v2.forecast_headers)
            if (n + 1) % 500 == 0 or n + 1 == len(rs):
                print(f"  {split}: {n + 1:,}/{len(rs):,} ranked ({time.monotonic() - began:,.0f}s, "
                      f"peak reserved {torch.cuda.max_memory_reserved() / 1e9:.1f} GB)", flush=True)
        details[split] = out

    validation = by_split["validation"]
    thresholds = choose_thresholds([d["kinds"] for d in details["validation"]], [r["kinds"] for r in validation])
    print("thresholds chosen on validation:", {k: thresholds[k] for k in KINDS}, flush=True)
    half = {k: 0.5 for k in KINDS}

    def named_forecasts(split: str, rs: list[dict]) -> dict[str, list[dict]]:
        return {
            "yesterday again": [forecasts(r, common)["yesterday again"] for r in rs],
            f"{label}, greedy)": [parse_forecast(t) for t in greedy_texts[split]],
            f"{label}, ranked)": [{"sections": d["sections"], "kinds": kinds_forecast(d["kinds"], thresholds)}
                                  for d in details[split]],
            f"{label}, ranked, thresholds 0.5)": [{"sections": d["sections"], "kinds": kinds_forecast(d["kinds"], half)}
                                                  for d in details[split]],
        }

    results: dict = {"variant": args.variant, "seed": args.seed, "thresholds": thresholds,
                     "tokenization_mismatches": mismatches, "seconds": None}
    for split, rs in by_split.items():
        named = named_forecasts(split, rs)
        summary, paired = evaluate(rs, named)
        results[split] = {"summary": summary, "paired": paired,
                          "mass": {"sections": float(np.mean([d["section_mass"] for d in details[split]])),
                                   "kinds": float(np.mean([d["kinds_mass"] for d in details[split]]))}}
        if split == "test":
            results["test"]["examples"] = [{"page_id": r["page_id"], "date": r["date"].isoformat()} for r in rs]
            results["test"]["forecasts"] = {n: fs for n, fs in named.items() if "ranked" in n}
            results["test"]["details"] = details["test"]
    results["seconds"] = time.monotonic() - start
    out_path = args.run_dir / ("ranked.json" if args.max_examples is None else "ranked.smoke.json")
    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")

    for split in by_split:
        print(f"\n{split} (all pages / the burst model's top pages); the probability the candidates hold: "
              f"sections {results[split]['mass']['sections']:.3f}, kinds {results[split]['mass']['kinds']:.3f}")
        s = results[split]["summary"]
        for n in named_forecasts(split, by_split[split]):
            print(f"  {n:46} " + ", ".join(f"{m} {s['all'][n][m]:.3f}/{s['top'][n][m]:.3f}" for m in METRICS))
    print("\nPaired differences on test (b - a, mean ± se), all pages / top pages:")
    for name, by_group in results["test"]["paired"].items():
        print(f"  {name}: " + "; ".join(
            f"{m} {by_group['all'][m]['mean']:+.3f} ± {by_group['all'][m]['se']:.3f} / "
            f"{by_group['top'][m]['mean']:+.3f} ± {by_group['top'][m]['se']:.3f}" for m in METRICS))
    print(f"Wrote {out_path} ({results['seconds']:,.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
