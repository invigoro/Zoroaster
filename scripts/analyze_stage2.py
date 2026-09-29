"""Analyses on top of `train_stage2.py`'s results.json (PLAN.md §5).

Works on whichever prompt variants the results file has.

- **Fair comparisons:** per-seed and pooled, on test examples whose prompt
  wasn't shortened in either compared variant. Longer prompts hit the token
  budget more often, and shortening removes the far end of the context,
  which biases the plain comparison against them.
- **Run-to-run shift:** where two variants' prompts are identical (no
  bursting neighbors; for `+relevant`, also no relevant sentence), any
  difference between their models is training noise, which the per-example
  standard errors don't cover.
- **Mechanism (conditions on the answer, so a diagnostic, not a score):**
  - trigger gains, by whether a shown neighbor's title appears in the
    inserted text;
  - snippet and relevant-sentence gains, by how much of the inserted text's
    vocabulary the shown text shares.

Usage:
    python scripts/analyze_stage2.py [results.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: F401  # isort: skip  (before pyarrow, see train_stage2.py)
import numpy as np
import pyarrow.parquet as pq

from scripts.train_stage2 import (CHANGES_DIR, EXAMPLES_DIR, MAX_CHANGES, OUT_DIR, TITLES_PATH, VARIANTS, attach_changes,
                                  attach_relevant, encode, paired, point_in_time_titles)
from src.stage2.examples import MAX_NEIGHBORS_SHOWN
from src.stage2.relevance import words as _words

PAIRS = [("context", "context+triggers"), ("context+triggers", "context+triggers+changes"),
         ("context", "context+triggers+changes"), ("context+triggers", "context+triggers+relevant"),
         ("context", "context+triggers+relevant")]


def title_in_target(row: dict) -> bool:
    """A shown bursting neighbor's title appears in the inserted text (typically as a new link)."""
    target = row["added_text"].lower().replace("_", " ")
    return any(t.replace("_", " ").lower() in target for t in row["bursting_neighbors"][:MAX_NEIGHBORS_SHOWN])


def _overlap(row: dict, shown: list[str]) -> float | None:
    target = _words(row["added_text"])
    return len(target & _words(" ".join(shown))) / len(target) if shown and target else None


def snippet_overlap(row: dict) -> float | None:
    """Share of the inserted text's words (4+ letters) found in the shown snippets; None without snippets."""
    return _overlap(row, [s for t in row["bursting_neighbors"] if (s := row["neighbor_changes"].get(t))][:MAX_CHANGES])


def relevant_overlap(row: dict) -> float | None:
    """The same, for the relevant sentences shown; None without any."""
    return _overlap(row, [text for _, text in row["relevant_changes"]])


def main(argv: list[str] | None = None) -> int:
    from transformers import AutoTokenizer

    argv = sys.argv[1:] if argv is None else argv
    results = json.loads(Path(argv[0] if argv else OUT_DIR / "results.json").read_text())
    scores = results["scores"]
    runs = [f"seed {s}" for s in results["seeds"]] + ["pooled"]
    variants = [v for v in VARIANTS if f"tuned / {v} / pooled" in scores]
    rows = pq.read_table(EXAMPLES_DIR).to_pylist()
    attach_changes(rows, CHANGES_DIR)
    point_in_time_titles(rows, TITLES_PATH)
    attach_relevant(rows, CHANGES_DIR)
    test = [r for r in rows if r["split"] == "test"]
    tokenizer = AutoTokenizer.from_pretrained(results["base_model"])
    shortened = {v: [encode(tokenizer, r, v)[2] for r in test] for v in variants}
    print(results["base_model"], "| test prompts shortened:", {v: sum(s) for v, s in shortened.items()})

    def report(a: str, b: str, groups: dict[str, list[bool]]) -> None:
        if a not in variants or b not in variants:
            return
        clean = [not (x or y) for x, y in zip(shortened[a], shortened[b])]
        print(f"\n{b} vs {a}, neither prompt shortened (mean ± se, share improved):")
        for name, group in groups.items():
            mask = [x and y for x, y in zip(group, clean)]
            cells = [f"{run}: {d['mean']:+.4f} ± {d['se']:.4f} ({d['share_improved']:.1%})"
                     for run in runs for d in [paired(scores[f"tuned / {a} / {run}"], scores[f"tuned / {b} / {run}"], mask)]]
            print(f"  {name:44} n={sum(mask):5}  " + " | ".join(cells))

    bursting = [bool(r["bursting_neighbors"]) for r in test]
    shown = [any(r["neighbor_changes"].get(t) for t in r["bursting_neighbors"]) for r in test]
    relevant = [bool(r["relevant_changes"]) for r in test]
    subsets = {"all": [True] * len(test), "bursting neighbors": bursting, "snippets shown": shown,
               "relevant shown": relevant, "no bursting neighbors": [not b for b in bursting]}
    for a, b in PAIRS:
        report(a, b, subsets)

    print("\nMechanism (splits on the answer):")
    in_target = [title_in_target(r) for r in test]
    report("context", "context+triggers", {
        "bursting neighbors, a title in the inserted text": [x and y for x, y in zip(bursting, in_target)],
        "bursting neighbors, no title in it": [x and not y for x, y in zip(bursting, in_target)],
    })
    for b, label, overlap_of in [("context+triggers+changes", "snippets", snippet_overlap),
                                 ("context+triggers+relevant", "relevant", relevant_overlap)]:
        if b not in variants:
            continue
        overlap = [overlap_of(r) for r in test]
        known = [o for o in overlap if o is not None]
        print(f"\nshare of inserted words found in the {label} text, 25/50/75/90th percentiles: "
              f"{np.percentile(known, [25, 50, 75, 90]).round(3).tolist()}")
        identical = [not b for b in bursting] if label == "snippets" else [not r for r in relevant]
        report("context+triggers", b, {
            f"{label} shown, no shared words": [o == 0 for o in overlap],
            f"{label} shown, shared words < 25%": [o is not None and 0 < o < 0.25 for o in overlap],
            f"{label} shown, shared words >= 25%": [o is not None and o >= 0.25 for o in overlap],
            f"none shown (identical prompts)": identical,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
