"""Analyses on top of `train_stage2.py`'s results.json (PLAN.md §5).

- **Fair comparisons:** per-seed and pooled, on test examples whose prompt
  wasn't shortened in either compared variant. Longer prompts hit the token
  budget more often, and shortening removes the far end of the context,
  which biases the plain comparison against them.
- **Run-to-run shift:** without bursting neighbors, the triggers and
  snippet prompts are identical. Any difference between their models there
  is training noise, which the per-example standard errors don't cover.
- **Mechanism (conditions on the answer, so a diagnostic, not a score):**
  - trigger gains, by whether a shown neighbor's title appears in the
    inserted text;
  - snippet gains, by how much of the inserted text's vocabulary the
    snippets share.

Usage:
    python scripts/analyze_stage2.py [results.json]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: F401  # isort: skip  (before pyarrow, see train_stage2.py)
import numpy as np
import pyarrow.parquet as pq

from scripts.train_stage2 import (BASE_MODEL, CHANGES_DIR, EXAMPLES_DIR, MAX_CHANGES, OUT_DIR, TITLES_PATH, VARIANTS,
                                  attach_changes, encode, paired, point_in_time_titles)
from src.stage2.examples import MAX_NEIGHBORS_SHOWN

_WORD = re.compile(r"[a-z][a-z'-]{3,}")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def title_in_target(row: dict) -> bool:
    """A shown bursting neighbor's title appears in the inserted text (typically as a new link)."""
    target = row["added_text"].lower().replace("_", " ")
    return any(t.replace("_", " ").lower() in target for t in row["bursting_neighbors"][:MAX_NEIGHBORS_SHOWN])


def snippet_overlap(row: dict) -> float | None:
    """Share of the inserted text's words (4+ letters) found in the shown snippets; None without snippets."""
    shown = [s for t in row["bursting_neighbors"] if (s := row["neighbor_changes"].get(t))][:MAX_CHANGES]
    target = _words(row["added_text"])
    return len(target & _words(" ".join(shown))) / len(target) if shown and target else None


def main(argv: list[str] | None = None) -> int:
    from transformers import AutoTokenizer

    argv = sys.argv[1:] if argv is None else argv
    results = json.loads(Path(argv[0] if argv else OUT_DIR / "results.json").read_text())
    scores = results["scores"]
    runs = [f"seed {s}" for s in results["seeds"]] + ["pooled"]
    rows = pq.read_table(EXAMPLES_DIR).to_pylist()
    attach_changes(rows, CHANGES_DIR)
    point_in_time_titles(rows, TITLES_PATH)
    test = [r for r in rows if r["split"] == "test"]
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    shortened = {v: [encode(tokenizer, r, v)[2] for r in test] for v in VARIANTS}
    print("test prompts shortened:", {v: sum(s) for v, s in shortened.items()})

    def report(a: str, b: str, groups: dict[str, list[bool]]) -> None:
        clean = [not (x or y) for x, y in zip(shortened[a], shortened[b])]
        print(f"\n{b} vs {a}, neither prompt shortened (mean ± se, share improved):")
        for name, group in groups.items():
            mask = [x and y for x, y in zip(group, clean)]
            cells = [f"{run}: {d['mean']:+.4f} ± {d['se']:.4f} ({d['share_improved']:.1%})"
                     for run in runs for d in [paired(scores[f"tuned / {a} / {run}"], scores[f"tuned / {b} / {run}"], mask)]]
            print(f"  {name:44} n={sum(mask):5}  " + " | ".join(cells))

    bursting = [bool(r["bursting_neighbors"]) for r in test]
    shown = [any(r["neighbor_changes"].get(t) for t in r["bursting_neighbors"]) for r in test]
    subsets = {"all": [True] * len(test), "bursting neighbors": bursting, "snippets shown": shown,
               "no bursting neighbors": [not b for b in bursting]}
    for a, b in [("context", "context+triggers"), ("context+triggers", "context+triggers+changes"),
                 ("context", "context+triggers+changes")]:
        report(a, b, subsets)

    print("\nMechanism (splits on the answer):")
    in_target = [title_in_target(r) for r in test]
    report("context", "context+triggers", {
        "bursting neighbors, a title in the inserted text": [x and y for x, y in zip(bursting, in_target)],
        "bursting neighbors, no title in it": [x and not y for x, y in zip(bursting, in_target)],
    })
    overlap = [snippet_overlap(r) for r in test]
    known = [o for o in overlap if o is not None]
    print(f"\nshare of inserted words found in the snippets, 25/50/75/90th percentiles: "
          f"{np.percentile(known, [25, 50, 75, 90]).round(3).tolist()}")
    report("context+triggers", "context+triggers+changes", {
        "snippets shown, no shared words": [o == 0 for o in overlap],
        "snippets shown, shared words < 25%": [o is not None and 0 < o < 0.25 for o in overlap],
        "snippets shown, shared words >= 25%": [o is not None and o >= 0.25 for o in overlap],
        "no bursting neighbors (identical prompts)": [not b for b in bursting],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
