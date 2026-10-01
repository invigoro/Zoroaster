"""Baselines for version 2's structured forecasts (PLAN.md §6 step 10, phase 2).

A model has to beat these to be worth showing (`src/forecast/metrics.py`):
- "yesterday again": yesterday's main sections and kinds of change, since
  ongoing stories continue. A quiet yesterday falls back to the next one.
- "most common": the lead, and every kind of change found on most training
  page-days.
Each is scored on the validation and test splits, overall and by
selection (the burst model's top pages vs random ones) and for pages about
living people. Writes `data/processed/enwiki/v2/baselines.json`.

Usage:
    python scripts/v2_baselines.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq

from scripts.fetch_v2_examples import EXAMPLES_DIR
from scripts.build_v2_targets import OUT_DIR
from src.forecast.changes import KINDS, LEAD
from src.forecast.metrics import main_sections, score, summarize

COLUMNS = ["split", "selection", "living", "sections", "section_chars", "kinds", "yesterday_sections",
           "yesterday_section_chars", "yesterday_kinds"]


def common_kinds(rows: list[dict]) -> list[str]:
    """The kinds found on most of `rows`."""
    counts = Counter(k for r in rows for k in r["kinds"])
    return [k for k in KINDS if counts[k] > len(rows) / 2]


def forecasts(row: dict, common: list[str]) -> dict[str, dict]:
    fallback = {"sections": [LEAD], "kinds": common}
    yesterday = ({"sections": main_sections(row["yesterday_sections"], row["yesterday_section_chars"]),
                  "kinds": row["yesterday_kinds"]} if row["yesterday_sections"] else fallback)
    return {"yesterday again": yesterday, "most common": fallback}


def evaluate(rows: list[dict], common: list[str]) -> dict:
    groups = {"all": rows, "top": [r for r in rows if r["selection"] == "top"],
              "random": [r for r in rows if r["selection"] == "random"], "living": [r for r in rows if r["living"]]}
    out = {}
    for name, members in groups.items():
        out[name] = {"examples": len(members)}
        for baseline in ("yesterday again", "most common"):
            out[name][baseline] = summarize([score(forecasts(r, common)[baseline], r) for r in members])
    return out


def main() -> int:
    rows = pq.read_table(EXAMPLES_DIR, columns=COLUMNS).to_pylist()
    common = common_kinds([r for r in rows if r["split"] == "train"])
    results = {"common_kinds": common}
    print(f"kinds on most training page-days: {common}")
    for split in ("validation", "test"):
        results[split] = evaluate([r for r in rows if r["split"] == split], common)
        print(f"\n{split}:")
        for group, by_baseline in results[split].items():
            for baseline in ("yesterday again", "most common"):
                m = by_baseline[baseline]
                print(f"  {group:7} n={by_baseline['examples']:5,}  {baseline:16} section precision "
                      f"{m['section_precision']:.3f}, main section {m['main_section_hit']:.3f}, "
                      f"kinds Jaccard {m['kinds_jaccard']:.3f}")
    (OUT_DIR / "baselines.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'baselines.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
