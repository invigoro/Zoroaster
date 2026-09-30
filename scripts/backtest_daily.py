"""Backtest the daily job: predict and score each day in a range.

For each day D, this runs `daily_predictions.py --day D`, then
`score_predictions.py --day D`, which needs D's own live file, so the
range must end before today. It then averages precision@k over the days,
model vs "most edits yesterday", with the paired per-day difference, and
writes `predictions/backtest-FIRST-LAST.json`.

Days before LIVE_START + LOOKBACK_DAYS have less than a full live lookback,
so their candidates miss pages last edited before LIVE_START.

Usage:
    python scripts/backtest_daily.py --days 2026-09-08 2026-09-29 [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from scripts import daily_predictions, score_predictions
from scripts.daily_predictions import PREDICTIONS_DIR


def summarize(outcomes: list[dict]) -> dict:
    """Mean precision@k per ranking, and the model's paired per-day gain over the baseline."""
    ks = list(outcomes[0]["precision"]["model"])
    summary: dict = {"days": len(outcomes), "mean_precision": {}, "model_minus_baseline": {}}
    for name in outcomes[0]["precision"]:
        summary["mean_precision"][name] = {k: float(np.mean([o["precision"][name][k] for o in outcomes])) for k in ks}
    for k in ks:
        diffs = np.array([o["precision"]["model"][k] - o["precision"]["edits yesterday"][k] for o in outcomes])
        se = float(diffs.std(ddof=1) / math.sqrt(len(diffs))) if len(diffs) > 1 else float("nan")
        summary["model_minus_baseline"][k] = {"mean": float(diffs.mean()), "se": se,
                                             "days_better": int((diffs > 0).sum()), "days_worse": int((diffs < 0).sum())}
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", nargs=2, type=date.fromisoformat, required=True, metavar=("FIRST", "LAST"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    first, last = args.days
    outcomes = []
    for i in range((last - first).days + 1):
        day = (first + timedelta(days=i)).isoformat()
        if not (PREDICTIONS_DIR / f"{day}.parquet").exists():
            daily_predictions.main(["--day", day, "--workers", str(args.workers)])
        score_predictions.main(["--day", day, "--workers", str(args.workers)])
        outcomes.append(json.loads((PREDICTIONS_DIR / f"{day}.outcomes.json").read_text(encoding="utf-8")))
    summary = summarize(outcomes) | {"first": first.isoformat(), "last": last.isoformat()}
    (PREDICTIONS_DIR / f"backtest-{first}-{last}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nBacktest {first} .. {last} ({summary['days']} days), mean precision:")
    for name, by_k in summary["mean_precision"].items():
        print(f"  {name:16} " + "  ".join(f"P@{k} {v:.3f}" for k, v in by_k.items()))
    for k, d in summary["model_minus_baseline"].items():
        print(f"  model - baseline, P@{k}: {d['mean']:+.3f} ± {d['se']:.3f} ({d['days_better']}/{d['days_worse']} days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
