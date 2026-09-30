"""Score a day's predictions against what actually burst that day.

Run once day D is over and its live file is fetched. It takes the top pages
of D's ranking (`daily_predictions.py`), and as a baseline the top pages by
"most edits yesterday", as in the Stage 1 harness. For each, it checks
whether the page burst on D by the Stage 1 target (`src.deploy.daily.bucket_outcomes`).
It reports precision at 10, 50, 100 and 1000, and writes
`predictions/D.outcomes.json`, which the page shows as the hit rate.

Mass editors come from D's live records (`live_mass_editors`), an
approximation of training's definition that also counts reverts.

Usage:
    python scripts/score_predictions.py --day 2026-09-07 [--top 1000] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import date, timedelta
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow.parquet as pq

from scripts.build_enwiki_revisions import bucket_files
from scripts.daily_predictions import PREDICTIONS_DIR, WORK_DIR
from scripts.fetch_recent_changes import LIVE_DIR
from src.deploy.daily import LIVE_START, N_BUCKETS, bucket_outcomes, live_files, live_mass_editors, split_by_bucket
from src.stage1.baselines import BASELINES

AT = (10, 50, 100, 1000)
SEED = 1234


def _job(args: tuple) -> list[dict]:
    bucket, day, work, pages, mass = args
    return bucket_outcomes(day, bucket_files(bucket), work / f"bucket={bucket:03d}.parquet", pages, mass)


def top_pages(scores: np.ndarray, page_ids: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    """The `k` highest-scoring pages, ties broken at random (as the harness's metrics do)."""
    order = np.lexsort((rng.random(len(scores)), -scores))
    return page_ids[order[:k]].tolist()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--day", type=date.fromisoformat, required=True)
    parser.add_argument("--top", type=int, default=max(AT))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    day, start = args.day, time.monotonic()
    predictions = pq.read_table(PREDICTIONS_DIR / f"{day.isoformat()}.parquet")
    ids = predictions["page_id"].to_numpy()
    rng = np.random.default_rng(SEED)
    rankings = {
        "model": top_pages(predictions["score"].to_numpy(), ids, args.top, rng),
        "edits yesterday": top_pages(BASELINES["edits yesterday"](predictions), ids, args.top, rng),
    }
    pages = sorted(set().union(*rankings.values()))
    work = WORK_DIR / f"outcomes-{day.isoformat()}"
    shutil.rmtree(work, ignore_errors=True)
    split_by_bucket(live_files(LIVE_DIR, LIVE_START, day), work)
    mass = live_mass_editors(LIVE_DIR / f"{day.isoformat()}.parquet", day)
    by_bucket: dict[int, list[int]] = {}
    for page in pages:
        by_bucket.setdefault(page % N_BUCKETS, []).append(page)
    with Pool(args.workers) as pool:
        outcomes = [o for part in pool.map(_job, [(b, day, work, p, mass) for b, p in by_bucket.items()]) for o in part]
    shutil.rmtree(work, ignore_errors=True)
    burst = {o["page_id"] for o in outcomes if o["burst"]}
    titles = dict(zip(ids.tolist(), predictions["page_title"].to_pylist()))
    payload = {
        "date": day.isoformat(),
        "target": "a burst of edits by 2+ editors (mass editors left out)",
        "mass_editors": len(mass),
        "precision": {name: {str(k): sum(p in burst for p in ranked[:k]) / k for k in AT if k <= args.top}
                      for name, ranked in rankings.items()},
        "hits": [titles[p].replace("_", " ") for p in rankings["model"][:100] if p in burst],
    }
    (PREDICTIONS_DIR / f"{day.isoformat()}.outcomes.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{day}: {len(pages):,} pages checked, {len(mass)} mass editors ({time.monotonic() - start:,.0f}s)")
    for name, by_k in payload["precision"].items():
        print(f"  {name:16} " + "  ".join(f"P@{k} {v:.3f}" for k, v in by_k.items()))
    print(f"  the model's top 100 that burst: {len(payload['hits'])}: " + "; ".join(payload["hits"][:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
