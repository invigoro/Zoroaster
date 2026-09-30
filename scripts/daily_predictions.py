"""Predict tomorrow's bursts on English Wikipedia: the local daily job.

PLAN.md §6 step 9, run on this machine. For prediction day D (default:
today, UTC), shortly after 00:00 UTC:
1. Fetch any live days from LIVE_START to D-1 that aren't fetched yet,
   oldest first (`fetch_recent_changes.py`). Recent changes keep 30 days.
2. Split them by page bucket, and compute every candidate page's features
   for D from dump history plus live days, one bucket per worker
   (`src/deploy/daily.py`).
3. Add the same context columns as training, and score the pages with the
   Stage 1 burst model (`train_stage1.py --target burst`).
4. Write every candidate's features and score to
   `data/processed/enwiki/predictions/D.parquet`, and the top pages to
   `D.json`, which the page will show.

Usage:
    python scripts/daily_predictions.py [--day 2026-09-30] [--workers 8] [--top 100]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import date, datetime, timedelta, timezone
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

from scripts import build_enwiki_features as enwiki
from scripts.build_enwiki_revisions import bucket_files
from scripts.fetch_recent_changes import LIVE_DIR, day_files, fetch_window
from src.deploy.daily import LIVE_START, N_BUCKETS, bucket_features, live_files, split_by_bucket
from src.stage1.features import FEATURE_SETS, add_context_columns, feature_matrix

MODEL_SET = "habits+burst"
MODEL_PATH = enwiki.OUT / "models" / f"stage1_burst2_{MODEL_SET.replace('+', '_')}.txt"
PREDICTIONS_DIR = enwiki.OUT / "predictions"
WORK_DIR = enwiki.OUT / "daily_work"
SHOWN = ("edits_1d", "edits_7d", "editors_1d", "burst_z_1d", "is_burst_1d", "page_age_days")


def _bucket_job(args: tuple) -> int:
    bucket, day, work = args
    return bucket_features(day, bucket_files(bucket), work / "live" / f"bucket={bucket:03d}.parquet",
                           work / "features" / f"part-{bucket:03d}.parquet")


def fetch_missing(day: date) -> None:
    todo = day_files(LIVE_START, day - timedelta(days=1), LIVE_DIR)
    oldest_available = datetime.now(timezone.utc).date() - timedelta(days=30)
    if todo and todo[0][0] < oldest_available:
        raise RuntimeError(f"{todo[0][0]} is older than recent changes keep; it needs the next history dump")
    session = requests.Session()
    for d, path in todo:
        fetch_window(f"{d.isoformat()}T00:00:00Z", f"{(d + timedelta(days=1)).isoformat()}T00:00:00Z", path, session)


def score(table: pa.Table, day: date) -> np.ndarray:
    """Burst-model scores, with site-wide context computed from the candidates themselves: every page
    edited on D-1 is a candidate, so their `edits_1d` sum to the site's total."""
    site_total = pc.sum(table["edits_1d"]).as_py() or 0
    table = add_context_columns(table, {(day - timedelta(days=1) - date(1970, 1, 1)).days: site_total})
    model = lgb.Booster(model_file=str(MODEL_PATH))
    return model.predict(feature_matrix(table, FEATURE_SETS[MODEL_SET]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--day", type=date.fromisoformat, default=datetime.now(timezone.utc).date())
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args(argv)
    day, start = args.day, time.monotonic()
    fetch_missing(day)
    work = WORK_DIR / day.isoformat()
    shutil.rmtree(work, ignore_errors=True)
    (work / "features").mkdir(parents=True)
    split_by_bucket(live_files(LIVE_DIR, LIVE_START, day - timedelta(days=1)), work / "live")
    print(f"live days {LIVE_START} .. {day - timedelta(days=1)} split by bucket ({time.monotonic() - start:,.0f}s)",
          flush=True)
    with Pool(args.workers) as pool:
        pages = 0
        for done, n in enumerate(pool.imap_unordered(_bucket_job, [(b, day, work) for b in range(N_BUCKETS)]), 1):
            pages += n
            if done % 16 == 0:
                print(f"  {done}/{N_BUCKETS} buckets, {pages:,} pages ({time.monotonic() - start:,.0f}s)", flush=True)
    table = pq.read_table(work / "features")
    scores = score(table, day)
    table = table.append_column("score", pa.array(scores))
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, PREDICTIONS_DIR / f"{day.isoformat()}.parquet", compression="zstd")
    top = table.take(np.argsort(-scores, kind="stable")[: args.top]).to_pylist()
    payload = {
        "date": day.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": f"stage1 burst target, {MODEL_SET}",
        "target": "a burst of edits by 2+ editors (mass editors left out) on this UTC day",
        "candidates": table.num_rows,
        "site_edits_yesterday": pc.sum(table["edits_1d"]).as_py(),
        "pages": [{"rank": i + 1, "page_id": r["page_id"], "title": r["page_title"].replace("_", " "),
                   "score": round(r["score"], 5)} | {k: r[k] for k in SHOWN} for i, r in enumerate(top)],
    }
    (PREDICTIONS_DIR / f"{day.isoformat()}.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                                                              encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    print(f"{table.num_rows:,} candidates scored for {day}; top {args.top} -> "
          f"{PREDICTIONS_DIR / (day.isoformat() + '.json')} ({time.monotonic() - start:,.0f}s)")
    for r in payload["pages"][:15]:
        print(f"  {r['rank']:3}. {r['title']:50} {r['score']:.4f}  (edits yesterday {r['edits_1d']}, z {r['burst_z_1d']:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
