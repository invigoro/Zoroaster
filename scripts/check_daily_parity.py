"""Feature-level parity: the daily job's features from recent changes vs the dumps'.

The daily job computes features with the training code
(`src/deploy/daily.py`), so the only difference is the input source. This
measures what that difference does to the features and the ranking. It uses
the window both sources cover
(`data/processed/enwiki/daily/rc_dump_overlap.parquet`), 2026-08-31T15:00Z
to the end of that day. Features for D = 2026-09-01 are computed for every
page edited in that window:
- once from the dump alone;
- once with the window's dump rows swapped for recent changes'.

It reports, per feature, how many pages differ, and how much the burst
model's top pages change. Both runs get the dump's full-day site total as
context, so only the per-page inputs differ.

Usage:
    python scripts/check_daily_parity.py [--workers 8]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from datetime import date
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from scripts import build_enwiki_features as enwiki
from scripts.build_enwiki_revisions import OUT_DIR as REVISIONS_DIR, bucket_files
from scripts.check_recent_changes_parity import OVERLAP_PATH
from src.deploy.daily import COLUMNS, N_BUCKETS, bucket_features, split_by_bucket
from src.features.activity import FEATURE_NAMES
from src.stage1.features import FEATURE_SETS, add_context_columns, feature_matrix, site_edit_totals

DAY = date(2026, 9, 1)
WINDOW_START = "2026-08-31T15:00:00Z"
DAY_START = "2026-09-01T00:00:00Z"
MODEL_SET = "habits+burst"
TOP = (100, 1000)


def _job(args: tuple) -> None:
    bucket, dump_window_dir, rc_dir, out_dir = args
    files = bucket_files(bucket)
    for name, live_dir in (("dump", dump_window_dir), ("rc", rc_dir)):
        bucket_features(DAY, files, live_dir / f"bucket={bucket:03d}.parquet",
                        out_dir / name / f"part-{bucket:03d}.parquet", dump_until=WINDOW_START)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    start = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rc = pq.read_table(OVERLAP_PATH, columns=list(COLUMNS))
        rc = rc.filter(pc.and_(pc.greater_equal(rc["timestamp"], WINDOW_START), pc.less(rc["timestamp"], DAY_START)))
        pq.write_table(rc, tmp / "rc.parquet")
        dump = ds.dataset(REVISIONS_DIR, format="parquet", partitioning="hive").to_table(
            columns=list(COLUMNS),
            filter=(pc.field("timestamp") >= WINDOW_START) & (pc.field("timestamp") < DAY_START))
        pq.write_table(dump, tmp / "dump.parquet")
        print(f"window {WINDOW_START} .. {DAY_START}: {rc.num_rows:,} recent-changes rows, {dump.num_rows:,} dump rows")
        split_by_bucket([tmp / "rc.parquet"], tmp / "rc")
        split_by_bucket([tmp / "dump.parquet"], tmp / "dump_window")
        for name in ("dump", "rc"):
            (tmp / "out" / name).mkdir(parents=True)
        with Pool(args.workers) as pool:
            pool.map(_job, [(b, tmp / "dump_window", tmp / "rc", tmp / "out") for b in range(N_BUCKETS)])
        a = pq.read_table(tmp / "out" / "dump").sort_by("page_id")
        b = pq.read_table(tmp / "out" / "rc").sort_by("page_id")
    print(f"features in {time.monotonic() - start:,.0f}s: {a.num_rows:,} pages from the dump, {b.num_rows:,} with "
          f"recent changes")
    common = np.intersect1d(a["page_id"].to_numpy(), b["page_id"].to_numpy())
    print(f"  in both: {len(common):,}; only from the dump: {a.num_rows - len(common):,}; "
          f"only with recent changes: {b.num_rows - len(common):,}")
    a = a.filter(pc.is_in(a["page_id"], value_set=pa.array(common))).sort_by("page_id")
    b = b.filter(pc.is_in(b["page_id"], value_set=pa.array(common))).sort_by("page_id")
    any_diff = np.zeros(len(common), dtype=bool)
    for name in FEATURE_NAMES:
        x, y = a[name].to_pylist(), b[name].to_pylist()
        diff = np.array([u != v for u, v in zip(x, y)])
        any_diff |= diff
        if diff.any():
            print(f"  {name:22} differs on {diff.sum():6,} pages ({diff.mean():.2%})")
    print(f"  any feature differs on {any_diff.sum():,} of {len(common):,} pages ({any_diff.mean():.2%})")

    site = site_edit_totals(pq.read_table(enwiki.SITE_PATH, columns=["date", "edits"]))
    model = lgb.Booster(model_file=str(enwiki.OUT / "models" / f"stage1_burst2_{MODEL_SET.replace('+', '_')}.txt"))
    columns = FEATURE_SETS[MODEL_SET]
    scores = [model.predict(feature_matrix(add_context_columns(t, site), columns)) for t in (a, b)]
    print(f"burst model ({MODEL_SET}) scores: {np.mean(scores[0] != scores[1]):.2%} of pages change; "
          f"rank correlation {np.corrcoef(np.argsort(np.argsort(-scores[0])), np.argsort(np.argsort(-scores[1])))[0, 1]:.4f}")
    for k in TOP:
        top_a, top_b = set(np.argsort(-scores[0])[:k]), set(np.argsort(-scores[1])[:k])
        print(f"  top {k}: {len(top_a & top_b)} of {k} pages in both rankings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
