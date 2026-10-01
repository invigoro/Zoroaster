"""Select version 2's page-days: pages like the ones the site forecasts, that did get edited.

Version 2 forecasts what a page gains over day D (PLAN.md §6 step 10). Its
examples come from the Stage 1 rows, scored by the burst model
(`train_stage1.py --target burst`), among page-days with a kept edit on D:
- each day's top-scored pages, like the ones the site forecasts;
- plus a random sample of the rest, for contrast.
Train and validation come from the panel's windows; test from the 14
evaluation days, after the base model's training data. The panel is a 20%
page sample, so its top 60 a day is about the whole wiki's top 300.

Each page-day gets three revision ids:
- `start_id`: the page at the end of D-2, the start of "yesterday's change";
- `prompt_id`: the page at the end of D-1, what a forecaster sees;
- `end_id`: the last revision on D that was never reverted. Vandalism
  reverted within D cancels out of the day's net change anyway; this also
  keeps out vandalism made late on D and reverted after midnight.

Writes `data/processed/enwiki/v2/targets.parquet`.

Usage:
    python scripts/build_v2_targets.py
"""

from __future__ import annotations

import json
import sys
import time
from bisect import bisect_left
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from scripts import build_enwiki_features as enwiki
from scripts.build_enwiki_labels import LABELS_DIR
from scripts.build_stage2_targets import FEATURES, bursting_neighbors
from scripts.train_stage1 import load
from src.stage1.features import FEATURE_SETS, feature_matrix, site_edit_totals
from src.stage1.splits import make_splits

OUT_DIR = Path("data/processed/enwiki/v2")
TARGETS_PATH = OUT_DIR / "targets.parquet"
MODEL_SET = "habits+burst"
MODEL_PATH = enwiki.OUT / "models" / f"stage1_burst2_{MODEL_SET.replace('+', '_')}.txt"
PER_DAY = {"train": (60, 20), "validation": (5, 2), "test": (150, 50)}  # (top-scored, random) edited pages a day
SEED = 1234


def choose(table: pa.Table, scores: np.ndarray, top: int, extra: int, rng: np.random.Generator) -> pa.Table:
    """Per day: the `top` highest-scored edited rows, then `extra` random ones from the rest."""
    days = pc.cast(table["date"], pa.int32()).to_numpy()
    picks, kinds = [], []
    for day in np.unique(days):
        rows = np.flatnonzero(days == day)
        order = rows[np.argsort(-scores[rows], kind="stable")]
        picks += order[:top].tolist()
        kinds += ["top"] * len(order[:top])
        rest = order[top:]
        sampled = rng.choice(rest, size=min(extra, len(rest)), replace=False) if len(rest) else []
        picks += sorted(sampled)
        kinds += ["random"] * len(sampled)
    chosen = table.take(pa.array(picks, pa.int64()))
    return chosen.append_column("score", pa.array(scores[picks])).append_column("selection", pa.array(kinds))


def day_bounds(page_days: list[tuple[int, date]], revisions: dict[int, list[tuple[str, int, bool]]]) -> list[dict]:
    """For each (page, D): its revision ids at the ends of D-2 and D-1, and its last never-reverted one on D.

    `revisions`: each page's (timestamp, revision id, is_reverted), sorted by timestamp.
    """
    out = []
    for page, day in page_days:
        revs = revisions.get(page, [])
        stamps = [r[0] for r in revs]
        before = lambda d: bisect_left(stamps, f"{d.isoformat()}T00:00:00Z")
        d1, d0, d2 = before(day - timedelta(days=1)), before(day), before(day + timedelta(days=1))
        kept_on_day = [r[1] for r in revs[d0:d2] if not r[2]]
        out.append({
            "start_id": revs[d1 - 1][1] if d1 else None,
            "prompt_id": revs[d0 - 1][1] if d0 else None,
            "end_id": kept_on_day[-1] if kept_on_day else None,
            "edits_on_day": d2 - d0,
        })
    return out


def main() -> int:
    start = time.monotonic()
    meta = json.loads(enwiki.META_PATH.read_text())
    splits = make_splits(date.fromisoformat(meta["labels_final_through"]), **enwiki.SPLIT_DAYS)
    site = site_edit_totals(pq.read_table(enwiki.SITE_PATH, columns=["date", "edits"]))
    model = lgb.Booster(model_file=str(MODEL_PATH))
    columns = FEATURE_SETS[MODEL_SET]
    rng = np.random.default_rng(SEED)
    panel, evaluation = load(enwiki.PANEL_DIR, site), load(enwiki.EVAL_DIR, site)
    parts = []
    for split, rows in [("train", panel.filter(splits.train.mask(panel["date"]))),
                        ("validation", panel.filter(splits.validation.mask(panel["date"]))),
                        ("test", evaluation)]:
        rows = rows.filter(rows["y"])  # a kept edit on D, so there's a change to forecast
        scores = model.predict(feature_matrix(rows, columns))
        chosen = choose(rows.select(["page_id", "date", *FEATURES]), scores, *PER_DAY[split], rng)
        parts.append(chosen.append_column("split", pa.array([split] * chosen.num_rows)))
        print(f"  {split}: {chosen.num_rows:,} of {rows.num_rows:,} edited page-days "
              f"({time.monotonic() - start:,.0f}s)", flush=True)
    selected = pa.concat_tables(parts)

    pages = pa.array(sorted(set(selected["page_id"].to_pylist())), pa.int64())
    labels = ds.dataset(LABELS_DIR, format="parquet").to_table(
        columns=["page_id", "page_title", "revision_id", "timestamp", "is_reverted"],
        filter=pc.field("page_id").isin(pages))
    labels = labels.sort_by([("page_id", "ascending"), ("timestamp", "ascending"), ("revision_id", "ascending")])
    revisions: dict[int, list] = {}
    titles: dict[int, str] = {}
    for page, title, rev, stamp, reverted in zip(*(labels[c].to_pylist() for c in
                                                  ("page_id", "page_title", "revision_id", "timestamp", "is_reverted"))):
        revisions.setdefault(page, []).append((stamp, rev, reverted))
        titles[page] = title
    bounds = day_bounds(list(zip(selected["page_id"].to_pylist(), selected["date"].to_pylist())), revisions)
    for name in ("start_id", "prompt_id", "end_id", "edits_on_day"):
        selected = selected.append_column(name, pa.array([b[name] for b in bounds], pa.int64()))
    selected = selected.append_column("page_title", pa.array([titles.get(p) for p in selected["page_id"].to_pylist()]))
    selected = selected.filter(pc.and_(pc.is_valid(selected["prompt_id"]), pc.is_valid(selected["end_id"])))
    selected = selected.append_column("bursting_neighbors", pa.array(bursting_neighbors(selected), pa.list_(pa.string())))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(selected, TARGETS_PATH)
    print(f"{selected.num_rows:,} page-days, {sum(1 for n in selected['bursting_neighbors'].to_pylist() if n):,} "
          f"with a bursting linked page, in {time.monotonic() - start:,.0f}s -> {TARGETS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
