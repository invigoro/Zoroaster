"""Stage 1 harness: next-day edit forecasting on the Simple Wikipedia test corpus.

For each prediction day D, rank every existing page by how likely it is to
get a kept edit on D (non-bot, not a revert, never reverted), using only
information from before D.

- Heuristic baselines (`src.stage1.baselines`): yesterday's edits, the last
  30 days' edits, recency, last year's edits.
- LightGBM on three nested feature sets (`src.stage1.features`): page
  habits, then the page's own bursts, then co-burst. Each is trained on the
  panel's train window and early-stopped on the validation window.
  Training is unweighted: the panel is a case-control sample, and sample
  weights would mostly shift the intercept, which per-day ranking ignores.
- Everything is scored on the full-day evaluation set (every page on 29 test
  days) with per-day ranking metrics (`src.stage1.metrics`).

Needs `build_test_features.py` and `build_test_eval_days.py` to have run.
Writes `data/processed/simplewiki_test_stage1_results.json` and saves the
models to `data/processed/models/`.

Usage:
    python scripts/train_stage1.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from scripts.build_test_eval_days import OUTPUT_PATH as EVAL_PATH
from scripts.build_test_features import DAILY_PATH, PANEL_META_PATH, PANEL_PATH
from src.stage1.baselines import BASELINES
from src.stage1.features import CATEGORICAL_FEATURES, FEATURE_SETS, add_context_columns, feature_matrix, site_edit_totals
from src.stage1.metrics import METRIC_NAMES, average, paired_difference, per_day_metrics
from src.stage1.splits import Splits, make_splits

RESULTS_PATH = Path("data/processed/simplewiki_test_stage1_results.json")
MODEL_DIR = Path("data/processed/models")

SEED = 1234
PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": SEED,
    "deterministic": True,
    "force_col_wise": True,
    "verbose": -1,
}
NUM_BOOST_ROUND = 3000
EARLY_STOPPING_ROUNDS = 100

# (baseline, candidate) pairs compared day by day: the model against the best
# heuristic, then each feature set against the one it extends.
COMPARISONS = [
    ("edits yesterday", "habits"),
    ("habits", "habits+burst"),
    ("habits+burst", "habits+burst+co_burst"),
]


def load(path: Path, site_totals: dict[int, int]) -> pa.Table:
    return add_context_columns(pq.read_table(path), site_totals)


def split_panel(panel: pa.Table, splits: Splits) -> tuple[pa.Table, pa.Table]:
    train = panel.filter(splits.train.mask(panel["date"]))
    validation = panel.filter(splits.validation.mask(panel["date"]))
    for name, part in (("train", train), ("validation", validation)):
        assert pc.all(part["label_is_final"]).as_py(), f"{name} window has immature labels"
    return train, validation


def labels(table: pa.Table) -> np.ndarray:
    return pc.cast(table["y"], pa.int8()).to_numpy(zero_copy_only=False)


def train_model(name: str, columns: tuple[str, ...], train: pa.Table, validation: pa.Table) -> lgb.Booster:
    categorical = [c for c in CATEGORICAL_FEATURES if c in columns]
    train_set = lgb.Dataset(
        feature_matrix(train, columns), labels(train), feature_name=list(columns), categorical_feature=categorical
    )
    validation_set = lgb.Dataset(feature_matrix(validation, columns), labels(validation), reference=train_set)
    start = time.monotonic()
    booster = lgb.train(
        PARAMS,
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[validation_set],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False)],
    )
    print(f"  {name}: {booster.best_iteration} rounds in {time.monotonic() - start:,.0f}s")
    return booster


def main() -> int:
    meta = json.loads(PANEL_META_PATH.read_text())
    splits = make_splits(date.fromisoformat(meta["labels_final_through"]))
    for name in ("train", "validation", "test"):
        window = getattr(splits, name)
        print(f"{name:>10}: {window.start} .. {window.end}")

    site_totals = site_edit_totals(pq.read_table(DAILY_PATH, columns=["date", "edits"]))
    train, validation = split_panel(load(PANEL_PATH, site_totals), splits)
    evaluation = load(EVAL_PATH, site_totals)
    y_eval = labels(evaluation)
    eval_days = pc.cast(evaluation["date"], pa.int32()).to_numpy()
    positives = pc.filter(evaluation, evaluation["y"])
    context = {
        "splits": {n: [str(getattr(splits, n).start), str(getattr(splits, n).end)] for n in ("train", "validation", "test")},
        "train_rows": train.num_rows,
        "validation_rows": validation.num_rows,
        "eval_days": len(np.unique(eval_days)),
        "eval_rows": evaluation.num_rows,
        "eval_positives_per_day": positives.num_rows / len(np.unique(eval_days)),
        "base_rate": positives.num_rows / evaluation.num_rows,
        "positives_with_no_edits_in_30d": pc.mean(pc.cast(pc.equal(positives["edits_30d"], 0), pa.float64())).as_py(),
        "positives_with_no_edits_in_365d": pc.mean(pc.cast(pc.equal(positives["edits_365d"], 0), pa.float64())).as_py(),
    }
    print(
        f"train {train.num_rows:,} rows, validation {validation.num_rows:,}; eval {evaluation.num_rows:,} rows over "
        f"{context['eval_days']} days ({context['eval_positives_per_day']:,.0f} positives/day, "
        f"base rate {context['base_rate']:.3%})"
    )

    results: dict[str, dict] = {}
    for name, score in BASELINES.items():
        per_day = per_day_metrics(score(evaluation), y_eval, eval_days, SEED)
        results[name] = {"kind": "baseline", "metrics": average(per_day), "per_day": per_day}

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("Training LightGBM:")
    for name, columns in FEATURE_SETS.items():
        booster = train_model(name, columns, train, validation)
        booster.save_model(str(MODEL_DIR / f"stage1_{name.replace('+', '_')}.txt"))
        scores = booster.predict(feature_matrix(evaluation, columns), num_iteration=booster.best_iteration)
        per_day = per_day_metrics(scores, y_eval, eval_days, SEED)
        gain = booster.feature_importance("gain")
        results[name] = {
            "kind": "lightgbm",
            "metrics": average(per_day),
            "per_day": per_day,
            "best_iteration": booster.best_iteration,
            "validation": {k: float(v) for k, v in booster.best_score["valid_0"].items()},
            "importance_share": {c: float(g / gain.sum()) for c, g in sorted(zip(columns, gain), key=lambda cg: -cg[1])},
        }

    comparisons = {
        f"{candidate} vs {baseline}": {
            m: paired_difference(results[baseline]["per_day"], results[candidate]["per_day"], m) for m in METRIC_NAMES
        }
        for baseline, candidate in COMPARISONS
    }
    RESULTS_PATH.write_text(json.dumps({"context": context, "results": results, "comparisons": comparisons}, indent=2))

    header = ["model"] + list(METRIC_NAMES)
    print("\n| " + " | ".join(header) + " |\n|" + "---|" * len(header))
    for name, result in results.items():
        print(f"| {name} | " + " | ".join(f"{result['metrics'][m]:.4f}" for m in METRIC_NAMES) + " |")
    print("\nPaired per-day differences (mean ± standard error; days better/worse):")
    for label, by_metric in comparisons.items():
        cells = [
            f"{m} {d['mean']:+.4f} ± {d['se']:.4f} ({d['days_better']}/{d['days_worse']})"
            for m, d in by_metric.items()
            if m in ("precision@100", "average_precision")
        ]
        print(f"  {label}: " + "; ".join(cells))
    print(f"\nWrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
