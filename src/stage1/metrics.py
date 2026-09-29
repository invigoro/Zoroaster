"""Per-day ranking metrics for "which pages get a kept edit tomorrow?".

Each evaluation day, every existing page is ranked by score. Metrics are
computed within the day and then averaged over days (macro), so a busy day
doesn't dominate. Ties are broken randomly with a fixed seed: heuristic
baselines tie massively (e.g. most pages had zero edits yesterday), and a
fixed order would bias them.
"""

from __future__ import annotations

import numpy as np

PRECISION_KS = (100, 1000)
RECALL_KS = (1000, 10000)
METRIC_NAMES = (
    [f"precision@{k}" for k in PRECISION_KS]
    + [f"recall@{k}" for k in RECALL_KS]
    + ["average_precision"]
)


def day_metrics(scores: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    """Ranking metrics for one day. Recall and average precision are NaN on
    a day with no positives; precision@k counts missing ranks as misses."""
    order = np.lexsort((rng.random(len(scores)), -scores))
    ranked = labels[order].astype(np.int64)
    hits = np.cumsum(ranked)
    n_pos = int(hits[-1]) if len(hits) else 0
    out = {}
    for k in PRECISION_KS:
        out[f"precision@{k}"] = hits[min(k, len(hits)) - 1] / k if len(hits) else 0.0
    for k in RECALL_KS:
        out[f"recall@{k}"] = hits[min(k, len(hits)) - 1] / n_pos if n_pos else float("nan")
    positive_ranks = np.flatnonzero(ranked) + 1
    out["average_precision"] = float(np.mean(hits[positive_ranks - 1] / positive_ranks)) if n_pos else float("nan")
    return out


def per_day_metrics(
    scores: np.ndarray, labels: np.ndarray, days: np.ndarray, seed: int = 0
) -> list[dict[str, float]]:
    """`day_metrics` for each distinct value of `days`, in sorted order."""
    rng = np.random.default_rng(seed)
    return [day_metrics(scores[days == d], labels[days == d], rng) for d in np.unique(days)]


def average(per_day: list[dict[str, float]]) -> dict[str, float]:
    """Macro-average over days, skipping NaNs (days without positives)."""
    return {name: float(np.nanmean([m[name] for m in per_day])) for name in METRIC_NAMES}


def mean_day_metrics(
    scores: np.ndarray, labels: np.ndarray, days: np.ndarray, seed: int = 0
) -> dict[str, float]:
    return average(per_day_metrics(scores, labels, days, seed))


def paired_difference(baseline: list[dict], candidate: list[dict], name: str) -> dict[str, float]:
    """Per-day `candidate - baseline` for metric `name`, over the days where
    both are defined: the mean, its standard error, and how many days the
    candidate was better or worse. Pairing by day removes most of the
    day-to-day variance that an unpaired comparison of two means would carry."""
    diffs = np.array(
        [c[name] - b[name] for b, c in zip(baseline, candidate, strict=True) if not (np.isnan(b[name]) or np.isnan(c[name]))]
    )
    return {
        "mean": float(diffs.mean()),
        "se": float(diffs.std(ddof=1) / np.sqrt(len(diffs))),
        "days_better": int((diffs > 0).sum()),
        "days_worse": int((diffs < 0).sum()),
        "days": len(diffs),
    }
