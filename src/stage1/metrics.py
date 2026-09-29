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


def mean_day_metrics(
    scores: np.ndarray, labels: np.ndarray, days: np.ndarray, seed: int = 0
) -> dict[str, float]:
    """Macro-average of `day_metrics` over the distinct values of `days`,
    skipping NaNs (days without positives)."""
    rng = np.random.default_rng(seed)
    per_day = [day_metrics(scores[days == d], labels[days == d], rng) for d in np.unique(days)]
    return {name: float(np.nanmean([m[name] for m in per_day])) for name in METRIC_NAMES}
