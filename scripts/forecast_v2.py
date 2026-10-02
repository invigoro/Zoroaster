"""Version 2's ranked forecasts for any set of examples, such as version 3's days.

This loads the full-prompt adapter, and the kinds thresholds that were chosen
on validation (`ranked.json`). Then, for each example:
1. a greedy header, which supplies its sections line;
2. the ranked sections and thresholded kinds (`rank_v2.rank_example`).

Nothing from the day being forecast is used: the prompt only describes the
page at the end of D-1. One JSON file is written.

Usage:
    python scripts/forecast_v2.py --examples data/processed/enwiki/v3/examples --out data/processed/enwiki/v3/forecasts.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# torch before pyarrow (see train_stage2.py).
import torch  # isort: skip
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from scripts.rank_v2 import ids, load_trained, rank_example
from scripts.train_v2 import BASE_MODEL, RUN_DIR, UNUSED_COLUMNS, encode, forecast_headers
from src.forecast.prompts import parse_forecast
from src.forecast.ranking import kinds_continuation, kinds_forecast, kinds_lines

SHOWN_SECTIONS = 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--variant", default="full")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    start = time.monotonic()
    columns = [c for c in pq.read_schema(next(args.examples.glob("*.parquet"))).names if c not in UNUSED_COLUMNS]
    rows = pq.read_table(args.examples, columns=columns).to_pylist()
    thresholds = json.loads((args.run_dir / "ranked.json").read_text(encoding="utf-8"))["thresholds"]
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    lines = kinds_lines()
    line_ids = [ids(tokenizer, kinds_continuation(k)) for k in lines]
    prompts = [encode(tokenizer, r, args.variant)[0] for r in rows]
    model = load_trained(BASE_MODEL, args.run_dir / "adapters" / f"{args.variant}_seed{args.seed}")
    headers = forecast_headers(model, tokenizer, prompts, pad_id)
    print(f"{len(rows):,} greedy headers ({time.monotonic() - start:,.0f}s)", flush=True)
    out: list = [None] * len(rows)
    for i in sorted(range(len(rows)), key=lambda i: -len(prompts[i])):  # longest first
        d = rank_example(model, tokenizer, prompts[i], rows[i], parse_forecast(headers[i]), line_ids, lines)
        torch.cuda.empty_cache()  # sizes change with every prompt (see train_v2.forecast_headers)
        out[i] = {"page_id": rows[i]["page_id"], "date": rows[i]["date"].isoformat(), "sections": d["sections"],
                  "kinds": kinds_forecast(d["kinds"], thresholds), "section_probs": d["section_probs"][:SHOWN_SECTIONS],
                  "kind_probs": d["kinds"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"thresholds": thresholds, "forecasts": out}, indent=1), encoding="utf-8")
    print(f"Wrote {args.out}: {len(out):,} forecasts ({time.monotonic() - start:,.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
