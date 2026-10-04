"""Version 3's daily step: the prophecy for day D, from what was known by the end of D-1 (PLAN.md §6 step 11,
phase 4).

Run by the daily job after Stage 1's prediction for D (`daily_predictions.py`), on this machine's GPU:
1. Stage 1's top pages for D, as the development days had them (`build_v3_days.py`): each page's signals,
   its linked pages that burst on D-1, and its revisions through D-1 from the dumps and the live days.
   Each page's text at the end of D-1, and its change on D-1, come from the API (`fetch_v2_examples.derive`).
   D itself is left out: it has barely begun, and the prophet never reads it.
2. Portal:Current events for the week before D, as it stood at the end of D-1 (`fetch_current_events.known`).
3. The prophet (`prophesy.prophesy`): questions, predictions, checks, rewrites in general terms, revisions
   of made-up details, and the day's selection of at most 10. Version 2's edit forecasts aren't made
   daily, so the pages' evidence goes without them.

Writes the day's whole record to `data/processed/enwiki/v3/daily/D.json`, as `prophesy.py` does for the
development days, and the public part to `D.prophecy.json`: the published predictions alone, each with its
due date, topic and confidence. Never what they cite (a story's or a page's title can name a person), nor a
rewrite's or a revision's original.

Usage:
    python scripts/prophesy_daily.py [--day 2026-10-04] [--model Qwen/Qwen2.5-7B-Instruct]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

from scripts import build_enwiki_features as enwiki
from scripts.build_site import PROPHECY_DIR, PUBLIC_SUFFIX
from scripts.build_v3_days import SIGNALS, TOP, bounds_or_unchanged, neighbors_bursting, page_revisions
from scripts.daily_predictions import PREDICTIONS_DIR
from scripts.fetch_current_events import KNOWN_DIR, known
from scripts.fetch_v2_examples import derive
from src.stage2.fetch import fetch_contents


def day_rows(day: date, session: requests.Session) -> list[dict]:
    """Stage 1's top pages for `day`, each as the prophet reads it: as it stood at the end of the day before."""
    prophecy = json.loads((PREDICTIONS_DIR / f"{day.isoformat()}.json").read_text(encoding="utf-8"))
    top = prophecy["pages"][:TOP]
    day_table = pq.read_table(PREDICTIONS_DIR / f"{day.isoformat()}.parquet",
                              columns=["page_id", "page_title", *dict.fromkeys(SIGNALS)])
    neighbors = neighbors_bursting([p["page_id"] for p in top], pq.read_table(enwiki.LINKS_PATH), day_table)
    top_ids = pa.array([p["page_id"] for p in top], pa.int64())
    signals = {r["page_id"]: r for r in day_table.filter(pc.is_in(day_table["page_id"], top_ids)).to_pylist()}
    rows = [{"page_id": p["page_id"], "date": day, "rank": p["rank"], "score": p["score"],
             "page_title": p["title"].replace(" ", "_"), **{k: signals[p["page_id"]][k] for k in SIGNALS},
             "bursting_neighbors": neighbors[p["page_id"]]} for p in top if p["page_id"] in signals]
    revisions = page_revisions([r["page_id"] for r in rows], day - timedelta(days=1))
    for row, b in zip(rows, bounds_or_unchanged([(r["page_id"], r["date"]) for r in rows], revisions)):
        row |= b | {"end_id": b["prompt_id"]}  # D is left out: the prophet reads up to the end of D-1
    rows = [r for r in rows if r["prompt_id"] is not None]
    texts = dict(fetch_contents(list(dict.fromkeys(i for r in rows for i in (r["start_id"], r["prompt_id"]) if i)),
                                session=session))
    return [row for row in (derive(r, texts) for r in rows) if row is not None]


def public(record: dict) -> dict:
    """What the site shows of a day's record: the published predictions, world events first, as chosen."""
    published = [p for p in record["predictions"] if p.get("published")]
    published.sort(key=lambda p: p.get("topic") == "sport")
    return {"date": record["date"], "generated_at": record["generated_at"], "model": record["model"],
            "predictions": [{"text": p["text"], "due": p.get("due") or record["date"], "topic": p.get("topic"),
                             "confidence": p.get("confidence")} for p in published]}


def main(argv: list[str] | None = None) -> int:
    from scripts.prophesy import DAY_COLUMNS, INSTRUCT_MODEL, load_instruct, prophesy, report

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--day", type=date.fromisoformat, default=datetime.now(timezone.utc).date())
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    args = parser.parse_args(argv)
    start = time.monotonic()
    session = requests.Session()
    rows = [{k: v for k, v in r.items() if k not in DAY_COLUMNS} for r in day_rows(args.day, session)]
    print(f"{len(rows)} pages as they stood at the end of {args.day - timedelta(days=1)} "
          f"({time.monotonic() - start:,.0f}s)", flush=True)
    KNOWN_DIR.mkdir(parents=True, exist_ok=True)
    events = known(session, args.day)
    (KNOWN_DIR / f"{args.day.isoformat()}.json").write_text(json.dumps(events, indent=1, ensure_ascii=False),
                                                            encoding="utf-8")
    print(f"{sum(len(d['items']) for d in events['days'])} current events from the week before, as known at "
          f"{events['known_at']} ({time.monotonic() - start:,.0f}s)", flush=True)
    model, tokenizer = load_instruct(args.model)
    record = prophesy(model, tokenizer, args.day, rows, {}, args.model)
    record["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    PROPHECY_DIR.mkdir(parents=True, exist_ok=True)
    (PROPHECY_DIR / f"{args.day.isoformat()}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False),
                                                               encoding="utf-8")
    (PROPHECY_DIR / f"{args.day.isoformat()}{PUBLIC_SUFFIX}").write_text(
        json.dumps(public(record), indent=1, ensure_ascii=False), encoding="utf-8")
    report(args.day, record)
    print(f"Done in {time.monotonic() - start:,.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
