"""Version 3's prophet: free-text predictions of real-world events, from a day's evidence.

PLAN.md §6 step 11. For each day:
1. The eligible pages: the prophecy's top 20, without biographies
   (milestone 1, `src/prophecy/evidence.py`).
2. Their evidence, including version 2's forecasts (`forecast_v2.py`).
3. The prophet's predictions (`src/prophecy/prophet.py`), from a local
   instruction-tuned model, Qwen2.5-7B-Instruct in 4-bit. Its training data
   ends long before these days, so it can't know what happened.
4. The checks (`src/prophecy/checks.py`), each a separate question to the
   same model, using only the cited evidence from the day before:
   - the topic list;
   - two person questions;
   - whether the evidence already settles the prediction;
   - whether the prediction is about its cited evidence at all;
   - copies of example sentences in the instructions, and repeats.

Writes `data/processed/enwiki/v3/prophecies/D.json` with every prediction,
kept or dropped and why, and prints them. `--rescreen` re-runs only the
checks on the saved predictions, without generating new ones.

Usage:
    python scripts/prophesy.py [--days 2026-09-20 2026-09-25] [--model Qwen/Qwen2.5-7B-Instruct] [--rescreen]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# torch before pyarrow (see train_stage2.py).
import torch  # isort: skip
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from scripts.build_v3_days import V3_DIR
from src.prophecy.checks import (NO_ANSWER, confirmed_people, grounded_messages, kind_messages, listed_names,
                                 novelty_messages, people_messages, person_messages, screen)
from src.prophecy.evidence import eligible, evidence_blocks
from src.prophecy.prophet import messages, parse_predictions

INSTRUCT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EXAMPLES_DIR = V3_DIR / "examples"
FORECASTS_PATH = V3_DIR / "forecasts.json"
OUT_DIR = V3_DIR / "prophecies"
DEFAULT_DAYS = (date(2026, 9, 20), date(2026, 9, 25), date(2026, 9, 30))
PROPHET_TOKENS = 1200
YES_NO_TOKENS, LIST_TOKENS = 4, 32
UNUSED_COLUMNS = {"spans", "blocks"}  # page_text stays: it shows which pages are biographies
CHECK_FIELDS = ("kept", "dropped_because", "person_check", "people_named", "novelty_check", "grounded_check",
                "people_check")


def load_instruct(name: str):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.padding_side = "left"
    return model, tokenizer


@torch.no_grad()
def chat(model, tokenizer, conversations: list[list[dict]], max_new_tokens: int) -> list[str]:
    """Greedy replies to a batch of chats."""
    if not conversations:
        return []
    texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in conversations]
    batch = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                         top_k=None, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    torch.cuda.empty_cache()
    return tokenizer.batch_decode(out[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)


def check(model, tokenizer, predictions: list[dict], by_title: dict[str, str], instructions: str) -> list[dict]:
    """Every check (`src/prophecy/checks.py`) on a day's predictions."""
    texts = [p["text"] for p in predictions]
    cited = [[by_title[t] for t in p["evidence"] if t in by_title] for p in predictions]
    person = chat(model, tokenizer, [person_messages(t) for t in texts], YES_NO_TOKENS)
    people = chat(model, tokenizer, [people_messages(t) for t in texts], LIST_TOKENS)
    names = [listed_names(a) for a in people]
    to_ask = [n for ns in names for n in ns if n != NO_ANSWER]
    verdicts = iter(chat(model, tokenizer, [kind_messages(n) for n in to_ask], YES_NO_TOKENS))
    confirmed = [confirmed_people(ns, [next(verdicts) if n != NO_ANSWER else "" for n in ns]) for ns in names]
    novelty = chat(model, tokenizer, [novelty_messages(t, c) for t, c in zip(texts, cited)], YES_NO_TOKENS)
    grounded = chat(model, tokenizer, [grounded_messages(t, c) for t, c in zip(texts, cited)], YES_NO_TOKENS)
    screened = screen(predictions, person, confirmed, novelty, grounded, instructions)
    return [s | {"people_check": a.strip()} for s, a in zip(screened, people)]


def report(day: date, record: dict) -> None:
    screened = record["predictions"]
    print(f"\n=== {day}: {len(record['pages'])} pages, {record['prompt_tokens']:,} prompt tokens, {len(screened)} "
          f"predictions, {sum(p['kept'] for p in screened)} kept ({record['seconds']:,}s)", flush=True)
    for p in screened:
        mark = "KEPT   " if p["kept"] else "DROPPED"
        why = f"  [{'; '.join(p['dropped_because'])}]" if not p["kept"] else ""
        print(f"  {mark} ({p['confidence'] or '?'}) {p['text']}  <- {'; '.join(p['evidence']) or 'no evidence cited'}{why}\n"
              f"          question: {p['question'] or '(none given)'}", flush=True)
    if not screened:
        print("  (no predictions parsed) raw answer:\n" + record["answer"][:1500], flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=date.fromisoformat, nargs="+", default=list(DEFAULT_DAYS))
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--rescreen", action="store_true", help="re-run only the checks on saved predictions")
    args = parser.parse_args(argv)
    model, tokenizer = load_instruct(args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.rescreen:
        for day in args.days:
            path = args.out / f"{day.isoformat()}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            by_title = dict(zip([p["title"] for p in record["pages"]], record["evidence"].split("\n\n")))
            bare = [{k: v for k, v in p.items() if k not in CHECK_FIELDS} for p in record["predictions"]]
            record["predictions"] = check(model, tokenizer, bare, by_title, record["instructions"])
            path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
            report(day, record)
        return 0
    columns = [c for c in pq.read_schema(next(EXAMPLES_DIR.glob("*.parquet"))).names if c not in UNUSED_COLUMNS]
    rows = pq.read_table(EXAMPLES_DIR, columns=columns).to_pylist()
    forecasts = {(f["page_id"], f["date"]): f for f in json.loads(FORECASTS_PATH.read_text(encoding="utf-8"))["forecasts"]}
    for day in args.days:
        start = time.monotonic()
        pages = eligible([r for r in rows if r["date"] == day])
        titles = [r["page_title"].replace("_", " ") for r in pages]
        blocks = evidence_blocks(pages, forecasts)
        evidence = "\n\n".join(blocks)
        conversation = messages(day, evidence)
        answer = chat(model, tokenizer, [conversation], PROPHET_TOKENS)[0]
        predictions = check(model, tokenizer, parse_predictions(answer, titles), dict(zip(titles, blocks)),
                            conversation[0]["content"])
        prompt_tokens = len(tokenizer(tokenizer.apply_chat_template(conversation, tokenize=False,
                                                                    add_generation_prompt=True)).input_ids)
        record = {"date": day.isoformat(), "model": args.model, "milestone": 1, "seconds": round(time.monotonic() - start),
                  "instructions": conversation[0]["content"], "prompt_tokens": prompt_tokens,
                  "pages": [{"rank": r["rank"], "title": t} for r, t in zip(pages, titles)],
                  "evidence": evidence, "answer": answer, "predictions": predictions}
        (args.out / f"{day.isoformat()}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
        report(day, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
