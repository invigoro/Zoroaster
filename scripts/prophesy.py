"""Version 3's prophet: free-text predictions of real-world events, from a day's evidence.

PLAN.md §6 step 11. For each day:
1. The eligible pages: the prophecy's top 20, without biographies
   (milestone 1, `src/prophecy/evidence.py`).
2. Each page's evidence, with its dates marked relative to the day, and
   version 2's forecast (`forecast_v2.py`).
3. The prophet (`src/prophecy/prophet.py`), a local instruction-tuned model,
   Qwen2.5-7B-Instruct in 4-bit, reads one page at a time:
   - for each page whose evidence dates something to the day itself, the
     question that day settles about it;
   - questions about a contest one person wins are dropped;
   - then, for each question left, a prediction.
   Its training data ends long before these days, so it can't know what
   happened.
4. The checks (`src/prophecy/checks.py`), each a separate question to the
   same model, using only the cited evidence from the day before:
   - specific people, and harm done to, by or alleged of a specific
     organization;
   - whether the evidence already settles the prediction;
   - copies of example sentences in the instructions, and repeats.

Writes `data/processed/enwiki/v3/prophecies/D.json` with each page's question,
every prediction, kept or dropped and why, and prints them. `--rescreen`
re-runs only the checks on saved predictions, without generating new ones.

Usage:
    python scripts/prophesy.py [--days 2026-09-20 2026-09-25] [--model Qwen/Qwen2.5-7B-Instruct] [--out DIR] [--rescreen]
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
from src.prophecy.checks import (NO_ANSWER, confirmed_orgs, confirmed_people, contest_messages, harm_messages,
                                 kind_messages, listed_names, novelty_messages, one_persons_contest, orgs_messages,
                                 people_messages, person_messages, screen)
from src.prophecy.evidence import TOP, eligible, evidence_blocks
from src.prophecy.prophet import (HORIZON, PREDICTION, QUESTION, marked_within, parse_prediction, parse_question,
                                  prediction_messages, question_messages)
from src.prophecy.selection import select, title_messages, topic_messages

INSTRUCT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EXAMPLES_DIR = V3_DIR / "examples"
FORECASTS_PATH = V3_DIR / "forecasts.json"
OUT_DIR = V3_DIR / "prophecies"
DEFAULT_DAYS = (date(2026, 9, 20), date(2026, 9, 25), date(2026, 9, 30))
QUESTION_TOKENS, PREDICTION_TOKENS = 120, 120
YES_NO_TOKENS, LIST_TOKENS = 4, 32
BATCH = 8
# What the day itself brought, for the judge only. The prophet never loads it, nor end_id, whose change
# would tell that the page was edited on the day. page_text stays: it shows which pages are biographies.
DAY_COLUMNS = {"prose", "sections", "section_chars", "kinds", "inserted_chars", "removed_chars", "spans", "blocks",
               "end_id"}
CHECK_FIELDS = ("kept", "dropped_because", "person_check", "people_named", "orgs_named", "harm_check", "novelty_check",
                "grounded_check", "people_check", "orgs_check",
                "topic_check", "sensitive",  # from a version of the checks that ran only on 2026-10-01
                "topic", "settles_a_title", "published", "unpublished_because")  # the selection (`publish`)


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


def chat_in_batches(model, tokenizer, conversations: list[list[dict]], max_new_tokens: int) -> list[str]:
    """`chat` over any number of conversations, BATCH at a time and longest first, so the GPU's memory peaks
    in the first batch (see train_v2.forecast_headers). Replies come back in the conversations' order."""
    order = sorted(range(len(conversations)), key=lambda i: -sum(len(m["content"]) for m in conversations[i]))
    replies = [""] * len(conversations)
    for start in range(0, len(order), BATCH):
        part = order[start : start + BATCH]
        for i, reply in zip(part, chat(model, tokenizer, [conversations[i] for i in part], max_new_tokens)):
            replies[i] = reply
    return replies


def check(model, tokenizer, predictions: list[dict], by_title: dict[str, str], instructions: str) -> list[dict]:
    """Every check (`src/prophecy/checks.py`) on a day's predictions."""
    def ask(conversations: list[list[dict]], tokens: int) -> list[str]:
        return chat_in_batches(model, tokenizer, conversations, tokens)

    texts = [p["text"] for p in predictions]
    cited = [[by_title[t] for t in p["evidence"] if t in by_title] for p in predictions]
    person = ask([person_messages(t) for t in texts], YES_NO_TOKENS)
    people = ask([people_messages(t) for t in texts], LIST_TOKENS)
    orgs_answers = ask([orgs_messages(t) for t in texts], LIST_TOKENS)
    person_names = [listed_names(a) for a in people]
    org_names = [listed_names(a) for a in orgs_answers]
    to_ask = list(dict.fromkeys(n for names in person_names + org_names for n in names if n != NO_ANSWER))
    kinds = dict(zip(to_ask, ask([kind_messages(n) for n in to_ask], YES_NO_TOKENS)))
    confirmed = [confirmed_people(names, [kinds.get(n, "") for n in names]) for names in person_names]
    orgs = [confirmed_orgs(names, [kinds.get(n, "") for n in names]) for names in org_names]
    questions_per_org = len(harm_messages("", ""))
    harm_said = iter(ask([chat for t, os in zip(texts, orgs) for o in os for chat in harm_messages(t, o)], YES_NO_TOKENS))
    harm = [[[next(harm_said) for _ in range(questions_per_org)] for _ in os] for os in orgs]
    novelty = ask([novelty_messages(t, c) for t, c in zip(texts, cited)], YES_NO_TOKENS)
    screened = screen(predictions, person, confirmed, orgs, harm, novelty, instructions)
    return [s | {"people_check": p.strip(), "orgs_check": o.strip()} for s, p, o in zip(screened, people, orgs_answers)]


def publish(model, tokenizer, predictions: list[dict], ranks: dict[str, int]) -> list[dict]:
    """The day's selection (`src/prophecy/selection.py`): each kept prediction's topic, whether a sports one
    settles a title, and which are published."""
    kept = [i for i, p in enumerate(predictions) if p["kept"]]
    asked = [(p["evidence"][0] if p["evidence"] else "", p["question"]) for p in (predictions[i] for i in kept)]
    topics, titles = [""] * len(predictions), [""] * len(predictions)
    for i, a in zip(kept, chat_in_batches(model, tokenizer, [topic_messages(*q) for q in asked], YES_NO_TOKENS)):
        topics[i] = a
    for i, a in zip(kept, chat_in_batches(model, tokenizer, [title_messages(*q) for q in asked], YES_NO_TOKENS)):
        titles[i] = a
    return select(predictions, ranks, topics, titles)


def prophesy(model, tokenizer, day: date, rows: list[dict], forecasts: dict, model_name: str, horizon: int = HORIZON,
             top: int = TOP) -> dict:
    """One day's record: each eligible page's question, if anything is due within the horizon, and the
    predictions, screened and selected."""
    def ask(conversations: list[list[dict]], tokens: int) -> list[str]:
        return chat_in_batches(model, tokenizer, conversations, tokens)

    start = time.monotonic()
    pages = eligible([r for r in rows if r["date"] == day], top=top)
    titles = [r["page_title"].replace("_", " ") for r in pages]
    blocks = evidence_blocks(pages, forecasts)
    dated = [i for i, b in enumerate(blocks) if marked_within(b, horizon)]
    asked = dict(zip(dated, ask([question_messages(day, blocks[i], horizon) for i in dated], QUESTION_TOKENS)))
    questions = {i: q for i, q in ((i, parse_question(a, day, horizon)) for i, a in asked.items()) if q}
    contests = dict(zip(questions, ask([contest_messages(titles[i], q) for i, (q, _) in questions.items()],
                                       YES_NO_TOKENS)))
    todo = [i for i, (q, _) in questions.items() if not one_persons_contest(titles[i], q, contests[i])]
    answers = ask([prediction_messages(day, blocks[i], *questions[i]) for i in todo], PREDICTION_TOKENS)
    parsed = ((parse_prediction(a, titles[i], questions[i][0]), questions[i][1]) for a, i in zip(answers, todo))
    predictions = [p | {"due": due.isoformat()} for p, due in parsed if p]
    predictions = check(model, tokenizer, predictions, dict(zip(titles, blocks)), PREDICTION)
    predictions = publish(model, tokenizer, predictions, {t: r["rank"] for r, t in zip(pages, titles)})
    page_notes = [{"rank": r["rank"], "title": t, "dated": i in asked, "asked": asked.get(i, "").strip(),
                   "question": questions[i][0] if i in questions else None,
                   "due": questions[i][1].isoformat() if i in questions else None,
                   "contest": contests.get(i, "").strip(), "one_persons_contest": i in questions and i not in todo}
                  for i, (r, t) in enumerate(zip(pages, titles))]
    return {"date": day.isoformat(), "model": model_name, "milestone": 1, "horizon": horizon, "top": top,
            "seconds": round(time.monotonic() - start), "instructions": PREDICTION, "question_instructions": QUESTION,
            "pages": page_notes, "evidence": "\n\n".join(blocks), "answers": answers, "predictions": predictions}


def report(day: date, record: dict) -> None:
    screened, pages = record["predictions"], record["pages"]
    print(f"\n=== {day}: {len(pages)} pages, {sum(p.get('dated', p.get('dated_today', False)) for p in pages)} with "
          f"something due, {sum(bool(p.get('question')) for p in pages)} questions "
          f"({sum(p.get('one_persons_contest', False) for p in pages)} one person's contest), {len(screened)} "
          f"predictions, {sum(p['kept'] for p in screened)} kept, {sum(p.get('published', False) for p in screened)} "
          f"published ({record['seconds']:,}s)", flush=True)
    for p in screened:
        mark = "PUBLISH" if p.get("published") else "KEPT   " if p["kept"] else "DROPPED"
        if not p["kept"]:
            why = f"  [{'; '.join(p['dropped_because'])}]"
        else:
            why = f"  [{p.get('topic')}, due {p.get('due', day)}{'; ' + p['unpublished_because'] if p.get('unpublished_because') else ''}]"
        print(f"  {mark} ({p['confidence'] or '?'}) {p['text']}  <- {'; '.join(p['evidence']) or 'no evidence cited'}{why}\n"
              f"          question: {p['question'] or '(none given)'}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=date.fromisoformat, nargs="+", default=list(DEFAULT_DAYS))
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--rescreen", action="store_true", help="re-run only the checks on saved predictions")
    parser.add_argument("--horizon", type=int, default=HORIZON, help="days after D within which a prediction may come due")
    parser.add_argument("--top", type=int, default=TOP, help="how many of the prophecy's pages to read, biographies left out")
    args = parser.parse_args(argv)
    model, tokenizer = load_instruct(args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.rescreen:
        for day in args.days:
            path = args.out / f"{day.isoformat()}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            by_title = dict(zip([p["title"] for p in record["pages"]], record["evidence"].split("\n\n")))
            bare = [{k: v for k, v in p.items() if k not in CHECK_FIELDS} for p in record["predictions"]]
            ranks = {p["title"]: p["rank"] for p in record["pages"]}
            screened = check(model, tokenizer, bare, by_title, record["instructions"])
            record["predictions"] = publish(model, tokenizer, screened, ranks)
            path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
            report(day, record)
        return 0
    columns = [c for c in pq.read_schema(next(EXAMPLES_DIR.glob("*.parquet"))).names if c not in DAY_COLUMNS]
    rows = pq.read_table(EXAMPLES_DIR, columns=columns).to_pylist()
    forecasts = {(f["page_id"], f["date"]): f for f in json.loads(FORECASTS_PATH.read_text(encoding="utf-8"))["forecasts"]}
    for day in args.days:
        record = prophesy(model, tokenizer, day, rows, forecasts, args.model, args.horizon, args.top)
        (args.out / f"{day.isoformat()}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
        report(day, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
