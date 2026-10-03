"""Version 3's prophet: free-text predictions of real-world events, from a day's evidence.

PLAN.md §6 step 11. For each day:
1. The eligible pages: the prophecy's top 100 (`TOP`), without biographies
   (milestone 1, `src/prophecy/evidence.py`).
2. Each page's evidence, with its dates marked relative to the day, and
   version 2's forecast (`forecast_v2.py`).
3. The prophet (`src/prophecy/prophet.py`), a local instruction-tuned model,
   Qwen2.5-7B-Instruct in 4-bit by default, reads one page at a time:
   - for each page whose evidence dates something to the day or the week
     after it (`HORIZON`), the question decided then, and its date, which
     the prediction is due on;
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
5. The selection (`src/prophecy/selection.py`): at most ten a day, world
   events first, and sport only when it settles a title.

Writes `data/processed/enwiki/v3/prophecies/D.json` with each page's question,
every prediction, kept or dropped and why, and prints them. `--rescreen`
re-runs only the checks on saved predictions, without generating new ones.

Usage:
    python scripts/prophesy.py [--days 2026-09-20 2026-09-25] [--model Qwen/Qwen2.5-7B-Instruct] [--out DIR]
                               [--rescreen] [--horizon 7] [--top 100] [--batch 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# torch before pyarrow (see train_stage2.py).
import torch  # isort: skip
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from scripts.build_v3_days import V3_DIR
from scripts.fetch_current_events import KNOWN_DIR
from src.prophecy.checks import (NO_ANSWER, apply_identifies, confirmed_orgs, confirmed_people, contest_messages,
                                 general_mention, generalize_messages, guarded_only, harm_messages,
                                 identifies_messages, kind_messages, listed_names, merge_rewrites, novelty_messages,
                                 one_persons_contest, orgs_messages, people_messages, person_messages, screen)
from src.prophecy.details import flagged, merge_revisions, revise_messages
from src.prophecy.evidence import TOP, eligible, evidence_blocks
from src.prophecy.prophet import (HORIZON, PREDICTION, QUESTION, STORY_PREDICTION, STORY_QUESTION, marked_offsets,
                                  marked_within, parse_prediction, parse_question, parse_rewrite, prediction_messages,
                                  question_messages, story_prediction_messages, story_question_messages)
from src.prophecy.selection import is_sport_page, select, title_messages, topic_messages
from src.prophecy.stories import stories as day_stories
from src.prophecy.stories import story_block

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
                "topic", "settles_a_title", "published", "unpublished_because",  # the selection (`publish`)
                "rewritten_from", "dropped_before_rewrite", "rewrite", "rewrite_dropped_because",  # `rewrite`
                "identifies_check",
                "revised_from", "unsupported", "revision", "revision_dropped_because")  # `revise`
# Both steps' instructions, for the check against copying their example sentences.
INSTRUCTIONS = PREDICTION + "\n\n" + STORY_PREDICTION


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
    # Qwen3 thinks aloud unless told not to; Qwen2.5's template ignores the switch.
    texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=False)
             for c in conversations]
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
    # A name the sentence uses only after "a" or "an" ("an armed group") is a general description, not one.
    person_names = [[n for n in listed_names(a) if not general_mention(n, t)] for a, t in zip(people, texts)]
    org_names = [[n for n in listed_names(a) if not general_mention(n, t)] for a, t in zip(orgs_answers, texts)]
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


def rewrite(model, tokenizer, screened: list[dict], by_title: dict[str, str], instructions: str) -> list[dict]:
    """Each prediction the guardrails alone dropped, rewritten in general terms and checked again
    (`checks.merge_rewrites`)."""
    guarded = [i for i, p in enumerate(screened) if guarded_only(p)]
    if not guarded:
        return screened
    answers = chat_in_batches(model, tokenizer, [generalize_messages(screened[i]["text"]) for i in guarded],
                              PREDICTION_TOKENS)
    drafts = [{k: v for k, v in screened[i].items() if k not in CHECK_FIELDS} | {"text": parse_rewrite(a)}
              for i, a in zip(guarded, answers)]
    rechecked = check(model, tokenizer, drafts, by_title, instructions)
    # Vague words aren't enough: does the rewrite still point to the one its original was about?
    asked = [i for i, r in enumerate(rechecked) if r["kept"]]
    said = dict(zip(asked, chat_in_batches(model, tokenizer, [
        identifies_messages(rechecked[i]["text"], screened[guarded[i]]["text"]) for i in asked], YES_NO_TOKENS)))
    rechecked = apply_identifies(rechecked, [said.get(i, "") for i in range(len(rechecked))])
    return merge_rewrites(screened, dict(zip(guarded, rechecked)))


def revise(model, tokenizer, screened: list[dict], by_title: dict[str, str], day: date, instructions: str) -> list[dict]:
    """Each kept story prediction that adds details its reports don't give (`details.flagged`), sent back to
    the prophet with those details named, and checked again (`details.merge_revisions`)."""
    found = flagged(screened, by_title)
    if not found:
        return screened
    evidence = {i: "\n\n".join(by_title.get(t, "") for t in screened[i]["evidence"]) for i in found}
    answers = chat_in_batches(model, tokenizer, [revise_messages(day, evidence[i], screened[i], details)
                                                 for i, details in found.items()], PREDICTION_TOKENS)
    drafts = []
    for i, answer in zip(found, answers):
        p = screened[i]
        parsed = parse_prediction(answer, p["evidence"][0], p["question"]) or {}  # none: still flagged, and dropped
        drafts.append({k: v for k, v in p.items() if k not in CHECK_FIELDS} | {"text": parsed.get("text", p["text"])}
                      | ({"confidence": parsed["confidence"]} if parsed.get("confidence") else {}))
    rechecked = check(model, tokenizer, drafts, by_title, instructions)
    return merge_revisions(screened, dict(zip(found, rechecked)), found, evidence)


def publish(model, tokenizer, predictions: list[dict], ranks: dict[str, int], sport_pages: set[str],
            story_topics: dict[str, str] | None = None) -> list[dict]:
    """The day's selection (`src/prophecy/selection.py`): each kept prediction's topic, whether a sports one
    settles a title, and which are published. A story's topic is its category's (`story_topics`), so the
    model isn't asked."""
    story_topics = story_topics or {}
    kept = [i for i, p in enumerate(predictions) if p["kept"] and (p["evidence"] or [""])[0] not in story_topics]
    asked = [(p["evidence"][0] if p["evidence"] else "", p["question"]) for p in (predictions[i] for i in kept)]
    topics, titles = [""] * len(predictions), [""] * len(predictions)
    for i, a in zip(kept, chat_in_batches(model, tokenizer, [topic_messages(*q) for q in asked], YES_NO_TOKENS)):
        topics[i] = a
    for i, a in zip(kept, chat_in_batches(model, tokenizer, [title_messages(*q) for q in asked], YES_NO_TOKENS)):
        titles[i] = a
    for i, p in enumerate(predictions):
        if p["evidence"] and p["evidence"][0] in story_topics:
            topics[i] = story_topics[p["evidence"][0]]
    return select(predictions, ranks, topics, titles, sport_pages)


def known_events(day: date) -> dict | None:
    """The week of Portal:Current events before `day`, as known at the end of the day before
    (`fetch_current_events.py --known`), or None if it hasn't been fetched."""
    path = KNOWN_DIR / f"{day.isoformat()}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def sport_titles(rows: list[dict]) -> set[str]:
    """The titles of the sports pages among `rows`, from their text at the end of D-1."""
    return {r["page_title"].replace("_", " ") for r in rows if is_sport_page(r.get("page_text") or "")}


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
    questions = {i: q for i, q in ((i, parse_question(a, day, horizon, marked_offsets(blocks[i], horizon)))
                                   for i, a in asked.items()) if q}
    contests = dict(zip(questions, ask([contest_messages(titles[i], q) for i, (q, _) in questions.items()],
                                       YES_NO_TOKENS)))
    todo = [i for i, (q, _) in questions.items() if not one_persons_contest(titles[i], q, contests[i])]
    answers = ask([prediction_messages(day, blocks[i], *questions[i]) for i in todo], PREDICTION_TOKENS)
    parsed = ((parse_prediction(a, titles[i], questions[i][0]), questions[i][1]) for a, i in zip(answers, todo))
    predictions = [p | {"due": due.isoformat()} for p, due in parsed if p]

    # The stories of the week's Portal:Current events, as known by the end of the day before (`stories.py`).
    known = known_events(day)
    found = day_stories(known, day) if known else []
    story_blocks = [story_block(n, s, day) for n, s in enumerate(found, start=1)]
    story_asked = ask([story_question_messages(day, b, horizon) for b in story_blocks], QUESTION_TOKENS)
    week_end = day + timedelta(days=horizon)
    story_questions = {i: q for i, q in ((i, parse_question(a, day, horizon, marked_offsets(story_blocks[i], horizon),
                                                            default_due=week_end))
                                         for i, a in enumerate(story_asked)) if q}
    story_answers = ask([story_prediction_messages(day, story_blocks[i], *story_questions[i]) for i in story_questions],
                        PREDICTION_TOKENS)
    parsed = ((parse_prediction(a, found[i]["title"], story_questions[i][0]), story_questions[i][1])
              for a, i in zip(story_answers, story_questions))
    predictions += [p | {"due": due.isoformat(), "source": "story"} for p, due in parsed if p]

    by_title = dict(zip(titles, blocks)) | {s["title"]: b for s, b in zip(found, story_blocks)}
    predictions = rewrite(model, tokenizer, check(model, tokenizer, predictions, by_title, INSTRUCTIONS), by_title,
                          INSTRUCTIONS)
    predictions = revise(model, tokenizer, predictions, by_title, day, INSTRUCTIONS)
    sport = sport_titles(pages)
    predictions = publish(model, tokenizer, predictions, {t: r["rank"] for r, t in zip(pages, titles)}, sport,
                          {s["title"]: s["topic"] for s in found})
    page_notes = [{"rank": r["rank"], "title": t, "sport": t in sport, "dated": i in asked, "asked": asked.get(i, "").strip(),
                   "question": questions[i][0] if i in questions else None,
                   "due": questions[i][1].isoformat() if i in questions else None,
                   "contest": contests.get(i, "").strip(), "one_persons_contest": i in questions and i not in todo}
                  for i, (r, t) in enumerate(zip(pages, titles))]
    story_notes = [{"title": s["title"], "story": s["story"], "category": s["category"], "topic": s["topic"],
                    "recent": s["recent"], "week": s["week"], "asked": a.strip(),
                    "question": story_questions[i][0] if i in story_questions else None,
                    "due": story_questions[i][1].isoformat() if i in story_questions else None}
                   for i, (s, a) in enumerate(zip(found, story_asked))]
    return {"date": day.isoformat(), "model": model_name, "checks_model": model_name, "milestone": 1, "horizon": horizon,
            "top": top, "seconds": round(time.monotonic() - start), "instructions": PREDICTION,
            "question_instructions": QUESTION, "story_instructions": STORY_PREDICTION,
            "story_question_instructions": STORY_QUESTION, "pages": page_notes, "evidence": "\n\n".join(blocks),
            "stories": story_notes, "story_blocks": story_blocks, "events_known_at": known and known["known_at"],
            "answers": answers, "story_answers": story_answers, "predictions": predictions}


def report(day: date, record: dict) -> None:
    screened, pages, found = record["predictions"], record["pages"], record.get("stories", [])
    print(f"\n=== {day}: {len(pages)} pages, {sum(p.get('dated', p.get('dated_today', False)) for p in pages)} with "
          f"something due, {sum(bool(p.get('question')) for p in pages)} questions "
          f"({sum(p.get('one_persons_contest', False) for p in pages)} one person's contest); {len(found)} stories, "
          f"{sum(bool(s.get('question')) for s in found)} questions; {len(screened)} predictions, "
          f"{sum(p['kept'] for p in screened)} kept ({sum(bool(p.get('rewritten_from')) for p in screened)} rewritten, "
          f"{sum(bool(p.get('revised_from')) for p in screened)} revised), "
          f"{sum(p.get('published', False) for p in screened)} published ({record['seconds']:,}s)", flush=True)
    for p in screened:
        mark = "PUBLISH" if p.get("published") else "KEPT   " if p["kept"] else "DROPPED"
        if not p["kept"]:
            why = f"  [{'; '.join(p['dropped_because'])}]"
        else:
            why = f"  [{p.get('topic')}, due {p.get('due', day)}{'; ' + p['unpublished_because'] if p.get('unpublished_because') else ''}]"
        print(f"  {mark} ({p['confidence'] or '?'}) {p['text']}  <- {'; '.join(p['evidence']) or 'no evidence cited'}{why}\n"
              f"          question: {p['question'] or '(none given)'}", flush=True)
        if p.get("rewritten_from"):
            print(f"          rewritten from: {p['rewritten_from']}", flush=True)
        elif p.get("rewrite"):
            print(f"          its rewrite, dropped too: {p['rewrite']} [{'; '.join(p['rewrite_dropped_because'])}]",
                  flush=True)
        if p.get("revised_from"):
            print(f"          revised for {', '.join(p['unsupported'])}, from: {p['revised_from']}", flush=True)
        elif p.get("revision"):
            print(f"          its revision, dropped too: {p['revision']} [{'; '.join(p['revision_dropped_because'])}]",
                  flush=True)


def main(argv: list[str] | None = None) -> int:
    global BATCH
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=date.fromisoformat, nargs="+", default=list(DEFAULT_DAYS))
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--rescreen", action="store_true",
                        help="re-run only the checks and the selection on saved predictions, with --model")
    parser.add_argument("--horizon", type=int, default=HORIZON, help="days after D within which a prediction may come due")
    parser.add_argument("--top", type=int, default=TOP, help="how many of the prophecy's pages to read, biographies left out")
    parser.add_argument("--batch", type=int, default=BATCH, help="chats per batch; fewer for a model that fills the GPU")
    args = parser.parse_args(argv)
    BATCH = args.batch
    model, tokenizer = load_instruct(args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.rescreen:
        texts = pq.read_table(EXAMPLES_DIR, columns=["page_title", "date", "page_text"],
                              filters=[("date", "in", args.days)]).to_pylist()
        for day in args.days:
            path = args.out / f"{day.isoformat()}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            stories_seen = record.get("stories", [])
            by_title = (dict(zip([p["title"] for p in record["pages"]], record["evidence"].split("\n\n")))
                        | dict(zip([s["title"] for s in stories_seen], record.get("story_blocks", []))))
            # A rewritten or revised prediction is checked again from its original, and may be changed again.
            bare = [{k: v for k, v in p.items() if k not in CHECK_FIELDS}
                    | {"text": p.get("rewritten_from") or p.get("revised_from") or p["text"]}
                    for p in record["predictions"]]
            ranks = {p["title"]: p["rank"] for p in record["pages"]}
            sport = sport_titles([r for r in texts if r["date"] == day])
            record["pages"] = [p | {"sport": p["title"] in sport} for p in record["pages"]]
            instructions = record["instructions"] + "\n\n" + record.get("story_instructions", "")
            screened = rewrite(model, tokenizer, check(model, tokenizer, bare, by_title, instructions), by_title,
                               instructions)
            screened = revise(model, tokenizer, screened, by_title, day, instructions)
            record["predictions"] = publish(model, tokenizer, screened, ranks, sport,
                                            {s["title"]: s["topic"] for s in stories_seen})
            record["checks_model"] = args.model  # the prophet's model stays in "model"
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
