"""Details a story's prediction adds that its reports don't give (PLAN.md §3 item 30; the user asked for this,
2026-10-02).

Run 11's story predictions missed mostly on made-up specifics: "at least 50 casualties", air quality "above
150" in Singapore, a trial in New York City. The reports gave none of them, so the predictions were
ungrounded, or unknowable because the portal never reports such a detail.

`unsupported_details` finds them without a model:
- **Numbers:** every number in the prediction must appear in its evidence, in digits or, from three up, in
  words. Dates are exempt, since the due date comes from the prophet's instructions, and so are years.
  "One" and "two" are exempt too ("at least one more strike", "the two countries").
- **Names:** every capitalized word (a place, a country, a party, a ship) must appear in its evidence,
  allowing for a nationality ("Ukrainian" for "Ukraine") and common abbreviations ("U.S.").

A flagged prediction goes back to the prophet, in its own conversation: the reports, the question, and its
answer, then a note naming what the reports don't give (`revise_messages`). In a trial on three days, it
replaced most such details with ones the reports give ("at least 2,400 additional refugees", the reports'
figure, for "at least 5,000"). A first design rewrote the sentence alone, without the reports: for numbers it
mostly gave the sentence back unchanged. A revision may restate what the reports already say (that more than
100,000 people have fled Yemen), whatever the note says; that's for the novelty check.

Only story predictions are checked. A page's evidence names teams by code ("teamA=CHN · teamB=INA"), so a
prediction naming China and Indonesia would be flagged though its evidence gives both; and it's short, so
nearly any score would be too, where the rubric rewards a predicted score. The trial's revisions of page
predictions lost what they predicted ("one country will defeat another in each matchup").
"""

from __future__ import annotations

import json
import re
from datetime import date

from src.prophecy.checks import UNSUPPORTED, repeats
from src.prophecy.evidence import DATE_PATTERNS, MONTHS
from src.prophecy.prophet import story_prediction_messages
from src.prophecy.stories import is_story

DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
# Words that start a sentence or a title, or that are capitalized without naming anything.
ORDINARY = {"i", "the", "a", "an", "and", "or", "of", "in", "on", "at", "by", "for", "to", "with", "this", "that",
            "it", "its", "their", "his", "her", "both", "after", "before", "during", "between", "from", "some",
            "more", "most", "least", "no", "not", "new", "prime", "minister", "president", "court", "party"}
NUMBER = re.compile(r"(?<![\w.,])\d[\d,]*(?:\.\d+)?")
WORD = re.compile(r"[^\W\d_][\w'’-]*")  # any script: "Niño", "Türkiye"
POSSESSIVE = re.compile(r"['’]s$")
YEAR = re.compile(r"^(?:19|20)\d\d$")
# "the 29th of September", which `DATE_PATTERNS` doesn't read.
ORDINAL_DATE = re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)\s+of\s+(?:{'|'.join(MONTHS)})\b")
PREFIX = "I predict that"
# Numbers in words, from three up: the trial's revisions wrote "ten air strikes" for a made-up "15".
WORD_NUMBERS = {w: str(n) for n, w in enumerate(
    ("three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen "
     "nineteen twenty").split(), start=3)} | {w: str(n) for n, w in zip(
    range(30, 100, 10), "thirty forty fifty sixty seventy eighty ninety".split())}
# Nationalities whose country starts differently, and abbreviations, each with what the evidence may say instead.
ALSO = {"french": ("france",), "british": ("britain", "united kingdom", "uk"), "dutch": ("netherlands",),
        "filipino": ("philippines",), "swiss": ("switzerland",), "welsh": ("wales",), "scottish": ("scotland",),
        "irish": ("ireland",), "greek": ("greece",), "turkish": ("turkey", "türkiye"), "polish": ("poland",),
        "danish": ("denmark",), "swedish": ("sweden",), "norwegian": ("norway",), "finnish": ("finland",),
        "spanish": ("spain",), "portuguese": ("portugal",), "thai": ("thailand",), "afghan": ("afghanistan",),
        "emirati": ("emirates", "uae"), "american": ("united states", "america", "u.s."),
        "u.s.": ("united states", "us", "american"), "us": ("united states", "u.s."), "uk": ("united kingdom", "british"),
        "un": ("united nations",), "eu": ("european union",), "uae": ("united arab emirates", "emirates"),
        "gdp": ("gross domestic product", "economy", "economic")}


def _numbers(text: str) -> set[str]:
    return {m.group(0).replace(",", "") for m in NUMBER.finditer(text)}


def _without_dates(text: str) -> str:
    for pattern in (*DATE_PATTERNS, ORDINAL_DATE):
        text = pattern.sub(" ", text)
    return text


def _known(word: str, evidence: str, words: set[str]) -> bool:
    """Whether a capitalized word, or its country or full name, appears in the evidence."""
    lower = POSSESSIVE.sub("", word.lower())
    if lower in ORDINARY or word in MONTHS or word in DAYS or len(lower) < 2:
        return True
    if any(other in evidence for other in ALSO.get(lower, ())):
        return True
    if len(lower) >= 4:  # "Ukrainian" for "Ukraine", "Houthis" for "Houthi"
        return any(w.startswith(lower[:4]) for w in words)
    return lower in words


def unsupported_details(text: str, evidence: str) -> list[str]:
    """The numbers and names in a prediction that its evidence doesn't contain, in order of appearance."""
    body = _without_dates(text[len(PREFIX):] if text.startswith(PREFIX) else text).replace("U.S.", "US")
    evidence_lower = evidence.replace("U.S.", "US").lower()
    words = {POSSESSIVE.sub("", w) for w in WORD.findall(evidence_lower)}
    known_numbers = _numbers(evidence) | {WORD_NUMBERS[w] for w in words if w in WORD_NUMBERS}
    found: list[str] = []
    for m in NUMBER.finditer(body):
        number = m.group(0).replace(",", "")
        if not YEAR.match(number) and number not in known_numbers and m.group(0) not in found:
            found.append(m.group(0))
    for m in WORD.finditer(body):
        word, lower = m.group(0), m.group(0).lower()
        if lower in WORD_NUMBERS:
            known = WORD_NUMBERS[lower] in known_numbers
        else:
            known = not word[0].isupper() or _known(word, evidence_lower, words)
        if not known and word not in found:
            found.append(word)
    return found


def flagged(predictions: list[dict], by_title: dict[str, str]) -> dict[int, list[str]]:
    """Each kept story prediction that adds details its reports don't give: index -> those details. Page
    predictions are left alone (see the module's docstring)."""
    out = {}
    for i, p in enumerate(predictions):
        if p["kept"] and p["evidence"] and all(is_story(t) for t in p["evidence"]):
            found = unsupported_details(p["text"], "\n\n".join(by_title.get(t, "") for t in p["evidence"]))
            if found:
                out[i] = found
    return out


REVISE = """Your prediction gives details that none of the reports give: {details}. Predict again, and this time take every place, country, group and number from the reports themselves. Predict what comes next, not what the reports already say has happened. If the reports give no number to go by, give none. Keep one detail that can be checked, taken from the reports.

Answer with only a JSON object: {{"prediction": "I predict that ...", "confidence": "low", "medium" or "high"}}"""


def revise_messages(day: date, block: str, prediction: dict, details: list[str]) -> list[dict]:
    """The story's prediction conversation (`prophet.story_prediction_messages`), its prediction as the answer,
    and a note naming the details its reports don't give (`unsupported_details`)."""
    answer = json.dumps({"prediction": prediction["text"], "confidence": prediction.get("confidence") or "medium"},
                        ensure_ascii=False)
    return (story_prediction_messages(day, block, prediction["question"], date.fromisoformat(prediction["due"]))
            + [{"role": "assistant", "content": answer},
               {"role": "user", "content": REVISE.format(details=", ".join(f'"{d}"' for d in details))}])


# Kept with the revision, from the prediction it replaces: a rewrite's original (`checks.merge_rewrites`).
CARRIED = ("rewritten_from", "dropped_before_rewrite")


def merge_revisions(screened: list[dict], revised: dict[int, dict], found: dict[int, list[str]],
                    evidence: dict[int, str]) -> list[dict]:
    """The day's predictions, each flagged one (`found`: index -> the details its evidence doesn't give)
    replaced by its revision (`revised`: index -> the revision as screened) if every check passed it, it adds
    no detail its evidence doesn't give, and it repeats no prediction kept that day. Otherwise the original is
    dropped: it adds details its evidence doesn't give, a quality reason (`checks.QUALITY_REASONS`), so it's
    still graded, to test this check. `evidence` holds each flagged prediction's cited evidence."""
    kept = [p["text"] for i, p in enumerate(screened) if p["kept"] and i not in found]
    out = []
    for i, p in enumerate(screened):
        if i not in found:
            out.append(p)
            continue
        r = revised[i]
        still = unsupported_details(r["text"], evidence[i])
        if r["kept"] and not still and not repeats(r["text"], kept):
            kept.append(r["text"])
            out.append(r | {k: p[k] for k in CARRIED if k in p} | {"revised_from": p["text"], "unsupported": found[i]})
        else:
            why = r["dropped_because"] or ([f"{UNSUPPORTED} ({'; '.join(still)})"] if still
                                           else ["repeats an earlier prediction"])
            out.append(p | {"kept": False, "dropped_because": [f"{UNSUPPORTED} ({'; '.join(found[i])})"],
                            "unsupported": found[i], "revision": r["text"], "revision_dropped_because": why})
    return out
