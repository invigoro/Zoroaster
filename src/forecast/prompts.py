"""Version 2's prompts and targets (PLAN.md §6 step 10, phase 3).

A prompt describes the page as a forecaster sees it at the end of D-1. Its
variants add information in steps:
- `page`: the title then, the date, the lead and the section headings;
- `yesterday`: plus yesterday's change (sections, kinds, a little of its
  prose);
- `full`: plus the Stage 1 signals and the bursting linked pages' titles.

The target puts the structured forecast first, which is all the site
shows, then the day's new prose, which is learned and measured but not
published:

    Sections: Career; Results
    Kinds: prose, references, table
    New text: In 2026 she was appointed ...

`parse_forecast` reads a generated header back for scoring.

Pages are named as they were titled at the end of D-1, not as of the dump
snapshot, which would leak later renames (`point_in_time_titles`).
"""

from __future__ import annotations

from datetime import timedelta

from src.forecast.changes import KINDS
from src.forecast.metrics import MAX_SECTIONS, main_sections
from src.stage2.examples import MAX_NEIGHBORS_SHOWN

VARIANTS = ("page", "yesterday", "full")
MAX_HEADINGS = 60
YESTERDAY_PROSE_CHARS = 300
NONE = "none"
TEXT_MARK = "New text:"


def _title(title: str) -> str:
    return title.replace("_", " ")


def _cut(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + " …"


def point_in_time_titles(rows: list[dict], titles: list[dict]) -> dict[str, int]:
    """Rename pages to their titles at the time (`build_stage2_titles.py --v2`):
    each page as of its revision at the end of D-1 (`prompt_id`), each
    bursting neighbor as of D-1. Returns how many titles changed."""
    page = {t["revision_id"]: t["title_then"] for t in titles if t["kind"] == "page" and t["title_then"]}
    neighbor = {(t["title"], t["date"]): t["title_then"] for t in titles if t["kind"] == "neighbor" and t["title_then"]}
    renamed = {"pages": 0, "neighbors": 0}
    for row in rows:
        then = page.get(row["prompt_id"], row["page_title"])
        renamed["pages"] += then != row["page_title"]
        row["page_title"] = then
        prev = row["date"] - timedelta(days=1)
        names = [neighbor.get((t, prev), t) for t in row["bursting_neighbors"]]
        renamed["neighbors"] += sum(a != b for a, b in zip(names, row["bursting_neighbors"]))
        row["bursting_neighbors"] = names
    return renamed


def build_prompt(example: dict, variant: str, lead_chars: int = 600) -> str:
    """The prompt for one page-day. `lead_chars` trims the lead to fit a token budget."""
    lines = [f"Wikipedia page: {_title(example['page_title'])}", f"Date: {example['date'].isoformat()}",
             f"Lead: {_cut(example['lead'], lead_chars) or NONE}",
             f"Sections: {'; '.join(example['heading_titles'][:MAX_HEADINGS]) or NONE}"]
    if variant in ("yesterday", "full"):
        if not example["yesterday_known"]:
            yesterday = "unknown"
        elif not example["yesterday_sections"]:
            yesterday = NONE
        else:
            main = "; ".join(main_sections(example["yesterday_sections"], example["yesterday_section_chars"]))
            yesterday = f"{main} ({', '.join(example['yesterday_kinds']) or NONE})"
            if example["yesterday_prose"]:
                yesterday += f': "{_cut(example["yesterday_prose"], YESTERDAY_PROSE_CHARS)}"'
        lines.append(f"Yesterday's changes: {yesterday}")
    if variant == "full":
        burst = "yes" if example["is_burst_1d"] else "no"
        lines.append(f"Edits yesterday: {example['edits_1d']}; last 7 days: {example['edits_7d']}; "
                     f"last 30 days: {example['edits_30d']}. Page bursting yesterday: {burst} "
                     f"(z = {example['burst_z_1d']:.1f}).")
        neighbors = "; ".join(_title(t) for t in example["bursting_neighbors"][:MAX_NEIGHBORS_SHOWN])
        lines.append(f"Linked pages bursting yesterday: {neighbors or NONE}")
    lines += ["Forecast for today's edits:", ""]
    return "\n".join(lines)


def header(example: dict) -> str:
    """The structured forecast the example's actual change makes."""
    sections = "; ".join(main_sections(example["sections"], example["section_chars"])) or NONE
    kinds = ", ".join(example["kinds"]) or NONE
    return f"Sections: {sections}\nKinds: {kinds}\n"


def new_text(example: dict) -> str:
    """The target's last line: the day's new prose."""
    return f"{TEXT_MARK} {example['prose'] or NONE}"


def target_text(example: dict) -> str:
    """The full target: the header, then the day's new prose."""
    return header(example) + new_text(example)


def parse_forecast(text: str) -> dict:
    """{"sections", "kinds"} from a generated header. Repeated sections and unknown kinds are dropped."""
    sections, kinds = [], []
    for line in text.split(TEXT_MARK, 1)[0].splitlines():
        key, _, value = line.partition(":")
        value = value.strip()
        if key.strip() == "Sections" and value and value != NONE:
            sections = list(dict.fromkeys(s.strip() for s in value.split(";") if s.strip()))[:MAX_SECTIONS]
        elif key.strip() == "Kinds" and value and value != NONE:
            kinds = [k for k in KINDS if k in {v.strip() for v in value.split(",")}]
    return {"sections": sections, "kinds": kinds}
