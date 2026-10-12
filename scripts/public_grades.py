"""What the site shows of the live predictions' grades, each grade's line checked like a prediction (PLAN.md §6 step 11).

The live site's predictions are graded about once a week, once their due day is over (README, "Grading the live
predictions"). Claude drafts each grade and a one-line public reason from its pack (`grading_packs.py`), and the user
confirms or edits both on the review page (`confirmed_grades.py --run live`). The site shows each graded prediction's
outcome, whether it was already known the night before or too vague to count, and the line (`build_site.py`).

Whatever is published about a prediction must pass the prophet's own guardrails (PLAN.md §2). So each line is asked
the same questions as a prediction (`prophesy.check`), on the GPU: it may name no person nor point to one, no company
or other organization doing or suffering harm, and none in a war, a disaster or a crime but countries' governments and
forces, and armed groups. A line that fails is left out, and the site shows the outcome alone until a new line is
confirmed. The quality checks (already known, vague words) don't apply to a line. A line checked before, unchanged,
isn't checked again, so the model loads only for new lines; never while the daily job runs (about 00:30 to 01:00 UTC).

Writes `data/processed/enwiki/v3/grades_live/public.json`, keyed like the packs ("D#n"): the prediction's day and its
text as published, the outcome, `already_known`, `vague` (specificity 0), and the line if it passed. Beside them,
`check`: the line as checked and why it was dropped, which stays on this machine.

Usage:
    python scripts/public_grades.py [--grades data/processed/enwiki/v3/grades_live]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_site import PROPHECY_DIR
from scripts.build_v3_days import V3_DIR
from src.prophecy.checks import GUARDRAIL_REASONS

LIVE_GRADES_DIR = V3_DIR / "grades_live"


def guardrail_reasons(screened: dict) -> list[str]:
    """Why the guardrails drop a checked line; the quality reasons don't apply to one."""
    return [r for r in screened["dropped_because"] if r.startswith(GUARDRAIL_REASONS)]


def to_check(confirmed: dict[str, dict], earlier: dict[str, dict]) -> dict[str, str]:
    """The lines not yet checked as they now read, by key."""
    return {key: g["public"] for key, g in confirmed.items()
            if g.get("public") and earlier.get(key, {}).get("check", {}).get("line") != g["public"]}


def public(confirmed: dict[str, dict], texts: dict[str, str], checks: dict[str, dict]) -> dict[str, dict]:
    """What the site shows of each confirmed grade, keyed like the packs. `texts`: each prediction's text as
    published; `checks`: each line's check, {"line", "dropped_because"}."""
    out = {}
    for key, grade in sorted(confirmed.items()):
        check = checks.get(key)
        passed = check and check["line"] == grade.get("public") and not check["dropped_because"]
        out[key] = {"date": key.split("#")[0], "text": texts[key], "outcome": grade["outcome"],
                    "already_known": grade["already_known"], "vague": grade["specificity"] == 0,
                    **({"line": grade["public"]} if passed else {}), **({"check": check} if check else {})}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grades", type=Path, default=LIVE_GRADES_DIR, help="the live grades' folder")
    args = parser.parse_args(argv)
    confirmed = json.loads((args.grades / "confirmed.json").read_text(encoding="utf-8"))
    path = args.grades / "public.json"
    earlier = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    records = {day: json.loads((PROPHECY_DIR / f"{day}.json").read_text(encoding="utf-8"))
               for day in {key.split("#")[0] for key in confirmed}}
    texts = {key: records[key.split("#")[0]]["predictions"][int(key.split("#")[1])]["text"] for key in confirmed}
    checks = {key: e["check"] for key, e in earlier.items() if "check" in e}
    new = to_check(confirmed, earlier)
    if new:
        from scripts.prophesy import INSTRUCT_MODEL, check, load_instruct  # loads the model onto the GPU

        model, tokenizer = load_instruct(INSTRUCT_MODEL)
        screened = check(model, tokenizer, [{"text": line, "evidence": []} for line in new.values()], {}, "")
        checks |= {key: {"line": line, "dropped_because": guardrail_reasons(s)}
                   for (key, line), s in zip(new.items(), screened, strict=True)}
    shown = public(confirmed, texts, checks)
    path.write_text(json.dumps(shown, indent=1, ensure_ascii=False), encoding="utf-8")
    dropped = {key: s["check"]["dropped_because"] for key, s in shown.items() if s.get("check", {}).get("dropped_because")}
    for key, reasons in dropped.items():
        print(f"  {key}: line left out ({'; '.join(reasons)}): {shown[key]['check']['line']}")
    print(f"{len(shown)} grades, {sum('line' in s for s in shown.values())} with their line, {len(new)} lines checked "
          f"now, {len(dropped)} left out -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
