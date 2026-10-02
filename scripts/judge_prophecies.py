"""Version 3's local judge: grade each kept prophecy against what its day brought.

PLAN.md §6 step 11, phase 2.
1. Read the packs (`grading_packs.py`).
2. Have the local model grade each one against the rubric
   (`src/prophecy/judge.py`). It's Qwen2.5-7B-Instruct, the prophet's model,
   but a separate step with its own instructions, so the prophet never
   grades itself.
3. Write `data/processed/enwiki/v3/grades/judge.json`.
4. If hand grades exist (`grades/confirmed.json`: the drafts as the user
   confirmed or adjusted them), report how often the judge agrees with them.

Usage:
    python scripts/judge_prophecies.py [--compare-only]
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

from scripts.grading_packs import GRADES_DIR
from scripts.prophesy import INSTRUCT_MODEL, chat, load_instruct
from src.prophecy.judge import credit, judge_messages, parse_grade

JUDGE_TOKENS = 220
BATCH = 4
FIELDS = ("outcome", "already_known", "specificity", "grounded")


def agreement(judge: dict[str, dict], hand: dict[str, dict]) -> dict:
    """How often the judge's grade matches the hand grade, field by field and on credit."""
    keys = [k for k in hand if k in judge and judge[k]]
    out = {"graded_by_both": len(keys)}
    for field in FIELDS:
        out[field] = sum(judge[k][field] == hand[k][field] for k in keys) / len(keys) if keys else None
    pairs = [(credit(judge[k]), credit(hand[k])) for k in keys]
    both = [(j, h) for j, h in pairs if j is not None and h is not None]
    out["credit_mean_abs_diff"] = sum(abs(j - h) for j, h in both) / len(both) if both else None
    out["mean_credit"] = {"judge": sum(j for j, _ in both) / len(both) if both else None,
                          "hand": sum(h for _, h in both) / len(both) if both else None}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compare-only", action="store_true", help="skip grading; compare the saved grades")
    args = parser.parse_args(argv)
    packs = json.loads((GRADES_DIR / "packs.json").read_text(encoding="utf-8"))
    judge_path = GRADES_DIR / "judge.json"
    if not args.compare_only:
        start = time.monotonic()
        model, tokenizer = load_instruct(INSTRUCT_MODEL)
        # Longest first, so the GPU's memory peaks in the first batch (see train_v2.forecast_headers).
        keys = sorted(packs, key=lambda k: -len(judge_messages(packs[k])[1]["content"]))
        grades, raw = {}, {}
        for i in range(0, len(keys), BATCH):
            batch = keys[i : i + BATCH]
            answers = chat(model, tokenizer, [judge_messages(packs[k]) for k in batch], JUDGE_TOKENS)
            for k, a in zip(batch, answers):
                grades[k], raw[k] = parse_grade(a), a
            print(f"  {min(i + BATCH, len(keys))}/{len(keys)} graded ({time.monotonic() - start:,.0f}s)", flush=True)
        judge_path.write_text(json.dumps({"model": INSTRUCT_MODEL, "grades": grades, "answers": raw}, indent=1,
                                         ensure_ascii=False), encoding="utf-8")
        unparsed = sum(g is None for g in grades.values())
        print(f"Wrote {judge_path}: {len(grades)} grades, {unparsed} unparsed ({time.monotonic() - start:,.0f}s)")
        torch.cuda.empty_cache()
    confirmed_path = GRADES_DIR / "confirmed.json"
    if confirmed_path.exists():
        judge = json.loads(judge_path.read_text(encoding="utf-8"))["grades"]
        hand = json.loads(confirmed_path.read_text(encoding="utf-8"))
        print("Agreement with the hand grades:", json.dumps(agreement(judge, hand), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
