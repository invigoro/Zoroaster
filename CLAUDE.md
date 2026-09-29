# Zoroaster — working notes for Claude

Read `PLAN.md` first. It's the source of truth for goals, decisions (§2),
status, findings and next steps. Keep it current as work lands.

## Working conventions (SOP)

- **Commit and push as you go.** Whenever a coherent chunk of work is done
  and its tests pass (a fix, a feature, a pipeline step, a doc update),
  commit it with a descriptive message and push. Don't mix unrelated changes
  in one commit, and don't leave finished work uncommitted.
- Work on the `stage1` branch and push it to `origin`. Merge to `main` only
  when the user says so.
- **Write tests alongside the code** in `tests/`, using stdlib `unittest`.
  Run `python -m unittest discover -s tests` from the repo root before every
  commit.
  - For anything that feeds a model, test the invariant that matters, e.g.
    the point-in-time leakage test in `tests/test_activity_features.py`.
  - Check that such a test can actually fail, e.g. by planting the bug it
    guards against.
- Never commit `data/`. It's git-ignored and regenerable (PLAN.md §4).

## Project invariants

- **Streaming:** process one page at a time and write Parquet through
  `src/parquet_io.RowGroupWriter`. Code has to scale to English Wikipedia,
  roughly 100× the test corpus.
- **Point-in-time:** features for prediction day D use only information
  knowable by the end of D−1. The final `is_reverted` flag and the
  `kept_edits` channel are label-side only.
- Don't tune co-burst against the Simple Wikipedia test corpus (PLAN.md §5).

## Environment

- Windows. Use the venv interpreter `.venv/Scripts/python.exe` (Python 3.10).
- Pipeline runs take minutes: run them in the background, since the scripts
  print progress.
