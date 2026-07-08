# Wikipedia Edit-Prediction Project — Status & Plan

This file is a handoff/status document for resuming this work on a different
machine. It captures the goal, the decisions already made, exactly what's
been built and validated so far, the findings from that validation, and the
concrete next steps. Read this before doing anything else if you're picking
this up cold.

Last updated: 2026-07-08.

## 1. Goal

Predict **which Wikipedia pages will be edited next** and **what content
that edit will add**, treating edits that were later reverted as noise to
exclude, not signal to learn from. The working hypothesis is that a
meaningful share of edits are reactions to real-world events unfolding, so
the model should pick up on "something is happening" signals rather than
only long-run per-page editing habits.

Two-stage approach:
- **Stage 1** (build first): forecast which pages are likely to be edited in
  an upcoming window, using activity-burst/attention signals.
- **Stage 2** (built on top of Stage 1): fine-tune a small causal LM to
  generate the actual added text (the diff insertion), conditioned on page
  context + trigger signals. Only trained on non-reverted edits.

## 2. Decisions already made (don't re-litigate these without reason)

- **Scope**: English Wikipedia, **mainspace only** (namespace 0).
- **Revert filtering**: a revision is treated as reverted (and excluded from
  training) if a later revision — within the next **15 revisions or 90
  days, whichever is tighter** — has a `sha1` matching an earlier revision's
  content. Both the undone revision and the mechanical revert itself are
  excluded from training targets (the revert doesn't add new content).
- **No exogenous news data.** "Current events" signal comes only from
  Wikipedia's own activity (edit-rate bursts per page, and how many *other*
  pages are bursting simultaneously — "co-burst"), not an external news
  corpus like GDELT. This was an explicit scope decision to keep the
  project Wikipedia-data-only.
- **Storage budget**: ≤25GB total for the training corpus. Achieved by
  storing **diffs** (inserted/removed text), not full page snapshots at
  every revision, and by sampling pages/time rather than taking everything.
- **Compute target for Stage 2 fine-tuning**: a single **RTX 3070 (8GB
  VRAM)**, via **QLoRA** (4-bit quantized base + LoRA adapters) on a
  ~0.5B-1.5B parameter causal LM (candidates: Qwen2.5-0.5B/1.5B, SmolLM2,
  Pythia-1B). An RTX 5090 is also available but is a fallback only, not the
  default target — don't assume its VRAM/throughput when sizing things.
- Stage 1 (LightGBM baseline, and features/sampling work) is CPU-only and
  can be done on any machine, including one without a GPU.

Full original plan detail (data sources, storage design rationale, model
architecture options, evaluation plan, milestones) — if you want the fuller
write-up, ask to have it regenerated; the essential decisions are captured
above and this file is now the source of truth for status.

## 3. What's been built so far

All of this lives in the repo (not gitignored) except the `data/` directory
itself (see §4 — it's regenerable, not committed).

```
src/
  common.py            # USER_AGENT constant + is_bot_edit() heuristic (shared)
  ingest/
    stub_stream.py      # streams a stub-meta-history dump -> mainspace RevisionRecord dicts
    revert_detect.py    # identity-revert detection (15-rev/90-day window)
    sampling.py          # stratified page sampling (edit-frequency x had_burst)
    fetch_diffs.py        # per-revision diff fetch via MediaWiki API — NOT YET RUN LIVE
  features/
    bursts.py             # per-page burst detection + cross-page co-burst counts
scripts/
  download_dump.py        # generic Wikimedia dump downloader (User-Agent-policy compliant)
  build_test_revert_labels.py  # orchestrates: download simplewiki dump -> parse -> revert-detect -> parquet
  build_test_features.py       # orchestrates: burst/co-burst features + stratified sample -> parquet + manifest
main.py                    # original Wikipedia-summary CLI (unchanged behavior, now imports USER_AGENT from src.common)
requirements.txt           # requests, mwxml, pyarrow
```

### Pipeline run so far (test scale, Simple English Wikipedia only)

Deliberately small-scale: **Simple English Wikipedia**
(`simplewiki-latest-stub-meta-history.xml.gz`, ~895MB compressed, dumped
2026-07-02), not English Wikipedia. This was a conscious choice to validate
the pipeline cheaply before committing to the 25GB English Wikipedia corpus
— **no English Wikipedia data has been downloaded yet.**

1. `python scripts/build_test_revert_labels.py`
   - Downloaded the dump to `data/raw/`.
   - Parsed **6,953,329 mainspace revisions across 397,196 pages**.
   - Flagged **993,827 revisions (14.29%) as reverted**.
   - Output: `data/processed/simplewiki_test_revert_labels.parquet`.

2. `python scripts/build_test_features.py`
   - Filtered to retained (non-reverted) **and** non-bot revisions for
     burst detection: 4,255,035 of the original 6,953,329.
   - Computed per-page burst z-scores and cross-page co-burst counts.
   - Result: 54,492 burst page-days across 43,633 pages.
   - Stratified all retained-revision pages by edit-frequency bucket
     (low/medium/high) × had-a-burst (True/False) → 5 strata, sampled 25
     pages per stratum (seed 1234) → 125 pages total.
   - Output: `data/processed/simplewiki_test_activity_features.parquet` and
     `data/processed/simplewiki_test_sample_manifest.json`.

3. `src/ingest/fetch_diffs.py` was written (fetches a revision + its parent's
   wikitext via the MediaWiki API, computes added/removed text via
   `difflib`) but **has only been validated offline with mocked HTTP
   responses** — it has never been run against live Wikipedia. This is the
   next real piece of work (see §5).

## 4. Data state — important for resuming on a new machine

`data/` is **git-ignored** (see `.gitignore`) — it does not travel with the
repo. On a fresh clone, `data/` will not exist. To regenerate the exact same
test dataset:

```bash
python -m venv .venv
# then activate it (see README.md for OS-specific activation commands)
pip install -r requirements.txt
python scripts/build_test_revert_labels.py   # ~35 min parse time, ~900MB download
python scripts/build_test_features.py         # runs against the local parquet, no download
```

Caveat: `build_test_revert_labels.py` downloads
`simplewiki-latest-stub-meta-history.xml.gz`, i.e. **whatever the current
latest dump is** — Wikimedia doesn't keep old "latest" dumps at that URL, so
the exact revision counts above may drift slightly on a re-run months later
as new edits accumulate. This hasn't mattered yet but would matter once
we care about exact reproducibility (at that point, pin to a dated dump
directory like `/simplewiki/20260701/` instead of `/simplewiki/latest/`).

## 5. Findings from validation (worth knowing before continuing)

### Revert detection — validated, looks correct

Hand-checked 5 small pages (Animals, Muslim, Hallucinations, Full English,
Jewish) by reading their real edit-comment history. **Every single revision
flagged `is_reverted=True` was immediately followed by a revision whose
comment literally said "Reverted N edit(s) by ..."**, and the mechanical
revert revision itself was correctly *not* flagged as reverted. High
confidence in this heuristic as implemented.

### Burst/co-burst detection — works as coded, but two real limitations found

1. **Bot/maintenance mass-edits look like bursts.** The first run (before
   filtering bots out) had its single largest co-burst day
   (2009-01-06, 1,177 pages) driven almost entirely by interwiki-link bots
   (`JAnDbot`, `ArthurBot`) and AutoWikiBrowser-assisted mass page creation
   (`Synergy` creating many Kentucky town stubs), not any real-world event.
   **Fix applied**: burst detection now runs only on retained + non-bot
   revisions (`is_bot_edit()` heuristic in `src/common.py`, matches
   usernames containing "bot" — a heuristic, not authoritative).

2. **Even after that fix, top "co-burst" days are still not genuine
   shared-event reactions.** After excluding bots, the top days
   (2011-07-19, 2012-03-13, etc.) are dominated by *coincidental overlap of
   unrelated single-editor sessions* — e.g. one editor (`Iantresman`) doing
   ~15 edits to "Natural number" in one day, another editor (`Zfish118`)
   separately doing a big content port into "Catholicism" the same day.
   These aren't multiple editors reacting to one shared trigger; they're
   independent editors who happened to pick the same calendar day.

   **This is an open question, not yet resolved**: is this a Simple
   Wikipedia-specific artifact (much lower edit volume/population than
   English Wikipedia, where breaking-news multi-editor bursts are a
   well-documented phenomenon), or does the co-burst signal need a smarter
   definition (e.g. requiring bursting pages to be topically/temporally
   clustered, or requiring *distinct* editors across pages rather than just
   counting bursting pages)? **Don't tune this further against the Simple
   Wikipedia test corpus** — it may simply lack the pattern we're looking
   for at this scale. Re-evaluate once real English Wikipedia data is in
   play.

## 6. Next steps, in order

1. **Wire up and run `fetch_diffs.py` for real**, but only for the 125
   already-sampled test pages (small, deliberate scope) — pull actual
   diff text for their retained revisions, to validate the diff-extraction
   step end-to-end before committing to it at scale. This requires live
   network calls (a few hundred, respecting `REQUEST_DELAY_SECONDS`), which
   is a step up from everything so far but still small.
2. Decide whether to keep prototyping on Simple Wikipedia a while longer, or
   move to a real (budget-limited, stratified) sample of **English**
   Wikipedia — this is the point where the "≤25GB" real corpus actually
   starts getting built, and is a bigger, more deliberate data-acquisition
   step than anything done so far. Get explicit go-ahead before starting it
   given the scale jump.
3. Build the pageview-based popularity stratum (currently deferred — see
   §2/§5 of the earlier plan discussion) once ready to pull the pageviews
   dumps.
4. Build the Stage 1 LightGBM baseline forecaster on the metadata + burst
   features (CPU-only, doable on any machine).
5. Only after Stage 1 shows the burst/co-burst signal is actually
   predictive: move to Stage 2 QLoRA fine-tuning on the RTX 3070 machine.

## 7. Open questions

- Co-burst signal validity at Simple-Wikipedia scale (§5) — needs English
  Wikipedia data to resolve, not further tuning on the current corpus.
- Whether the current bot-detection heuristic (`"bot" in username`) is good
  enough, or whether it's worth pulling actual user-group data — deferred,
  revisit if bot mass-edits keep leaking into signal at larger scale.
- Everything in §6 is unstarted; pick up there.
