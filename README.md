# Zoroaster

A small Python project that reads Wikipedia page data via Wikipedia's free
[REST API](https://en.wikipedia.org/api/rest_v1/) (no API key required).

`main.py` fetches a Wikipedia page and prints its summary.

## Setup

Requires Python 3.10+.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Command Prompt (`cmd.exe`):

```bat
py -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

To leave the virtual environment later, run `deactivate`.

## Usage

```bash
# Default page (Zoroastrianism)
python main.py

# A specific page
python main.py "Albert Einstein"

# A different language edition
python main.py "Albert Einstein" --lang de
```

## Edit-prediction pipeline (in progress)

See [`PLAN.md`](PLAN.md) for full status, decisions made, validation
findings, and next steps — read that first if you're resuming this work.

This repo is being expanded into a pipeline that trains a model to predict
which Wikipedia pages will be edited next and what content the edit will add,
using only edits that were not later reverted. See `data/` (git-ignored —
regenerate locally, don't commit) and `src/ingest/` for the current pieces:

- `src/ingest/stub_stream.py` — streams a MediaWiki `stub-meta-history` dump
  (revision metadata only, no article text) into mainspace revision records
  without loading the whole file into memory.
Every step streams one page at a time and writes Parquet in row groups
(`src/parquet_io.py`), so memory is bounded by the largest single page
history rather than the corpus. The same code has to run on English
Wikipedia, which is roughly 100× the test corpus.

- `src/ingest/revert_detect.py` — identity-revert detection from metadata
  alone, with no article text needed. A revision is a *revert*
  (`is_revert`) if it restores an earlier content hash. The revisions it
  undoes, back to the most recent earlier copy of that content and within
  15 revisions or 90 days (whichever is tighter), are *reverted*
  (`is_reverted`, with `reverted_by_revision_id`). Neither kind is a
  training target.
- `scripts/build_test_revert_labels.py` — end-to-end test run of the above
  against the small Simple English Wikipedia dump (~900MB compressed), used
  to validate the pipeline before pointing it at English Wikipedia at scale.
  Run it with:

  ```bash
  python scripts/build_test_revert_labels.py
  # or re-label an existing labels file (~2 min, no dump re-parse):
  python scripts/build_test_revert_labels.py --from-parquet <labels.parquet>
  ```

  It downloads the dump to `data/raw/` (skipped if already present) and
  writes labeled revisions to `data/processed/simplewiki_test_revert_labels.parquet`.

- `src/features/activity.py` — aggregates each page's revisions into daily
  channels (non-bot edits, same-day-reverted edits, reverts, bot edits, and
  the label-only `kept_edits`). It computes **point-in-time** features for a
  prediction day D from days before D only: recent edit counts, distinct
  editors, days since the last edit, page age, and burst features.
- `src/features/bursts.py` — causal burst z-scores: each day is compared to
  the page's previous 90 calendar days, zero-edit days included. Also
  counts cross-page "co-burst": how many *other* pages are bursting the
  same day, the endogenous stand-in for an external news signal. It has
  a variant that counts only bursts with at least two distinct editors.
- `src/ingest/sampling.py` — stratifies pages by edit-frequency bucket and
  historical burst activity (popularity/pageview-based stratification is
  deferred until we're ready to pull that dump), and draws a reproducible
  per-stratum random sample.
- `scripts/build_test_features.py` — runs the above against the existing
  `simplewiki_test_revert_labels.parquet` (no new downloads). Run it with:

  ```bash
  python scripts/build_test_features.py
  ```

  Writes these to `data/processed/`:
  - `simplewiki_test_daily_activity.parquet`: per page-day channels and
    bursts
  - `simplewiki_test_co_burst.parquet`: per-day co-burst counts
  - `simplewiki_test_stage1_panel.parquet` plus a `.json` metadata file:
    Stage 1 rows with features, label and sample weight
  - `simplewiki_test_sample_manifest.json`: the Stage 2 page sample

  **Known limitation from the first test run**: on the small Simple
  Wikipedia corpus, the highest-ranked "co-burst" days were dominated by
  single editors doing long editing sessions that happened to fall on the
  same calendar day as other unrelated sessions. They weren't several
  editors reacting to a shared real-world event. This may just be an
  artifact of a low-edit-volume wiki, so it needs re-checking on English
  Wikipedia, where breaking-news editing is a well-documented multi-editor,
  multi-page phenomenon.

Run the tests (no network or data files needed) with:

```bash
python -m unittest discover -s tests
```

- `src/ingest/fetch_diffs.py` — fetches a revision's wikitext plus its
  parent's via the MediaWiki API and computes the added/removed text
  between them. This is the per-revision fetch path used once a sample is
  chosen. `scripts/build_test_diffs.py` validated it live on 300 revisions
  of the Simple Wikipedia sample (300/300 fetched). The full run is
  deliberately skipped for now; see `PLAN.md` §6 for what changes before any
  large fetch.

### Known issues (code review, 2026-09-28)

Checking the code against the test data turned up five problems. Details
and numbers are in `PLAN.md` §5, and the fix order is in §6.

1. **Fixed.** Revert labels didn't match the planned rule. Revert revisions
   weren't flagged, so 335K of them counted as retained. Separately, 174K
   `is_reverted` flags landed on revisions that are themselves reverts,
   because the code matched the earliest earlier copy of a `sha1` instead of
   the most recent.
2. **Fixed.** Burst z-scores used each page's whole history as the baseline,
   including future days, so they leaked the future into any forecaster.
   Features are now point-in-time, and a test checks that.
3. **Fixed.** Pages with fewer than 10 active days could never register a
   burst. That was 81% of pages, including brand-new, heavily edited ones.
4. **Open.** Diff targets include bot edits (39% of the current sample),
   and line-level diffs overstate the inserted text about 3.6×. Reverts are
   now excluded. To be fixed before any Stage 2 fetch.
5. **Fixed.** Every script loaded the whole dataset into memory. They now
   stream one page at a time.

On top of these, the top co-burst days on the test corpus turn out to be
mass maintenance edits from human accounts, not events (`PLAN.md` §5).

## Dependencies

- [`requests`](https://pypi.org/project/requests/) — HTTP client used to call
  the Wikipedia REST API and download Wikimedia dumps.
- [`mwxml`](https://pypi.org/project/mwxml/) — streaming parser for MediaWiki
  XML dumps.
- [`pyarrow`](https://pypi.org/project/pyarrow/) — Parquet I/O for the
  processed revision datasets.
