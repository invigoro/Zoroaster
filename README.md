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

This repo is being expanded into a pipeline that trains a model to predict
which Wikipedia pages will be edited next and what content the edit will add,
using only edits that were not later reverted. See `data/` (git-ignored —
regenerate locally, don't commit) and `src/ingest/` for the current pieces:

- `src/ingest/stub_stream.py` — streams a MediaWiki `stub-meta-history` dump
  (revision metadata only, no article text) into mainspace revision records
  without loading the whole file into memory.
- `src/ingest/revert_detect.py` — flags reverted revisions using an
  identity-revert heuristic (a later revision's content hash matches an
  earlier one within 15 revisions or 90 days, whichever is tighter), so
  reverted edits can be excluded from training data using metadata alone —
  no full article text needed for this step.
- `scripts/build_test_revert_labels.py` — end-to-end test run of the above
  against the small Simple English Wikipedia dump (~900MB compressed), used
  to validate the pipeline before pointing it at English Wikipedia at scale.
  Run it with:

  ```bash
  python scripts/build_test_revert_labels.py
  ```

  It downloads the dump to `data/raw/` (skipped if already present) and
  writes labeled revisions to `data/processed/simplewiki_test_revert_labels.parquet`.

- `src/features/bursts.py` — per-page edit-rate burst detection and
  cross-page "co-burst" counts (how many *other* pages are also bursting the
  same day), the endogenous stand-in for an external news signal. Only
  meaningful on retained, non-bot revisions — bot maintenance runs and AWB
  mass edits otherwise dominate the "burst" signal.
- `src/ingest/sampling.py` — stratifies pages by edit-frequency bucket and
  historical burst activity (popularity/pageview-based stratification is
  deferred until we're ready to pull that dump), and draws a reproducible
  per-stratum random sample.
- `scripts/build_test_features.py` — runs the above against the existing
  `simplewiki_test_revert_labels.parquet` (no new downloads). Run it with:

  ```bash
  python scripts/build_test_features.py
  ```

  Writes `data/processed/simplewiki_test_activity_features.parquet` and
  `data/processed/simplewiki_test_sample_manifest.json`.

  **Known limitation from this test run**: on the small Simple Wikipedia
  corpus, the highest-ranked "co-burst" days are dominated by a single
  editor doing a long editing session on one page, coincidentally
  overlapping with other unrelated single-editor sessions on the same
  calendar day — not genuine multi-editor reactions to a shared real-world
  event. This may simply be a low-edit-volume-project artifact; it needs
  re-checking once this points at English Wikipedia, where breaking-news
  editing is a well-documented multi-editor, multi-page phenomenon.

- `src/ingest/fetch_diffs.py` — fetches a revision's wikitext plus its
  parent's via the MediaWiki API and computes the added/removed text
  between them. This is the per-revision fetch path used once a sample is
  chosen; it is **not yet invoked against live Wikipedia** (only offline,
  mocked-response checks so far) since running it for real means pulling
  actual content for every sampled revision.

## Dependencies

- [`requests`](https://pypi.org/project/requests/) — HTTP client used to call
  the Wikipedia REST API and download Wikimedia dumps.
- [`mwxml`](https://pypi.org/project/mwxml/) — streaming parser for MediaWiki
  XML dumps.
- [`pyarrow`](https://pypi.org/project/pyarrow/) — Parquet I/O for the
  processed revision datasets.
