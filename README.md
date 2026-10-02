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

- **Stage 1 harness** (`src/stage1/`, next-day horizon). The task: for
  each day D, rank every existing page by how likely it is to get a kept
  edit on D, using only information from before D.
  - `scripts/build_test_eval_days.py` scores every page on 29 test days,
    which makes the per-day ranking metrics exact.
  - `scripts/train_stage1.py` compares heuristic baselines with LightGBM on
    three nested feature sets: page habits, then the page's own bursts, then
    co-burst. It uses a time-based split with 90-day embargo gaps.
  - Results go to `data/processed/simplewiki_test_stage1_results.json` and
    are summarized in `PLAN.md` §5.
  - First results on Simple Wikipedia: a LightGBM model on page habits
    beats the best heuristic by about 29% on precision@100. Neither the
    page's own burst features nor the site-wide co-burst count add
    measurable value.
- **Link-neighbor signal** (step 4).
  - `scripts/build_test_links.py` downloads the July 2026 SQL tables
    (~154MB) and builds the mainspace link graph
    (`src/ingest/sql_dump.py`, `src/ingest/link_graph.py`).
  - `src/features/neighbors.py` counts, for each page and day, how many of
    its link neighbors were bursting.
  - `src/features/activity.py` adds a second burst definition that ignores
    mass-editing sessions (editors touching more than 25 pages in a day).
  - Result: neighbor bursts add only a sliver of deep recall; details in
    `PLAN.md` §5.

  ```bash
  pip install -r requirements.txt   # adds numpy + lightgbm
  python scripts/build_test_eval_days.py
  python scripts/train_stage1.py
  ```

- **English Wikipedia** (step 5). This uses Wikimedia's MediaWiki history
  dumps instead of stub XML: monthly TSV files with source bot flags, page
  creation dates and content hashes (`src/ingest/mediawiki_history.py`).
  The window is June 2023 through August 2026, about 22GB. Each step runs
  in parallel and skips work already done:

  ```bash
  python scripts/download_enwiki_history.py     # ~50 min
  python scripts/build_enwiki_revisions.py      # months -> 128 page buckets
  python scripts/build_enwiki_labels.py         # same revert rule as Simple Wikipedia
  python scripts/build_enwiki_features.py       # 20% page sample, 6-month windows
  python scripts/train_stage1.py --corpus enwiki
  ```

  Results (133M revisions, 9.7M pages): 74 of the model's top 100 pages
  get a kept edit the next day, against 58 for the best simple rule. A
  page's own bursts add a tiny but consistent gain. Site-wide co-burst adds
  nothing, even though its top days are clearly real events (elections,
  disasters, the World Cup final). Details are in `PLAN.md` §5.

- **Stage 2: generating the edit's text** (`src/stage2/`). This fine-tunes
  Qwen2.5-0.5B or 1.5B with QLoRA on an 8GB GPU to write the text an edit
  inserts, given the page, date, section and surrounding text. It compares
  prompts, each trained with two seeds:
  - context only;
  - plus *trigger text*: the Stage 1 signals, including the titles of
    linked pages that were bursting the day before;
  - plus what changed on those linked pages that day, either as a snippet
    of the new prose each one gained (`src/stage2/wikitext.py` strips the
    markup), or as their new sentences most relevant to the page being
    edited (`src/stage2/relevance.py`).

  Pages are named by their titles at the time, from the history dumps. The
  snapshot's titles would leak later renames.
  - The inserted text comes from word-level diffs of revisions fetched
    from the MediaWiki API, 50 per request.
  - Only the changed text and its context are stored.

  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  pip install -r requirements-stage2.txt
  python scripts/build_enwiki_links.py      # 11 GB of link tables (needed for trigger titles)
  python scripts/build_stage2_targets.py    # 16,000 kept edits with their Stage 1 signals
  python scripts/fetch_stage2_diffs.py      # ~20 min of polite API fetching
  python scripts/build_stage2_neighbor_changes.py  # ~20 min: what changed on bursting linked pages
  python scripts/build_stage2_titles.py     # ~4 min: titles at the time, from the history dumps
  python scripts/train_stage2.py            # ~3.3 hours on an RTX 3070: 3 prompts x 2 seeds
  python scripts/train_stage2.py --model Qwen/Qwen2.5-1.5B --out data/processed/enwiki/stage2/qwen2.5-1.5b
  python scripts/analyze_stage2.py          # per-seed and pooled comparisons from results.json
  ```

  Results, on 2,809 test edits from Dec 2025–Jun 2026:
  - Fine-tuning Qwen2.5-0.5B cuts the perplexity of the inserted text
    from 15.6 to 6.6. Qwen2.5-1.5B gets to 5.3.
  - Trigger text helps on edits whose linked pages were bursting, by about
    0.01 nats per token at both sizes and in every seed. It does nothing
    measurable elsewhere.
  - What helps is the neighbor's name. Most of the gain comes from edits
    that add that name, often as a link (e.g. a singer's page gaining "She
    performed at the [[2026 Winter Olympics opening ceremony]]").
  - What changed on those pages adds nothing detectable, whether as each
    page's longest new prose or as the sentences most relevant to the page
    being edited.

  Details are in `PLAN.md` §5.

- **The daily prophecy** (`src/deploy/`, `web/`). Each day, this ranks
  English Wikipedia's recently edited pages by how likely each is to
  *burst* the next day: far more edits than usual, from 2+ editors. It uses
  the Stage 1 burst model (`train_stage1.py --target burst`). The page,
  headed "Also sprach Zarathustra", shows the top pages and how the
  previous day's prophecy turned out.
  - Yesterday's edits come from the API's recent changes, turned into the
    dumps' records (`src/ingest/recent_changes.py`).
  - Features are computed by the training code itself, from dump history
    plus those live days (`src/deploy/daily.py`).
  - The page's art is kept full size in `assets/img/`.
    `python scripts/prepare_web_images.py` makes the small WebP versions in
    `web/img/` that the site uses. It keys the banner's white out to
    transparent, so the page's red glow shows through.
  - The Faravahar emblem and favicon come from
    [`Faravahar.svg`](https://commons.wikimedia.org/wiki/File:Faravahar.svg)
    on Wikimedia Commons, by Ploxhoi and Kevin McCormick, under CC BY-SA 3.0.
    The site credits it in its footers, and versions made from it carry the
    same license.
  - A second page, `how.html`, shows how the prophet works and every
    version 2 test forecast next to what happened. Its data is
    `data/forecasts.json`, from `v2_report.py --site`, which the daily run
    refreshes.

  ```bash
  python scripts/fetch_recent_changes.py --days 2026-09-01 2026-09-29   # live days (recent changes keep 30)
  python scripts/daily_predictions.py --day 2026-09-30                  # rank ~1.5M pages, ~2 min
  python scripts/score_predictions.py --day 2026-09-29                  # how a finished day's prophecy did
  python scripts/build_site.py --serve                                  # preview the page at localhost:8000
  python scripts/run_daily.py                                           # a whole day: predict, score, build, publish
  ```

  A Windows scheduled task runs `run_daily.py` every evening, shortly after
  00:00 UTC. `publish_site.py` pushes the page to the `gh-pages` branch,
  which GitHub Pages serves.

  A 22-day backtest (September 2026) found about one in five of each day's
  top 100 burst the next day, and one in three of the top 10. That's double
  the "most edits yesterday" baseline, better on every day. Details are in
  `PLAN.md` §5, "Daily job backtest".

- **Version 2: what the edits will be** (in progress, `src/forecast/`).
  For a page in the prophecy, this forecasts which sections the day's
  edits will touch and what kinds of change they'll make (new prose,
  references, tables, …), from the page as it stood the day before. When
  it's published, each forecast will be labeled as a machine-generated
  guess, not news, and kept structured for pages about living people.

  ```bash
  python scripts/build_v2_targets.py      # choose page-days: the burst model's top pages, plus random ones
  python scripts/fetch_v2_examples.py     # fetch each page at the ends of D-2, D-1 and D, and derive the changes
  python scripts/build_stage2_titles.py --v2   # titles as of D-1, from the history dumps
  python scripts/v2_baselines.py          # score the baselines
  python scripts/train_v2.py              # QLoRA on Qwen2.5-1.5B; greedy-decoded test forecasts
  python scripts/rank_v2.py               # forecasts ranked by the model's own probabilities
  python scripts/v2_report.py             # a readable page of every test forecast next to what happened
  ```

  Results so far: the model's prompt holds the page's recent changes and
  the Stage 1 signals.
  - With its forecasts ranked by its own probabilities, it beats the
    baseline of repeating yesterday's changes on the kinds of change.
  - It ties that baseline on which sections change, once both name as many
    sections.
  - Nothing is published yet.

  Details are in `PLAN.md` §5, "Version 2 model: structured forecasts"
  and "Version 2: ranked forecasts".

- **Version 3: prophecies of real-world events** (in progress,
  `src/prophecy/`). A local model, Qwen2.5-7B-Instruct, reads the day's top
  100 pages one at a time, as they stood the night before. Where a page
  dates something to the day or the week after (the code marks every date,
  such as "October 4 [in 4 days]"), it writes predictions such as "I
  predict that the United States team will win the 2026 Presidents Cup".
  Each is due on the date the evidence gives.
  - It also reads the stories of Wikipedia's Portal:Current events from
    the week before, as they stood the night before, and asks what each
    brings next ("the Iran war", "the Moroccan general election").
  - A prediction the guardrails drop may be rewritten in general terms and
    checked again ("a prominent actor will die tomorrow").
  - Separate checks drop any prediction that names a specific person (none
    for now), or has an organization doing or suffering harm. They also
    drop predictions that restate their evidence, or copy or repeat
    another. Wars, disasters and crime are fine in general terms.
  - At most ten a day are published, world events first. Sport, read from
    the page itself, comes only when it settles a title, at most three a
    day.
  - On its due day a judge grades each one against what the cited pages
    gained by then and that day's Portal:Current events.
  - The judge is being checked against hand grades before anything is
    published.

  ```bash
  python scripts/build_v3_days.py           # the development days: the site's own prophecies, 2026-09-18 to 10-01
  python scripts/prophesy.py --days 2026-09-20 2026-09-25   # predictions, checks and selection; --model, --horizon
  python scripts/prophesy.py --rescreen --model Qwen/Qwen2.5-7B-Instruct --out DIR   # re-run only the checks and selection
  python scripts/fetch_current_events.py --days 2026-09-18 2026-10-01          # the judge's record, latest
  python scripts/fetch_current_events.py --days 2026-09-18 2026-10-01 --known  # the prophet's, as known the night before
  python scripts/fetch_due_pages.py --prophecies DIR   # each cited page at the end of a later due day
  python scripts/grading_packs.py --published          # each prediction's evidence, before and after, for grading
  python scripts/judge_prophecies.py        # the local judge; compares with confirmed hand grades if present
  python scripts/grading_review_docs.py     # documents for the hand-grading review page
  ```

  On the development days, the first version mostly named results due days
  or weeks later. Reading one page at a time, with the dates worked out,
  nearly every prediction is about something the evidence dates. With the
  stories of Portal:Current events, world events fill the day. Details are
  in `PLAN.md` §3 items 24–28 and §6 step 11.

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
