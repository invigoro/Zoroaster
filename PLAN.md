# Wikipedia Edit-Prediction Project — Status & Plan

This file is a handoff/status document for resuming this work on a different
machine. It captures the goal, the decisions already made, exactly what's
been built and validated so far, the findings from that validation, and the
concrete next steps. Read this before doing anything else if you're picking
this up cold.

Last updated: 2026-09-28.
- The code-vs-data review found five problems (§5).
- Revert labels and point-in-time features were rebuilt and validated
  (§3).
- The Stage 1 harness runs on both Simple Wikipedia and English Wikipedia.
  Results are in §5.
  - Page habits dominate. On English Wikipedia, 74 of the top 100 pages
    get edited the next day.
  - A page's own bursts add a tiny but consistent gain on English
    Wikipedia, and nothing on Simple Wikipedia.
  - Site-wide co-burst adds nothing on either, although on English
    Wikipedia its top days are plainly real events.
- Next candidate: link-neighbor bursts on English Wikipedia (§6 step 6),
  which needs a go-ahead for 11 GB of link tables.

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
  Precisely (clarified 2026-09-28, see §5): a revert restores the *most
  recent* earlier revision with its `sha1`, and it undoes the revisions
  strictly between that one and itself. The revert is flagged `is_revert`
  whatever the window. The undone revisions it reaches within the window
  are flagged `is_reverted`.
- **Point-in-time framing (confirmed 2026-09-28)**: for a prediction day D,
  the target is D's edits, and every input feature uses only information
  knowable by the end of day D−1. Note that the final `is_reverted` flag is
  *not* knowable then, because a revert can land up to 90 days later. So
  inputs only drop edits that were reverted by the end of the day they were
  made, while labels use the final flags. Consequence: labels for the last
  90 days before a dump aren't final, so evaluation windows must end at
  least 90 days before the dump date.
- **Stage 1 horizon: next day (decided 2026-09-28)**: the target is "does
  page P get a kept edit on day D?", rather than "within the next 7 days".
  The hypothesis is about short-lived "something is happening" signals, and
  bursts decay within days. Over a 7-day window, a page's long-run editing
  rate dominates and would mask exactly the effect being tested. A 7-day
  variant is a cheap relabel from the daily activity table if it's wanted
  later.
- **Streaming pipeline (decided 2026-09-28)**: pipeline code processes one
  page at a time and writes Parquet in row groups, so memory is bounded by
  the largest single page history rather than the corpus. The same code has
  to run on English Wikipedia, which is roughly 100× the test corpus.
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
  parquet_io.py        # RowGroupWriter: streaming Parquet output in row groups, atomic replace
  ingest/
    stub_stream.py      # streams a stub-meta-history dump -> mainspace RevisionRecord dicts
    revert_detect.py    # identity reverts, one page at a time: is_revert, is_reverted,
                        #   reverted_by_revision_id (15-rev/90-day window, §2)
    sampling.py          # stratified page sampling (edit-frequency x had_burst)
    fetch_diffs.py        # per-revision diff fetch via MediaWiki API — validated live, see §5
    sql_dump.py           # streaming reader for MediaWiki SQL table dumps (*.sql.gz)
    link_graph.py         # mainspace page->page links from page/redirect/linktarget/pagelinks
    mediawiki_history.py  # parser for the 78-column MediaWiki history TSV (English Wikipedia)
  features/
    activity.py           # per-page daily channels + point-in-time Stage 1 features;
                          #   mass-editor days and the mass-editor-discounted burst definition
    bursts.py             # causal burst z-scores + cross-page co-burst counter
    neighbors.py          # link-neighbor burst features (vectorized over CSR adjacency)
  stage1/
    panel.py              # the (page, day) row contract shared by the panel and eval builders
    splits.py             # train/validation/test windows with 90-day embargo gaps
    features.py           # nested feature sets (habits / +burst / +co_burst) + context columns
    baselines.py          # heuristic rankings (yesterday's edits, recency, ...)
    metrics.py            # per-day ranking metrics (precision@k, recall@k, average precision)
scripts/
  download_dump.py        # generic Wikimedia dump downloader (User-Agent-policy compliant)
  build_test_revert_labels.py  # orchestrates: simplewiki dump -> parse -> revert-detect -> parquet
                                # (--from-parquet re-labels an existing file in ~2 min, no re-parse)
  build_test_features.py       # orchestrates: labels -> daily activity, co-burst, Stage 1 panel,
                                # stratified sample manifest
  build_test_eval_days.py      # every existing page on 29 test days (exact per-day ranking metrics)
  build_test_links.py          # downloads the 20260701 SQL tables (~154MB) -> link graph parquet
  download_enwiki_history.py   # English Wikipedia MediaWiki history, 2023-06..2026-09 (~22GB)
  build_enwiki_revisions.py    # months in parallel -> page-bucketed Parquet (128 buckets)
  build_enwiki_labels.py       # per bucket: revert_detect on content hashes + editor-day counts
  build_enwiki_features.py     # per bucket: sampled panel, full-day eval set, co-burst, site totals
  train_stage1.py              # Stage 1 harness: baselines + LightGBM ablation (--corpus simplewiki|enwiki)
  build_test_diffs.py          # orchestrates: sampled pages' retained revisions -> live fetch_diffs.py calls -> parquet
                                # takes an optional `max_revisions` arg to cap a run short of the full sample
tests/
  test_revert_detect.py        # revert semantics, incl. the repeated-vandalism case the old code got wrong
  test_activity_features.py    # leakage test: features for day D computed from the full history must
                                # equal features from a history cut off at midnight before D
  test_stage1.py               # splits/embargo, metrics, feature sets (no label columns), baselines
  test_link_graph.py           # SQL dump parsing (tricky strings, fails loudly) + link resolution
  test_neighbors.py            # neighbor-burst features, incl. a randomized point-in-time check
  test_mediawiki_history.py    # history TSV parsing, filtering, flags
  test_sampling.py             # hash page sample: rate, independence from buckets, nesting
  test_download_dump.py        # downloads are atomic and size-checked (against a local server)
CLAUDE.md                  # working SOP for Claude sessions (commit/push as you go, tests alongside code)
main.py                    # original Wikipedia-summary CLI (unchanged behavior, now imports USER_AGENT from src.common)
requirements.txt           # requests, mwxml, pyarrow, numpy, lightgbm
```

Everything under `src/` and `scripts/` streams one page at a time (§2), so
memory is bounded by the largest single page history. Run the tests with
`python -m unittest discover -s tests` from the repo root. They need no
network and no data files.

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
   `difflib`). Originally validated only offline with mocked HTTP responses;
   **now also validated live** (see §5) via `scripts/build_test_diffs.py`
   against the first 300 of the 4,076 retained revisions across the 125
   sampled pages — 300/300 fetched, 0 failures. The full 4,076-revision run
   (~2+ hours at the polite 1 req/sec rate) has **not** been run yet; that
   was deliberately deferred pending a decision (see §6) since it's a much
   longer live-network job than the original "a few hundred calls" estimate.

Items 1–3 above are the **v1 run (2026-07-08)**. Its outputs were moved to
`data/processed/v1_2026-07-08/` when they were superseded:

4. **Re-label (2026-09-28)**: `python scripts/build_test_revert_labels.py
   --from-parquet data/processed/v1_2026-07-08/simplewiki_test_revert_labels.parquet`
   took 112s, streaming.
   - Same 6,953,329 revisions and 397,196 pages.
   - **819,543 reverted (11.79%)** and **607,603 reverts (8.74%)**, of which
     101,321 are both (revert wars). 5,627,504 revisions carry neither flag.
   - Every difference from v1 is accounted for:
     - 170,803 v1 flags were on reverts;
     - 3,633 were on null revisions (page moves), which are neither;
     - 152 revisions are newly flagged, all on the 186 pages whose
       timestamps run out of order. There the new pairwise 90-day check
       reaches a revert that v1's stop-at-first-late-revision scan missed.
   - Parsing straight from the dump gives identical labels (checked on the
     first 30K revisions).
5. **Features (2026-09-28)**: `python scripts/build_test_features.py` took
   199s, streaming.
   - **193,166 burst page-days across 124,405 pages**. 76,060 of those
     pages have fewer than 10 active days; v1 made that impossible.
   - **Panel: 8,837,634 rows, 0.80GB in memory.** That's all 2,190,870
     positive (page, day) rows, plus 6,646,764 negatives sampled at 0.5% of
     the 1.33B negative page-days (`sample_weight` 200).
     - 68% of positive days follow zero human edits in the prior 30 days.
     - Labels are final through 2026-04-02.
   - Validation:
     - the panel's positives equal the daily table's kept-edit days exactly;
     - the leakage check (features from the full history vs. a history cut
       at midnight before D) passed on 21,283 (page, day) pairs from 2,000
       random real pages.
   - New stratified sample: 6 strata (`low`+burst now exists) × 25 = 150
     pages, with 4,140 diff targets (38.6% bot edits).
6. **Evaluation set (2026-09-28)**: `python scripts/build_test_eval_days.py`
   took 159s.
   - Every existing page on 29 test days (every 13th day from 2025-04-03 to
     2026-04-02): 11,090,976 rows and 12,589 positives (208–743 per day).
   - Every panel row on those days reappears with identical values.
7. **Stage 1 harness (2026-09-28)**: `python scripts/train_stage1.py`.
   - Windows: train 2018-10-07..2023-10-05 (3.02M panel rows), validation
     2024-01-04..2025-01-02 (824K), test 2025-04-03..2026-04-02.
   - Deterministic: two runs gave identical metrics.
   - Results and interpretation are in §5, "Stage 1 harness results".
8. **Link graph (2026-09-28)**: `python scripts/build_test_links.py`.
   - Downloaded 153.4MB in about a minute and built the graph in 45s.
   - 12,879,453 unique mainspace links. The median page has 15 out-links
     and 8 in-links.
   - Every link endpoint is a page in the revision data.
   - Top hubs are template-driven: *Geographic coordinate system* (35,534
     in-links), *United States*, *Wayback Machine*.
9. **Mass-editor discount (2026-09-28)**: rebuilt `build_test_features.py`
   with a pre-pass.
   - Adds `is_burst_excl_mass` / `editors_excl_mass` to the daily table
     and `pages_bursting_excl_mass` to the co-burst table.
   - 146,488 burst page-days once mass editing is excluded, vs 193,166.
   - The panel, manifest and all existing columns are unchanged
     (verified).
10. **Step 4 ablation (2026-09-28)**: `python scripts/train_stage1.py`
    again, now with the link-count and link-neighbor feature sets. Results
    are in §5, "Step 4 results".
11. **English Wikipedia data (2026-09-28)**:
    - Downloaded 39 monthly files (22.3 GB) in about 50 minutes, two at a
      time.
    - `build_enwiki_revisions.py`: 133,031,385 mainspace revisions into 128
      page buckets.
    - `build_enwiki_labels.py`: 555s with 8 workers.
    - `build_enwiki_features.py`: 349s. 1,936,499 of 9,681,562 pages
      sampled; panel 7.59M rows; evaluation set 25.2M rows over 14 test
      days.
    - Disk: 21 GB raw, 15 GB processed. The 7 GB of bucketed revisions are
      intermediate (the labels contain them) and can be deleted unless
      labels need re-running.
12. **English Wikipedia harness (2026-09-28)**: `python
    scripts/train_stage1.py --corpus enwiki`.
    - Windows: train 2024-06-07..2024-12-05, validation
      2025-03-06..2025-09-03, test 2025-12-03..2026-06-02.
    - Results are in §5, "English Wikipedia results".

## 4. Data state — important for resuming on a new machine

`data/` is **git-ignored** (see `.gitignore`) — it does not travel with the
repo. On a fresh clone, `data/` will not exist. To regenerate the exact same
test dataset:

```bash
python -m venv .venv
# then activate it (see README.md for OS-specific activation commands)
pip install -r requirements.txt
python scripts/build_test_revert_labels.py   # ~35 min parse time, ~900MB download
python scripts/build_test_features.py         # ~3.5 min, runs against the local parquet, no download
python scripts/build_test_eval_days.py        # ~3 min, full-day Stage 1 evaluation set
python scripts/build_test_links.py            # ~2 min, ~154MB download, link graph
python scripts/train_stage1.py                # ~4 min, Stage 1 baselines + LightGBM ablation
python -m unittest discover -s tests          # ~5s, no network or data needed
```

English Wikipedia (about 22GB of downloads and roughly 10GB of Parquet;
each step skips work that's already done):

```bash
python scripts/download_enwiki_history.py      # ~50 min at ~7 MB/s, two files at a time
python scripts/build_enwiki_revisions.py       # months -> page buckets, parallel
python scripts/build_enwiki_labels.py          # revert labels per bucket
python scripts/build_enwiki_features.py        # panel, eval set, co-burst, site totals
python scripts/train_stage1.py --corpus enwiki
```

To re-run revert detection after changing it, use `--from-parquet` on the
existing labels file (~2 min) instead of re-parsing the dump. The
superseded v1 outputs (2026-07-08) are in `data/processed/v1_2026-07-08/`
for comparison. They're safe to delete.

Caveat: `build_test_revert_labels.py` downloads
`simplewiki-latest-stub-meta-history.xml.gz`, i.e. **whatever the current
latest dump is** — Wikimedia doesn't keep old "latest" dumps at that URL, so
the exact revision counts above may drift slightly on a re-run months later
as new edits accumulate. This hasn't mattered yet but would matter once
we care about exact reproducibility (at that point, pin to a dated dump
directory like `/simplewiki/20260701/` instead of `/simplewiki/latest/`).

## 5. Findings from validation (worth knowing before continuing)

### Revert detection — hand-validated (see the 2026-09-28 correction below)

Hand-checked 5 small pages (Animals, Muslim, Hallucinations, Full English,
Jewish) by reading their real edit-comment history. **Every single revision
flagged `is_reverted=True` was immediately followed by a revision whose
comment literally said "Reverted N edit(s) by ..."**, and the mechanical
revert revision itself was correctly *not* flagged as reverted. High
confidence in this heuristic as implemented.

**Correction (2026-09-28):** that confidence was misplaced for pages with
repeated vandalize/restore cycles, which none of the five small hand-checked
pages had. See "Code review against the data" below.

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

### fetch_diffs.py — validated live, but full-sample scope is bigger than planned

Ran `scripts/build_test_diffs.py 300` against real `simple.wikipedia.org`:
300/300 retained revisions fetched, 0 failures, output written to
`data/processed/simplewiki_test_diffs.parquet` (now in
`data/processed/v1_2026-07-08/`, since the sample it was drawn from has been
superseded). Spot-checked the content —
added/removed text is sane wikitext (page-creation text, interwiki links,
`{{msg:stub}}` template additions correctly showing as added-only diffs with
no removed text). One false alarm during review: a page title containing
"ü" (`Baden-Württemberg`) printed as `�` in the terminal — checked the raw
codepoints and UTF-8 bytes directly, confirmed this is a Windows console
codepage display artifact only, not a data-encoding bug.

**Scope correction**: the plan's original "a few hundred network calls"
estimate for the 125-page sample was wrong. The actual count is **4,076
retained revisions**, which at the polite 1 req/sec rate (with ~2 HTTP
calls per revision — the revision itself plus its parent) is **~2+ hours**
of continuous live-network runtime, not a quick validation step. The 300-row
cap was used instead to keep the validation pass small; the full run was
deliberately not started pending a decision on whether to just run it, or
fold it into whatever sample gets pulled for the real corpus (see §6).

### Code review against the data (2026-09-28)

I re-checked the code against the local test corpus after a ~12-week
break. Every headline count above reproduces from the files on disk (the
burst tally was off by 2 of 54K, which doesn't matter). But the check found
five problems. None of them break the plumbing, but each gets much more
expensive once it's baked into an English Wikipedia corpus. Items 1, 2, 3
and 5 were fixed the same day (§3 items 4–5). Item 4 stays open until the
Stage 2 fetch is reworked (§6 step 8).

1. **The revert rule from §2 wasn't implemented as decided.**
   - Nothing flagged the revert revisions themselves. So **335,488 identity
     reverts** (5.63% of the revisions not flagged `is_reverted`) counted as
     retained: they fed burst detection and were eligible as Stage 2
     targets.
   - `revert_detect.py` stored the *earliest* index at which each `sha1`
     appeared, not the most recent. On pages that are vandalized and
     restored repeatedly, each intermediate restore got flagged as reverted
     by the next one. **174,439 of the 993,827 `is_reverted` flags (17.55%)**
     sat on revisions that are themselves reverts, e.g. "Reverted 1 edit by
     … identified as vandalism" on *April*. The true rate of undone edits is
     ~11.8%, not 14.29%.
2. **The burst features used future data.** `compute_page_bursts` used each
   page's whole history, including later days and the day itself, as its
   baseline. Any forecaster trained on those features would be seeing the
   future.
3. **Pages with fewer than 10 active days could never burst.** The z-score
   was a population z-score over active days only. By Samuelson's
   inequality, one value's z-score can't exceed √(n−1), so reaching z ≥ 3
   takes at least 10 active days. **318,601 pages (81%)** couldn't burst at
   all, including 102 with 50+ edits in a single day, which is exactly what
   a breaking-news article looks like.
4. **The Stage 2 diff targets are noisy.**
   - Of the 4,076-revision fetch target set, **1,381 (33.9%) are bot edits**
     (nothing filters bots from Stage 2 targets) and 263 (6.5%) are reverts
     (see 1).
   - Diffs are line-level, and a wikitext paragraph is one line, so a
     one-word fix records the whole paragraph as added. On the 200 fetched
     diffs that modify existing text, `added_text` had **3.6× the words
     actually inserted** (median 2.9× per diff, p90 25×). That inflates both
     the Stage 2 targets and the storage math behind the 25GB budget.
5. **The scripts don't scale.** Every script loaded the whole dataset into
   memory (`list(...)`, `to_pylist()`). That's fine for 7M revisions but not
   for English Wikipedia, at roughly 100× that.

Two more findings came up while designing the fixes:

- **A point-in-time trap in the revert flags.** `is_reverted` depends on
  revisions up to 90 days *later*. So even features built only from days
  before the target day leak if they filter on the final flag. The rule
  adopted is in §2.
- **Most edits land on dormant pages.** Of the 2.82M page-days with a
  non-bot, non-reverted edit, only 17% follow any human edit to that page
  in the previous 7 days. That rises to 30% with a 30-day lookback and 44%
  with 90 days (these figures use the old revert flags). So a Stage 1
  candidate set of "recently active pages" would miss most of the target.
  The panel has to cover every page's lifetime, keeping all positive days
  and sampling negative days with weights.

### Top co-burst days after the rewrite (2026-09-28)

With the causal burst definition, the top co-burst days are dominated by
**mass editing from human accounts**, which the name-based bot filter can't
catch, rather than by coincident single-page sessions:

- **2019-02-27** (278 pages): one editor made 929 "fix template param case"
  edits across 272 of the pages.
- **2012-02-04** (169 pages): two separate maintenance runs overlapped on the
  same ice-hockey biographies (AWB category sorting plus a category
  removal).
- **2023-01-01** (149 pages): a category-move run.
- **2021-02-13 and 2010-01-05**: batches of new pages (74 and 72 created
  that day), plus new-page patrol.

The "≥2 distinct editors" variant drops the 2019 day (7 of 278 pages qualify)
but is fooled when two maintenance runs overlap (2012-02-04: 149 of 169
qualify). Only a sliver of the top days is news: 2010-01-05 includes
*Casey Johnson*, who had died the day before, and *Deaths in 2010*.

The natural next filter is to discount editors who touch many distinct
pages in one day, since a single human can't react to that many separate
events. Per the rule above, it's recorded as an open question (§7) rather
than tuned here.

### Stage 1 harness results (2026-09-28, Simple Wikipedia, next-day)

The task: rank every existing page each day by how likely it is to get a
kept edit tomorrow. On a test day, about 434 of ~382K pages get one (a
0.114% base rate), so a random ranking scores about 0.001 precision@100.
The table shows per-day metrics averaged over the 29 evaluation days:

| model | P@100 | P@1000 | R@1000 | R@10000 | AP |
|---|---|---|---|---|---|
| edits yesterday (best heuristic) | 0.146 | 0.051 | 0.127 | 0.332 | 0.0295 |
| most recent edit | 0.139 | 0.044 | 0.109 | 0.305 | 0.0267 |
| edits last 30 days | 0.119 | 0.048 | 0.117 | 0.329 | 0.0239 |
| edits last year | 0.072 | 0.033 | 0.082 | 0.270 | 0.0148 |
| LightGBM: habits | 0.188 | 0.061 | 0.152 | 0.351 | 0.0402 |
| LightGBM: + burst | 0.192 | 0.062 | 0.153 | 0.353 | 0.0407 |
| LightGBM: + co-burst | 0.186 | 0.061 | 0.151 | 0.349 | 0.0404 |

Paired per-day differences (mean ± standard error; days better/worse):

- **habits vs the best heuristic:** P@100 +0.042 ± 0.007 (26/3), AP
  +0.0107 ± 0.0016 (27/2). The learned model is clearly better, by about 6
  standard errors.
- **+ burst vs habits:** P@100 +0.004 ± 0.003 (15/6), AP +0.0004 ± 0.0003.
  Not significant. Burst features get ~0.5% of the model's gain, because
  recent raw counts already capture a burst.
- **+ co-burst vs + burst:** P@100 −0.006 ± 0.004 (6/14), AP −0.0003 ±
  0.0004. Not significant.

What this means:

- **Next-day edits are predictable mostly from page habits.** Days since
  the last edit carries 41–53% of the gain, then site-wide edits
  yesterday (16–19%), edits in the last year, and page age.
- **Co-burst can't help as defined, by construction.** Every page gets the
  same count on a given day, so it can't reorder pages within that day. It
  does improve validation loss (0.4193 → 0.4165, and it takes ~10% of the
  gain), because it predicts how busy each *day* is, but that doesn't
  change a within-day ranking. `site_edits_1d` works the same way.
  Testing the "something is happening" hypothesis needs a
  **page-specific** cross-page signal: whether pages *related to this one*
  were bursting (§6 step 4).
- **The ceiling from a page's own history is low.** 66% of test-day
  positives had no human edit in the previous 30 days, and 28% had none in
  the previous year. Even the top 10,000 pages (2.6% of all pages) catch
  only 35% of the next day's edits.

This is one wiki at small scale, so it isn't a verdict on the hypothesis.
But the harness is working, leak-free and reproducible, and the negative
co-burst result is structural, not a Simple Wikipedia artifact: it would
recur on English Wikipedia.

### Step 4 results: link-neighbor bursts and the mass-editor discount (2026-09-28)

**Mass editing is common.**
- Editor-days with non-revert edits to more than 25 distinct pages are
  only 1.6% of editor-days (16,189 of them, by 2,515 editors). But they
  account for **41% of all human editor-page-days** (1.33M of 3.22M).
- Excluding them removes 24% of burst page-days (193,166 → 146,488),
  including all three maintenance days at the top of the co-burst list.
- What's left at the top is a mix. Some are real events:
  - 2022-08-01: *Ayman al-Zawahiri*, whose killing was announced that day;
  - 2024-01-01: *2024 Noto earthquake*, the year pages *2023* and *2024*,
    and *Cale Yarborough*, who had died the day before.

  Others are diffuse busy days of unrelated pages. 2009-02-23, for
  instance, is 97 general-topic pages (*Cat*, *Bicycle*, *Black hole*),
  probably a class editing project.
- On a small wiki, then, site-wide co-burst mostly measures busy days.

**Link neighbors carry raw signal.** 0.82% of evaluation rows had a link
neighbor bursting the day before. Those rows were positive 6.8× as often as
average (6.4× with mass editing excluded).

**But they add little beyond what the model already knows.** The link sets
extend a control that has the page's own link counts, so the neighbor-burst
effect is isolated. Paired per-day differences:

| comparison | recall@10000 | P@100, P@1000, R@1000, AP |
|---|---|---|
| + link counts vs habits + burst | +0.0068 ± 0.0021 (t = 3.2) | no change |
| + neighbor bursts vs link counts | +0.0045 ± 0.0022 (t = 2.0) | no change |
| + neighbor bursts, mass editing excluded, vs link counts | +0.0055 ± 0.0019 (t = 2.9) | no change |
| mass editing excluded vs not | +0.0010 ± 0.0011 | no change |

What this means:
- **The page-specific cross-page signal helps slightly**, in the direction
  the hypothesis predicts, and only deep in the ranking: about half a
  percentage point of recall within the top 10,000 pages. It doesn't
  improve the model's top picks.
- **Excluding mass editing cleans up the co-burst lists** but doesn't
  measurably change the model.
- **The page's own link counts help more than neighbor bursts, but that
  gain is suspect.** The counts come from the July 2026 snapshot, so pages
  edited heavily during the test year had grown links by then; `out_links`
  took 29% of the gain. A point-in-time size measure would test whether the
  gain is real: the byte size of the latest revision before D, which the
  revision metadata already has (§6 step 5).
- **Both link-based gains are optimistic**, because the link snapshot
  postdates most of the data.
- **Several metrics and comparisons were checked.** A t of 2 on one metric
  is weak evidence on its own.

Bottom line for Simple Wikipedia: next-day edits are predictable mainly
from each page's own habits. "Something is happening" signals, whether
site-wide or through links, add little. As §5 anticipated, that may be a
property of a small, low-traffic wiki, so English Wikipedia is the real
test.

### Point-in-time page size (2026-09-28)

`page_bytes` (the page's size at the end of D−1, from the revision
metadata) was added to the habits features. It's point-in-time, unlike the
link counts. The tables above predate it. With it:

- **It's the model's top feature,** with 35% of the gain. Validation AUC
  for habits rises from 0.787 to 0.807, and recall@10000 from 0.351 to
  0.357. Precision@100 is unchanged (0.187).
- **Most of what looked like a link-count effect was article size.**
  `out_links`' share of the gain falls from 29% to 4.7%. Link counts still
  add recall@10000 +0.0052 ± 0.0015 (t = 3.4), down from +0.0068. That
  remainder is either centrality or the snapshot leak.
- **Neighbor bursts are unchanged:** +0.0054 ± 0.0019 recall@10000 over
  link counts (t = 2.9), or +0.0035 ± 0.0017 with mass editing excluded,
  and nothing at the top of the ranking.

### English Wikipedia results (2026-09-28)

**The corpus**
- 133.0M mainspace revisions across 9.68M pages, June 2023 to August 2026
  (MediaWiki history dumps, 2026-08 snapshot).
- 7.14% of revisions reverted, 5.05% reverts.
- Every revert and reverted edit our `revert_detect` finds is also flagged
  by Wikimedia. Wikimedia flags about 12% more reverted edits and 6% more
  reverts, as expected: their definition has no 15-revision/90-day window,
  and they can see states from before our window.
- Mass editing is 1.6% of editor-days but 41% of human editor-page-days,
  the same split as Simple Wikipedia.

**The phenomenon is real here.** With mass editing excluded, the top
co-burst days are real-world events, with dozens to over a hundred editors
on the central page:
- 2024-07-05, UK general election: the election page (158 editors),
  *Keir Starmer* (103), *Starmer ministry*.
- 2024-06-04, Indian general election results.
- 2024-02-05: the Grammys, *The Tortured Poets Department* (announced
  there) and *Charles III* (cancer diagnosis announced).
- 2025-01-02, the Las Vegas Cybertruck explosion.
- 2025-01-06, Trudeau's resignation.

The regular list looks similar: 2026-07-19 (the World Cup final, *Spain
national football team*), 2024-11-30 (the Syrian offensive and *Battle of
Aleppo (2024)*), 2024-12-06 (the annulled Romanian presidential election).
This is unlike Simple Wikipedia, where top days were maintenance runs and
busy days.

**Window-start artifact.** Burst baselines are truncated in the first 90
days of data, because no history before June 2023 was downloaded. So early
June 2023 days top the discounted co-burst list. This only affects the
lookback year, never a train, validation or test row, whose 90-day
baselines are complete.

**Harness.** Every sampled page (20%) is ranked on 14 test days. About
11,200 pages a day get a kept edit, a base rate of 0.62%:

| model | P@100 | P@1000 | R@10000 | AP |
|---|---|---|---|---|
| edits yesterday (best heuristic by P@100) | 0.578 | 0.315 | 0.092 | 0.0572 |
| LightGBM: habits | 0.742 | 0.435 | 0.149 | 0.0859 |
| + burst | 0.741 | 0.436 | 0.150 | 0.0865 |
| + co-burst | 0.739 | 0.435 | 0.149 | 0.0864 |

Paired per-day differences:
- **habits vs the best heuristic:** P@100 +0.164 ± 0.011 and AP +0.0287 ±
  0.0011, better on all 14 days (t = 15–25).
- **+ burst vs habits:** AP **+0.0006 ± 0.0001, better on 14 of 14 days
  (t = 9.9)**; recall@10000 +0.0007 ± 0.0002 (t = 2.9). P@100 and P@1000
  don't change. `burst_z_1d` takes 0.9% of the gain.
- **+ co-burst vs + burst:** nothing on any metric (|t| ≤ 1.3).

What this means:
- **Next-day edits are very predictable from page habits.** 74 of the top
  100 sampled pages get a kept edit the next day. The gain goes to edits in
  the last year (49%), page size (22%) and days since the last edit.
- **A page's own burst signal gives the first unambiguous evidence for the
  hypothesis:** a gain on every test day. But it's tiny, about 0.7% of AP.
- **Site-wide co-burst again adds nothing.** It's the same number for every
  page on a day. The events are plainly visible in co-burst, but a count
  can't say which pages they'll touch. The page-specific version, link
  neighbors, is the natural next test (§6 step 6).
- **Checked:** the leakage check on real pages (features from the full
  history vs a history cut at midnight before D, including source bot
  flags, creation dates and the mass-editor discount) passed on 7,143
  (page, day) pairs from 600 pages.
- **Comparability with Simple Wikipedia:** the universe here is pages with
  a revision since June 2023, sampled at 20%. 55% of next-day positives had
  no human edit in the prior 30 days (66% on Simple Wikipedia), and 9.7% had
  none in the prior year (28%).

## 6. Next steps, in order

Re-planned 2026-09-28 after the code review in §5, and again after the
Stage 1 results. The old step 1 (run the full 4,076-revision Simple
Wikipedia diff fetch) is **skipped**. Stage 1 needs no diff text, and the
fetch method needs the changes listed in step 7 before any large run. The
old step 2 (move to English Wikipedia) is now step 5.

1. **Fix the revert labels** (**done 2026-09-28**, see §3 item 4). Split them into `is_reverted`
   (most-recent-match semantics), `reverted_by_revision_id` (when the revert
   happened, which point-in-time features need) and `is_revert`. Exclude
   both kinds of revision from targets. Recompute from the existing labels
   parquet, with no dump re-parse.
2. **Point-in-time features** (**done 2026-09-28**, see §3 item 5):
   - a causal burst baseline over calendar days, with zero-edit days
     included and new pages able to burst;
   - raw activity features: edits in the last 1/7/30/365 days, days since
     the last edit, page age, distinct editors;
   - co-burst counts from the causal burst flags.

   Build a Stage 1 panel that keeps every positive (page, day) and samples
   negative days across each page's whole lifetime, with weights (see
   "most edits land on dormant pages" in §5).
3. **Stage 1 harness on Simple Wikipedia** (**done 2026-09-28**, see §3
   items 6–7 and §5). The setup:
   - next-day horizon (§2);
   - LightGBM trained unweighted on the case-control panel;
   - exact per-day ranking metrics on a full-day evaluation set, with
     paired per-day comparisons.

   Result: a learned model on page habits clearly beats every heuristic.
   Neither burst nor co-burst features add measurable value, and co-burst
   *can't* help as defined (§5).
4. **Page-specific cross-page signal** (**done 2026-09-28**, see §3 items
   8–10 and §5 "Step 4 results"). Neighbor bursts add a small,
   borderline-significant amount of deep recall and nothing at the top of
   the ranking. The mass-editor discount doesn't change the model. As
   planned:
   - Pull Simple Wikipedia's `pagelinks` dump, plus `linktarget` / `page`
     to resolve link targets to page ids. That's Wikipedia-internal data, so
     it's within the §2 scope decision. `categorylinks` is an optional
     second relation.
   - Features for page P on day D: how many of P's in- and out-link
     neighbors were bursting on D−1 and over the last 7 days, raw and
     normalized by degree.
   - Caveat: link dumps are a *current* snapshot, so links created after D
     (including ones added *because of* an event) leak into features for
     historical days. Evaluate on the test year only, which is closest to
     the snapshot, and treat the result as optimistic. Reconstructing
     historical links from revision text would remove the leak, at much
     higher cost.
   - Also try the mass-editor discount from §5 in the burst definition,
     then rerun the ablation.
5. **Move to English Wikipedia** (**done 2026-09-28**, see §3 items 11–12
   and §5 "English Wikipedia results"). Page habits are very predictive
   (P@100 0.74). The page's own bursts add a tiny but consistent gain, and
   site-wide co-burst adds nothing, although the top co-burst days are
   clearly real events. What was planned:

   **Sizes, checked 2026-09-28:**
   - MediaWiki history, 2026-08 snapshot: 309 English Wikipedia files,
     137.9 GB for all history. The 24 months of 2024–2025 are 14.2 GB,
     about 0.6 GB per month.
   - Features need about a year of history before the first prediction
     day, so a two-year evaluation setup means roughly three years of
     files, around 20 GB.
   - The link tables would add 11.15 GB (`pagelinks` alone is 7.13 GB).
     Given how little links added here, start without them.

   **Done first, on Simple Wikipedia:** the point-in-time article size
   feature `page_bytes` (§5, "Point-in-time page size"). It absorbed most
   of the link counts' apparent value and strengthened the habits baseline.

   **Needed for English Wikipedia:**
   - Restrict any link graph to the pages being scored; it won't fit in
     memory whole.
   - Vectorize or sample pages in the feature build (pure Python took 3.5
     minutes here; ~100× that is hours).

   Evaluate Wikimedia's MediaWiki history dumps
   (https://dumps.wikimedia.org/other/mediawiki_history/readme.html) as the
   source instead of parsing ~25 years of stub XML:
   - English Wikipedia is split into monthly TSV.bz2 files (≲2GB each), so
     you can pull only the window Stage 1 needs.
   - Each revision already has `revision_text_sha1`,
     `revision_is_identity_revert`, `revision_is_identity_reverted`,
     `revision_seconds_to_identity_revert` and `event_user_is_bot_by`
     (`name` or `group`, i.e. real bot user-group membership).
   - The dumps document no revert window, so apply the §2 rule on top, or
     rerun `revert_detect.py` on the sha1 column so both wikis use one
     definition.
   - Records change retroactively between monthly snapshots (renames,
     reverts, moves), so take every month from a single snapshot.
6. **Link-neighbor bursts on English Wikipedia** (**next candidate**; needs
   go-ahead for 11.15 GB of link tables, `pagelinks` alone 7.13 GB). On
   English Wikipedia the events are visible in co-burst, but a site-wide
   count can't point at pages. "Pages linked to this one are bursting" is
   the page-specific version, and the direct test of the hypothesis where
   it can actually show up.
   - The code exists (`src/ingest/link_graph.py`, `src/features/neighbors.py`).
   - The graph needs restricting to pages near the sampled ones to fit in
     memory: roughly 1B+ links in total.
   - The snapshot-leak caveat from §5 applies.
   - Alternative, cheaper next steps: a 7-day horizon variant, or moving
     on to Stage 2 (step 8). Stage 2's gate below is only partly met: the
     burst signal is real but tiny.
7. **Pageview-based popularity stratum.** Unchanged; deferred until ready
   to pull the pageviews dumps.
8. **Stage 2 QLoRA fine-tuning** on the RTX 3070, only after Stage 1 shows
   the burst/co-burst signal is predictive. Before any large diff fetch:
   - switch to word-level diffs (keep line-level context separately if
     useful as conditioning);
   - exclude bot edits and reverts from targets;
   - batch up to 50 revision ids per API request (a revision's parent is
     usually the previous revision already being fetched);
   - write output incrementally (`build_test_diffs.py` currently writes only
     at the end, so a crash loses the whole run).

## 7. Open questions

- Co-burst signal validity at Simple-Wikipedia scale (§5). Needs English
  Wikipedia data to resolve, not more tuning on the current corpus. The
  2026-09-28 rewrite adds a "≥2 distinct editors" co-burst variant,
  targeting the single-editor-session failure mode from §5, as an extra
  feature rather than a replacement. After the rewrite, the top co-burst days
  are mass maintenance by human accounts (§5). **Tried 2026-09-28** (step
  4): discounting editors who touch more than 25 pages in a day removes the
  maintenance days, but leaves a mix of real events and diffuse busy days.
  It didn't change the model on Simple Wikipedia. Worth re-testing on
  English Wikipedia; edit tags (e.g. AWB) are an alternative where the
  source has them.
- How much of the link-based gains is snapshot leakage (§5 "Step 4
  results")? Two ways to find out:
  - a point-in-time article-size feature, for the link-count part (§6
    step 5);
  - links reconstructed as of each day from revision text, for the
    neighbor-burst part (expensive).
- Whether the bot heuristic (`"bot" in username`) is good enough. This goes
  away for English Wikipedia if the MediaWiki history dumps are adopted
  (`event_user_is_bot_by` is group-based). The name heuristic stays for the
  Simple Wikipedia test corpus.
- Should Stage 2 targets exclude bot edits? Recommended yes: 34% of the v1
  target set and 39% of the current one are bot maintenance (interwiki
  links, stub tags), not reactions to events.
- Stage 1 task details. **Settled 2026-09-28:** next-day horizon (§2) and
  per-day ranking metrics on a full-day evaluation set. **Still open:**
  - whether a 7-day variant is worth adding;
  - how to treat the ~66% of next-day edits that land on pages with no
    human edit in the previous 30 days (§5). A page's own history can't
    anticipate those; only page-specific cross-page signals (§6 step 4)
    might.
- Whether to refit the final model on train + validation before scoring
  the test window. It currently trains on data ending ~18 months before
  the test year. That's conservative, and fine while the features are all
  relative (counts, recency).
