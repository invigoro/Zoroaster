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
- On English Wikipedia, link-neighbor bursts add a small, consistent gain
  (+1.6% relative AP in total).
- Stage 2's first run works: fine-tuning cuts inserted-text perplexity
  15.6 → 6.6. Trigger text helps only on edits whose linked pages were
  bursting (−0.010 nats/token, t ≈ 2.5). Next steps are in §6 step 8.

## 1. Goal

Predict **which Wikipedia pages will be edited next** and **what content
that edit will add**, treating edits that were later reverted as noise to
exclude, not signal to learn from. The working hypothesis is that a
meaningful share of edits are reactions to real-world events unfolding, so
the model should pick up on "something is happening" signals rather than
only long-run per-page editing habits.

**The end goal** (stated 2026-10-01) is to predict major real-world events,
not Wikipedia itself. Wikipedia's edits are a fast, public trace of
events. The next day's edits are also a record to check a prediction
against. Predicting edits (Stages 1 and 2, version 2) is the means. The
prophecy is meant to become free text, "I predict that X will happen",
written and graded the next day by local LLM steps (§6 step 11).

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
- **Stage 2 design (decided 2026-09-28)**:
  - **Model:** QLoRA on Qwen2.5-0.5B (Apache-2.0): a 4-bit NF4 base with
    LoRA r=16 on every attention/MLP projection, on the RTX 3070.
  - **Examples:** random Stage 1 positive (page, day) rows on English
    Wikipedia: 12,000 train, 1,000 validation, and 3,000 test (from the
    Stage 1 test days, Dec 2025–Jun 2026, after the base model's training
    data). The target is the page's first kept revision that day.
  - **Target:** its inserted text from a whitespace-word diff against the
    parent, with nearby changes merged.
  - **Prompt:** page, date, section, and the parent text around the change,
    marked where the edit goes.
  - **Comparison:** the same model trained with and without *trigger text*:
    the Stage 1 signals, including titles of linked pages bursting the day
    before. It's scored by per-token NLL of the inserted text on the test
    split, pairwise, and on the subset naming a bursting linked page.
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
- **Version 3's guardrails (decided 2026-10-01, in the user's words):**
  "It's fine if those topics—wars, disasters, and crimes—are part of
  predictions, so long as they're not too specific to a person or an
  organization."
  - Blocked: "I predict President Trump will rob the Bank of America", and
    "I predict Vladimir Putin will be killed by a Ukrainian drone attack".
  - Permitted: "I predict an important politician will rob a major bank",
    and "I predict a major Ukrainian drone attack will take place".
  - Living people come in two milestones: none at first, then public-role
    events only (winning an election, being sworn in, a team playing in the
    final). Never health issues, death, crime or personal life, a list that
    may grow.
  - The user is open to predictions due later than the next day (§7).

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
  stage2/
    diff.py               # whitespace-word diffs with merged spans + the changed blocks
    fetch.py              # batched revision text from the MediaWiki API (50/request, maxlag, retries)
    examples.py           # edit context (section + marked window) and prompts with/without triggers
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
  build_enwiki_links.py        # 20260901 SQL tables (~11GB) -> links touching the 20% page sample
  build_stage2_targets.py      # Stage 2 examples: kept edits with their Stage 1 signals + bursting neighbors
  fetch_stage2_diffs.py        # revision text via the API -> word diffs + context (resumable)
  train_stage2.py              # QLoRA fine-tune (context vs context+triggers) + test NLL + samples
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
  test_stage2_*.py             # word diffs, API fetch (fake server), prompts, label/logit alignment
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
13. **English Wikipedia link graph (2026-09-28)**: `python
    scripts/build_enwiki_links.py`.
    - The 20260901 SQL tables, 11.15 GB.
    - 255,571,218 mainspace links touching the 20% page sample, built in
      22 minutes. The SQL parser needed a fix first: English Wikipedia's
      dumps put one row per line.
    - The harness then reruns with the link feature sets. Neighbor features
      took about 2.5 minutes per burst definition, over 1.87M receiving
      pages.
    - Results are in §5, "English Wikipedia link-neighbor results".
14. **Stage 2, first run (2026-09-29)**:
    - `build_stage2_targets.py`: 16,000 targets. 40% name a linked page
      that was bursting the day before.
    - `fetch_stage2_diffs.py`: 21 minutes of sequential API fetching, 50
      revisions per request. That gave 14,760 examples: 11,012 train, 939
      validation, 2,809 test. 1,229 targets only removed text and 11 were
      unavailable. The median insertion is 44 characters.
    - `train_stage2.py`: 38 and 32 minutes per variant, 1 epoch.
    - Results are in §5, "Stage 2 results".
15. **Stage 2, seeds and change snippets (2026-09-29)**:
    - `build_stage2_neighbor_changes.py`: what changed on each example's top
      3 bursting neighbors the day before. 9,959 (page, day) pairs; 9,937
      fetched in 20.5 minutes. 6,664 (67%) have a prose snippet, and 228
      pages were created that day.
    - A first fetch was stopped after 9 of 20 parts. It took the longest
      inserted span as it was, and 59% of its snippets held citation,
      infobox or table markup. The rerun cleans wikitext
      (`src/stage2/wikitext.py`) and skips moved text.
    - `build_stage2_titles.py`: 4 minutes over 20 monthly history dumps with
      10 workers. Every needed revision was found, and the dumps' snapshot
      titles matched the revision data's in every case.
    - Titles were different at the time for 373 of 14,760 example pages
      (2.5%) and 540 of 13,496 neighbor-days (4.0%). In the test split, 1.0%
      of pages had one, and 5.9% of examples with bursting neighbors named a
      neighbor by a later title. Some renames encode later events, e.g.
      "2026 California billionaire tax" became "2026 California Proposition
      40".
    - `train_stage2.py`: three prompt variants × two seeds, 1 epoch each:
      28.5, 31.6 and 33.5 minutes per run, 3.3 hours in all, with a 4.6 GB
      peak. Results are in §5, "Stage 2: seeds and change snippets".
16. **Stage 2, relevance and model size (2026-09-29 to 30)**:
    - **What the seeds run showed** (`scripts/analyze_stage2.py`):
      - The trigger gain comes from edits whose inserted text names a
        bursting neighbor: 40 test edits, −0.19 per token.
      - The snippets shared no words with the answer for 75% of edits.
    - **A mention filter alone is too thin to test.** Only 3.9% of test edits
      with bursting neighbors have a neighbor whose new text links to the
      page. `src/stage2/relevance.py` ranks all new neighbor sentences
      instead. Its rules were chosen on train:
      - Relevant text appears for 30% of edits with bursting neighbors, and
        for 8% using another edit's neighbors (chance).
      - 14.5% of shown texts share a quarter of the answer's words, against
        about 4% for the old snippets and 4.9% for chance.
    - `build_stage2_neighbor_changes.py` now covers all shown neighbors (up
      to 8): 3,559 more page-days, 13,464 in all, 8,905 with a prose
      snippet.
    - **Runs**, launched as a one-off Windows scheduled task so they'd
      survive a disconnect. Logs are in `data/processed/enwiki/stage2/logs/`.
      1. Qwen2.5-0.5B, +triggers vs +relevant, 2 seeds: 32–33 minutes of
         training per run, 3 hours in all.
      2. Qwen2.5-1.5B, context, +triggers and +relevant, 2 seeds: 58–80
         minutes of training per run (5.1–7.3 s/step; longer prompts are
         slower), a 5.6 GB peak, 9 hours in all.
    - **GPU memory ran short while scoring.** Training's cached blocks plus
      desktop apps filled the 8 GB, and test scoring took 2–60 minutes
      instead of about 3, most likely from spilling into system RAM (GPU
      memory read 7.9 of 8 GB whenever it was slow). Nothing failed, and
      `score()` now frees the cache and chunks the loss.
    - Results are in §5, "Stage 2: relevance-ranked sentences and
      Qwen2.5-1.5B".
17. **Step 9, daily input (2026-09-30)**:
    - `fetch_recent_changes.py` fetched the 11.5 hours that recent changes
      (30 days kept) and the history dump still both covered, from
      2026-08-31T15:00Z: 91,515 edits in 3.6 minutes, 40% of them
      bot-flagged. That window was unusually busy: the September days that
      followed averaged 114K mainspace edits, 4–7 minutes each to fetch.
    - `check_recent_changes_parity.py` compares the two sources edit by
      edit. The results are in §5, "Recent changes vs the history dumps".
    - The overlap was about to expire: from 2026-10-01, recent changes no
      longer reach the dump's last hours. The window is kept in
      `data/processed/enwiki/daily/rc_dump_overlap.parquet`.
18. **Step 9, the local daily job (2026-09-30)**:
    - **Backfill:** `fetch_recent_changes.py --days 2026-09-01 2026-09-29`
      bridged the dump's end to today, oldest first, since September 1 was
      about to expire. That's 3,317,159 mainspace edits, 94K–175K a day.
      - A dropped connection ended the first run after 10 days, and
        `api_get` now retries those.
      - It moved to a one-off scheduled task so an SSH disconnect wouldn't
        kill it.
    - **The job** (`daily_predictions.py`) ranks every page edited in the
      30 days before D: 1.2–1.6M pages, in about 2 minutes on 8 workers.
    - **Feature parity:** the burst model's top 100 is identical whether
      the features come from recent changes or the dump (§5, "Recent
      changes vs the history dumps").
    - **Backtest:** `backtest_daily.py --days 2026-09-08 2026-09-29`
      predicted and scored 22 days in 40 minutes. The results are in §5,
      "Daily job backtest".
    - **The page** (`web/`, assembled by `build_site.py`) was checked in
      headless Edge with the 2026-09-30 prophecy and 2026-09-29's record.
19. **Step 9, the daily schedule and publishing (2026-10-01)**:
    - `run_daily.py` orchestrates a day: predict D (fetching missing live
      days first), score D−1, build and publish the site, and prune
      candidate tables after 14 days.
    - A Windows scheduled task, "Zoroaster daily", runs it windowless with
      pythonw at 18:30 local time, which is 00:30 UTC (01:30 after daylight
      saving ends). It catches up after a missed start.
    - **First run (00:24–00:32 UTC):**
      - It fetched 2026-09-30 (121,388 edits) and ranked 1,603,914 pages
        for 2026-10-01.
      - It scored 2026-09-30: 15 of the top 100 burst, against 7 for the
        baseline.
      - It published.
    - **Publishing:** `publish_site.py` pushes the built site to an orphan
      `gh-pages` branch, from a worktree inside the ignored `data/`.
20. **Version 2, phases 1 and 2 (2026-10-01)**:
    - **Selection:** `build_v2_targets.py` chose 18,634 edited page-days in
      43 seconds: each day's top pages by the burst model, plus a random
      sample. Of these, 68% had a bursting linked page the day before.
    - **Point-in-time titles:** `build_stage2_titles.py --v2` scanned 36
      monthly dumps in 8 minutes. It found every revision, and 4.5% of
      pages had a different title then.
    - **The fetch:** `fetch_v2_examples.py` fetched 44,010 revisions in 50
      minutes, as a one-off scheduled task, giving 18,575 examples (314 MB).
      - It was restarted twice in its first minutes, after checking real
        examples. Word fixes had counted as new prose (now "copyedits"),
        and deletion-only days had no kind (now "removals").
      - It also stores the changed paragraphs, for whole-sentence targets
        later, and yesterday's per-section sizes.
    - **Baselines:** `v2_baselines.py`. The results are in §5, "Version 2
      data and baselines".
21. **Version 2, phase 3 (2026-10-01)**:
    - **Prompts** (`src/forecast/prompts.py`) are far shorter than planned.
      - The full variant averages 318 tokens (99th percentile 650), and
        the page alone 206.
      - Targets average 64 tokens, 22 of them the header.
      - So `train_v2.py` allows 1,024 prompt tokens and 256 target tokens.
        Only 6 of 18,575 prompts needed a shorter lead.
      - Prompts list up to 60 headings. At 40, 1.7% of the most-changed
        sections would have been cut off; at 60, 0.5% are.
    - **Memory on 8 GB:**
      - Training on the longest sequences, four at a time, peaked at
        4.8 GB.
      - Generating for 16 of the longest prompts at once peaked at 6.1 GB.
        So generation batches prompts by length under a 10K-token budget,
        which peaks at 2.8 GB and takes 0.45 s a header.
    - **Smoke run** (1,000 training examples): 97% of generated headers
      parsed. One forecast repeated "(lead)" up to the token cap, so
      parsing now drops repeated sections.
      - By my mistake, a backgrounded command chain started a second copy
        of the smoke run. The two shared the GPU, both spilled into system
        RAM, and both slowed about 3×. The second copy was stopped at its
        time limit.
    - **Runs:** the full prompt, then the page-only control, with one seed
      each. They were launched at 08:19 as a one-off scheduled task, with
      the log in `data/processed/enwiki/logs/v2_train.log`.
      - Full prompt: 107 minutes of training (7.1 s/step), 2.5 hours in
        all.
      - Page alone: 82 minutes of training, 1.8 hours in all.
      - Peak allocated memory was 4.7 GB.
      - Results and adapters are in `data/processed/enwiki/v2/qwen2.5-1.5b/`,
        including every test header in `generations.json`. The results are
        in §5, "Version 2 model: structured forecasts".
    - **Generation spilled into system RAM** in both runs. Each run's
      generation took about 40 minutes instead of 25.
      - At the peak, the process held 7.8 GB on the GPU and 13.4 GB in
        shared memory, and free RAM fell to 6.9 GB. The memory log is
        `data/processed/enwiki/logs/v2_train_gpu_memory.csv`.
      - **Cause:** batches went shortest first, so each needed bigger
        blocks than PyTorch had cached, and its cache kept growing. On
        Windows the driver backs allocations with system RAM instead of
        failing them, so PyTorch never frees its cache to retry. The same
        thing probably slowed Stage 2's scoring (item 16).
      - **Fix:** batches now go longest first, and the cache is freed
        after each one.
      - **Re-run on 1,200 test prompts:**
        - The old loop's reserved memory grew from 2.3 to 6.3 GB over 38
          batches, with 1.7 GB allocated between them.
        - The fixed loop's stayed at 2.2 GB between batches.
        - The headers were identical.
22. **Version 2, ranking and a readable report (2026-10-01)**:
    - **The report:** `v2_report.py` writes `report.html` into the run
      directory, a local page of every test forecast next to what
      actually changed.
      - It's labeled as machine-generated forecasts, not facts, and it
        isn't part of the site.
      - It can be filtered by top or random pages, living people and
        yesterday's activity.
    - **Ranking** (`src/forecast/ranking.py`, `rank_v2.py`), chosen after
      the greedy results (§5 "Version 2 model: structured forecasts"):
      - **Sections:** each heading on the page, plus the lead, is scored
        by its probability of being named first, and the top 3 are named.
      - **Kinds:** all 512 possible kinds lines are scored, which gives
        each kind's exact probability. Per-kind thresholds are set on
        validation.
    - **Speed:** each forward pass of the 4-bit LoRA model has a fixed cost
      of about 0.14 s here. So the first version took 3.6 s an example:
      17 batched passes over a copied cache, which also spilled out of
      GPU memory.
      - `tree_logprobs` puts every continuation into one token trie after
        the prompt: 1,360 nodes for the kinds lines instead of 5,890
        tokens. A custom attention mask lets each node see only the prompt
        and its ancestors.
      - That's one pass per example: 0.5 s, peaking at 3.2 GB.
      - Its scores are within 0.14 nats of scoring each sequence alone;
        bf16 noise moves those by 0.17 when they're batched.
      - A tiny Qwen2 test checks it against scoring each sequence alone,
        and fails on a wrong mask or wrong positions.
    - **The run:** a one-off scheduled task from 18:47, 44 minutes in all.
      The results are in §5, "Version 2: ranked forecasts", and the
      report now shows the ranked forecasts too.
    - **Publishing failed on 2026-10-02.** The prophecy was made and
      scored, but the push was rejected: setting the custom domain had
      committed a CNAME file to `gh-pages` on GitHub.
      - `publish_site.py` now rebases onto GitHub's branch before building.
      - It also pushes whenever the branch is ahead, so the stranded commit
        went out on a rerun at 18:50.
23. **The site's design and its second page (2026-10-01)**:
    - **Design:** dark stone (SVG noise drawn by the browser, no download),
      light text and red accents. The two images supplied for the site are
      used, and the Faravahar sits above the title.
      - The banner's white is keyed out, so a red glow behind it fills the
        sky and lights the outlines in the crowd.
      - `zoroaster_1` is set on black before the explainer.
      - The Faravahar is from Wikimedia Commons (CC BY-SA 3.0, Ploxhoi and
        Kevin McCormick) and is credited in the footers. It's also the
        favicon.
      - `prepare_web_images.py` makes the site's images from the originals
        in `assets/img/`: 2.4 MB of PNG became 100 KB. The whole page is
        about 150 KB.
    - **Two bugs found on the way:**
      - `build_site.py` couldn't copy a folder, so adding `web/img` would
        have broken the daily build.
      - The explainer said half a million pages are ranked; it's about 1.5
        million.
    - **"How the prophet works"** (`web/how.html`) explains the burst
      prophecy and the edit forecasts, shows their scores, and lists every
      version 2 test forecast next to what happened.
      - Its data comes from `v2_report.py --site`, which follows the
        publishing guardrails. Made-up and sensitive section names are
        withheld (`src/forecast/guardrails.py`), and there are no quotes
        for living people.
      - It's 1.5 MB, served gzipped at 260 KB.
      - The daily run refreshes it where version 2's ranked forecasts
        exist.
      - The main page ends with a link to it.
24. **Version 3, phase 1 begins (2026-10-01)**:
    - **The development set is the site's own prophecies**
      (`build_v3_days.py`): the top 30 for each day from 2026-09-18 to
      2026-10-01, which is 420 page-days and 313 pages. These are exactly
      what the prophet will read each night.
      - **Nothing about D picks the pages:** 84% were edited on D, 16% not.
        Version 2's examples were chosen among pages edited on D, which
        would tell the prophet that something happened.
      - **Not version 2's validation days:**
        - The panel keeps every edited page-day but only a sample of the
          unedited ones, so its "top 20" would still lean toward pages
          edited on D.
        - It's also a 20% page sample, so its top 20 is about the whole
          wiki's top 100.
      - **The fetch:** `fetch_v2_examples.py --targets --out` took 63
        seconds. Version 2's forecasts (`forecast_v2.py`) took 7 minutes on
        the GPU.
    - **The model is Qwen2.5-7B-Instruct** (Apache 2.0), 4-bit, about 6 GB
      on the GPU. Its training data predates both the development days
      and the test days, so it can't remember what happened. Newer models
      trained into 2025 might.
    - **Milestone 1 leaves out every biography,** living or dead (living
      people, births or deaths categories). In the first run, recently
      dead people's pages, bursting for their deaths, drew "will be
      remembered" predictions.
    - **The person check asks two questions,** and either can drop a
      prediction:
      - yes or no;
      - list the people, or "none".
      On 12 known sentences the list was right 12 times and yes-or-no 10.
      Yes-or-no alone kept "…with Luke Hodge carrying the premiership cup".
    - **First run, 3 days** (`prophesy.py`; 3–5 minutes a day for about
      4,500 prompt tokens and 8 predictions):
      - An extra brace broke one day's JSON, so parsing now reads each
        object separately.
      - The model broke rules: predictions without "I predict that", and
        predictions mixing unrelated pages.
      - **Many predictions restated the evidence** (a venue, a release's
        platforms, a cyclone already formed). Novelty is what the
        instructions need most work on, on development days only.
    - **Second run, after the fixes** (the same 3 days, 4–4.5 minutes
      each):
      - All 24 predictions parsed.
      - 3 were dropped, each correctly, for naming a person: South
        Ossetia's president, a footballer, and the Junior Eurovision hosts.
      - Of the 21 kept, a few are specific, checkable outcomes ("Georgia
        will beat Arkansas", "the White Sox will reach the ALCS").
      - About half restate the evidence (dates, venues, matchups already
        set) or are trivial ("the Asian Games will award medals in
        badminton").
      - One is impossible: three gold medals from a single tournament.
      - So the next round of instruction work is about novelty and
        sanity, on development days only.
    - **Hand grades:** Claude drafts them, and the user confirms or adjusts
      them (decided 2026-10-01).
25. **Version 3: new checks, the judge, and the first hand grades
    (2026-10-01)**:
    - **The instructions' examples leaked into the predictions.** The two
      example predictions came from development days: a Wild Card game
      (2026-09-30) and Tropical Storm Fay (09-20). On 09-18 the model copied
      both, citing unrelated pages, and on 09-26 and 09-28 it predicted the
      Astros from them.
      - The instructions now describe good and bad predictions in general
        terms, with no real-world examples.
      - Each prediction first names the unsettled question it answers.
    - **The person check, redesigned:**
      - It lists the people, then asks what kind of thing each listed name
        is. On 15 known names this got all 15 right; yes-or-no phrasings got
        13 and 14.
      - Words that always mean one person ("coach", "defending champion")
        drop a prediction outright, unless they begin a name. "The 2026
        Presidents Cup" was dropped for "presidents", though the US won it
        that day.
    - **Three more checks:**
      - grounding: is the prediction about its cited evidence?
      - copies of the instructions' example sentences;
      - repeats within a day, dropped only if what they repeat was kept.
        On 09-18 the first of two identical predictions cited the wrong
        page.
    - **The 14-day run** used the old instructions, since it began before
      they changed. It took 5.4 minutes a day and wrote 81 predictions.
      - Re-screened with the new checks (80 seconds for all 14 days), 17 are
        kept.
      - The grounding check alone dropped 49. Persons dropped 8 and copies
        5; novelty was the only reason for 2.
    - **The judge** (`src/prophecy/judge.py`, `judge_prophecies.py`):
      - Qwen2.5-7B-Instruct, as a separate step with its own instructions.
      - Each prediction's pack holds:
        - the cited pages as the prophet saw them;
        - what each page gained on the day, with the diff;
        - the day's Portal:Current events items (`fetch_current_events.py`;
          11–28 a day, few of them sports).
    - **What gets graded:** the 17 kept and the 51 dropped only by the
      quality checks (novelty, grounding), so the hand grades test those
      checks too. Guardrail drops, copies and repeats aren't graded.
    - **Drafting the hand grades refined the rubric** before the judge ran:
      - **Outcome** is whether the prediction had come true by the end of
        the day. "On the day" would have called Georgia's win the day
        before a miss.
      - **Already known** includes a match already played, but not an
        election held the day before: Morocco voted on 09-23 and its
        results came out on the 24th.
      - **Grounded** is false if the prediction is about something else,
        picks a team or party its evidence never mentions, or predicts for
        the day what the evidence says comes later.
      - **Credit** is 0 when already known, even if the pack can't tell the
        outcome.
    - **What Claude's 68 draft grades show** (not yet confirmed by the
      user):
      - **Outcomes:** 8 happened, 1 partly, 47 didn't, 12 unknown. 12 were
        already known. Only 4 earn credit:
        - China's women's team badminton gold (09-24), which the grounding
          check dropped;
        - the PAM winning the most seats in Morocco (09-24);
        - the US winning the Presidents Cup (09-27);
        - half credit for "the NL Wild Card Series ends by 30 September".
      - **The main failure is timing.** 41 of the 47 misses name a result
        that comes after the day: a final days or weeks off, a whole
        season, an election in November. The evidence usually gives the
        dates.
      - **26 pick a team or party the evidence never mentions**, from the
        model's memory, which ends years earlier: Mercedes EQ in Formula E,
        LA Galaxy, Al-Ahly in an Asian competition.
      - **The checks separate somewhat.** Kept predictions average 0.078
        credit and dropped ones 0.012.
        - Grounding agrees with the hand "grounded" on 51 of 68.
        - Novelty flags only 2 of the 12 already-known predictions.
      - **A guardrail gap:** a team or country "winning" an individual
        event points to a person ("Red Bull will win the Azerbaijan Grand
        Prix", "China will win the men's singles"). No check catches it.
    - **The judge's first grades** (17 kept predictions, compared with the
      drafts) are poor:
      - Outcome agrees 41% of the time and grounded 47%; already known
        and specificity 76%.
      - It calls every prediction grounded and none already known.
      - It called Georgia's win, stated in the evidence, "did not happen",
        and a grand final dated 4 October "happened".
    - **The review page** is a private claude.ai artifact. Each card shows a
      prediction, its evidence, the day's diff links, and Claude's draft
      grade with its reasoning. The user confirms or adjusts each grade, and
      the page keeps the results in its database, from which they're read
      back into `grades/confirmed.json`.

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

Stage 2 (on top of the English Wikipedia steps; needs a CUDA GPU):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-stage2.txt
python scripts/build_enwiki_links.py       # ~22 min, 11GB of SQL tables
python scripts/build_stage2_targets.py     # ~30s
python scripts/fetch_stage2_diffs.py       # ~21 min of sequential API requests
python scripts/build_stage2_neighbor_changes.py  # ~21 min of API requests: what changed on bursting neighbors
python scripts/build_stage2_titles.py      # ~4 min: titles at the time, from the history dumps
python scripts/train_stage2.py             # ~3.3 h on an RTX 3070: 3 prompt variants x 2 seeds
python scripts/analyze_stage2.py           # ~1 min: the §5 comparisons, from results.json
python scripts/train_stage2.py --model Qwen/Qwen2.5-1.5B --variants context context+triggers context+triggers+relevant --out data/processed/enwiki/stage2/qwen2.5-1.5b  # ~8 h
python scripts/fetch_recent_changes.py --start 2026-09-29T00:00:00Z --end 2026-09-30T00:00:00Z --out data/processed/enwiki/daily/2026-09-29.parquet  # ~7 min a day
python scripts/check_recent_changes_parity.py   # recent changes vs the dump on their overlap (the saved window)
python scripts/fetch_recent_changes.py --days 2026-09-01 2026-09-29 --out-dir data/processed/enwiki/live  # live days
python scripts/check_daily_parity.py        # the daily job's features, recent changes vs dump
python scripts/daily_predictions.py --day 2026-09-30   # ~2 min once the live days are fetched
python scripts/score_predictions.py --day 2026-09-29   # after the day is over
python scripts/backtest_daily.py --days 2026-09-08 2026-09-29
python scripts/build_site.py --serve        # the page, previewed at localhost:8000
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

### English Wikipedia link-neighbor results (2026-09-28)

**Raw lift.** 11.1% of evaluation rows had a link neighbor bursting the day
before. Those rows got a kept edit at 3.0× the base rate (1.90% vs
0.62%).

| model | P@100 | P@1000 | R@10000 | AP |
|---|---|---|---|---|
| habits + burst | 0.741 | 0.436 | 0.150 | 0.0865 |
| + link counts | 0.746 | 0.436 | 0.150 | 0.0867 |
| + neighbor bursts | 0.741 | 0.434 | 0.151 | 0.0871 |
| + neighbor bursts, mass editing excluded | 0.744 | 0.436 | 0.151 | **0.0873** |

Paired per-day differences in AP:
- **link counts vs habits + burst:** +0.0002 ± 0.0001 (better 13/1 days).
- **neighbor bursts vs link counts:** +0.0005 ± 0.0001 (11/3).
- **neighbor bursts with mass editing excluded vs link counts:**
  **+0.0007 ± 0.0001 (13/1)**.
- **the discount vs no discount:** +0.0002 ± 0.0001 (12/2).

P@100 differences are all within noise.

What this means:
- **The page-specific cross-page signal is real on English Wikipedia.**
  Neighbor bursts improve AP on 13 of 14 test days once mass editing is
  excluded, where the site-wide co-burst count added nothing. Here the
  mass-editor discount does help, if modestly.
- **The gains are small.** From habits to the full model, AP goes 0.0859 →
  0.0873, +1.6% relative, and the top of the ranking doesn't move. Next-day
  "which pages" prediction is dominated by page habits.
- **Still optimistic:** the link snapshot is from 2026-09-01, after the test
  window (§5 "Step 4 results").

### Stage 2 results (2026-09-29, English Wikipedia)

The task: given the page, date, section, and parent text around the change,
write the text a kept edit inserted. The test set is 2,809 edits from the
Stage 1 test days (Dec 2025–Jun 2026), after the base model's training
data. The score is NLL per inserted-text token:

| model | test NLL/token | perplexity |
|---|---|---|
| Qwen2.5-0.5B untuned, context prompt | 2.750 | 15.6 |
| untuned, context + triggers | 2.767 | 15.9 |
| QLoRA-tuned, context | 1.893 | 6.6 |
| QLoRA-tuned, context + triggers | 1.894 | 6.6 |

Paired per-example differences in mean NLL/token, triggers minus context,
both tuned:
- **all test examples:** −0.0030 ± 0.0020 (better on 51.2%). Not
  significant.
- **with a bursting linked page (940):** **−0.0104 ± 0.0042** (better on
  54.3%, t ≈ 2.5).
- **without one (1,869):** +0.0007 ± 0.0021. No effect.

For comparison, tuning vs untuned, both with the context prompt: −1.64 ±
0.03 (better on 99.6%).

What this means:
- **Fine-tuning works.** Perplexity on inserted text drops from 15.6 to
  6.6. Greedy samples reproduce edit types well: citations, wikilinks,
  categories, "See also" entries, table rows. Some are exact, often by
  copying from the surrounding context.
- **Trigger text helps only where it has something to say.** It helps on
  examples whose prompt names bursting linked pages, and not elsewhere.
  That mirrors Stage 1: the "something is happening" signal is real but
  small.
- **Caveat: one training run per variant.** The paired SE covers variation
  across examples, not across training runs. The effect being confined to
  the subset where the triggers differ argues against a global fluke, but a
  second seed would settle it. **Settled** by the next run, below.
- **Caveat: prompts used the snapshot's titles**, so a page renamed after
  the event showed its later name. The next run uses titles at the time,
  and the effect holds.
- **Triggers are underused.** One test edit added "Commissioner of Mental
  Health under Mayor [[Wilson Fisk (Marvel Cinematic Universe)]]" while
  that very page was a bursting neighbor, yet both variants wrote
  "Murdock.". The trigger text carries only *titles*.
- **Greedy decoding sometimes loops** ("2026, … 2026, …").

### Stage 2: seeds and change snippets (2026-09-29, English Wikipedia)

This run repeats the first run's setup with three changes (§3 item 15):
- **Two seeds per prompt variant.**
- **A third variant** adds what changed on bursting linked pages the day
  before: a prose snippet each, for up to three of them.
- **Titles at the time.** Pages are named by their titles then, not the
  snapshot's.

The prompt budget is 512 tokens. Some test prompts had their context
shortened to fit: 3 for context, 14 with triggers, 66 with changes. The
comparisons below exclude examples shortened in either prompt. Including
them moves the trigger rows by at most 0.0004, but makes the snippet
penalty about 0.002 larger (+0.0047 pooled), since shortening removes
context. `python scripts/analyze_stage2.py` reproduces these numbers from
`results.json`.

Test NLL per inserted-text token:

| variant | untuned | tuned, seed 1234 | tuned, seed 2345 |
|---|---|---|---|
| context | 2.750 | 1.893 | 1.895 |
| + triggers | 2.762 | 1.889 | 1.891 |
| + triggers + changes | 2.761 | 1.891 | 1.894 |

Paired per-example differences in mean NLL/token (negative is better):

| comparison | subset | n | seed 1234 | seed 2345 | pooled |
|---|---|---|---|---|---|
| triggers − context | bursting linked page | 927 | −0.0132 ± 0.0035 | −0.0144 ± 0.0034 | **−0.0138 ± 0.0032** |
| triggers − context | none | 1,868 | +0.0004 ± 0.0016 | −0.0036 ± 0.0014 | −0.0016 ± 0.0010 |
| changes − triggers | snippet shown | 684 | +0.0003 ± 0.0022 | +0.0047 ± 0.0019 | +0.0025 ± 0.0017 |
| changes − triggers | none (identical prompts) | 1,868 | −0.0011 ± 0.0009 | +0.0030 ± 0.0010 | +0.0009 ± 0.0007 |

What this means:
- **The trigger effect replicates, and is larger:** −0.014 per token on
  edits with a bursting linked page, in both seeds (t ≈ 4 each). The
  titles are now point-in-time, so the first run's −0.010 didn't come from
  leaked renames.
- **Run-to-run noise is about ±0.003 per token.** The cleanest measure is
  the last row. The two variants' prompts are identical there, yet the
  seeds differ by −0.0011 and +0.0030. The per-example SE doesn't cover
  this, so effects below ~0.005 need more seeds. The trigger effect is
  about four times that.
- **Snippets of what changed add nothing detectable.** If anything they
  make things slightly worse (+0.0025 ± 0.0017 pooled). The samples
  suggest why:
  - Snippets are about the linked page's own news, and rarely say what
    this page's edit will. Take the share of an inserted text's words (4+
    letters) that appear in its snippets: the median is 0, and the 90th
    percentile is 5%.
  - High-degree pages get incidental neighbors: Formula One's were Patrick
    Mahomes, New York Red Bulls and Russia.
  - What helps is the neighbor's *name*. For example, "Deccan thorn scrub
    forests" had [[Acacia planifrons]] changed to [[Vachellia …, the day
    after "Vachellia planifrons" burst.
- **Most of the trigger gain is a neighbor's name being copied.** 40 test
  edits' inserted text contains a shown neighbor's title, 25 of them as a
  link. For example, Cecilia Bartoli's page gained "She performed at the
  [[2026 Winter Olympics opening ceremony]]", which was a bursting neighbor
  the day before.
  - On those 40 edits, triggers cut NLL by 0.19 per token in both seeds.
  - On the other 887 edits with bursting neighbors, the cut is 0.0058 ±
    0.0020.
  - This split conditions on the answer, so it shows where the gain comes
    from; it isn't a score.
- **The untuned model gets slightly worse with trigger text** (2.750 →
  2.762). Only a fine-tuned model can use it.
- **The repetition penalty (1.2) fixed looping:** none of the 12 saved
  generations loop, in any variant. In the first run, 3 of 24 did.
- **Still optimistic in one way:** which pages count as linked comes from
  the 2026-09 link snapshot (§5, "Step 4 results").
- **Snippet coverage:** a neighbor's snippet can come from another example
  that day, so a snippet shown isn't always one of the top three
  neighbors. That's harmless: it's still a bursting neighbor, point-in-time.

### Stage 2: relevance-ranked sentences and Qwen2.5-1.5B (2026-09-29 to 30, English Wikipedia)

Qwen2.5-0.5B, `+relevant` vs `+triggers`, two seeds (§3 item 16).
`+relevant` shows the bursting neighbors' new sentences most relevant to
the page, or nothing if nothing relates (`src/stage2/relevance.py`). It
shows text for 265 of the 940 test edits with bursting neighbors. The
comparisons exclude prompts shortened in either variant: 14 with triggers
and 35 with relevant sentences, on test. `python scripts/analyze_stage2.py
data/processed/enwiki/stage2/relevant_0.5b/results.json` reproduces them.

Paired differences in mean NLL/token, `+relevant` − `+triggers`:

| subset | n | seed 1234 | seed 2345 | pooled |
|---|---|---|---|---|
| relevant text shown | 239 | −0.0071 ± 0.0067 | −0.0069 ± 0.0070 | −0.0070 ± 0.0066 |
| … sharing ≥25% of the answer's words | 35 | −0.048 ± 0.040 | −0.039 ± 0.040 | −0.043 ± 0.040 |
| … sharing fewer | 47 | +0.0010 ± 0.0024 | −0.0040 ± 0.0024 | −0.0015 ± 0.0022 |
| … sharing none | 145 | +0.0006 ± 0.0048 | −0.0019 ± 0.0059 | −0.0007 ± 0.0049 |
| none shown (identical prompts) | 2,535 | −0.0015 ± 0.0006 | +0.0029 ± 0.0007 | +0.0007 ± 0.0005 |

What this means:
- **Relevance ranking fixes what went wrong with snippets, but the gain
  isn't significant.** Nothing is harmed, and the sign is right where text
  is shown, in both seeds. But −0.007 ± 0.007 on 239 edits is about 1 SE.
  An effect this size needs roughly 4× as many edits with relevant text to
  resolve.
- **The model uses content when the content holds the answer.** The whole
  gain is in the 35 edits whose shown text shares at least a quarter of
  the answer's words: −0.043, in both seeds, and nothing elsewhere. That's
  13% of the edits with text. The 90th-percentile share of answer words is
  0.43, against 0.05 for the longest-prose snippets.
- **Same-seed reruns agree in aggregate.** `+triggers` with seed 1234
  scored 1.88934 test NLL/token, against 1.88919 in the seeds run. GPU
  nondeterminism moves individual examples by 0.008 on average, but the
  mean by only ±0.0004, far less than a different seed does (±0.003).

**Qwen2.5-1.5B**, all three prompts, two seeds, on the same examples and
comparisons (`python scripts/analyze_stage2.py
data/processed/enwiki/stage2/qwen2.5-1.5b/results.json`). Test NLL per
inserted-text token, pooled over seeds:

| model | untuned | tuned, context | tuned, + triggers | tuned, + relevant |
|---|---|---|---|---|
| Qwen2.5-0.5B | 2.750 | 1.894 | 1.890 | 1.891 |
| Qwen2.5-1.5B | 2.372 | 1.667 | 1.665 | 1.666 |

(The 0.5B's context score is from the seeds run, the rest from this one.)

Paired per-example differences in mean NLL/token, neither prompt
shortened:

| comparison | subset | n | 1.5B, seed 1234 | 1.5B, seed 2345 | 1.5B, pooled | 0.5B, pooled |
|---|---|---|---|---|---|---|
| triggers − context | bursting linked page | 927 | −0.0118 ± 0.0033 | −0.0077 ± 0.0031 | **−0.0098 ± 0.0031** | −0.0138 ± 0.0032 |
| triggers − context | … its title in the inserted text | 40 | −0.186 ± 0.056 | −0.178 ± 0.053 | −0.182 ± 0.055 | −0.193 ± 0.053 |
| triggers − context | … no title in it | 887 | −0.0040 ± 0.0020 | −0.0001 ± 0.0019 | −0.0020 ± 0.0018 | −0.0058 ± 0.0020 |
| relevant − triggers | relevant text shown | 239 | +0.0007 ± 0.0043 | −0.0030 ± 0.0061 | −0.0012 ± 0.0051 | −0.0070 ± 0.0066 |
| relevant − triggers | … sharing ≥25% of the answer's words | 35 | −0.015 ± 0.025 | −0.041 ± 0.039 | −0.028 ± 0.032 | −0.043 ± 0.040 |
| relevant − triggers | none shown (identical prompts) | 2,535 | +0.0006 ± 0.0005 | +0.0011 ± 0.0004 | +0.0008 ± 0.0003 | +0.0007 ± 0.0005 |

What this means:
- **The larger model is much better at the task.** Fine-tuned, it cuts
  NLL per inserted token from 1.894 to 1.667 (perplexity 6.6 → 5.3), and
  untuned from 2.750 to 2.372.
- **It gets no more from the "something is happening" signal.** The
  trigger gain is −0.010 per token, against −0.014 at 0.5B, within noise.
  It comes from the same place: the 40 edits whose inserted text names a
  bursting neighbor (−0.18 in both seeds). On the rest, triggers do almost
  nothing (−0.002).
- **Relevant content still shows no detectable gain** (−0.001 ± 0.005 on
  239 edits). Where the shown text shares the answer's words, both sizes
  lean the same way (−0.03 and −0.04), but 35 edits are too few to tell.
- **For Stage 2, then, the useful signal is which linked pages are
  bursting:** names the edit is likely to add, not what those pages said.
  Content might help where it overlaps the edit, but this test set has
  too few such edits to show it.
- **Seed effects are larger at 1.5B.** Seed 2345 scores 0.005–0.007 worse
  than seed 1234 in every variant. Paired, within-seed comparisons cancel
  that out.

### Recent changes vs the history dumps (2026-09-30)

The deployed pipeline reads yesterday's edits from the API's recent changes
(`src/ingest/recent_changes.py`), but the models were trained on the
history dumps. The two sources overlapped for 11.5 hours when checked,
2026-08-31T15:00Z to 2026-09-01T02:30Z. That gave 91,515 recent-changes
records and 92,162 dump records, 91,469 in both:

| field | agreement | why they differ |
|---|---|---|
| page id, parent id, timestamp | 100% | |
| sha1 | 100% where both have one | the dump lacks 2,677 in its last hours; 17 are hidden by revision deletion |
| byte size | 99.99% | 12 missing in the dump |
| anonymous | 99.93% | |
| bot | 99.83% | the dumps also flag bot-named or bot-group accounts' unflagged edits |
| title | 99.86% | pages moved since; the dump has snapshot titles |
| user name | 99.77% | users renamed since |
| page creation time | 96% of 889 creations | |

- **Hashes needed converting.** The API gives sha1 in hex and the dumps in
  base 36. Unconverted, 99.99% disagreed, which would have hidden every
  revert crossing from dump history into daily data.
- **The dumps count 0.75% more revisions** (693 not in recent changes).
  They're mostly revisions that page moves and protections create: 229
  repeat their parent's content exactly. Recent changes list those as log
  entries, so daily edit counts will run about 0.75% low. A feature-level
  check should confirm the model doesn't notice.
- **Missing hashes are rare in the dumps**, 0.04–0.31% a month, except in
  the snapshot's final hours (10.5%). So they don't affect training.
- **Feature-level parity** (`scripts/check_daily_parity.py`): the daily
  job's features for 2026-09-01 compared on the 50,347 pages edited in the
  window, once from the dump and once with the window's recent changes in
  its place.
  - Edit counts and burst z differ on 0.33% of pages, editors on 0.1%, and
    burst flags on 0.04%, from the missing log-action revisions and
    renamed users.
  - The burst model's top 100 is identical, and 989 of its top 1,000 match
    (rank correlation 0.995).
  - The check caught one skew, now fixed. Recent changes give a creation
    date only for pages created that day, so a page first edited in years
    looked brand new. Such a page must predate the history, so it now gets
    HISTORY_START (2023-06-01) as a lower bound.
  - That still leaves `page_age_days` short of the true age on 8% of pages
    (4.4% of the model's importance). Page ids are assigned in creation
    order, so ids could give estimated dates if it ever matters.

### Stage 1 burst target (2026-09-30, English Wikipedia)

`python scripts/train_stage1.py --corpus enwiki --target burst` ranks pages
by how likely they are to burst on day D: a burst by 2+ editors, with mass
editors left out (the target decided in §6 step 9). It uses the same panel,
splits and 14 full test days as the edit target. The 20% page sample has
137 such bursts a day, a 0.008% base rate.

| model | P@100 | P@1000 | R@10000 | AP |
|---|---|---|---|---|
| edits yesterday | 0.066 | 0.025 | 0.326 | 0.017 |
| burst z yesterday (persistence) | 0.043 | 0.021 | 0.264 | 0.010 |
| habits | 0.102 | 0.028 | 0.451 | 0.036 |
| habits+burst | **0.110** | 0.029 | 0.450 | **0.037** |
| habits+burst+links (mass editors left out) | 0.096 | 0.029 | 0.463 | 0.030 |

What this means:
- **Bursts are predictable, modestly.** In the page sample, the model's top
  100 pages hold 11 of the next day's bursts on average, against 6.6 for
  the busiest pages yesterday and 4.3 for yesterday's bursts.
  - habits+burst beats persistence on all 14 days (+0.067 ± 0.009 P@100).
  - habits beats "edits yesterday" on 11 days and loses on 2 (+0.036 ±
    0.008).
- **The page's own burst features help a little at the top** (+0.008 ±
  0.004 P@100, 9 days better and 3 worse). Co-burst adds nothing.
- **Link features don't help this target.** The link sets score below
  habits+burst (P@100 0.091–0.096 vs 0.110). So the daily job needs no
  link graph.
- **Train unweighted.** With the panel's sample weights, per-day AP halved
  on both validation (0.021 vs 0.041, habits+burst) and test (0.013 vs
  0.037). The weight-200 negatives dominate the loss, and two feature sets
  early-stopped after 32 and 52 rounds. The difference showed up on test
  first, and the 182 validation days confirmed it.
- **Weekly lags don't reliably help** (habits+burst+weekly: the same
  weekday's edits, editors and bursts one and two weeks back). On test they
  add +0.0026 ± 0.0006 AP (12 of 14 days), with P@100 unchanged at 0.110.
  On the 182 validation days they add +0.0002 ± 0.0003 (98 better, 84
  worse). So the test gain is seasonal or chance, and habits+burst stays
  the production model. The features remain in the panel.

### Daily job backtest (2026-09-30, English Wikipedia)

The local daily job predicted each day from 2026-09-08 to 2026-09-29, then
scored its top pages against what actually burst that day. It ranked the
whole wiki (1.2–1.6M recently edited pages a day), not a sample. The
target is the Stage 1 burst: a burst by 2+ editors, with mass editors left
out.

| mean precision over 22 days | P@10 | P@50 | P@100 | P@1000 |
|---|---|---|---|---|
| **model** (habits+burst) | **0.318** | **0.249** | **0.203** | 0.070 |
| most edits yesterday | 0.118 | 0.107 | 0.097 | 0.049 |
| model − baseline | +0.200 ± 0.047 (16/2 days) | +0.142 ± 0.017 (21/0) | +0.106 ± 0.007 (22/0) | +0.021 ± 0.002 (22/0) |

- **About one in five of each day's top 100 bursts the next day**, and
  about one in three of the top 10. That's double the baseline, better on
  all 22 days at P@100.
- **Higher than the offline P@100 (0.110), as expected.** Ranking the whole
  wiki makes the top 100 far more selective than in a 20% page sample.
- **The hits are recognizably the day's events:** elections the day after
  the vote, sports fixtures, launches, crashes, anniversaries (the 125th of
  William McKinley's assassination).
- **Caveats:**
  - Mass editors are approximated from each day's live records, reverts
    included (`live_mass_editors`).
  - Before 2026-10-01, candidates come only from live days since
    2026-09-01, so pages last edited in August are missed. That's
    irrelevant at the top of the list.

### Version 2 data and baselines (2026-10-01, English Wikipedia)

Version 2's page-days, from `fetch_v2_examples.py`:

| split | examples | gained new prose | about living people | changed the day before |
|---|---|---|---|---|
| train | 14,507 | 41% | 23% | 75% |
| validation | 1,272 | 47% | 24% | 73% |
| test | 2,796 | 38% | 20% | 68% |

The kinds of change, across all examples: copyedits 59%, removals 58%,
template fields 57%, links 51%, new prose 41%, references 37%, tables 11%,
new sections 10%, categories 7%. About 9% of page-days have no detected
kind. They look like small value changes inside templates (a number in an
infobox), and the stored spans allow a "data" kind to be added without
refetching.

Structured-forecast baselines on the test days (`src/forecast/metrics.py`):

| baseline | section precision | main section named | kinds Jaccard |
|---|---|---|---|
| yesterday again | 0.448 | **0.490** | **0.412** |
| most common (the lead; copyedits, removals, template fields, links) | **0.474** | 0.313 | 0.398 |

- **The bar for a model** is "yesterday again" on the burst model's top
  pages, the site's kind: it names the main changed section for half of
  them (0.503), with a kinds Jaccard of 0.457.
- **The lead changes often**, so always naming it gives decent section
  precision but rarely the main section.
- **Validation runs higher than test** (main section 0.540 for "yesterday
  again"), so all comparisons are within one split.

### Version 2 model: structured forecasts (2026-10-01, English Wikipedia)

QLoRA on Qwen2.5-1.5B (`train_v2.py`), with one seed (§3 item 21). The
headers were greedy-decoded on all 2,796 test days, so 2,097 of them are
the burst model's top pages, the kind the site shows. Each cell gives all
test days / top pages:

| forecaster | section precision | main section named | kinds Jaccard |
|---|---|---|---|
| yesterday again | 0.448 / 0.451 | **0.490 / 0.503** | 0.412 / 0.457 |
| most common | **0.474 / 0.486** | 0.313 / 0.279 | 0.398 / 0.438 |
| model, full prompt | 0.449 / 0.458 | 0.447 / 0.459 | **0.430 / 0.486** |
| model, page alone | 0.431 / 0.441 | 0.360 / 0.343 | 0.400 / 0.444 |

Paired per-example differences, with standard errors (all / top):

| comparison | section precision | main section named | kinds Jaccard |
|---|---|---|---|
| full prompt − yesterday again | −0.001 ± 0.005 / +0.007 ± 0.006 | **−0.042 ± 0.008 / −0.044 ± 0.010** | **+0.018 ± 0.004 / +0.029 ± 0.004** |
| full prompt − page alone | +0.017 ± 0.006 / +0.017 ± 0.008 | +0.088 ± 0.009 / +0.116 ± 0.011 | +0.030 ± 0.004 / +0.042 ± 0.005 |

- **The full prompt doesn't beat "yesterday again" overall.** It's better
  on the kinds of change, ties on section precision, and is worse at
  naming the main section.
  - The agreed rule was that the middle variant and second seeds would
    run only if it won. So they haven't run.
  - On random pages it trails "yesterday again" on all three metrics.
- **It names fewer sections:** 1.76 on average, against up to 3 for
  "yesterday again".
  - It names exactly yesterday's sections on 45% of days.
  - When it names fewer, it misses the main section more often.
  - On days the page changed the day before (1,914), it beats "yesterday
    again" on kinds (0.504 vs 0.467) and on section precision (0.453 vs
    0.442), but names the main section less often (0.466 vs 0.518).
- **Greedy decoding never predicts the rare kinds.**
  - "New section" (10% of days) is never predicted, and categories almost
    never (2% recall).
  - The common kinds are nearly always predicted: copyedits, removals and
    template fields each get 0.94 recall.
  - So ranking candidates by the model's own probabilities may do better
    than its single most likely header.
- **The prompt's extra information helps.** The full prompt beats the page
  alone on every metric, and on likelihood:
  - header: −0.016 ± 0.001 nats per token;
  - new text: −0.085 ± 0.009 nats per token, on days with new prose.
  - Mean NLL per token was 0.460 vs 0.476 for the header, and 2.545 vs
    2.610 for the text.
- **Living people (557 test days):**
  - The full prompt is better on kinds (+0.029 ± 0.008) and worse on the
    main section (−0.053 ± 0.017).
  - All headers parsed, and 0.7% (full) and 1.0% (page) of the sections
    they named aren't on the page.
  - Most of those invented names state outcomes, e.g. "2026 Four
    Continents champion", "2025–2026 season: World bronze" and "2026:
    Return to the Cup Series" (page alone). So a section name is free
    text in disguise.
  - The full prompt's "2026: Dementia diagnosis and guardianship"
    (Wendy Williams) added a year to an existing section, one that the
    day before's edits had touched.

### Version 2: ranked forecasts (2026-10-01, English Wikipedia)

`rank_v2.py` reads the full-prompt model's own probabilities instead of
decoding greedily (§3 item 22):
- **Sections:** the top 3 candidates by the probability of being named
  first. Candidates hold 83% of that probability; the rest goes to names
  not on the page, "none", and other tokenizations.
- **Kinds:** each kind is forecast when its exact probability passes a
  threshold. The thresholds were chosen on validation to maximize the mean
  Jaccard, and most are low:
  - categories 0.8, table 0.4, template fields 0.3;
  - new section, references and links 0.25;
  - prose and removals 0.2, copyedits 0.15.
  - The 512 kinds lines hold 99% of the probability.

Each cell gives all test days / top pages:

| forecaster | section precision | main section named | kinds Jaccard |
|---|---|---|---|
| yesterday again | **0.448 / 0.451** | 0.490 / 0.503 | 0.412 / 0.457 |
| model, greedy | 0.449 / 0.458 | 0.447 / 0.459 | 0.430 / 0.486 |
| model, ranked | 0.344 / 0.371 | **0.621 / 0.605** | **0.444 / 0.498** |
| model, ranked, every threshold 0.5 | 0.344 / 0.371 | 0.621 / 0.605 | 0.352 / 0.392 |

Paired against "yesterday again" (all / top):
- section precision −0.104 ± 0.006 / −0.080 ± 0.006;
- main section named +0.131 ± 0.008 / +0.102 ± 0.009;
- kinds Jaccard +0.032 ± 0.004 / +0.041 ± 0.004.

- **Kinds: ranking wins.** It beats "yesterday again", and greedy decoding
  by +0.014 ± 0.003.
  - The validation thresholds matter. With every threshold at 0.5, it's
    worse than "yesterday again" by 0.060.
  - On random pages there's no gain (+0.005 ± 0.005).
- **Sections: a tie, once the count is equal.** Ranking names the main
  section far more often only because it names 2.97 sections a day,
  against 1.88 for "yesterday again".
  - Trimmed each day to as many sections as "yesterday again" names, it
    ties: precision −0.001 ± 0.004, main section +0.003 ± 0.007 (top pages
    +0.004 ± 0.005 and +0.010 ± 0.008).
  - So for sections, the model knows about what yesterday's change says.
- **Rare kinds are still missed.** "New section" is now forecast
  sometimes, but it's rarely right (precision 0.06). Categories never are.
- **Living people (557 days):** main section +0.141 ± 0.018, kinds +0.053
  ± 0.008, precision −0.123 ± 0.014.
- **Cost:** 44 minutes for 4,068 page-days, including validation's greedy
  headers. That's 0.5 s each, peaking at 4.7 GB reserved.

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
6. **Link-neighbor bursts on English Wikipedia** (**done 2026-09-28**, see §3
   item 13 and §5 "English Wikipedia link-neighbor results"). Small and
   consistent: +1.6% relative AP in total, with neighbor bursts better on
   13 of 14 days once mass editing is excluded. The original rationale: on
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
8. **Stage 2 QLoRA fine-tuning** on the RTX 3070 (**done through the
   model-size run, 2026-09-30**; see §3 items 14–16 and §5's Stage 2
   sections). It started once the Stage 1 burst and link signals proved
   real, if small.

   Done:
   - A second seed: the trigger effect replicates, −0.014 per token.
   - Snippets of what changed on bursting linked pages: no detectable gain.
   - Relevance-ranked sentences instead: no detectable gain either.
   - Qwen2.5-1.5B: much better at the task (perplexity 6.6 → 5.3), with
     the same trigger effect.
   - Point-in-time titles, a repetition penalty for generations, and
     scoring that stays inside GPU memory.

   Left, if Stage 2 is revisited:
   - **A targeted test set** of edits whose neighbors' new text shares
     their content, to tell whether content helps where it exists. This
     test set has only about 35 such edits.
   - **More seeds, or averaged runs,** for effects under ~0.005 per token.
     Run-to-run noise is about ±0.003, and seeds differ by up to 0.007 at
     1.5B.
   - **Room for longer prompts.** At 512 tokens, 35 relevant-sentence
     prompts were shortened on test, against 14 with triggers alone. The
     1.5B peaked at 5.6 of 8 GB, so there's room to raise the budget.

   The pre-fetch checklist below is done (word-level diffs, bots and
   reverts excluded, 50-revision batches, incremental output). Before any
   large diff fetch:
   - switch to word-level diffs (keep line-level context separately if
     useful as conditioning);
   - exclude bot edits and reverts from targets;
   - batch up to 50 revision ids per API request (a revision's parent is
     usually the previous revision already being fetched);
   - write output incrementally (`build_test_diffs.py` currently writes only
     at the end, so a crash loses the whole run).

9. **Deployment: a static "predicted events for tomorrow" page** (planned
   2026-09-29; **running daily since 2026-10-01**, see §3 items 17–19 and
   §5 "Recent changes vs the history dumps" and "Daily job backtest"). Host on GitHub Pages only (free,
   static), not an app hosting service.
   - **Precompute, don't serve.** Tomorrow's prediction is the same for
     every visitor. A daily batch job writes it as static JSON and the page
     only displays it, so no inference server is needed. The model could run
     in the browser (transformers.js, WebLLM), but every visitor would
     download hundreds of MB for the same answer.
   - **The daily job** is a scheduled GitHub Actions workflow, free on
     standard runners for a public repo (2,000 minutes a month if private).
     The local machine is the fallback. Steps:
     1. Just after 00:00 UTC, pull day D−1's edits from the MediaWiki API
        (`scripts/fetch_recent_changes.py`). About 114K mainspace edits a
        day take ~230 requests, 4–7 minutes.
     2. Update a rolling state of per-page daily counts.
     3. Compute the Stage 1 features and score them with LightGBM. The model
        is a few MB and cheap on CPU.
     4. Optionally, write Stage 2 text for the top pages. The LoRA adapter is
        merged into the base model and quantized for llama.cpp (~400 MB,
        pulled from the Hugging Face Hub each run). That's a few dozen
        short generations on CPU.
     5. Publish `predictions/YYYY-MM-DD.json` and the page (see
        "Hosting").
   - **Limits:**
     - Pages: sites up to 1 GB and 100 GB/month of bandwidth (soft). Git
       rejects files over 100 MB.
     - Runners: 4 CPUs, 16 GB RAM, ~14 GB of disk, 6-hour jobs.
     - Scheduled workflows in *public* repos are disabled after 60 days
       without repository activity. That doesn't apply while the workflow
       lives in this private repo.
   - **Decided 2026-09-30:**
     - **Rank predicted bursts**, not predicted edits. Most edited pages
       are edited out of habit, so the page would show much the same list
       every day. Bursts are mostly new: of the 797 a day by 2+ editors
       (mass editors left out, test half-year), only 6.8% were bursting the
       day before. `train_stage1.py --target burst` trains and scores that
       target.
     - **Run the daily job locally first**, on the dump data already here,
       then move it to GitHub Actions once it works.
     - **The model is habits+burst, trained unweighted on the burst target**
       (§5 "Stage 1 burst target"). It needs no link graph.
   - **Hosting (decided 2026-09-30):** this repo goes public once the page
     works, and the site is published from an orphan `gh-pages` branch that
     only the daily job writes. Its history is then a record of every past
     prediction, kept out of `main`. Pages is free only for public repos.
     - The history was rewritten on 2026-09-30 so that only GitHub noreply
       addresses appear. The personal emails were on four early commits'
       metadata and in the User-Agent, which now gives the repo URL as its
       contact. Messages, dates and every other file are unchanged. A
       bundle of the old history is in
       `C:\Users\Tim\Documents\Zoroaster-backups\`.
   - **The page's voice:** the repo is named for Zoroaster, the prophet, so
     the page frames its predictions as his prophecy: "Also sprach
     Zarathustra" (thus spoke Zarathustra), over tomorrow's list.
   - **Work needed first:**
     - **Daily input: done.** `src/ingest/recent_changes.py` turns recent
       changes into the dumps' revision records, checked edit by edit
       against the dump (§5).
     - **The local daily job: done** (§3 item 18). It computes features
       with the training code from dump history plus live days, so it
       carries no state of its own yet. Its features match the dump path's
       on real data (`check_daily_parity.py`), and backtesting puts 20 of
       each day's top 100 bursts in the next day (§5).
     - **The page: done, locally** (`web/`, `build_site.py`).
     - **The daily schedule and publishing: done** (§3 item 19). The
       repo is public, and the site is pushed to `gh-pages` daily.
       - Pages serves it at the custom domain `zoroaster.invigoro.me`
         (set 2026-10-01). `invigoro.github.io/Zoroaster` redirects
         there.
       - "Enforce HTTPS" is off, so the redirect lands on plain HTTP.
       - Setting the domain committed a CNAME file to `gh-pages` on
         GitHub, so the 2026-10-02 publish was rejected. `publish_site.py`
         now rebases onto GitHub's branch first (§3 item 22).
       - **The site's shape** (decided 2026-10-01): the main page lists
         the prophecies and links to "How the prophet works"
         (`how.html`). That page explains the method and shows the
         forecasts' track record (§3 item 23). Version 3's prophecies and
         their grades will join it there.
     - **Still to do:**
       - Choose a license (none yet, so the code is all rights reserved).
       - For GitHub Actions: a compact rolling state, since recent changes
         keep only 30 days and a runner can't hold the 7 GB of dump
         revisions. It needs the per-page daily counts for the 90-day burst
         baseline and the 365-day windows, a few hundred MB. It should be
         checked against the local job's features, as the local job was
         checked against the dump path's.
     - **Link features: not needed.** They don't help the burst target
       (§5), which is just as well, since the full English Wikipedia graph
       (1B+ links) is too big for a free runner.
     - **Weekly periodicity: tried, no reliable gain** (§5 "Stage 1 burst
       target"). Same-weekday lags help on test but not on validation.

10. **Version 2: forecast what the edits will say** (planned 2026-10-01).
    For the top pages of each day's prophecy, forecast what each page will
    gain that day, not just whether it bursts.
    - **Decided 2026-10-01:**
      - **The most important guardrail is labeling.** Every forecast is
        shown as non-factual: a machine-generated guess at what editors
        might add, not news and not a claim about anyone.
        - The label sits next to each forecast, so a screenshot of one
          still carries it.
        - The page's header and explainer say the same.
      - **Structured forecasts only, for every page, to start** (decided
        later the same day): which sections, and what kind of edit. The
        model still learns the day's new prose after the header, and its
        quality is measured but not published.
        - Header tokens come first, so the prose doesn't affect how they're
          predicted.
        - Switching to free text later needs no new data or retraining:
          it's a display change plus the free-text guardrails below.
      - **Living people get structured forecasts only, even then**: never
        free text. They're not left out, but nothing is generated that
        reads as a claim about a person.
        - Free text is the riskiest form: fluent and specific, in
          Wikipedia's voice, and often wrong.
        - The base model knows nothing after 2024, so for 2026 it can only
          extrapolate.
      - **20 pages a day, short output**, to start.
    - **Why Stage 2 doesn't drop in:** it predicts an edit's text given
      where it goes (the parent revision around the insertion point).
      Tomorrow's edit locations and parent states are unknown. So the
      target becomes the page's net change over day D, from its state at
      the end of D−1. That's the same start-to-end diff as the neighbor
      snippets (`build_stage2_neighbor_changes.py`).
    - **The task:**
      - **Input**, all as of the end of D−1:
        - the title as of then, and the date;
        - the page's lead and section headings;
        - its own net changes over the previous days, likely the strongest
          signal, since ongoing stories continue;
        - the Stage 1 signals;
        - the titles of bursting linked pages. They helped Stage 2, while
          the neighbors' text didn't.
      - **Output:**
        - first, a structured header: the sections the day's new text goes
          into, and the kinds of change (prose, new section, table rows,
          infobox, references);
        - then the day's new prose, in page order, cleaned of markup and
          capped at about 256 tokens.
        - Living people's pages show only the header.
    - **Phases:**
      1. **Data (done 2026-10-01, §3 item 20).** Page-days like the ones
         the page will forecast.
         - On each day: the burst model's top-ranked pages that did get
           edited, plus a random sample of edited pages for contrast.
         - Train and validation come from the panel's windows, test from
           the 14 evaluation days.
         - Each example needs the page at the ends of D−2, D−1 and D, about
           60K revisions and an hour of API fetching.
      2. **Baselines and metrics (done 2026-10-01, §5 "Version 2 data and
         baselines").**
         - Baselines: "yesterday again" (today's change says what
           yesterday's did, in the same sections), and title only.
         - For the structured forecast, which is what's published: section
           precision and recall (did it name the sections that got new
           text?), and accuracy on the kinds of change.
         - For the prose, measured but not published: NLL per token, and
           new-word recall, the share of the day's new content words absent
           from the page at the end of D−1 that a generation contains.
      3. **Model (first runs done 2026-10-01, §3 item 21 and §5 "Version
         2 model: structured forecasts").** QLoRA on Qwen2.5-1.5B
         (`train_v2.py`), with prompts of about 300 tokens.
         - Variants: the page alone; plus its own recent changes; plus the
           signals and neighbor names.
         - First: the full prompt and the page-only control, one seed each.
           The middle variant and second seeds follow only if the full
           prompt beats "yesterday again".
         - **Result:** it didn't. It's better on kinds and worse at naming
           the main section, so the follow-up runs haven't been started.
         - **Ranking (done 2026-10-01, §5 "Version 2: ranked forecasts").**
           The model's own probabilities name the top 3 sections, and
           per-kind thresholds come from validation.
           - Kinds now beat "yesterday again" (+0.032 ± 0.004).
           - Sections tie it once both name as many.
         - **Open: what next.** Version 3 (step 11) is to use these
           forecasts as evidence. Options:
           - the middle variant and a second seed;
           - a forecast of yesterday's sections plus the ranked kinds,
             judged on validation, not test;
           - or leave version 2 as it is until version 3 shows what its
             prophet needs.
      4. **Deployment.** A "foretell" step in `run_daily.py`, after the
         predictions.
         - Fetch the top 20 pages' current text (one request), generate
           (about a minute on the GPU), and add the result to `D.json`.
         - The page shows each forecast with its label.
         - Score the forecasts the next day by new-word recall, and publish
           that too.
      5. **With the Actions move:** CPU inference via llama.cpp, with a
         quantized model of about 1 GB.
    - **Other guardrails:**
      - Filter generations that touch death, crime, legal trouble, health
        and similar topics.
      - For living people, also leave sensitive section names ("Death",
        "Legal issues", "Controversies", "Personal life", and health ones
        such as diagnoses) out of the structured forecast.
      - **Publish only section names already on the page**, verbatim, for
        every page. Phase 3's invented names mostly stated outcomes
        ("2026 Four Continents champion"), so a predicted new section is
        shown only as the kind "new section", without its name.
    - **Free text now comes from step 11** (2026-10-01): a separate LLM
      writes event prophecies, and version 2's forecasts are part of its
      evidence.
11. **Version 3: free-text prophecies of real-world events** (planned
    2026-10-01). This is the project's real aim (§1): predictions about
    events, with Wikipedia as the sensor, written as free text and graded
    the next day. Two local LLM steps sit on top of the edit predictor
    and stay separate from it, so each part can be changed and tested on
    its own.
    - **The prophet** runs on day D, after the daily predictions.
      - **Input**, all as of the end of D−1, for the burst model's top
        pages:
        - each page's lead, yesterday's changes and their new text;
        - its bursting linked pages;
        - its version 2 structured forecast.
      - **Output:** 5–10 predictions in a fixed form: "I predict that
        [something specific] will happen by [date]". Each one cites the
        pages and signals it rests on.
      - It may use only the evidence it's given. The base models know
        nothing after 2024, so anything else would be invented.
    - **The judge** runs at the next daily run, once D is over. Grading
      is a separate step from writing, so the prophet never grades
      itself.
      - **Input:** each prediction, the evidence it cited, and what D
        actually brought:
        - the cited pages' changes over D, derived as version 2 does from
          each page at the ends of D−1 and D. Only those pages are
          fetched.
        - that day's Portal:Current events page, a daily list of major
          events that editors curate. It's outside the mainspace scope
          (§2) and is used only for grading. It may fill in late, so the
          judge may need to wait an extra day.
      - **Output:** a score for each prediction against the rubric, with
        the reasoning, published as the day's record.
    - **The rubric** will be refined on validation days:
      - **Outcome:** happened as stated, happened in part, didn't happen,
        or can't be told from D's data.
      - **Novelty:** was it already known at the end of D−1? Predicting
        something already on the page earns nothing.
      - **Specificity:** "news about X continues" is cheap; a named
        outcome or number is worth more. Credit is outcome times
        specificity, so vague predictions can't score well.
      - **Grounding:** does the cited evidence support it?
    - **Checks before anything is published:**
      - **Point-in-time:** the prophet sees only data up to the end of
        D−1, and the judge only what D brought.
      - **The judge is checked against hand grades,** e.g. on 100
        predictions, before its scores are trusted. Claude drafts the
        grades, and the user confirms or adjusts them.
      - **Baselines are graded the same way:**
        - "yesterday's stories continue";
        - the same model writing without the edit signals.
        The prophet has to beat both to show any skill.
    - **Models:** a local instruction-tuned model, e.g. Qwen2.5-7B-Instruct
      in 4-bit (about 5 GB, so it fits the 8 GB GPU), or a 3B one for
      speed. Ten predictions and their grades should take a few minutes a
      day. After the Actions move, it runs on CPU via llama.cpp.
    - **Guardrails:**
      - **From version 2:** every prediction is labeled as a
        machine-generated guess, not news.
      - **Never predicted, about anyone:** health, death, crime, legal
        trouble, personal life and the like.
        - The list will grow, so it lives in one place in the code.
        - Both the prophet's instructions and a separate check on its
          output use it.
      - **Living people, in two milestones** (decided 2026-10-01):
        1. **First, none.** The prophet names no living person. Its
           instructions say so, and a separate check drops any prediction
           that names or points to a person.
        2. **Then, public-role events only:** winning an election, being
           sworn in, a team playing in the final, and the like. The
           never-predicted list still applies.
    - **Phases:**
      1. **A prophet prototype** on development days, read by hand
         (started 2026-10-01, §3 items 24–25). The development days are the
         site's own prophecies, 2026-09-18 to 2026-10-01.
         - **Next:** a timing check. Ask whether the evidence says the
           prediction's result comes on the day or later, and drop "later".
           41 of the 47 misses drafted so far were results due after the
           day.
         - **Next:** a check for team or country wins in individual events,
           which point to a person. Try it on known sentences first, as the
           person check was.
      2. **The judge and rubric,** checked against hand grades (started
         2026-10-01, §3 item 25).
         - The rubric and 68 draft grades are done. The user is confirming
           or adjusting them on the review page.
         - The 7B judge's first grades agree poorly with the drafts. Next,
           try one focused question per field, as the checks do, or a
           different model.
      3. **A backtest** of prophet and judge over the test days, against
         the baselines. Prompts are tuned on development days only.
         - **Open: which test days.** Version 2's 14 test days are
           unbiased (every sampled page), but they come from the 20% page
           sample, so their top 20 is about the whole wiki's top 100. The
           alternative is days still to come, as the daily job runs.
      4. **Daily:** a "prophesy" step in `run_daily.py` after the
         predictions, and a "judge" step for the day before. The site
         shows both, labeled. Phases 1–4 leave living people out (the
         first milestone).
      5. **Living people, public-role events only** (the second
         milestone). The person check and the topic list are checked by
         hand on past days before this goes daily.
      6. **The Actions move:** llama.cpp on CPU.
    - **How it relates to version 2:** version 2's forecasts become the
      prophet's evidence. Its own phase 4, publishing them on the site,
      may not be needed.

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
- **Version 3: predictions due later than the next day, and how updates are
  scored** (raised 2026-10-01). The user is open to longer horizons, and
  asked how to grade a forecast the prophet may change. Claude's proposal,
  not yet decided:
  - A forecast is never edited. An update is a new forecast with its own
    date, and every version stays on the record.
  - Each day a question is open, the forecast standing that day is graded
    when it resolves, and the question's score is the mean over those days.
    Forecasting tournaments (the Good Judgment Project, Metaculus) score
    this way.
    - Being right early and staying right scores best.
    - A late switch to the right answer earns credit only from the switch
      on, and flip-flopping doesn't pay.
  - Scores are also reported by lead time (the day before, within a week,
    longer), so easy last-minute forecasts can't hide weak long-range ones.
