"""A readable page of version 2's test forecasts, next to what happened.

For each test page-day it shows:
- the page, titled as it was then;
- what the day's edits actually changed;
- what changed the day before, which is the "yesterday again" baseline;
- each model forecast, with hits marked.

It writes `report.html` into the run directory. This is a local file, not
part of the site: the forecasts are machine-generated guesses about real
pages, some of them about living people.

Usage:
    python scripts/v2_report.py [--run-dir data/processed/enwiki/v2/qwen2.5-1.5b]
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq

from scripts.build_v2_targets import OUT_DIR
from scripts.fetch_v2_examples import EXAMPLES_DIR
from scripts.v2_baselines import common_kinds, forecasts
from src.forecast.changes import LEAD
from src.forecast.metrics import main_sections
from src.forecast.prompts import parse_forecast, point_in_time_titles

RUN_DIR = OUT_DIR / "qwen2.5-1.5b"
COLUMNS = ["page_id", "date", "split", "selection", "living", "page_title", "prompt_id", "bursting_neighbors",
           "heading_titles", "sections", "section_chars", "kinds", "prose", "yesterday_sections",
           "yesterday_section_chars", "yesterday_kinds"]
PROSE_CHARS = 300
METRICS = (("section_precision", "section precision"), ("main_section_hit", "main section named"),
           ("kinds_jaccard", "kinds Jaccard"))


def forecast_view(forecast: dict, row: dict) -> dict:
    """A forecast with each section marked main, hit or miss (and whether
    it's on the page at the end of D-1), and each kind marked right or wrong."""
    main = main_sections(row["sections"], row["section_chars"], 1)
    sections = [{"name": s, "mark": "main" if main and s == main[0] else "hit" if s in row["sections"] else "miss",
                 "on_page": s == LEAD or s in row["heading_titles"]} for s in forecast["sections"]]
    return {"sections": sections, "kinds": [{"name": k, "hit": k in row["kinds"]} for k in forecast["kinds"]]}


def display(run: str) -> str:
    """A run's name as shown: "full / seed 1234" -> "model (full prompt, greedy)"."""
    return f"model ({run.split(' / ')[0]} prompt, greedy)"


def view_row(row: dict, named: dict[str, dict]) -> dict:
    sizes = dict(zip(row["sections"], row["section_chars"]))
    actual = main_sections(row["sections"], row["section_chars"])
    prose = row["prose"][:PROSE_CHARS] + ("…" if len(row["prose"]) > PROSE_CHARS else "")
    return {
        "title": row["page_title"].replace("_", " "), "date": row["date"].isoformat(), "top": row["selection"] == "top",
        "living": bool(row["living"]), "changed_yesterday": bool(row["yesterday_sections"]),
        "actual": {"sections": [[s, sizes[s]] for s in actual], "kinds": row["kinds"], "prose": prose},
        "forecasts": {name: forecast_view(f, row) for name, f in named.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    args = parser.parse_args(argv)
    rows = pq.read_table(EXAMPLES_DIR, columns=COLUMNS).to_pylist()
    point_in_time_titles(rows, pq.read_table(OUT_DIR / "titles.parquet").to_pylist())
    common = common_kinds([r for r in rows if r["split"] == "train"])
    test = [r for r in rows if r["split"] == "test"]
    generations = json.loads((args.run_dir / "generations.json").read_text(encoding="utf-8"))
    if [(r["page_id"], r["date"].isoformat()) for r in test] != [(k["page_id"], k["date"]) for k in generations["examples"]]:
        raise SystemExit("generations.json doesn't match the test examples' order")
    named = {"yesterday again": [forecasts(r, common)["yesterday again"] for r in test]}
    named |= {display(n): [parse_forecast(t) for t in texts] for n, texts in generations["runs"].items()}
    summary = json.loads((args.run_dir / "results.json").read_text(encoding="utf-8"))["forecasts"]
    table = {"yesterday again": summary["all"]["yesterday again"] | {"top": summary["top"]["yesterday again"]}}
    table |= {display(n): summary["all"][n] | {"top": summary["top"][n]} for n in generations["runs"]}
    ranked_path = args.run_dir / "ranked.json"
    if ranked_path.exists():
        ranked = json.loads(ranked_path.read_text(encoding="utf-8"))
        keys = [(k["page_id"], k["date"]) for k in ranked["test"]["examples"]]
        if keys != [(r["page_id"], r["date"].isoformat()) for r in test]:
            raise SystemExit("ranked.json doesn't match the test examples' order")
        for name, fs in ranked["test"]["forecasts"].items():
            if "thresholds 0.5" in name:  # a reference for rank_v2.py's comparisons, not worth reading
                continue
            named[name] = fs
            table[name] = ranked["test"]["summary"]["all"][name] | {"top": ranked["test"]["summary"]["top"][name]}
    names = list(named)
    data = [view_row(r, {n: named[n][i] for n in names}) for i, r in enumerate(test)]
    page = TEMPLATE.replace("__DATA__", json.dumps({"names": names, "rows": data}, ensure_ascii=False).replace("</", "<\\/"))
    page = page.replace("__SUMMARY__", summary_table(table)).replace("__COUNT__", f"{len(test):,}")
    out = args.run_dir / "report.html"
    out.write_text(page, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(test):,} page-days, {len(names)} forecasters)")
    return 0


def summary_table(table: dict[str, dict]) -> str:
    head = "".join(f"<th>{html.escape(label)}<br><small>all / top pages</small></th>" for _, label in METRICS)
    body = "".join(
        f"<tr><td>{html.escape(n)}</td>"
        + "".join(f"<td>{s[m]:.3f} / {s['top'][m]:.3f}</td>" for m, _ in METRICS) + "</tr>"
        for n, s in table.items())
    return f"<table class=summary><tr><th>forecaster</th>{head}</tr>{body}</table>"


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Version 2 forecasts</title>
<style>
:root { --bg: #fbfaf7; --fg: #1d1b18; --muted: #6b665d; --line: #e3dfd6; --card: #ffffff;
  --hit: #1f7a3a; --hit-bg: #e3f3e7; --miss: #a33a2a; --miss-bg: #f8e6e2; --warn-bg: #fff4d6; --warn: #6b4e00; }
@media (prefers-color-scheme: dark) { :root { --bg: #171614; --fg: #ece8e1; --muted: #a39d92; --line: #34312c;
  --card: #201f1c; --hit: #7fd394; --hit-bg: #1f3325; --miss: #f0a091; --miss-bg: #3a2420; --warn-bg: #3a3218; --warn: #f2d27a; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 1.5rem; margin: 0 0 4px; }
.label { background: var(--warn-bg); color: var(--warn); border-radius: 8px; padding: 10px 14px; margin: 16px 0; }
table.summary { border-collapse: collapse; margin: 12px 0 20px; font-size: 0.9rem; }
table.summary td, table.summary th { border-bottom: 1px solid var(--line); padding: 6px 12px; text-align: left; }
.controls { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; margin: 12px 0; position: sticky; top: 0;
  background: var(--bg); padding: 8px 0; border-bottom: 1px solid var(--line); }
.controls input[type=search] { padding: 4px 8px; min-width: 200px; }
.day { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px 16px; margin: 12px 0; }
.day h2 { font-size: 1.05rem; margin: 0; } .day h2 a { color: inherit; }
.meta { color: var(--muted); font-size: 0.85rem; }
.grid { display: grid; grid-template-columns: 220px 1fr; gap: 4px 12px; margin-top: 8px; font-size: 0.9rem; }
.grid .who { color: var(--muted); }
.chip { display: inline-block; border-radius: 6px; padding: 0 6px; margin: 1px 2px; border: 1px solid var(--line); }
.chip.main { background: var(--hit-bg); color: var(--hit); font-weight: 700; } .chip.hit { background: var(--hit-bg); color: var(--hit); }
.chip.miss { background: var(--miss-bg); color: var(--miss); } .chip.off::after { content: " (not on the page)"; font-size: 0.8em; }
.kinds { color: var(--muted); } .prose { color: var(--muted); font-style: italic; margin-top: 6px; font-size: 0.85rem; }
.badge { font-size: 0.75rem; border: 1px solid var(--line); border-radius: 999px; padding: 0 8px; margin-left: 6px; color: var(--muted); }
@media (max-width: 640px) { .grid { grid-template-columns: 1fr; } .grid .who { margin-top: 6px; } }
</style></head>
<body><main>
<h1>Version 2: test forecasts</h1>
<div class="meta">__COUNT__ test page-days (December 2025 to June 2026), each forecast made from the page as it stood the day before.</div>
<div class="label"><strong>Machine-generated forecasts, not facts.</strong> Each forecast is a model's guess at which
sections Wikipedia editors would change that day, and how. It's shown next to what they actually changed. A forecast
makes no claim about any person or event. Section names marked "not on the page" were made up by the model.</div>
__SUMMARY__
<div class="meta">Green: the section changed that day (bold: it changed most) or the kind of change happened. Red: it didn't.</div>
<div class="controls">
  <label>Pages <select id="sel"><option value="all">all</option><option value="top">burst model's top</option><option value="random">random</option></select></label>
  <label>People <select id="liv"><option value="all">all</option><option value="yes">living people only</option><option value="no">no living people</option></select></label>
  <label>Yesterday <select id="yes"><option value="all">any</option><option value="changed">changed</option><option value="quiet">quiet</option></select></label>
  <input type="search" id="q" placeholder="Search titles">
  <span class="meta" id="count"></span>
</div>
<div id="days"></div>
<button id="more" hidden>Show more</button>
</main>
<script>
const DATA = __DATA__;
const PAGE = 50;
let shown = PAGE;
const $ = (id) => document.getElementById(id);
function chip(text, cls) { const s = document.createElement("span"); s.className = "chip " + cls; s.textContent = text; return s; }
function line(container, who, sections, kinds, extra) {
  const w = document.createElement("div"); w.className = "who"; w.textContent = who; container.append(w);
  const v = document.createElement("div");
  sections.forEach((s) => v.append(s));
  if (!sections.length) v.append(chip("no section", "miss"));
  const k = document.createElement("span"); k.className = "kinds"; k.textContent = " · ";
  kinds.forEach((c) => k.append(c)); if (!kinds.length) k.append("no kinds named");
  v.append(k); if (extra) v.append(extra); container.append(v);
}
function render() {
  const sel = $("sel").value, liv = $("liv").value, yes = $("yes").value, q = $("q").value.trim().toLowerCase();
  const rows = DATA.rows.filter((r) => (sel === "all" || (sel === "top") === r.top)
    && (liv === "all" || (liv === "yes") === r.living) && (yes === "all" || (yes === "changed") === r.changed_yesterday)
    && (!q || r.title.toLowerCase().includes(q)));
  $("count").textContent = rows.length.toLocaleString() + " page-days";
  const box = $("days"); box.replaceChildren();
  rows.slice(0, shown).forEach((r) => {
    const day = document.createElement("section"); day.className = "day";
    const h = document.createElement("h2"); const a = document.createElement("a");
    a.href = "https://en.wikipedia.org/wiki/" + encodeURIComponent(r.title.replaceAll(" ", "_")); a.textContent = r.title;
    h.append(a); const b = document.createElement("span"); b.className = "badge";
    b.textContent = r.date + (r.top ? " · top page" : " · random page") + (r.living ? " · living person" : "");
    h.append(b); day.append(h);
    const g = document.createElement("div"); g.className = "grid";
    const prose = r.actual.prose ? Object.assign(document.createElement("div"), {className: "prose", textContent: "New text: " + r.actual.prose}) : null;
    line(g, "what happened", r.actual.sections.map(([s, n], i) => chip(s + " (" + n.toLocaleString() + " chars)", i ? "hit" : "main")),
      r.actual.kinds.map((k) => chip(k, "hit")), prose);
    DATA.names.forEach((n) => {
      const f = r.forecasts[n];
      line(g, n, f.sections.map((s) => chip(s.name, s.mark + (s.on_page ? "" : " off"))), f.kinds.map((k) => chip(k.name, k.hit ? "hit" : "miss")));
    });
    day.append(g); box.append(day);
  });
  $("more").hidden = rows.length <= shown;
}
["sel", "liv", "yes"].forEach((id) => $(id).addEventListener("change", () => { shown = PAGE; render(); }));
$("q").addEventListener("input", () => { shown = PAGE; render(); });
$("more").addEventListener("click", () => { shown += PAGE; render(); });
render();
</script>
</body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
