// Renders data/forecasts.json, which the daily run writes from version 2's test
// forecasts (scripts/v2_report.py --site): the scores, then each page-day's
// forecasts next to what happened, a page at a time.
"use strict";

const PAGE = 40;
const METRICS = [["section_precision", "Section precision"], ["main_section_hit", "Main section named"],
  ["kinds_jaccard", "Kinds of change"]];
let data = null;
let shown = PAGE;

const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  node.append(...children);
  return node;
}

function chip(text, cls) {
  return el("span", { class: `chip ${cls}` }, text);
}

function renderScores() {
  const table = $("scores");
  table.replaceChildren(el("tr", {}, el("th", {}, "Forecaster"), ...METRICS.map(([, label]) => el("th", {}, label))));
  for (const s of data.scores) {
    table.append(el("tr", {}, el("td", {}, s.name),
      ...METRICS.map(([key]) => el("td", {}, `${s[key][0].toFixed(3)} / ${s[key][1].toFixed(3)}`))));
  }
  $("count").textContent = data.count.toLocaleString("en-US");
}

function line(grid, who, sections, kinds, extra) {
  grid.append(el("div", { class: "who" }, who));
  const value = el("div");
  value.append(...(sections.length ? sections : [chip("no section named", "miss")]));
  const ks = el("span", { class: "kinds" }, " · ");
  ks.append(...(kinds.length ? kinds : ["no kinds named"]));
  value.append(ks);
  if (extra) value.append(extra);
  grid.append(value);
}

function renderDays() {
  const sel = $("sel").value, liv = $("liv").value, yes = $("yes").value, q = $("q").value.trim().toLowerCase();
  const rows = data.rows.filter((r) => (sel === "all" || (sel === "top") === r.top)
    && (liv === "all" || (liv === "yes") === r.liv) && (yes === "all" || (yes === "changed") === r.y)
    && (!q || r.t.toLowerCase().includes(q)));
  $("shown").textContent = `${rows.length.toLocaleString("en-US")} page-days`;
  const box = $("days");
  box.replaceChildren();
  for (const r of rows.slice(0, shown)) {
    const url = "https://en.wikipedia.org/wiki/" + encodeURIComponent(r.t.replaceAll(" ", "_"));
    const badge = `${r.d} · ${r.top ? "top page" : "random page"}${r.liv ? " · living person" : ""}`;
    const day = el("article", { class: "day" }, el("h3", {}, el("a", { href: url }, r.t), el("span", { class: "badge" }, badge)));
    const grid = el("div", { class: "grid" });
    const quote = r.a.p ? el("p", { class: "quote" }, `New text: ${r.a.p}`) : null;
    line(grid, "What happened", r.a.s.map(([name, cls, n]) => chip(`${name} (${n.toLocaleString("en-US")} characters)`, cls)),
      r.a.k.map((k) => chip(k, "hit")), quote);
    data.shown.forEach((who, i) => {
      const f = r.f[i];
      line(grid, who, f.s.map(([name, cls]) => chip(name, cls)), f.k.map(([k, hit]) => chip(k, hit ? "hit" : "miss")));
    });
    day.append(grid);
    box.append(day);
  }
  $("more").hidden = rows.length <= shown;
}

["sel", "liv", "yes"].forEach((id) => $(id).addEventListener("change", () => { shown = PAGE; renderDays(); }));
$("q").addEventListener("input", () => { shown = PAGE; renderDays(); });
$("more").addEventListener("click", () => { shown += PAGE; renderDays(); });

fetch("data/forecasts.json", { cache: "no-cache" })
  .then((response) => { if (!response.ok) throw new Error(response.status); return response.json(); })
  .then((json) => { data = json; renderScores(); renderDays(); })
  .catch(() => { $("days").replaceChildren(el("p", { class: "placeholder" }, "The forecasts aren't available right now.")); });
