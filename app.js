// Renders data/latest.json (tomorrow's prophecy) and data/latest.outcomes.json
// (how yesterday's turned out), both written by the daily job.
"use strict";

const SHOWN = 50;

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  node.append(...children);
  return node;
}

function formatDay(iso) {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d)).toLocaleDateString("en-US", {
    weekday: "long", year: "numeric", month: "long", day: "numeric", timeZone: "UTC",
  });
}

function plural(n, word) {
  return `${n.toLocaleString("en-US")} ${word}${n === 1 ? "" : "s"}`;
}

function renderPages(prophecy) {
  document.getElementById("day").textContent = formatDay(prophecy.date);
  const list = document.getElementById("pages");
  list.replaceChildren();
  const top = prophecy.pages[0] ? prophecy.pages[0].score : 1;
  for (const page of prophecy.pages.slice(0, SHOWN)) {
    const url = "https://en.wikipedia.org/wiki/" + encodeURIComponent(page.title.replaceAll(" ", "_"));
    const activity = page.edits_1d > 0
      ? `yesterday ${plural(page.edits_1d, "edit")} by ${plural(page.editors_1d, "editor")}`
      : "no edits yesterday";
    const meter = el("div", { class: "meter", "aria-hidden": "true" }, el("span"));
    meter.firstChild.style.width = `${Math.max(2, (100 * page.score) / top)}%`;
    list.append(el("li", {},
      el("span", { class: "rank" }, String(page.rank)),
      el("a", { class: "title", href: url }, page.title),
      el("span", { class: "chance", title: "the model's estimate of a burst tomorrow" }, `${Math.round(100 * page.score)}%`),
      meter,
      el("span", { class: "detail" }, `${activity}; ${plural(page.edits_7d, "edit")} in the last week`),
    ));
  }
  document.getElementById("generated").textContent =
    `Foretold ${new Date(prophecy.generated_at).toUTCString()} from ${plural(prophecy.candidates, "candidate page")}.`;
}

function renderRecord(outcomes) {
  const model = outcomes.precision.model["100"];
  const baseline = outcomes.precision["edits yesterday"]["100"];
  const text = document.getElementById("record-text");
  text.replaceChildren(
    `For ${formatDay(outcomes.date)}, `,
    el("strong", { class: "hit" }, `${Math.round(100 * model)} of the top 100`),
    ` came true: those pages burst. Ranking by yesterday's edits alone would have caught ${Math.round(100 * baseline)}.`,
  );
  if (outcomes.hits.length) {
    text.append(" Among them: " + outcomes.hits.slice(0, 5).join("; ") + ".");
  }
  document.getElementById("record").hidden = false;
}

async function load(name) {
  const response = await fetch(`data/${name}`, { cache: "no-cache" });
  if (!response.ok) throw new Error(`${name}: ${response.status}`);
  return response.json();
}

load("latest.json").then(renderPages).catch(() => {
  document.getElementById("pages").replaceChildren(el("li", { class: "placeholder" }, "The prophecy isn't available right now."));
});
load("latest.outcomes.json").then(renderRecord).catch(() => {});
