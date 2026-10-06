// Renders data/prophecies.json: version 3's prophecy for each of the last days, newest first, which the daily
// job writes (scripts/prophesy_daily.py, then build_site.py). Each holds its published predictions alone.
"use strict";

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  node.append(...children);
  return node;
}

function formatDay(iso, options = { weekday: "long", year: "numeric", month: "long", day: "numeric" }) {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d)).toLocaleDateString("en-US", { ...options, timeZone: "UTC" });
}

function daysBetween(a, b) {
  return Math.round((Date.parse(b + "T00:00:00Z") - Date.parse(a + "T00:00:00Z")) / 86400000);
}

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
  "November", "December"];

// A prediction's own "… by the end of Sunday, 11 October 2026." at its end, when that's the due date shown below it.
function withoutDue(text, dueDay) {
  const [y, m, d] = dueDay.split("-").map(Number);
  const tail = new RegExp(`,? by the end of [A-Z][a-z]+day, ${d} ${MONTHS[m - 1]} ${y}\\.$`);
  return text.replace(tail, ".");
}

function due(prophecy, prediction) {
  const n = daysBetween(prophecy.date, prediction.due);
  const when = formatDay(prediction.due, { weekday: "long", month: "long", day: "numeric" });
  return n === 0 ? `due by the end of ${when}, the same day` : `due by the end of ${when}`;
}

// Each day's prophecy has its own comment thread in the repo's GitHub Discussions, through giscus
// (https://giscus.app): found by its title, "Prophecy for 2026-10-05", in the Announcements category, where only
// the maintainer and giscus may start a thread. Strict, so a day's title never matches another's.
const GISCUS = {
  "data-repo": "invigoro/Zoroaster", "data-repo-id": "R_kgDOTFSlwA",
  "data-category": "Announcements", "data-category-id": "DIC_kwDOTFSlwM4DHBq8",
  "data-mapping": "specific", "data-strict": "1", "data-reactions-enabled": "1", "data-emit-metadata": "0",
  "data-input-position": "top", "data-theme": "transparent_dark", "data-lang": "en", "data-loading": "lazy",
  crossorigin: "anonymous",
};
let discussion = null;

function discuss(prophecy) {
  if (discussion) discussion.remove();
  discussion = document.createElement("script");
  discussion.src = "https://giscus.app/client.js";
  discussion.async = true;
  for (const [key, value] of Object.entries(GISCUS)) discussion.setAttribute(key, value);
  discussion.setAttribute("data-term", `Prophecy for ${prophecy.date}`);
  document.body.append(discussion);  // giscus puts its frame in the .giscus box, in place of the last day's
}

function render(prophecy) {
  document.getElementById("day").textContent = formatDay(prophecy.date);
  const list = document.getElementById("prophecy");
  list.replaceChildren();
  if (!prophecy.predictions.length) {
    list.append(el("li", { class: "placeholder" }, "The prophet foretold nothing that passed its checks that day."));
  }
  for (const p of prophecy.predictions) {
    const meta = el("p", { class: "meta" }, el("span", { class: "topic" }, p.topic || "other"), ` · ${due(prophecy, p)}`);
    if (p.confidence) meta.append(` · ${p.confidence} confidence`);
    list.append(el("li", {}, el("p", { class: "text" }, withoutDue(p.text, p.due)), meta));
  }
  const late = document.getElementById("late");
  late.hidden = !prophecy.late;
  if (prophecy.late) {
    const made = prophecy.generated_at.slice(0, 10);
    const dayBefore = new Date(Date.parse(prophecy.date + "T00:00:00Z") - 86400000).toISOString().slice(0, 10);
    late.replaceChildren(el("strong", {}, "Foretold late."), ` Made at ${new Date(prophecy.generated_at)
      .toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", timeZone: "UTC" })} UTC on `
      + `${formatDay(made, { month: "long", day: "numeric" })}, after its day had begun. As every night, the `
      + `prophet read only what Wikipedia said by the end of ${formatDay(dayBefore, { weekday: "long", month: "long", day: "numeric" })}.`);
  }
  document.getElementById("generated").textContent = prophecy.generated_at
    ? `Foretold ${new Date(prophecy.generated_at).toUTCString()} by ${prophecy.model}.` : "";
  discuss(prophecy);
}

function renderDays(prophecies) {
  const select = document.getElementById("days");
  select.replaceChildren(...prophecies.map((p, i) => el("option", { value: String(i) }, formatDay(p.date) + (p.late ? " (late)" : ""))));
  select.addEventListener("change", () => render(prophecies[Number(select.value)]));
  document.getElementById("day-pick").hidden = prophecies.length < 2;
  render(prophecies[0]);
}

fetch("data/prophecies.json", { cache: "no-cache" })
  .then((response) => { if (!response.ok) throw new Error(response.status); return response.json(); })
  .then((prophecies) => { if (!prophecies.length) throw new Error("none yet"); renderDays(prophecies); })
  .catch(() => {
    document.getElementById("prophecy").replaceChildren(
      el("li", { class: "placeholder" }, "The prophecy isn't available right now."));
  });
