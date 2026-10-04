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

function due(prophecy, prediction) {
  const n = daysBetween(prophecy.date, prediction.due);
  const when = formatDay(prediction.due, { weekday: "long", month: "long", day: "numeric" });
  return n === 0 ? `due by the end of ${when}, the same day` : `due by the end of ${when}`;
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
    list.append(el("li", {}, el("p", { class: "text" }, p.text), meta));
  }
  document.getElementById("generated").textContent = prophecy.generated_at
    ? `Foretold ${new Date(prophecy.generated_at).toUTCString()} by ${prophecy.model}.` : "";
}

function renderDays(prophecies) {
  const select = document.getElementById("days");
  select.replaceChildren(...prophecies.map((p, i) => el("option", { value: String(i) }, formatDay(p.date))));
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
