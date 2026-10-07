#!/usr/bin/env python3
"""
Self-contained HTML report for a benchmark run file. No network access: the
data is embedded as JSON and drawn with inline SVG, so the file can be opened
offline, archived with the run, or attached to a review.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BEAM benchmarks</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
  --series: #2a78d6; --series-wash: rgba(42,120,214,0.12);
  --good: #0ca30c; --good-text: #006300; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
    --series: #3987e5; --series-wash: rgba(57,135,229,0.16); --good-text: #0ca30c;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
  --series: #3987e5; --series-wash: rgba(57,135,229,0.16); --good-text: #0ca30c;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
       font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1280px; margin: 0 auto; padding: 20px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; font-weight: 600; }
h2 { font-size: 16px; margin: 28px 0 10px; font-weight: 600; }
.sub { color: var(--ink-2); margin: 0 0 16px; }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 14px 16px; }
.grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: 2px 12px; font-size: 13px; }
.kv dt { color: var(--muted); } .kv dd { margin: 0; overflow-wrap: anywhere; }
code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; }
.tiles { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 12px; }
.tile { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 10px 14px; min-width: 120px; }
.tile .label { color: var(--muted); font-size: 12px; } .tile .value { font-size: 22px; font-weight: 600; }
.warn { border-left: 3px solid var(--warning); background: var(--surface); padding: 8px 12px; margin: 6px 0;
        border-radius: 0 8px 8px 0; }
.warn b { color: var(--ink); }
.badge { display: inline-flex; align-items: center; gap: 4px; font-size: 12px; font-weight: 600; white-space: nowrap; }
.badge::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
.st-ok::before { background: var(--good); } .st-noisy::before, .st-contended::before { background: var(--warning); }
.st-error::before, .st-crashed::before { background: var(--critical); }
.st-refused::before { background: var(--serious); } .st-skipped::before { background: var(--axis); }
.flag { display: inline-block; font-size: 11px; color: var(--ink-2); border: 1px solid var(--ring);
        border-radius: 4px; padding: 0 5px; margin: 1px 2px 1px 0; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 10px; align-items: center; }
.controls input, .controls select { font: inherit; background: var(--surface); color: var(--ink);
  border: 1px solid var(--ring); border-radius: 6px; padding: 5px 8px; }
.tablewrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--grid); vertical-align: top; }
th { color: var(--ink-2); font-weight: 600; position: sticky; top: 0; background: var(--surface); cursor: pointer; user-select: none; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
tr.main:hover { background: var(--series-wash); cursor: pointer; }
tr.detail td { background: var(--page); }
.detail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; padding: 6px 0; }
.detail-grid h4 { margin: 0 0 4px; font-size: 12px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
.detail-grid ul { margin: 0; padding-left: 18px; }
svg text { fill: var(--muted); font-size: 11px; font-family: system-ui, sans-serif; }
svg .lbl { fill: var(--ink-2); font-size: 12px; }
svg .val { fill: var(--ink); font-size: 11.5px; }
#tip { position: fixed; pointer-events: none; background: var(--surface); color: var(--ink);
       border: 1px solid var(--ring); border-radius: 8px; padding: 6px 9px; font-size: 12px;
       box-shadow: 0 4px 14px rgba(0,0,0,.18); display: none; max-width: 360px; z-index: 10; }
.up { color: var(--critical); font-weight: 600; } .down { color: var(--good-text); font-weight: 600; }
.small { font-size: 12px; color: var(--ink-2); }
.empty { color: var(--muted); padding: 12px; }
</style>
</head>
<body>
<main>
  <h1>BEAM C benchmarks</h1>
  <p class="sub" id="subtitle"></p>
  <div id="warnings"></div>
  <div class="tiles" id="tiles"></div>

  <h2>Provenance</h2>
  <div class="grid" id="provenance"></div>

  <div id="comparison"></div>

  <h2>Slowest per call at this workload</h2>
  <p class="small">Median time per call (dot) on a log scale; the line runs to the first call after the model's caches were dropped. Hover a row for details. Ranking is per call: it does not know how often each function is called in practice. * contended: other processes competed for cores, so the time may be inflated.</p>
  <div class="card" id="chart-time"></div>

  <h2>Largest peak heap per call</h2>
  <p class="small">High-water mark of memory allocated inside one call and live at once, measured by wrapping the allocator (log scale).</p>
  <div class="card" id="chart-mem"></div>

  <h2>All results</h2>
  <div class="controls">
    <input id="q" type="search" placeholder="Filter benchmarks…" aria-label="Filter benchmarks">
    <select id="suite" aria-label="Suite"><option value="">All suites</option></select>
    <select id="status" aria-label="Status"><option value="">All statuses</option></select>
    <span class="small" id="count"></span>
  </div>
  <div class="tablewrap"><table id="results"><thead><tr>
    <th data-k="domain">Suite</th><th data-k="name">Benchmark</th><th data-k="status">Status</th>
    <th class="num" data-k="median">Median / call</th><th class="num" data-k="ci">95% CI</th>
    <th class="num" data-k="cold">First call</th><th class="num" data-k="peak">Peak heap</th>
    <th class="num" data-k="tput">Throughput</th><th>Notes</th></tr></thead><tbody></tbody></table></div>

  <div id="scaling"></div>

  <h2>Coverage</h2>
  <div class="card" id="coverage"></div>
</main>
<div id="tip" role="tooltip"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
"use strict";
const RUN = JSON.parse(document.getElementById("data").textContent);
const R = RUN.run, SWEEP = RUN.sweep;
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function ns(v) { if (v == null) return "–"; if (v < 1e3) return v.toFixed(1) + " ns"; if (v < 1e6) return (v/1e3).toFixed(2) + " µs";
  if (v < 1e9) return (v/1e6).toFixed(2) + " ms"; return (v/1e9).toFixed(2) + " s"; }
function bytes(b) { if (b == null) return "–"; const a = Math.abs(b);
  if (a >= 2**30) return (b/2**30).toFixed(2) + " GiB"; if (a >= 2**20) return (b/2**20).toFixed(1) + " MiB";
  if (a >= 1024) return (b/1024).toFixed(1) + " KiB"; return b.toFixed(0) + " B"; }
const statusOf = r => r.status !== "ok" ? r.status : (r.flags || []).includes("contended") ? "contended"
  : (r.flags || []).includes("noisy") ? "noisy" : "ok";
const badge = s => `<span class="badge st-${esc(s)}">${esc(s.toUpperCase())}</span>`;

/* tooltip */
const tip = document.getElementById("tip");
document.addEventListener("mousemove", e => {
  const t = e.target.closest("[data-tip]");
  if (!t) { tip.style.display = "none"; return; }
  tip.innerHTML = t.getAttribute("data-tip"); tip.style.display = "block";
  const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8);
  const y = Math.min(e.clientY + 14, innerHeight - tip.offsetHeight - 8);
  tip.style.left = x + "px"; tip.style.top = y + "px";
});

/* header */
const w = R.workload, b = R.build, env = R.environment;
document.getElementById("subtitle").textContent =
  `Run ${R.run_id}${R.label ? " (" + R.label + ")" : ""} · ${R.created} · ${w.alphabet} K=${w.order} max_length=${w.max_length} L=${w.length} · ${w.threads_effective} threads`;
document.getElementById("warnings").innerHTML = (R.warnings || []).map(x => `<div class="warn"><b>⚠ Warning</b> ${esc(x)}</div>`).join("");
const counts = (R.summary && R.summary.counts) || {};
const nOk = R.results.filter(r => statusOf(r) === "ok").length;
const tiles = [["Benchmarks", R.results.length], ["OK", nOk], ["Noisy", R.results.filter(r => statusOf(r) === "noisy").length],
  ["Contended", counts.contended || 0],
  ["Errors", counts.error || 0], ["Refused (budget)", counts.refused || 0], ["Crashed", counts.crashed || 0], ["Skipped", counts.skipped || 0]];
document.getElementById("tiles").innerHTML = tiles.filter(([k, v]) => v || k === "Benchmarks" || k === "OK" || k === "Errors")
  .map(([k, v]) => `<div class="tile"><div class="label">${k}</div><div class="value">${v}</div></div>`).join("");

const kv = rows => `<dl class="kv">${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>`;
const fx = Object.entries(R.fixtures || {}).map(([v, f]) => [`model ${v}`,
  `${f.n_states.toLocaleString()} states · ${f.n_nonzero_cells.toLocaleString()} non-zero cells · ${bytes(f.memory_bytes)} tables · built ${ns(f.observe_ns)} + ${ns(f.finalize_ns)}`]);
document.getElementById("provenance").innerHTML = [
  `<div class="card"><h4 style="margin:0 0 6px">Code measured</h4>${kv([
    ["commit", `<code>${esc((b.git_commit || "?").slice(0, 12))}</code> ${esc(b.git_branch || "")}${b.git_dirty ? " · dirty tree" : ""}`],
    ["libbeam.a sha256", `<code>${esc((b.libbeam_sha256 || "?").slice(0, 16))}…</code>`],
    ["harness sha256", `<code>${esc((b.beam_bench_sha256 || "?").slice(0, 16))}…</code>`],
    ["compiler", esc(b.compiler || "?")],
    ["uncommitted library files", b.library_sources_dirty && b.library_sources_dirty.length ? esc(b.library_sources_dirty.length) : "none"]])}</div>`,
  `<div class="card"><h4 style="margin:0 0 6px">Workload</h4>${kv([
    ["alphabet / order", `${esc(w.alphabet)} / K=${esc(w.order)}`],
    ["max_length / L", `${esc(w.max_length)} / ${esc(w.length)}`],
    ["corpus", `${esc(w.corpus_source)} · ${esc(w.corpus_sequences)} sequences · mean length ${(w.corpus_mean_length || 0).toFixed(1)}`],
    ...fx,
    ["threads", `${esc(w.threads_effective)} effective (requested ${esc(w.threads_requested)}; 0 = library default)`],
    ["memory budget", bytes(w.memory_budget_bytes)]])}</div>`,
  `<div class="card"><h4 style="margin:0 0 6px">Machine and method</h4>${kv([
    ["CPU", `${esc(env.cpu_model)} · ${esc(env.logical_cpus)} logical`],
    ["governor", esc(env.governor || "?")],
    ["busy CPUs at start", env.background_cpus_at_start != null ? env.background_cpus_at_start.toFixed(2) : (env.load_average_1m != null ? "load " + env.load_average_1m.toFixed(2) : "?")],
    ["timer floor", ns(R.timer_overhead_ns)],
    ["sampling", `${esc(R.sampling.min_samples)}–${esc(R.sampling.max_samples)} samples, ${esc(R.sampling.time_budget_s)} s budget, fast calls batched to ${(R.sampling.target_window_s * 1e3).toFixed(1)} ms windows, calls over ${esc(R.sampling.slow_call_s)} s sampled once`],
    ["process peak RSS", bytes(R.process_peak_rss_bytes)]])}</div>`].join("");

/* comparison */
(function () {
  const c = R.comparison; if (!c) return;
  let h = `<h2>Comparison with ${esc(c.baseline_run)}</h2>`;
  if (c.problems.length) h += c.problems.map(p => `<div class="warn"><b>⚠ Not comparable</b> ${esc(p)}</div>`).join("");
  if (c.same_library && c.same_harness) h += `<div class="warn"><b>Identical binaries</b> Both sides ran the same harness and libbeam.a; any difference here is measurement noise.</div>`;
  else if (c.same_library) h += `<div class="warn"><b>Same library</b> Same libbeam.a but a different harness build: differences come from the benchmark code or from code layout, not from BEAM.</div>`;
  if (c.kind !== "ab") h += `<div class="warn"><b>Untested differences</b> Each side is one process. The same binary's median moves between processes (up to about 2× for short multi-threaded kernels on this machine), so these are differences, not findings. Use <code>--save-binary</code> and <code>--ab</code> to decide.</div>`;
  else h += `<p class="small">${esc(c.rounds)} processes per side, alternating. A benchmark is faster or slower only when the exact Mann–Whitney test on the per-process medians gives p ≤ 0.05 and the change is at least 1%.</p>`;
  const rows = c.rows.filter(r => r.rel_change != null).sort((a, b) => a.rel_change - b.rel_change);
  h += `<div class="tablewrap"><table><thead><tr><th>Benchmark</th><th>Verdict</th><th class="num">Baseline</th><th class="num">This build</th><th class="num">Change</th><th>How decided</th></tr></thead><tbody>` +
    rows.map(r => { const tested = r.verdict === "faster" || r.verdict === "slower";
      const cls = tested ? (r.verdict === "slower" ? "up" : "down") : "";
      const icon = tested ? (r.verdict === "slower" ? "▲ " : "▼ ") : "";
      return `<tr><td>${esc(r.key)}</td><td class="${cls}">${icon}${esc(r.verdict)}</td><td class="num">${ns(r.base_median)}</td><td class="num">${ns(r.cur_median)}</td><td class="num ${cls}">${(100 * r.rel_change).toFixed(1)}%</td><td class="small">${esc(r.method)}</td></tr>`; }).join("") +
    `</tbody></table></div>`;
  const nc = c.rows.filter(r => r.verdict === "not comparable");
  if (nc.length) h += `<p class="small">Not comparable: ${nc.map(r => esc(r.key) + " (" + esc(r.method) + ")").join(", ")}.</p>`;
  document.getElementById("comparison").innerHTML = h;
})();

/* log-scale dot plot */
function dotPlot(el, items, valueOf, extentOf, fmt, tipOf, tickBase = 10, tickFmt = fmt) {
  if (!items.length) { el.innerHTML = `<div class="empty">Nothing to show.</div>`; return; }
  const rowH = 24, labelW = 260, right = 90, top = 22;
  const width = Math.max(560, el.clientWidth - 32);
  const plotW = width - labelW - right;
  const vals = items.flatMap(r => [valueOf(r), extentOf ? extentOf(r) : null]).filter(v => v > 0);
  const lg = v => Math.log(v) / Math.log(tickBase);
  let lo = Math.floor(lg(Math.min(...vals))), hi = Math.ceil(lg(Math.max(...vals)));
  if (hi === lo) hi = lo + 1;
  const x = v => labelW + (lg(v) - lo) / (hi - lo) * plotW;
  const h = top + items.length * rowH + 8;
  let s = `<svg width="${width}" height="${h}" role="img" aria-label="dot plot">`;
  for (let e = lo; e <= hi; e++) {
    const X = x(tickBase ** e);
    s += `<line x1="${X}" x2="${X}" y1="${top - 6}" y2="${h - 4}" stroke="var(--grid)" stroke-width="1"/>`;
    s += `<text x="${X}" y="${top - 10}" text-anchor="middle">${tickFmt(tickBase ** e)}</text>`;
  }
  items.forEach((r, i) => {
    const y = top + i * rowH + rowH / 2, v = valueOf(r), X = x(v);
    s += `<g data-tip="${esc(tipOf(r))}">`;
    s += `<rect x="0" y="${y - rowH / 2}" width="${width}" height="${rowH}" fill="transparent"/>`;
    const mark = (r.flags || []).includes("contended") ? " *" : "";
    s += `<text class="lbl" x="${labelW - 10}" y="${y + 4}" text-anchor="end">${esc(r.key.length > 36 ? r.key.slice(0, 35) + "…" : r.key)}${mark}</text>`;
    const ext = extentOf ? extentOf(r) : null;
    if (ext && ext > v) s += `<line x1="${X}" x2="${x(ext)}" y1="${y}" y2="${y}" stroke="var(--series)" stroke-opacity=".45" stroke-width="2" stroke-linecap="round"/>`;
    s += `<circle cx="${X}" cy="${y}" r="5" fill="var(--series)" stroke="var(--surface)" stroke-width="2"/>`;
    const lx = Math.max(X, ext && ext > v ? x(ext) : X) + 8;
    s += `<text class="val" x="${lx}" y="${y + 4}">${fmt(v)}</text>`;
    s += `</g>`;
  });
  el.innerHTML = s + `</svg>`;
}
const timed = R.results.filter(r => r.status === "ok" && r.stats);
const nsTick = v => { const u = [[1e9, " s"], [1e6, " ms"], [1e3, " µs"], [1, " ns"]].find(([k]) => v >= k) || [1, " ns"];
  return +(v / u[0]).toPrecision(3) + u[1]; };
const byteTick = v => { const u = [[2 ** 30, " GiB"], [2 ** 20, " MiB"], [2 ** 10, " KiB"], [1, " B"]].find(([k]) => v >= k) || [1, " B"];
  return +(v / u[0]).toPrecision(3) + u[1]; };
const tipFor = r => `<b>${esc(r.key)}</b><br>${esc(r.description)}<br>median ${ns(r.stats.median)} (n=${r.stats.n})` +
  (r.stats.median_ci ? `, 95% CI ${ns(r.stats.median_ci.lo)}–${ns(r.stats.median_ci.hi)}` : "") +
  `<br>first call ${ns(r.cold_ns)} · peak heap ${bytes(r.memory.peak_bytes)}` +
  ((r.flags || []).includes("contended") ? `<br>⚠ contended: other processes competed for cores; may be inflated` : "");
function drawCharts() {
  dotPlot(document.getElementById("chart-time"), [...timed].sort((a, b) => b.stats.median - a.stats.median).slice(0, 20),
    r => r.stats.median, r => (r.flags || []).includes("cold_at_floor") ? null : r.cold_ns, ns, tipFor, 10, nsTick);
  dotPlot(document.getElementById("chart-mem"), timed.filter(r => r.memory.peak_bytes > 0)
    .sort((a, b) => b.memory.peak_bytes - a.memory.peak_bytes).slice(0, 15),
    r => r.memory.peak_bytes, null, bytes,
    r => `<b>${esc(r.key)}</b><br>peak ${bytes(r.memory.peak_bytes)} over ${r.memory.alloc_calls} allocations<br>left allocated after the call: ${bytes(r.memory.retained_bytes)}`,
    32, byteTick);
}
drawCharts();
addEventListener("resize", () => { clearTimeout(window._rz); window._rz = setTimeout(drawCharts, 150); });

/* results table */
const tbody = document.querySelector("#results tbody");
const suites = [...new Set(R.results.map(r => r.domain))].sort();
document.getElementById("suite").innerHTML += suites.map(s => `<option>${esc(s)}</option>`).join("");
const statuses = [...new Set(R.results.map(statusOf))].sort();
document.getElementById("status").innerHTML += statuses.map(s => `<option>${esc(s)}</option>`).join("");
let sortKey = null, sortDir = 1;
const sortVal = (r, k) => ({ domain: r.domain, name: r.name, status: statusOf(r), median: r.stats ? r.stats.median : -1,
  ci: r.stats && r.stats.median_ci ? (r.stats.median_ci.hi - r.stats.median_ci.lo) / r.stats.median : -1,
  cold: r.cold_ns ?? -1, peak: (r.memory && r.memory.peak_bytes) ?? -1, tput: r.throughput ?? -1 })[k];
function detail(r) {
  const st = r.stats, m = r.memory || {};
  const src = r.source ? `<code>${esc(r.source.file)}:${esc(r.source.line)}</code>` : "–";
  const calls = (r.timed_calls || []).map(c => `<li><code>${esc(c.symbol)}</code>${c.definition ? ` — <code>${esc(c.definition)}</code>` : ""}</li>`).join("");
  const stats = st ? kv([["samples", `${st.n} × ${r.reps} call${r.reps > 1 ? "s" : ""} per window`],
    ["min / q1 / median", `${ns(st.min)} / ${ns(st.q1)} / ${ns(st.median)}`], ["q3 / max", `${ns(st.q3)} / ${ns(st.max)}`],
    ["mean ± sd", `${ns(st.mean)} ± ${ns(st.stddev)}`],
    ["95% CI of median", st.median_ci ? `${ns(st.median_ci.lo)} – ${ns(st.median_ci.hi)}` : "needs ≥ 6 samples"]]) : "";
  const mem = kv([["peak heap", bytes(m.peak_bytes)], ["left allocated", bytes(m.retained_bytes)], ["allocated in total", bytes(m.alloc_bytes)],
    ["allocations", m.alloc_calls ?? "–"], ["first call: peak / left", `${bytes(m.cold_peak_bytes)} / ${bytes(m.cold_retained_bytes)}`]]);
  const msg = r.error_message || r.skip_reason || r.crash_reason;
  return `<div class="detail-grid">
    <div><h4>What is timed</h4><p style="margin:0 0 6px">${esc(r.description)}</p><div class="small">benchmark ${src}${r.single_shot ? " · single-shot (setup re-run every call)" : ""}</div>${calls ? `<ul>${calls}</ul>` : ""}</div>
    <div><h4>${msg ? "Outcome" : "Observations"}</h4>${msg ? `<p style="margin:0">${esc(msg)}</p>` : ""}${(r.observations || []).length ? `<ul>${r.observations.map(o => `<li>${esc(o)}</li>`).join("")}</ul>` : (msg ? "" : `<span class="small">Nothing notable.</span>`)}</div>
    ${st ? `<div><h4>Time per call</h4>${stats}</div>` : ""}
    <div><h4>Memory</h4>${mem}</div></div>`;
}
function renderTable() {
  const q = document.getElementById("q").value.toLowerCase(), su = document.getElementById("suite").value, ss = document.getElementById("status").value;
  let rows = R.results.filter(r => (!q || r.key.toLowerCase().includes(q) || (r.description || "").toLowerCase().includes(q))
    && (!su || r.domain === su) && (!ss || statusOf(r) === ss));
  if (sortKey) rows.sort((a, b) => { const x = sortVal(a, sortKey), y = sortVal(b, sortKey); return (x > y ? 1 : x < y ? -1 : 0) * sortDir; });
  document.getElementById("count").textContent = `${rows.length} of ${R.results.length}`;
  tbody.innerHTML = rows.map((r, i) => {
    const st = r.stats, ci = st && st.median_ci ? `±${(50 * (st.median_ci.hi - st.median_ci.lo) / st.median).toFixed(1)}%` : (st ? `n=${st.n}` : "");
    const flags = (r.flags || []).filter(f => f !== "noisy" && f !== "contended").map(f => `<span class="flag">${esc(f.replace(/_/g, " "))}</span>`).join("");
    const note = st ? flags : `<span class="small">${esc((r.error_message || r.skip_reason || r.crash_reason || "").slice(0, 110))}</span>`;
    return `<tr class="main" data-i="${i}"><td>${esc(r.domain)}</td><td><b>${esc(r.name)}</b></td><td>${badge(statusOf(r))}</td>
      <td class="num">${st ? ns(st.median) : "–"}</td><td class="num">${ci}</td><td class="num">${ns(r.cold_ns)}</td>
      <td class="num">${bytes(r.memory && r.memory.peak_bytes)}</td><td class="num">${esc(r.throughput_human || "")}</td><td>${note}</td></tr>
      <tr class="detail" hidden><td colspan="9">${detail(r)}</td></tr>`;
  }).join("");
}
tbody.addEventListener("click", e => { const tr = e.target.closest("tr.main"); if (tr) tr.nextElementSibling.hidden = !tr.nextElementSibling.hidden; });
document.querySelectorAll("#results th[data-k]").forEach(th => th.addEventListener("click", () => {
  const k = th.dataset.k; sortDir = sortKey === k ? -sortDir : (["median", "ci", "cold", "peak", "tput"].includes(k) ? -1 : 1); sortKey = k; renderTable(); }));
["q", "suite", "status"].forEach(id => document.getElementById(id).addEventListener("input", renderTable));
renderTable();

/* scaling sweep */
function spark(points, xk) {
  const pts = points.filter(p => p.status === "ok" && p.median_ns > 0);
  if (pts.length < 2) return `<span class="small">${pts.length} valid point${pts.length === 1 ? "" : "s"}</span>`;
  const W = 120, H = 30, xs = pts.map(p => Math.log(p[xk])), ys = pts.map(p => Math.log(p.median_ns));
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys) || y0 + 1;
  const X = v => 4 + (v - x0) / ((x1 - x0) || 1) * (W - 8), Y = v => H - 4 - (v - y0) / ((y1 - y0) || 1) * (H - 8);
  const d = pts.map((p, i) => `${i ? "L" : "M"}${X(xs[i]).toFixed(1)},${Y(ys[i]).toFixed(1)}`).join("");
  const tipTxt = pts.map(p => `${xk}=${p[xk]}: ${ns(p.median_ns)}`).join("<br>");
  return `<svg width="${W}" height="${H}" data-tip="${esc(tipTxt)}"><path d="${d}" fill="none" stroke="var(--series)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>` +
    pts.map((p, i) => `<circle cx="${X(xs[i])}" cy="${Y(ys[i])}" r="3" fill="var(--series)"/>`).join("") + `</svg>`;
}
(function () {
  if (!SWEEP) return;
  const k = SWEEP.kernels || [];
  let h = `<h2>Scaling</h2><p class="small">Separate runs at each point; L is also the model's max_length, so whole-model analytics grow with it. Exponents are least-squares fits on log-log axes over valid points only; the order factor is the median time ratio per +1 order (for comparison, the radix is ${esc(SWEEP.radix)}).</p>`;
  const hasT = (SWEEP.threads || []).length > 1;
  h += `<div class="tablewrap"><table><thead><tr><th>Benchmark</th><th>Time vs L (K=${esc(SWEEP.base_order)})</th><th class="num">L exponent</th><th class="num">R²</th><th>Time vs K (L=${esc(SWEEP.base_length)})</th><th class="num">× per order</th>` +
    (hasT ? `<th>Time vs threads</th><th class="num">Speed-up</th><th class="num">Efficiency</th>` : "") + `<th>Notes</th></tr></thead><tbody>` +
    k.map(r => `<tr><td>${esc(r.key)}</td><td>${spark(r.L_points, "L")}</td><td class="num">${r.L_exponent != null ? r.L_exponent.toFixed(2) : "–"}</td><td class="num">${r.L_r2 != null ? r.L_r2.toFixed(3) : "–"}</td>
      <td>${spark(r.K_points, "K")}</td><td class="num">${r.K_factor != null ? r.K_factor.toFixed(1) + "×" : "–"}</td>` +
      (hasT ? `<td>${spark(r.T_points || [], "T")}</td><td class="num">${r.speedup != null ? r.speedup.toFixed(1) + "×" : "–"}</td><td class="num">${r.parallel_efficiency != null ? (100 * r.parallel_efficiency).toFixed(0) + "%" : "–"}</td>` : "") +
      `<td class="small">${esc((r.notes || []).join("; "))}</td></tr>`).join("") + `</tbody></table></div>`;
  document.getElementById("scaling").innerHTML = h;
})();

/* coverage */
(function () {
  const c = R.coverage; if (!c) return;
  const by = Object.entries(c.by_header).sort().map(([h, v]) => `<tr><td><code>${esc(h)}</code></td><td class="num">${v.covered} / ${v.total}</td></tr>`).join("");
  document.getElementById("coverage").innerHTML = `<p style="margin-top:0">${c.covered} of ${c.total} public functions are called inside some benchmark's timed block (${c.percent.toFixed(1)}%). Read from the sources at report time.</p>
    <div class="grid"><div><table><tbody>${by}</tbody></table></div>
    <div><h4 style="margin:0 0 6px">Not timed by any benchmark</h4>${c.uncovered.length ? `<ul>${c.uncovered.map(u => `<li><code>${esc(u.symbol)}</code> <span class="small">${esc(u.header)}</span></li>`).join("")}</ul>` : "<p>None.</p>"}</div></div>`;
})();
</script>
</body>
</html>
"""


def write(run: Dict[str, Any], path: Path, sweep: Optional[Dict[str, Any]] = None) -> None:
    payload = json.dumps({"run": run, "sweep": sweep}, default=str)
    payload = payload.replace("</", "<\\/")   # cannot close the <script> element
    path.write_text(TEMPLATE.replace("__DATA__", payload), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Write the HTML report for a run file")
    p.add_argument("run_json")
    p.add_argument("--output", default=None)
    p.add_argument("--sweep", default=None, help="scaling_sweep.json to include")
    a = p.parse_args()
    run = json.loads(Path(a.run_json).read_text())
    sweep = json.loads(Path(a.sweep).read_text()) if a.sweep else None
    out = Path(a.output) if a.output else Path(a.run_json).with_name(f"report_{run['run_id']}.html")
    write(run, out, sweep)
    print(f"wrote {out}")
