"""
Statistics, status, observations, rankings and A/B comparison for a
benchmark run. Everything the reports show is computed here, from what the
harness measured; nothing is looked up from a table of expectations.

Conventions
-----------
- Every time is nanoseconds PER CALL of a benchmark's timed body.
- A result's `status` is the harness's (ok / error / skipped), `refused`
  when BEAM declined the workload under its memory budget (BEAM_ERR_BUDGET:
  a real answer about the workload, not a harness fault), or `crashed`.
  Only `ok` results carry times that may be ranked or compared.
- `flags` qualify an ok result: few_samples, single_sample, noisy,
  timer_floor, cold_at_floor, retains_memory, contended.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA = 2

# A result is NOISY when its interquartile range exceeds this share of the median.
NOISY_RIQR = 0.10
# A timing window shorter than this many timer overheads is mostly clock.
FLOOR_FACTOR = 50.0
# A result is CONTENDED when other processes used more than CONTENDED_CPUS on
# average while it ran AND this process's CPUs plus theirs exceeded the
# PHYSICAL cores, so threads shared execution units. Physical, not logical:
# with 7 busy processes on an 8-core/16-thread machine, single-threaded RNG
# benchmarks slowed 20-25% through their hyperthread siblings. A desktop idles
# at 0.3-2 CPUs, which leaves single-threaded work alone but does reach a
# kernel running a thread on every CPU (layer_weights 135 us -> 6 ms under
# heavy load). Contended times stay in rankings, marked: they are inflated by
# a factor, not orders of magnitude.
CONTENDED_CPUS = 0.5


def physical_cores() -> Optional[int]:
    """Distinct (package, core) pairs in sysfs; None if unreadable."""
    try:
        cores = set()
        for d in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/topology"):
            cores.add(((d / "physical_package_id").read_text().strip(), (d / "core_id").read_text().strip()))
        return len(cores) or None
    except Exception:
        return None
# BeamError value of BEAM_ERR_BUDGET (include/beam/errors.h).
BEAM_ERR_BUDGET = 23


# ── Reading the harness output ─────────────────────────────────────────

def load_jsonl(path: Path) -> Dict[str, Any]:
    """Parse harness JSON lines. Several processes may have appended to one
    file (the runner restarts the harness after a crash); the first meta,
    corpus and successful fixture records win, and so does the last result
    per benchmark. A benchmark that began but has no result crashed."""
    out: Dict[str, Any] = {"meta": None, "corpus": None, "fixtures": {}, "fixture_errors": [],
                           "results": {}, "begun": [], "ends": [], "processes": 0}
    open_bench: Optional[str] = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut short by a crash
            t = rec.get("type")
            if t == "meta":
                out["processes"] += 1
                if out["meta"] is None:
                    out["meta"] = rec
                if open_bench:
                    out["begun"].append(open_bench)
                open_bench = None
            elif t == "corpus" and out["corpus"] is None:
                out["corpus"] = rec
            elif t == "fixture":
                if "error" in rec:
                    out["fixture_errors"].append(rec)
                else:
                    out["fixtures"].setdefault(str(rec.get("variant")), rec)
            elif t == "begin":
                open_bench = f"{rec['domain']}::{rec['name']}"
            elif t == "result":
                key = f"{rec['domain']}::{rec['name']}"
                out["results"][key] = rec
                open_bench = None
            elif t == "end":
                out["ends"].append(rec)
    if open_bench:
        out["begun"].append(open_bench)
    # A benchmark that later produced a result (after a restart) did not crash.
    out["crashed"] = [k for k in out["begun"] if k not in out["results"]]
    return out


# ── Statistics ────────────────────────────────────────────────────────

def _quantile(sorted_xs: List[float], q: float) -> float:
    """Linear interpolation between order statistics (type 7)."""
    n = len(sorted_xs)
    if n == 1:
        return sorted_xs[0]
    h = (n - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, n - 1)
    return sorted_xs[lo] + (h - lo) * (sorted_xs[hi] - sorted_xs[lo])


def _binom_cdf(k: int, n: int) -> float:
    """P(X <= k) for X ~ Binomial(n, 1/2)."""
    if k < 0:
        return 0.0
    return sum(math.comb(n, i) for i in range(0, min(k, n) + 1)) / (2.0 ** n)


def median_ci(sorted_xs: List[float], level: float = 0.95) -> Optional[Tuple[float, float, float]]:
    """Distribution-free confidence interval for the median from order
    statistics: [x_(k), x_(n-k+1)] with the largest k whose two tails each
    hold at most (1 - level) / 2. Needs n >= 6 at 95%. Returns (lo, hi,
    actual coverage) or None."""
    n = len(sorted_xs)
    alpha = 1.0 - level
    k = 0
    while k + 1 <= n // 2 and _binom_cdf(k, n) <= alpha / 2.0:
        k += 1
    if k == 0:
        return None
    coverage = 1.0 - 2.0 * _binom_cdf(k - 1, n)
    return sorted_xs[k - 1], sorted_xs[n - k], coverage


def summarize(samples: List[float]) -> Optional[Dict[str, Any]]:
    if not samples:
        return None
    xs = sorted(samples)
    n = len(xs)
    mean = sum(xs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    med = _quantile(xs, 0.5)
    q1, q3 = _quantile(xs, 0.25), _quantile(xs, 0.75)
    ci = median_ci(xs)
    return {
        "n": n,
        "min": xs[0], "q1": q1, "median": med, "q3": q3, "max": xs[-1],
        "mean": mean, "stddev": sd,
        "cv": sd / mean if mean > 0 else 0.0,
        "riqr": (q3 - q1) / med if med > 0 else 0.0,
        "rrange": (xs[-1] - xs[0]) / med if med > 0 else 0.0,
        "median_ci": {"lo": ci[0], "hi": ci[1], "coverage": ci[2]} if ci else None,
    }


# ── Per-result interpretation ─────────────────────────────────────────

def fmt_ns(ns: Optional[float]) -> str:
    if ns is None:
        return "-"
    if ns < 1e3:
        return f"{ns:.1f} ns"
    if ns < 1e6:
        return f"{ns / 1e3:.2f} µs"
    if ns < 1e9:
        return f"{ns / 1e6:.2f} ms"
    return f"{ns / 1e9:.2f} s"


def fmt_bytes(b: Optional[float]) -> str:
    if b is None:
        return "-"
    a = abs(b)
    sign = "-" if b < 0 else ""
    for unit, scale in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if a >= scale:
            return f"{sign}{a / scale:.1f} {unit}" if unit != "GiB" else f"{sign}{a / scale:.2f} {unit}"
    return f"{sign}{a:.0f} B"


def fmt_rate(per_s: float, unit: str) -> str:
    for suffix, scale in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if per_s >= scale:
            return f"{per_s / scale:.2f} {suffix}{unit}/s"
    return f"{per_s:.1f} {unit}/s"


def interpret(raw: Dict[str, Any], meta: Dict[str, Any], fixture: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Turn one harness result record into the report's view of it."""
    overhead = float(meta.get("timer_overhead_ns") or 0.0)
    status = raw.get("status", "error")
    if status == "error" and raw.get("error_code") == BEAM_ERR_BUDGET:
        status = "refused"
    stats = summarize([float(x) for x in raw.get("samples_ns", [])]) if status == "ok" else None
    reps = int(raw.get("reps") or 1)
    flags: List[str] = []
    obs: List[str] = []
    r: Dict[str, Any] = {
        "key": f"{raw['domain']}::{raw['name']}",
        "domain": raw["domain"], "name": raw["name"],
        "description": raw.get("description", ""),
        "status": status,
        "single_shot": bool(raw.get("single_shot")),
        "reps": reps,
        "warmup_calls": raw.get("warmup_calls", 0),
        "stats": stats,
        "cold_ns": raw.get("cold_ns") if status == "ok" else None,
        "memory": {
            "peak_bytes": raw.get("peak_bytes"),
            "retained_bytes": raw.get("retained_bytes"),
            "alloc_bytes": raw.get("alloc_bytes"),
            "alloc_calls": raw.get("alloc_calls"),
            "cold_peak_bytes": raw.get("cold_peak_bytes"),
            "cold_retained_bytes": raw.get("cold_retained_bytes"),
        },
        "items_per_op": raw.get("items_per_op"),
        "item_unit": raw.get("item_unit"),
        "wall_ns": raw.get("wall_ns"),
    }
    if status in ("error", "refused"):
        r["error_code"] = raw.get("error_code")
        r["error_message"] = raw.get("error_message", "")
    if status == "skipped":
        r["skip_reason"] = raw.get("skip_reason", "")
    if not stats:
        r["flags"], r["observations"] = flags, obs
        return r

    med = stats["median"]
    foreign, mine = raw.get("foreign_cpus"), raw.get("self_cpus")
    cores = meta.get("physical_cores") or physical_cores() or meta.get("logical_cpus") or 0
    r["foreign_cpus"], r["self_cpus"] = foreign, mine
    if foreign is not None and foreign > CONTENDED_CPUS:
        if mine is not None and cores and mine + foreign > cores:
            flags.append("contended")
            obs.append(f"Contended: it used {mine:.1f} CPUs and other processes {foreign:.1f}, more than the "
                       f"{cores} physical cores, so threads shared cores. Expect the time to be inflated "
                       f"(up to ~2x for short multi-threaded kernels); fine for ranking, not for A/B.")
        else:
            obs.append(f"Other processes used {foreign:.1f} CPUs while this ran (it used {mine:.1f}); "
                       f"together within the {cores} physical cores.")
    if stats["n"] == 1:
        flags.append("single_sample")
        obs.append("One sample (the call is slower than --slow-call): no dispersion estimate.")
    elif stats["n"] < 5:
        flags.append("few_samples")
    if stats["n"] >= 3 and stats["riqr"] > NOISY_RIQR:
        flags.append("noisy")
        obs.append(f"Spread is high: interquartile range is {100 * stats['riqr']:.0f}% of the median.")
    if overhead > 0 and reps == 1 and med < FLOOR_FACTOR * overhead:
        flags.append("timer_floor")
        obs.append(f"Single-shot call of {fmt_ns(med)} is within {FLOOR_FACTOR:.0f}x the {fmt_ns(overhead)} "
                   f"timer overhead; treat it as an upper bound.")

    cold = raw.get("cold_ns") or 0
    if overhead > 0 and cold < FLOOR_FACTOR * overhead:
        flags.append("cold_at_floor")
    elif med > 0 and cold > 1.5 * med:
        cr = raw.get("cold_retained_bytes") or 0
        tail = f"; it left {fmt_bytes(cr)} allocated (caches built on first use)" if cr > 0 else ""
        obs.append(f"First call after dropping caches takes {fmt_ns(cold)}, {cold / med:.1f}x a warm call{tail}.")

    peak = raw.get("peak_bytes")
    calls = raw.get("alloc_calls")
    if peak:
        msg = f"Peak heap {fmt_bytes(peak)} per call over {calls} allocations"
        mem_tables = (fixture or {}).get("memory_bytes")
        if mem_tables and peak > mem_tables:
            msg += f" ({peak / mem_tables:.1f}x the model's {fmt_bytes(mem_tables)} of tables)"
        obs.append(msg + ".")
    retained = raw.get("retained_bytes") or 0
    if retained > 0:
        flags.append("retains_memory")
        obs.append(f"A warm call leaves {fmt_bytes(retained)} allocated after it returns.")
    if calls and calls >= 1000:
        obs.append(f"{calls:,} allocations per call.")
    ipo = raw.get("items_per_op")
    if ipo and med > 0:
        r["throughput"] = ipo / (med * 1e-9)
        r["throughput_human"] = fmt_rate(r["throughput"], raw.get("item_unit") or "item")
    r["flags"], r["observations"] = flags, obs
    return r


# ── Run-level views ───────────────────────────────────────────────────

def timed(results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in results if r["status"] == "ok" and r.get("stats")]


def rank_by_time(results: List[Dict[str, Any]], n: int = 10) -> List[Dict[str, Any]]:
    """Contended results stay in, marked by their flag: inflated by a factor at
    most, while the ranking spans orders of magnitude."""
    return sorted(timed(results), key=lambda r: -r["stats"]["median"])[:n]


def rank_by_memory(results: List[Dict[str, Any]], n: int = 10) -> List[Dict[str, Any]]:
    rs = [r for r in timed(results) if r["memory"].get("peak_bytes")]
    return sorted(rs, key=lambda r: -r["memory"]["peak_bytes"])[:n]


def rank_by_cold_penalty(results: List[Dict[str, Any]], n: int = 10) -> List[Dict[str, Any]]:
    rs = [r for r in timed(results)
          if "cold_at_floor" not in r["flags"] and r["cold_ns"] and r["stats"]["median"] > 0]
    rs = [r for r in rs if r["cold_ns"] > 1.5 * r["stats"]["median"]]
    return sorted(rs, key=lambda r: -(r["cold_ns"] - r["stats"]["median"]))[:n]


def counts_by_status(results: List[Dict[str, Any]]) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for r in results:
        c[r["status"]] = c.get(r["status"], 0) + 1
    c["noisy"] = sum(1 for r in results if "noisy" in r.get("flags", []))
    c["contended"] = sum(1 for r in results if "contended" in r.get("flags", []))
    return c


# ── Comparisons ───────────────────────────────────────────────────────
#
# Within one harness process the samples of a benchmark are close together,
# but the same binary's median moves between PROCESSES: on an 8-core/16-thread
# machine under the powersave governor, an A/A test measured up to 1.8x for a
# 3 ms parallel kernel and 2.8x for a 20 ms one at 16 threads, and about 2% at
# one thread. A within-run confidence interval says nothing about that, so a
# verdict needs replicated processes: --ab interleaves the two binaries and
# applies an exact rank test to the per-process medians. --compare of two
# single runs reports differences only.

# Workload fields that must agree for times to be comparable at all.
_WORKLOAD_KEYS = ("alphabet", "order", "max_length", "length", "corpus_source", "corpus_size", "seed",
                  "threads_effective")
# A change smaller than this is never called a change.
MIN_EFFECT = 0.01


def comparability(cur: Dict[str, Any], base: Dict[str, Any]) -> List[str]:
    """Reasons the two runs' times should not be compared; empty if none."""
    problems = []
    if base.get("schema") != SCHEMA:
        return [f"baseline has schema {base.get('schema', 1)}, this report needs {SCHEMA}; "
                f"re-run the baseline with the current harness"]
    cw, bw = cur.get("workload", {}), base.get("workload", {})
    for k in _WORKLOAD_KEYS:
        if cw.get(k) != bw.get(k):
            problems.append(f"workload {k}: {bw.get(k)!r} (baseline) vs {cw.get(k)!r}")
    ce, be = cur.get("environment", {}), base.get("environment", {})
    for k in ("cpu_model", "logical_cpus", "governor"):
        if ce.get(k) != be.get(k):
            problems.append(f"{k}: {be.get(k)!r} (baseline) vs {ce.get(k)!r}")
    return problems


def compare(cur: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    """Differences between two single runs, WITHOUT verdicts: one process per
    side cannot separate a change from between-process variation."""
    problems = comparability(cur, base)
    same_lib = (cur.get("build", {}).get("libbeam_sha256")
                and cur["build"]["libbeam_sha256"] == base.get("build", {}).get("libbeam_sha256"))
    bmap = {r["key"]: r for r in base.get("results", [])}
    rows = []
    for r in cur.get("results", []):
        b = bmap.get(r["key"])
        if not b:
            continue
        row = {"key": r["key"], "domain": r["domain"], "name": r["name"]}
        if r["status"] != "ok" or b["status"] != "ok" or not r.get("stats") or not b.get("stats"):
            row.update(verdict="not comparable", method=f"{b['status']} -> {r['status']}")
        else:
            cs, bs = r["stats"], b["stats"]
            row.update(cur_median=cs["median"], base_median=bs["median"],
                       rel_change=cs["median"] / bs["median"] - 1.0 if bs["median"] > 0 else 0.0,
                       verdict="difference", method="single runs: not tested (use --ab)")
        rows.append(row)
    same_harness = (cur.get("build", {}).get("beam_bench_sha256")
                    and cur["build"]["beam_bench_sha256"] == base.get("build", {}).get("beam_bench_sha256"))
    return {"kind": "single", "problems": problems, "same_library": bool(same_lib),
            "same_harness": bool(same_harness), "rows": rows,
            "baseline_run": base.get("run_id"), "baseline_commit": base.get("build", {}).get("git_commit")}


def mann_whitney_exact(a: List[float], b: List[float]) -> float:
    """Two-sided exact permutation p-value of the Mann-Whitney U statistic
    (ties count one half), enumerating every split of the pooled values."""
    from itertools import combinations
    pooled = list(a) + list(b)
    n, na = len(pooled), len(a)

    def u_of(idx_a: Tuple[int, ...]) -> float:
        sa = set(idx_a)
        xa = [pooled[i] for i in idx_a]
        xb = [pooled[i] for i in range(n) if i not in sa]
        return sum(1.0 if x < y else 0.5 if x == y else 0.0 for x in xa for y in xb)

    centre = na * (n - na) / 2.0
    observed = abs(u_of(tuple(range(na))) - centre)
    total = extreme = 0
    for idx in combinations(range(n), na):
        total += 1
        if abs(u_of(idx) - centre) >= observed - 1e-12:
            extreme += 1
    return extreme / total


def ab_compare(base_runs: List[Dict[str, Dict[str, Any]]], cur_runs: List[Dict[str, Dict[str, Any]]],
               alpha: float = 0.05) -> List[Dict[str, Any]]:
    """Per benchmark: the per-process medians of each side, their spread, the
    exact rank test, and a verdict. Each argument is one dict per process,
    mapping benchmark key to an interpreted result."""
    keys = sorted({k for run in base_runs + cur_runs for k in run})
    rows = []
    for k in keys:
        b = [run[k] for run in base_runs if k in run]
        c = [run[k] for run in cur_runs if k in run]
        bm = [r["stats"]["median"] for r in b if r["status"] == "ok" and r.get("stats")]
        cm = [r["stats"]["median"] for r in c if r["status"] == "ok" and r.get("stats")]
        d, n = k.split("::", 1)
        row: Dict[str, Any] = {"key": k, "domain": d, "name": n, "base_medians": bm, "cur_medians": cm}
        if len(bm) < 2 or len(cm) < 2 or len(bm) != len(b) or len(cm) != len(c):
            sb = sorted({r["status"] for r in b}); sc = sorted({r["status"] for r in c})
            row.update(verdict="not comparable", method=f"statuses {'/'.join(sb)} -> {'/'.join(sc)}")
            rows.append(row)
            continue
        mb, mc = _quantile(sorted(bm), 0.5), _quantile(sorted(cm), 0.5)
        rel = mc / mb - 1.0 if mb > 0 else 0.0
        p = mann_whitney_exact(bm, cm)
        spread = max((max(x) - min(x)) / _quantile(sorted(x), 0.5) for x in (bm, cm))
        if p <= alpha and abs(rel) >= MIN_EFFECT:
            verdict = "faster" if rel < 0 else "slower"
        else:
            verdict = "no detectable change"
        row.update(base_median=mb, cur_median=mc, rel_change=rel, p_value=p, process_spread=spread,
                   verdict=verdict,
                   method=f"exact Mann-Whitney p={p:.3f} over {len(bm)}+{len(cm)} processes; "
                          f"between-process spread {spread:.0%}")
        rows.append(row)
    return rows
