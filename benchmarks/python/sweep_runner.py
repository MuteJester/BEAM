#!/usr/bin/env python3
"""
Scaling sweep: the same benchmarks at several lengths and orders.

Each point is a separate harness run whose fixtures are built for exactly that
order and length (L is also the model's max_length, so whole-model analytics
grow with it). A point where a benchmark failed, was refused or crashed is
kept in the record with its status and left out of every fit -- the first
version of this sweep fitted error returns and reported them as O(1). So is
a point measured while other processes were using CPU ("contended").

  L exponent  least-squares slope of log(time) on log(L) over valid points,
              with R^2; needs 3 points
  K factor    exp(slope of log(time) on K): the time ratio per +1 order, to
              set beside the alphabet's radix (21 for aa, 6 for nt), which is
              the factor by which the state space grows
  speed-up    time at the fewest threads / time at the most (--sweep-threads),
              and parallel efficiency = speed-up / thread ratio. A kernel that
              is slow AND scales poorly is the first place to look.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import analysis
import runner


def _fit(xs: List[float], ys: List[float]) -> Optional[Tuple[float, float]]:
    """Slope and R^2 of y on x by least squares."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (my + slope * (x - mx))) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, r2


def _point(a: argparse.Namespace, order: int, length: int, tag: str,
           threads: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    args = runner.harness_args(a)
    # The sweep sets the workload axes itself.
    for flag in ("--order", "--length", "--max-length") + (("--threads",) if threads is not None else ()):
        if flag in args:
            i = args.index(flag)
            del args[i:i + 2]
    args += ["--order", str(order), "--length", str(length)]
    if threads is not None:
        args += ["--threads", str(threads)]
    jsonl = runner.RUNS / "raw" / f"sweep_{tag}.jsonl"
    parsed = runner.run_harness(args, jsonl, (a.exclude or "").split(",") if a.exclude else [], quiet=True)
    meta = parsed["meta"] or {}
    out = {}
    for key, raw in parsed["results"].items():
        r = analysis.interpret(raw, meta, parsed["fixtures"].get("0"))
        status = "contended" if "contended" in r.get("flags", []) else r["status"]
        out[key] = {"status": status, "median_ns": r["stats"]["median"] if r.get("stats") else None,
                    "peak_bytes": r["memory"].get("peak_bytes"),
                    "message": r.get("error_message") or r.get("skip_reason")}
    for key in parsed.get("crashed", []):
        out[key] = {"status": "crashed", "median_ns": None, "peak_bytes": None, "message": "crashed"}
    if parsed.get("fixture_errors"):
        print(f"   fixture failed at K={order} L={length}: {parsed['fixture_errors'][0].get('error')}")
    return out


def main(a: argparse.Namespace) -> int:
    alphabet = a.alphabet or "aa"
    lengths = [int(x) for x in a.sweep_lengths.split(",") if x.strip()]
    orders = [int(x) for x in a.sweep_orders.split(",") if x.strip()]
    default_k, default_l = (4, 27) if alphabet == "aa" else (6, 81)
    base_order = a.order if a.order else (default_k if default_k in orders else max(orders))
    base_length = a.length if a.length else (default_l if default_l in lengths else max(lengths))
    if not (a.suite or a.filter):
        print(">> note: sweeping every benchmark runs the whole suite once per point; "
              "--suite or --filter keeps it short")

    runner.RUNS.mkdir(parents=True, exist_ok=True)
    (runner.RUNS / "raw").mkdir(exist_ok=True)
    b, env = runner.build_info(), runner.environment()

    by_l: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for L in lengths:
        print(f">> K={base_order} L={L}")
        by_l[L] = _point(a, base_order, L, f"K{base_order}_L{L}")
    by_k: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for K in orders:
        if K == base_order and base_length in by_l:
            by_k[K] = by_l[base_length]
            continue
        print(f">> K={K} L={base_length}")
        by_k[K] = _point(a, K, base_length, f"K{K}_L{base_length}")

    threads = [int(x) for x in (a.sweep_threads or "").split(",") if x.strip()]
    by_t: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for T in threads:
        print(f">> K={base_order} L={base_length} threads={T}")
        by_t[T] = _point(a, base_order, base_length, f"K{base_order}_L{base_length}_T{T}", threads=T)

    keys = sorted({k for pts in list(by_l.values()) + list(by_k.values()) + list(by_t.values()) for k in pts})
    kernels = []
    for key in keys:
        lp = [{"L": L, **by_l[L].get(key, {"status": "absent", "median_ns": None})} for L in lengths]
        kp = [{"K": K, **by_k[K].get(key, {"status": "absent", "median_ns": None})} for K in orders]
        notes = [f"{p['status']} at L={p['L']}" for p in lp if p["status"] not in ("ok", "absent")]
        notes += [f"{p['status']} at K={p['K']}" for p in kp if p["status"] not in ("ok", "absent")]
        okl = [p for p in lp if p["status"] == "ok" and p["median_ns"]]
        okk = [p for p in kp if p["status"] == "ok" and p["median_ns"]]
        l_exp = l_r2 = k_fac = None
        if len(okl) >= 3:
            f = _fit([math.log(p["L"]) for p in okl], [math.log(p["median_ns"]) for p in okl])
            if f:
                l_exp, l_r2 = f
        elif okl:
            notes.append(f"{len(okl)} valid length point(s): no exponent fitted")
        if len(okk) >= 2:
            f = _fit([float(p["K"]) for p in okk], [math.log(p["median_ns"]) for p in okk])
            if f:
                k_fac = math.exp(f[0])
        tp = [{"T": T, **by_t[T].get(key, {"status": "absent", "median_ns": None})} for T in threads]
        okt = [p for p in tp if p["status"] == "ok" and p["median_ns"]]
        speedup = efficiency = None
        if len(okt) >= 2:
            lo_t, hi_t = min(okt, key=lambda p: p["T"]), max(okt, key=lambda p: p["T"])
            if hi_t["T"] > lo_t["T"]:
                speedup = lo_t["median_ns"] / hi_t["median_ns"]
                efficiency = speedup / (hi_t["T"] / lo_t["T"])
        notes += [f"{p['status']} at threads={p['T']}" for p in tp if p["status"] not in ("ok", "absent")]
        d, n = key.split("::", 1)
        kernels.append({"key": key, "domain": d, "name": n, "L_points": lp, "K_points": kp, "T_points": tp,
                        "L_exponent": l_exp, "L_r2": l_r2, "K_factor": k_fac,
                        "speedup": speedup, "parallel_efficiency": efficiency, "notes": notes})

    sweep = {
        "schema": analysis.SCHEMA,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "alphabet": alphabet, "radix": 21 if alphabet == "aa" else 6,
        "lengths": lengths, "orders": orders, "threads": threads,
        "base_order": base_order, "base_length": base_length,
        "build": b, "environment": env, "kernels": kernels,
    }
    out = runner.RUNS / "scaling_sweep.json"
    out.write_text(json.dumps(sweep, indent=1))
    print(f"{'benchmark':<44} {'L exp':>7} {'R2':>6} {'x/order':>8} {'speed-up':>9} {'effic.':>7}  notes")
    for k in kernels:
        le = f"{k['L_exponent']:.2f}" if k["L_exponent"] is not None else "-"
        r2 = f"{k['L_r2']:.3f}" if k["L_r2"] is not None else "-"
        kf = f"{k['K_factor']:.1f}" if k["K_factor"] is not None else "-"
        su = f"{k['speedup']:.1f}x" if k["speedup"] is not None else "-"
        ef = f"{100 * k['parallel_efficiency']:.0f}%" if k["parallel_efficiency"] is not None else "-"
        print(f"{k['key']:<44} {le:>7} {r2:>6} {kf:>8} {su:>9} {ef:>7}  {'; '.join(k['notes'])}")
    print(f">> Sweep saved to {out}; the next run's HTML report includes it if the library is unchanged")
    return 0
