#!/usr/bin/env python3
"""
Everything known about one C function or one benchmark:

    ./benchmarks/run --inspect beam_model_entropy
    ./benchmarks/run --inspect math::entropy
    ./benchmarks/run --inspect entropy            (every match)

Locations are read from the sources; measurements come from the latest run
file (benchmarks/runs/latest.json). Nothing is inferred beyond those.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_sources  # noqa: E402
from analysis import fmt_bytes, fmt_ns  # noqa: E402
from function_catalog import FUNCTION_CATALOG  # noqa: E402

LATEST = bench_sources.REPO_ROOT / "benchmarks" / "runs" / "latest.json"


def _latest() -> Optional[Dict[str, Any]]:
    try:
        run = json.loads(LATEST.read_text())
        return run if run.get("schema") == 2 else None
    except Exception:
        return None


def _result_lines(r: Dict[str, Any]) -> List[str]:
    out = [f"    status: {r['status']}" + (f"  ({', '.join(r['flags'])})" if r.get("flags") else "")]
    if r.get("error_message"):
        out.append(f"    message: {r['error_message']}")
    st = r.get("stats")
    if st:
        ci = st.get("median_ci")
        out.append(f"    median per call: {fmt_ns(st['median'])}  (n={st['n']}, reps={r['reps']}"
                   + (f", 95% CI {fmt_ns(ci['lo'])}-{fmt_ns(ci['hi'])}" if ci else "") + ")")
        out.append(f"    min / max: {fmt_ns(st['min'])} / {fmt_ns(st['max'])}   first call: {fmt_ns(r.get('cold_ns'))}")
        m = r.get("memory", {})
        out.append(f"    peak heap: {fmt_bytes(m.get('peak_bytes'))} over {m.get('alloc_calls')} allocations; "
                   f"left allocated: {fmt_bytes(m.get('retained_bytes'))}")
        if r.get("throughput_human"):
            out.append(f"    throughput: {r['throughput_human']}")
    for o in r.get("observations", []):
        out.append(f"    - {o}")
    return out


def _bench_block(b: bench_sources.BenchSource, run: Optional[Dict[str, Any]], defs: Dict[str, Any]) -> List[str]:
    lines = [f"benchmark {b.key}", f"  source: {b.file}:{b.line}"]
    if b.timed_calls:
        lines.append("  times:")
        for c in b.timed_calls:
            d = defs.get(c)
            lines.append(f"    {c}" + (f"  ({d[0]}:{d[1]})" if d else ""))
    if run:
        r = next((x for x in run["results"] if x["key"] == b.key), None)
        if r:
            lines.append(f"  latest run {run['run_id']} ({run['workload']['alphabet']} K={run['workload']['order']} "
                         f"L={run['workload']['length']}):")
            lines.extend(_result_lines(r))
        else:
            lines.append("  not in the latest run")
    lines.append(f"  reproduce: ./benchmarks/run --filter {b.key}")
    lines.append(f"  profile:   perf record -g -- ./build/benchmarks/beam_bench --filter {b.key} --samples 5"
                 "   (needs perf_event_paranoid <= 2)")
    lines.append(f"  heap:      valgrind --tool=massif ./build/benchmarks/beam_bench --filter {b.key} --samples 1")
    return lines


def main(argv: List[str]) -> int:
    if not argv:
        print("usage: --inspect SYMBOL_OR_BENCHMARK")
        return 2
    q = argv[0]
    benches = bench_sources.benchmarks()
    defs = bench_sources.definitions()
    pub = bench_sources.public_symbols()
    run = _latest()
    if run is None:
        print("(no schema-2 run in benchmarks/runs/latest.json yet: showing locations only)\n")

    symbols = [s for s in sorted(set(defs) | set(pub)) if s == q] or \
              [s for s in sorted(set(defs) | set(pub)) if q in s and q.startswith("beam_")]
    matches = [b for k, b in benches.items() if k == q] or \
              [b for k, b in benches.items() if q in k and not q.startswith("beam_")]
    if not symbols and not matches:
        print(f"nothing matches {q!r}")
        return 1

    for s in symbols:
        cat = FUNCTION_CATALOG.get(s, {})
        d = defs.get(s)
        print(f"function {s}")
        if cat.get("signature"):
            print(f"  {cat['signature']}")
        print(f"  declared: {pub.get(s, '(not in a public header)')}")
        print(f"  defined:  {d[0]}:{d[1]}" if d else "  defined:  (not found)")
        if cat.get("purpose"):
            print(f"  purpose:  {cat['purpose']}")
        if cat.get("time_complexity"):
            print(f"  expected complexity (catalog, unverified): time {cat['time_complexity']}, "
                  f"space {cat.get('space_complexity', '?')}")
        timed_by = [b for b in benches.values() if s in b.timed_calls]
        print(f"  timed by: {', '.join(b.key for b in timed_by) or 'NO BENCHMARK'}")
        print()
        for b in timed_by:
            print("\n".join(_bench_block(b, run, defs)))
            print()
    for b in matches:
        if any(b in [x for x in benches.values() if s in x.timed_calls] for s in symbols):
            continue
        print("\n".join(_bench_block(b, run, defs)))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
