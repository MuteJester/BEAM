#!/usr/bin/env python3
"""Terminal summary of a run file: provenance, warnings, every result, rankings."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import analysis  # noqa: E402
from analysis import fmt_bytes, fmt_ns  # noqa: E402

try:
    from rich import box
    from rich.console import Console
    from rich.table import Table
    HAVE_RICH = True
except ImportError:  # plain text fallback
    HAVE_RICH = False

STATUS_STYLE = {"ok": "green", "noisy": "yellow", "contended": "yellow", "error": "bold red", "refused": "magenta",
                "skipped": "dim", "crashed": "bold red"}


def _status(r: Dict[str, Any]) -> str:
    if r["status"] == "ok" and "contended" in r.get("flags", []):
        return "contended"
    if r["status"] == "ok" and "noisy" in r.get("flags", []):
        return "noisy"
    return r["status"]


def _row(r: Dict[str, Any]) -> List[str]:
    st = r.get("stats")
    if not st:
        msg = r.get("error_message") or r.get("skip_reason") or r.get("crash_reason") or ""
        return [r["domain"], r["name"], _status(r).upper(), "", "", "", "", msg[:90]]
    ci = st.get("median_ci")
    spread = f"±{100 * (ci['hi'] - ci['lo']) / 2 / st['median']:.1f}%" if ci else f"n={st['n']}"
    notes = ", ".join(f for f in r.get("flags", []) if f not in ("noisy", "contended"))
    if r.get("throughput_human"):
        notes = (r["throughput_human"] + ("; " + notes if notes else ""))
    return [r["domain"], r["name"], _status(r).upper(), fmt_ns(st["median"]), spread,
            fmt_ns(r.get("cold_ns")), fmt_bytes(r["memory"].get("peak_bytes")), notes]


def header_lines(run: Dict[str, Any]) -> List[str]:
    w, b, e = run["workload"], run["build"], run["environment"]
    commit = (b.get("git_commit") or "?")[:10]
    lines = [
        f"Run {run['run_id']}  |  commit {commit}{' (dirty)' if b.get('git_dirty') else ''}  |  "
        f"libbeam {(b.get('libbeam_sha256') or '?')[:12]}",
        f"Workload: {w['alphabet']} K={w['order']} max_length={w['max_length']} L={w['length']}  |  corpus "
        f"{w['corpus_source']} ({w.get('corpus_sequences')} seqs, mean length {w.get('corpus_mean_length') or 0:.1f})"
        f"  |  threads {w['threads_effective']} effective",
        f"Machine: {e.get('cpu_model')}  |  {e.get('logical_cpus')} logical CPUs  |  governor {e.get('governor')}"
        f"  |  timer floor {fmt_ns(run.get('timer_overhead_ns'))}",
    ]
    for v, f in sorted(run.get("fixtures", {}).items()):
        lines.append(f"Model {v}: {f['n_states']:,} states, {f['n_nonzero_cells']:,} non-zero cells, "
                     f"{fmt_bytes(f['memory_bytes'])} tables; built in {fmt_ns(f['observe_ns'])} observe + "
                     f"{fmt_ns(f['finalize_ns'])} finalize")
    cov = run.get("coverage", {})
    if cov:
        lines.append(f"Coverage: {cov['covered']}/{cov['total']} public functions timed ({cov['percent']:.1f}%)")
    return lines


def render(run: Dict[str, Any]) -> None:
    results = run["results"]
    counts = run.get("summary", {}).get("counts") or analysis.counts_by_status(results)
    if not HAVE_RICH:
        print("\n".join(header_lines(run)))
        for wmsg in run.get("warnings", []):
            print("WARNING: " + wmsg)
        for r in results:
            print("  ".join(_row(r)))
        return

    con = Console()
    if not con.is_terminal:
        con = Console(width=170)   # piped or captured: do not squeeze the table to 80 columns
    con.print()
    for ln in header_lines(run):
        con.print(ln, highlight=False)
    for wmsg in run.get("warnings", []):
        con.print(f"[yellow]WARNING[/] {wmsg}", highlight=False)

    t = Table(box=box.SIMPLE_HEAD, header_style="bold", pad_edge=False)
    for col, just in (("suite", "left"), ("benchmark", "left"), ("status", "left"), ("median/call", "right"),
                      ("95% CI", "right"), ("cold call", "right"), ("peak heap", "right"), ("notes", "left")):
        t.add_column(col, justify=just, no_wrap=col != "notes")
    for r in results:
        row = _row(r)
        row[2] = f"[{STATUS_STYLE.get(_status(r), 'white')}]{row[2]}[/]"
        t.add_row(*row)
    con.print(t)

    slow = analysis.rank_by_time(results, 8)
    if slow:
        con.print("[bold]Slowest per call at this workload[/]")
        for i, r in enumerate(slow, 1):
            mark = "*" if "contended" in r.get("flags", []) else " "
            con.print(f"  {i:>2}. {r['key']:<42} {fmt_ns(r['stats']['median']):>10}{mark}  {r['description']}",
                      highlight=False)
        if any("contended" in r.get("flags", []) for r in slow):
            con.print("      * contended: may be inflated (other processes competed for cores)")
    heavy = analysis.rank_by_memory(results, 5)
    if heavy:
        con.print("[bold]Largest peak heap per call[/]")
        for i, r in enumerate(heavy, 1):
            con.print(f"  {i}. {r['key']:<42} {fmt_bytes(r['memory']['peak_bytes']):>10}", highlight=False)

    summary = "  ".join(f"{k} {v}" for k, v in counts.items() if v)
    con.print(f"[bold]Results:[/] {summary}")
    comp = run.get("comparison")
    if comp:
        render_comparison(con, comp)


def render_comparison(con: "Console", comp: Dict[str, Any]) -> None:
    con.print(f"\n[bold]Comparison with {comp.get('baseline_run')}[/]")
    for p in comp["problems"]:
        con.print(f"[bold red]NOT COMPARABLE[/] {p}", highlight=False)
    if comp.get("same_library") and comp.get("same_harness"):
        con.print("[yellow]Both sides ran the identical binary: any difference is measurement noise.[/]")
    elif comp.get("same_library"):
        con.print("[yellow]Same libbeam.a, different harness build: differences come from the benchmark code or "
                  "from code layout, not from BEAM.[/]")
    rows = [r for r in comp["rows"] if r.get("rel_change") is not None]
    if comp.get("kind") == "single":
        con.print("[yellow]Single runs: differences below are NOT tested. The same binary's median moves between "
                  "processes (up to ~2x for short multi-threaded kernels), so decide with --ab.[/]")
        for r in sorted(rows, key=lambda r: r["rel_change"]):
            con.print(f"  {r['key']:<44} {100 * r['rel_change']:+7.1f}%  "
                      f"{fmt_ns(r['base_median'])} -> {fmt_ns(r['cur_median'])}", highlight=False)
    else:
        changed = [r for r in rows if r["verdict"] in ("faster", "slower")]
        for r in sorted(changed, key=lambda r: r["rel_change"]):
            colour, icon = ("green", "▼") if r["verdict"] == "faster" else ("red", "▲")
            con.print(f"  [{colour}]{icon} {r['verdict']:<7}[/] {r['key']:<44} {100 * r['rel_change']:+7.1f}%  "
                      f"{fmt_ns(r['base_median'])} -> {fmt_ns(r['cur_median'])}   ({r['method']})", highlight=False)
        quiet = [r for r in rows if r["verdict"] == "no detectable change"]
        con.print(f"  {len(changed)} changed, {len(quiet)} with no detectable change "
                  f"(over {comp.get('rounds')} processes per side), "
                  f"{sum(1 for r in comp['rows'] if r.get('verdict') == 'not comparable')} not comparable")
    nc = [r for r in comp["rows"] if r.get("verdict") == "not comparable"]
    for r in nc:
        con.print(f"  [dim]not comparable[/] {r['key']}: {r.get('method', '')}", highlight=False)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Print a benchmark run file")
    p.add_argument("run_json", nargs="?", default=str(Path(__file__).resolve().parent.parent.parent / "runs" / "latest.json"))
    render(json.loads(Path(p.parse_args().run_json).read_text()))
