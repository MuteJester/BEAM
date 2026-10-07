#!/usr/bin/env python3
"""
BEAM benchmark runner: build, run, record, report.

    ./benchmarks/run                      every benchmark, default workload (aa, K=4, L=27)
    ./benchmarks/run --suite math,dp      selected suites
    ./benchmarks/run --compare runs/run_X.json
    ./benchmarks/run --check              fast self-test of the harness itself

What it guarantees:
- The harness and libbeam.a are rebuilt through make on every run, and the
  SHA-256 of both is recorded, so a result always names the exact library it
  measured (the commit alone does not when the tree is dirty).
- A benchmark that crashes the harness is recorded as CRASHED and the run
  continues with the rest.
- The run file (runs/run_<id>.json) holds raw per-call samples and the
  environment, so it can be re-analysed or compared later.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis  # noqa: E402
import bench_sources  # noqa: E402

REPO = bench_sources.REPO_ROOT
BENCH_DIR = REPO / "benchmarks"
RUNS = BENCH_DIR / "runs"
BINARY = REPO / "build" / "benchmarks" / "beam_bench"
LIBRARY = REPO / "build" / "libbeam.a"
MAX_RESTARTS = 20


# ── Build and provenance ───────────────────────────────────────────────

def build() -> None:
    """make decides what is stale; the benchmark Makefile always consults the
    library's own Makefile, so a library edit is never benchmarked stale."""
    res = subprocess.run(["make", "-C", str(REPO), "-f", "benchmarks/Makefile", "-j", str(os.cpu_count() or 4)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout)
        raise SystemExit("build failed")
    rebuilt = [ln.split(" -o ", 1)[1].split()[0] for ln in res.stdout.splitlines() if " -o " in ln]
    rebuilt += ["build/libbeam.a" for ln in res.stdout.splitlines() if ln.startswith("ar ")]
    if rebuilt:
        print(f">> rebuilt {len(rebuilt)} target(s): " + ", ".join(sorted(set(rebuilt))[:6])
              + (" ..." if len(set(rebuilt)) > 6 else ""))
    if any(ln.startswith("warning") or ": warning:" in ln for ln in res.stdout.splitlines()):
        print(">> the build printed warnings:\n" + "\n".join(ln for ln in res.stdout.splitlines() if "warning" in ln))


def sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def build_info() -> Dict[str, Any]:
    status = _git("status", "--porcelain") or ""
    lib_dirty = []
    for line in status.splitlines():
        path = line[3:].strip()
        if path.startswith(("src/native", "include/", "build_support/", "Makefile")):
            lib_dirty.append(path)
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status.strip()),
        "library_sources_dirty": lib_dirty[:50],
        "libbeam_sha256": sha256(LIBRARY),
        "beam_bench_sha256": sha256(BINARY),
    }


def _read(path: str) -> Optional[str]:
    try:
        return Path(path).read_text().strip()
    except Exception:
        return None


def background_cpus(seconds: float = 1.0) -> Optional[float]:
    """CPUs busy on the machine over one second before the run, measured from
    /proc/stat. Not the load average, which still carries minutes of whatever
    ran before -- including the previous benchmark run."""
    import time

    def busy() -> int:
        v = [int(x) for x in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:9]]
        return v[0] + v[1] + v[2] + v[5] + v[6] + v[7]
    try:
        b0, t0 = busy(), time.time()
        time.sleep(seconds)
        b1, t1 = busy(), time.time()
        return (b1 - b0) / os.sysconf("SC_CLK_TCK") / (t1 - t0)
    except Exception:
        return None


def environment() -> Dict[str, Any]:
    cpu = None
    cpuinfo = _read("/proc/cpuinfo") or ""
    for line in cpuinfo.splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    return {
        "host": socket.gethostname(),
        "cpu_model": cpu,
        "logical_cpus": os.cpu_count(),
        "physical_cores": analysis.physical_cores(),
        "kernel": platform.release(),
        "governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "frequency_driver": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver"),
        "energy_preference": _read("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"),
        "no_turbo": _read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
        "perf_event_paranoid": _read("/proc/sys/kernel/perf_event_paranoid"),
        "background_cpus_at_start": background_cpus(),
        "python": platform.python_version(),
        "omp_env": {k: v for k, v in os.environ.items() if k.startswith(("OMP_", "GOMP_", "KMP_"))},
    }


# ── Running the harness ────────────────────────────────────────────────

def harness_args(a: argparse.Namespace) -> List[str]:
    out: List[str] = []
    def opt(flag: str, val: Any) -> None:
        if val is not None:
            out.extend([flag, str(val)])
    opt("--suite", a.suite); opt("--filter", a.filter)
    opt("--alphabet", a.alphabet); opt("--order", a.order)
    opt("--length", a.length); opt("--max-length", a.max_length)
    opt("--corpus", a.corpus); opt("--corpus-file", a.corpus_file); opt("--seed", a.seed)
    opt("--threads", a.threads)
    opt("--samples", a.samples); opt("--min-samples", a.min_samples)
    opt("--time-budget", a.time_budget); opt("--warmup-time", a.warmup_time)
    opt("--slow-call", a.slow_call)
    return out


def run_harness(base_args: List[str], jsonl: Path, exclude: List[str], quiet: bool = False) -> Dict[str, Any]:
    """Run the harness, restarting it past any benchmark that crashes it."""
    if jsonl.exists():
        jsonl.unlink()
    crashed: List[Dict[str, Any]] = []
    for attempt in range(MAX_RESTARTS + 1):
        parsed = analysis.load_jsonl(jsonl) if jsonl.exists() else {"results": {}, "crashed": []}
        skip = list(exclude) + list(parsed["results"].keys()) + [c["key"] for c in crashed]
        cmd = [str(BINARY), *base_args, "--jsonl", str(jsonl)]
        if skip:
            cmd += ["--exclude", ",".join(skip)]
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL if quiet else None)
        if proc.returncode == 0:
            break
        parsed = analysis.load_jsonl(jsonl) if jsonl.exists() else {"crashed": []}
        new = [k for k in parsed.get("crashed", []) if k not in {c["key"] for c in crashed}]
        sig = -proc.returncode if proc.returncode < 0 else None
        if not new:
            raise SystemExit(f"harness exited with status {proc.returncode} outside any benchmark "
                             f"(workload or fixture build failed?); see the output above")
        why = signal.Signals(sig).name if sig else f"exit status {proc.returncode}"
        for k in new:
            print(f">> CRASHED in {k} ({why}); continuing with the remaining benchmarks", file=sys.stderr)
            crashed.append({"key": k, "reason": why})
    parsed = analysis.load_jsonl(jsonl)
    parsed["crash_reasons"] = {c["key"]: c["reason"] for c in crashed}
    return parsed


def assemble(parsed: Dict[str, Any], run_id: str, label: Optional[str], args_used: List[str],
             build: Dict[str, Any], env: Dict[str, Any]) -> Dict[str, Any]:
    meta = parsed["meta"] or {}
    fixture0 = parsed["fixtures"].get("0")
    sources = bench_sources.benchmarks()
    defs = bench_sources.definitions()
    results: List[Dict[str, Any]] = []
    for key, raw in parsed["results"].items():
        r = analysis.interpret(raw, meta, fixture0)
        results.append(r)
    for key in parsed.get("crashed", []):
        d, n = key.split("::", 1)
        results.append({"key": key, "domain": d, "name": n, "status": "crashed",
                        "crash_reason": parsed.get("crash_reasons", {}).get(key, "unknown"),
                        "flags": [], "observations": [], "stats": None, "memory": {}, "cold_ns": None})
    for r in results:
        s = sources.get(r["key"])
        if s:
            r["source"] = {"file": s.file, "line": s.line}
            r["timed_calls"] = [{"symbol": c, "definition": "%s:%d" % defs[c] if c in defs else None}
                                for c in s.timed_calls]
        if not r.get("description"):
            r["description"] = ""
    order = {k: i for i, k in enumerate(sources)}
    results.sort(key=lambda r: (r["domain"], order.get(r["key"], 1 << 30)))

    corpus = parsed.get("corpus") or {}
    workload = {
        "alphabet": meta.get("alphabet"), "order": meta.get("order"),
        "max_length": meta.get("max_length"), "length": meta.get("length"),
        "corpus_source": meta.get("corpus_file") or "synthetic",
        "corpus_size": meta.get("corpus_size"), "seed": meta.get("seed"),
        "corpus_sequences": corpus.get("n_sequences_0"), "corpus_mean_length": corpus.get("mean_length_0"),
        "threads_requested": meta.get("threads_requested"), "threads_effective": meta.get("threads_effective"),
        "memory_budget_bytes": meta.get("memory_budget_bytes"),
    }
    sampling = {k: meta.get(k) for k in ("max_samples", "min_samples", "time_budget_s", "warmup_time_s",
                                         "min_warmup", "target_window_s", "slow_call_s")}
    run = {
        "schema": analysis.SCHEMA,
        "run_id": run_id,
        "label": label,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "harness_args": args_used,
        "build": {**build, "compiler": meta.get("compiler"), "harness_built": meta.get("harness_built")},
        "environment": env,
        "workload": workload,
        "fixtures": parsed["fixtures"],
        "fixture_errors": parsed["fixture_errors"],
        "sampling": sampling,
        "timer_overhead_ns": meta.get("timer_overhead_ns"),
        "harness_processes": parsed.get("processes"),
        "process_peak_rss_bytes": max((e.get("process_peak_rss_bytes", 0) for e in parsed["ends"]), default=None),
        "results": results,
        "coverage": bench_sources.coverage(),
    }
    run["warnings"] = warnings(run)
    run["summary"] = {
        "counts": analysis.counts_by_status(results),
        "slowest": [r["key"] for r in analysis.rank_by_time(results)],
        "largest_heap": [r["key"] for r in analysis.rank_by_memory(results)],
        "largest_cold_penalty": [r["key"] for r in analysis.rank_by_cold_penalty(results)],
    }
    return run


def warnings(run: Dict[str, Any]) -> List[str]:
    w: List[str] = []
    env, b, wl = run["environment"], run["build"], run["workload"]
    if env.get("governor") and env["governor"] != "performance":
        how = (f"{env.get('frequency_driver')} '{env['governor']}' mode, energy preference "
               f"'{env.get('energy_preference') or '?'}', turbo {'off' if env.get('no_turbo') == '1' else 'on'}")
        w.append(f"CPU clock speed is dynamic ({how}): each core runs between idle and boost speed "
                 f"depending on how many cores are busy and on temperature. This is the normal Linux setting, "
                 f"not a power-saving mode, but it adds run-to-run noise; compare runs made under the same setting.")
    bg = env.get("background_cpus_at_start")
    if bg is not None and bg > 1.0:
        w.append(f"Other processes were using {bg:.1f} CPUs when the run started; multi-threaded kernels "
                 f"compete with them for cores (see the per-benchmark 'contended' flag).")
    if b.get("library_sources_dirty"):
        w.append(f"The library was built from uncommitted changes ({len(b['library_sources_dirty'])} files); "
                 f"the commit hash does not identify the code measured, libbeam sha256 does.")
    c = run["summary"]["counts"] if "summary" in run else analysis.counts_by_status(run["results"])
    if c.get("error"):
        w.append(f"{c['error']} benchmark(s) failed with a BEAM error; their times are not reported.")
    if c.get("contended"):
        w.append(f"{c['contended']} benchmark(s) ran while other processes competed for the "
                 f"{env.get('physical_cores') or '?'} physical cores (flag 'contended'): their times may be "
                 f"inflated (up to ~2x for short multi-threaded kernels). Rankings hold to that factor; for "
                 f"a decision use --ab, whose interleaving exposes both sides to the same load.")
    if c.get("refused"):
        w.append(f"{c['refused']} benchmark(s) were refused by BEAM's memory budget at this workload "
                 f"(the message says how much they needed).")
    if c.get("crashed"):
        w.append(f"{c['crashed']} benchmark(s) crashed the harness.")
    if run.get("fixture_errors"):
        w.append("A fixture could not be built: " + "; ".join(e.get("error", "") for e in run["fixture_errors"]))
    if env.get("perf_event_paranoid") and env["perf_event_paranoid"].lstrip("-").isdigit() \
            and int(env["perf_event_paranoid"]) > 2:
        w.append("Hardware counters are not available to this user (perf_event_paranoid="
                 f"{env['perf_event_paranoid']}); see benchmarks/DESIGN.md for profiling one benchmark with perf.")
    return w


# ── Entry point ────────────────────────────────────────────────────────

def parse(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="benchmarks/run", description="Build, run and report BEAM's C benchmarks.")
    g = p.add_argument_group("selection")
    g.add_argument("--suite", help="comma-separated suites (core, model, dp, math, algorithms, mixture, query, occupancy, insights)")
    g.add_argument("--filter", help="comma-separated substrings of domain::name")
    g.add_argument("--exclude", help="comma-separated domain::name to skip")
    g = p.add_argument_group("workload")
    g.add_argument("--alphabet", choices=["aa", "nt"])
    g.add_argument("--order", type=int)
    g.add_argument("--length", type=int, help="per-length argument; also max_length unless --max-length")
    g.add_argument("--max-length", type=int)
    g.add_argument("--corpus", type=int, help="synthetic corpus size, or cap on --corpus-file (default 20000)")
    g.add_argument("--corpus-file", help="train fixtures on real sequences, one per line")
    g.add_argument("--seed", type=int)
    g.add_argument("--threads", type=int, help="BEAM thread cap (0 = library default, all cores)")
    g = p.add_argument_group("sampling")
    g.add_argument("--samples", "--iterations", "-n", dest="samples", type=int, help="max samples per benchmark (default 20)")
    g.add_argument("--min-samples", type=int, help="min samples per benchmark (default 3)")
    g.add_argument("--time-budget", type=float, help="seconds of sampling per benchmark once min-samples is met (default 2)")
    g.add_argument("--warmup-time", type=float)
    g.add_argument("--slow-call", type=float, help="a cold call slower than this many seconds gets one sample (default 30)")
    g = p.add_argument_group("output")
    g.add_argument("--label", help="a name for this run, e.g. before-fix")
    g.add_argument("--compare", metavar="RUN_JSON",
                   help="list differences from a baseline run file (single runs: no verdicts; use --ab)")
    g.add_argument("--no-html", action="store_true")
    g.add_argument("--quiet", action="store_true", help="do not stream the harness's per-benchmark lines")
    g = p.add_argument_group("A/B testing a change")
    g.add_argument("--save-binary", metavar="NAME",
                   help="keep the current harness (libbeam.a is linked in) as baseline NAME, then exit")
    g.add_argument("--ab", metavar="NAME", help="interleave baseline NAME and the current build, decide per benchmark")
    g.add_argument("--rounds", type=int, default=4, help="processes per side for --ab (default 4; 4 is the "
                   "smallest count whose exact two-sided test can reach p < 0.05)")
    g = p.add_argument_group("other modes")
    g.add_argument("--check", action="store_true", help="self-test: every benchmark once on a tiny workload")
    g.add_argument("--audit", action="store_true", help="which public C functions no benchmark times")
    g.add_argument("--inspect", metavar="SYMBOL_OR_BENCHMARK", help="everything known about one function or benchmark")
    g.add_argument("--list", action="store_true", help="list benchmarks")
    g.add_argument("--sweep", action="store_true", help="scaling sweep over lengths and orders")
    g.add_argument("--sweep-lengths", default="15,21,27", help="lengths for --sweep (each is also max_length)")
    g.add_argument("--sweep-orders", default="2,3,4", help="orders for --sweep")
    g.add_argument("--sweep-threads", default="", help="thread counts for --sweep, e.g. 1,2,4,8,16")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    a = parse(argv)
    if a.inspect:
        import inspect_tool
        return inspect_tool.main([a.inspect])
    if a.audit:
        import audit_coverage
        return audit_coverage.main()

    build()
    if a.list:
        return subprocess.call([str(BINARY), "--list"])
    if a.sweep:
        import sweep_runner
        return sweep_runner.main(a)
    if a.check:
        return self_check(a)
    if a.save_binary:
        return save_binary(a.save_binary)

    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "raw").mkdir(exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = stamp + (f"_{a.label}" if a.label else "")
    jsonl = RUNS / "raw" / f"run_{run_id}.jsonl"
    args_used = harness_args(a)
    b, env = build_info(), environment()

    exclude = (a.exclude or "").split(",") if a.exclude else []
    if a.ab:
        run = ab_test(a, run_id, args_used, exclude, b, env)
    else:
        parsed = run_harness(args_used, jsonl, exclude, quiet=a.quiet)
        run = assemble(parsed, run_id, a.label, args_used, b, env)

    if a.compare:
        base_path = Path(a.compare)
        base = json.loads(base_path.read_text())
        run["comparison"] = analysis.compare(run, base)

    out = RUNS / f"run_{run_id}.json"
    out.write_text(json.dumps(run, indent=1))
    latest = RUNS / "latest.json"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out.name)

    from dashboard import terminal_tui
    terminal_tui.render(run)
    if not a.no_html:
        from dashboard import html_generator
        sweep = load_matching_sweep(run)
        html = RUNS / f"report_{run_id}.html"
        html_generator.write(run, html, sweep=sweep)
        shutil.copyfile(html, RUNS / "latest_report.html")
        print(f">> HTML report: file://{html}")
    print(f">> Run file: {out}")
    return 0


BINARIES = RUNS / "binaries"


def save_binary(name: str) -> int:
    """The harness links libbeam.a statically, so a copy of the binary is a
    frozen copy of the library as it is now: build the change afterwards and
    --ab compares the two without rebuilding the old one."""
    BINARIES.mkdir(parents=True, exist_ok=True)
    dst = BINARIES / name
    shutil.copy2(BINARY, dst)
    info = {"name": name, "saved": _dt.datetime.now().astimezone().isoformat(timespec="seconds"), **build_info()}
    (BINARIES / f"{name}.json").write_text(json.dumps(info, indent=1))
    print(f">> saved {dst} (libbeam {info['libbeam_sha256'][:12]}, commit {(info['git_commit'] or '?')[:10]}"
          f"{', dirty' if info['git_dirty'] else ''})")
    return 0


def ab_test(a: argparse.Namespace, run_id: str, args_used: List[str], exclude: List[str],
            b: Dict[str, Any], env: Dict[str, Any]) -> Dict[str, Any]:
    """Alternate baseline and current processes, `rounds` of each, starting
    side swapped every round so drift over time hits both equally."""
    global BINARY
    base_bin = BINARIES / a.ab
    if not base_bin.exists():
        raise SystemExit(f"no saved baseline {a.ab!r}: run --save-binary {a.ab} on the baseline build first")
    base_info = json.loads((BINARIES / f"{a.ab}.json").read_text()) if (BINARIES / f"{a.ab}.json").exists() else {}
    cur_bin = BINARY
    rounds = max(2, a.rounds)
    base_runs: List[Dict[str, Any]] = []
    cur_runs: List[Dict[str, Any]] = []
    last_cur = None
    for i in range(rounds):
        order = [("base", base_bin), ("cur", cur_bin)] if i % 2 == 0 else [("cur", cur_bin), ("base", base_bin)]
        for side, binary in order:
            print(f">> round {i + 1}/{rounds}: {side}")
            BINARY = binary
            try:
                parsed = run_harness(args_used, RUNS / "raw" / f"ab_{run_id}_{side}{i}.jsonl", exclude, quiet=True)
            finally:
                BINARY = cur_bin
            meta, fx = parsed["meta"] or {}, parsed["fixtures"].get("0")
            per = {k: analysis.interpret(r, meta, fx) for k, r in parsed["results"].items()}
            for k in parsed.get("crashed", []):
                per[k] = {"status": "crashed", "stats": None}
            (base_runs if side == "base" else cur_runs).append(per)
            if side == "cur":
                last_cur = parsed
    run = assemble(last_cur, run_id, a.label, args_used, b, env)
    rows = analysis.ab_compare(base_runs, cur_runs)
    run["comparison"] = {
        "kind": "ab", "rounds": rounds, "problems": [],
        "same_library": base_info.get("libbeam_sha256") == b.get("libbeam_sha256"),
        "same_harness": base_info.get("beam_bench_sha256") == sha256(cur_bin),
        "baseline_run": f"saved binary {a.ab}", "baseline_commit": base_info.get("git_commit"),
        "baseline_libbeam_sha256": base_info.get("libbeam_sha256"), "rows": rows,
    }
    return run


def load_matching_sweep(run: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The last scaling sweep, only if it measured the same library build."""
    path = RUNS / "scaling_sweep.json"
    if not path.exists():
        return None
    try:
        sweep = json.loads(path.read_text())
    except Exception:
        return None
    if sweep.get("schema") != analysis.SCHEMA:
        print(">> scaling_sweep.json predates the current harness; not attached (re-run --sweep)")
        return None
    if sweep.get("build", {}).get("libbeam_sha256") != run["build"].get("libbeam_sha256"):
        print(">> scaling_sweep.json measured a different libbeam.a; not attached (re-run --sweep)")
        return None
    return sweep


def self_check(a: argparse.Namespace) -> int:
    """Every benchmark once, on a workload small enough to finish in seconds.
    Fails if any benchmark errors or crashes: run it after changing the C API."""
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "raw").mkdir(exist_ok=True)
    rc = 0
    for alpha, extra in (("aa", ["--order", "2", "--length", "12"]), ("nt", ["--order", "3", "--length", "30"])):
        jsonl = RUNS / "raw" / f"check_{alpha}.jsonl"
        args = ["--alphabet", alpha, *extra, "--corpus", "2000", "--samples", "1", "--min-samples", "1",
                "--time-budget", "0", "--warmup-time", "0"]
        if a.suite:
            args += ["--suite", a.suite]
        if a.filter:
            args += ["--filter", a.filter]
        parsed = run_harness(args, jsonl, [], quiet=True)
        bad = [(k, r.get("error_message", "")) for k, r in parsed["results"].items()
               if r.get("status") == "error" and r.get("error_code") != analysis.BEAM_ERR_BUDGET]
        bad += [(k, "CRASHED") for k in parsed.get("crashed", [])]
        print(f"[{alpha}] {len(parsed['results'])} benchmarks, {len(bad)} failed")
        for k, msg in bad:
            print(f"    {k}: {msg}")
        rc |= 1 if bad else 0
    print("check: " + ("FAILED" if rc else "all benchmarks ran without error"))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
