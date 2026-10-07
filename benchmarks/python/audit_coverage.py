#!/usr/bin/env python3
"""
Which public C functions does no benchmark time?

Read from the sources, not from a hand-kept table: a function counts as
covered only if some benchmark calls it INSIDE its timed block. Exit status
is 1 when anything is uncovered, so this can gate a check.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_sources  # noqa: E402


def main() -> int:
    cov = bench_sources.coverage()
    benches = bench_sources.benchmarks()
    print(f"{'header':<34} {'timed':>7} {'total':>7}")
    for h, v in sorted(cov["by_header"].items()):
        print(f"{h:<34} {v['covered']:>7} {v['total']:>7}")
    print(f"{'all public headers':<34} {cov['covered']:>7} {cov['total']:>7}   ({cov['percent']:.1f}%)")
    print(f"{len(benches)} benchmarks registered in benchmarks/c/suites")
    if cov["uncovered"]:
        print("\nNot timed by any benchmark:")
        for u in cov["uncovered"]:
            print(f"  {u['symbol']:<44} {u['header']}")
        return 1
    print("\nEvery public function is timed by at least one benchmark.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
