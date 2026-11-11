"""What the budget charges for a draw must cover what the draw actually costs.

The budget used to charge the flat result buffer alone -- 30 MB against a
157 MB peak, 0.19x -- because the `str` objects and the list built from that
buffer are allocated by CPython after the C code returns. `sample(2**28)`
therefore passed a 4.29 GB budget and then asked for roughly 16 GB.

The refusal tests next door check that a refusal HAPPENS and says something
useful. Neither notices if the arithmetic behind it drifts: drop the pymalloc
rounding or the buffer term and every one of them still passes, while the
number quietly returns to being wrong in the direction that caused the
original failure. This is the test that would notice.

It measures the true high-water mark in a CHILD PROCESS, because `ru_maxrss`
never decreases -- taking two readings in one process reports the second as
near-zero, which produced a 7.28x figure that was pure artefact while this was
being written.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

CHILD = textwrap.dedent("""
    import gc, json, random, sys, threading, time
    from BEAM import Builder
    maxlen, slen, n = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

    def rss():
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * 4096

    random.seed(0)
    AA = "ACDEFGHIKLMNPQRSTVWY"
    seqs = ["C" + "".join(random.choice(AA) for _ in range(slen)) + "F"
            for _ in range(3000)]
    b = Builder(order=3, max_length=maxlen)
    b.observe(seqs, [1.0] * len(seqs))
    m = b.build()
    gc.collect()

    # Sampled by a watcher thread, NOT from ru_maxrss. On Linux a child keeps
    # the high-water mark it inherited, so under a full test run the parent is
    # already above anything this allocates and the delta reads zero -- which
    # is exactly how this test first failed.
    base = rss()
    peak = [base]
    stop = threading.Event()
    def watch():
        while not stop.is_set():
            r = rss()
            if r > peak[0]:
                peak[0] = r
            time.sleep(0.001)
    t = threading.Thread(target=watch, daemon=True)
    t.start()
    out = m.sample(n)
    stop.set(); t.join()
    if rss() > peak[0]:
        peak[0] = rss()
    print(json.dumps({"peak": peak[0] - base, "n": len(out)}))
""")


def _peak(maxlen: int, slen: int, n: int) -> int:
    r = subprocess.run([sys.executable, "-c", CHILD, str(maxlen), str(slen), str(n)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["n"] == n
    return out["peak"]


def _charged(maxlen: int, n: int) -> int:
    """The binding's OWN figure, read back out of its refusal.

    Restating the formula here would test this file against itself: change
    guards.c and forget this, and the comparison still passes because it never
    consulted the thing it claims to check. The refusal prints the byte count
    it computed, so lower the budget until it refuses and read it.
    """
    import re, random
    import BEAM
    from BEAM import Builder
    random.seed(0)
    seqs = ["C" + "".join(random.choice("ACDEFGHIKLMNPQRSTVWY")
                          for _ in range(maxlen - 4)) + "F" for _ in range(300)]
    b = Builder(order=3, max_length=maxlen)
    b.observe(seqs, [1.0] * len(seqs))
    m = b.build()
    was = BEAM.memory_budget()
    try:
        BEAM.set_memory_budget(1 << 20)       # certainly too small for n
        from BEAM.errors import BudgetExceeded
        with pytest.raises(BudgetExceeded) as e:
            m.sample(n)
        got = re.search(r"needs (\d+) bytes", str(e.value))
        assert got, f"the refusal no longer prints its byte figure: {e.value}"
        return int(got.group(1))
    finally:
        BEAM.set_memory_budget(was)


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="reads resident memory from /proc/self/statm (Linux)")
@pytest.mark.parametrize("maxlen,slen", [(14, 10), (27, 20)])
def test_the_charge_covers_the_true_peak(maxlen, slen):
    n = 2_000_000
    peak = _peak(maxlen, slen, n)
    charged = _charged(maxlen, n)
    ratio = charged / peak
    print(f"\n  stride {maxlen + 1}: charged {charged/1e6:.1f} MB, "
          f"peak {peak/1e6:.1f} MB, ratio {ratio:.2f}")
    # Loose on purpose. Measured 1.11-1.42x; the regression it guards is 0.19x.
    # Anything at or above 1.0 separates them, and a tight bound here would
    # fail on a machine whose allocator rounds differently.
    assert ratio >= 1.0, (
        f"the budget charges {charged/1e6:.1f} MB for a draw that peaks at "
        f"{peak/1e6:.1f} MB -- it is promising what it cannot keep")
    assert ratio <= 3.0, (
        f"charging {ratio:.1f}x the real peak refuses work that would fit")


def test_the_charge_counts_the_representation_not_just_the_buffer():
    """The objects dominate, so omitting them is not a rounding error. If this
    comes out near 1.0, the representation term is gone and the charge is back
    to the buffer alone -- which is the state that let sample(2**28) through."""
    n, maxlen = 1_000_000, 14
    buffer_only = n * (maxlen + 1)
    assert _charged(maxlen, n) / buffer_only >= 3.0
