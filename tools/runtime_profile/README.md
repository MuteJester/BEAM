# BEAM runtime profile

A package-wide timing, memory and **validity** record, so that optimisation
work can be done without fearing regression: every run is stamped with the
commit it measured, and every run stores what each function *returned*, not
just how long it took.

The scripts are tracked; everything they PRODUCE is not. `runs/`, `cache/` and `TIMINGS.md` are gitignored, so a measurement stays on the machine that took it while the tool that took it travels with the repository. They used to live in `.dev/`, untracked, which meant a fix to the methodology reached no other machine.

## Why the return values are stored

A benchmark tells you a change was faster. It does not tell you the change was
correct. Every case here writes its return value under `runs/<id>/results/`,
arrays as `.npy` and everything else as JSON. `compare.py` then reports two
independent things:

- **speed**: A vs B, per case
- **value**: `same` / `drift` (below tolerance) / **`CHANGED`**

A `CHANGED` verdict is a broken optimisation regardless of how fast it got.

## Layout

```
tools/runtime_profile/
  profile.py            run one profile
  cases.py              the case registry - every function BEAM exposes
  compare.py            diff two runs: speed, memory, and values
  render_table.py       rebuild TIMINGS.md from all runs
  build_foundation.py   build + cache the 927M-sequence model (slow, once)
  cache/                model artefacts (tiny.beam, small.beam, foundation.beam)
  runs/<id>/            one profile run
      manifest.json     commit, date, machine, threads, dirty flag
      timings.csv       one row per (tier, case)
      results/          the captured return value of every case
  TIMINGS.md            wide table: rows are cases, one column per run
```

Run ids are `YYYYmmdd-HHMMSS__<short-commit>[__label]`, so the table header
names the revision to revert to.

## Tiers

| tier | source | sequences |
|---|---|---|
| `tiny` | one 10x donor | ~33k |
| `small` | all four 10x donors pooled | ~101k |
| `foundation` | `release_20260812_censored` parquet | 926,909,993 |

`tiny` and `small` are built fresh each run, so their build/finalize timings
are measured. `foundation` is built once by `build_foundation.py` and cached;
its build timings are carried forward from that cache and its `load` is
measured live.

## Usage

```bash
V=.venv/bin/python

# baseline, before touching anything
$V tools/runtime_profile/profile.py --tiers tiny,small --reps 3

# the foundation model, once (~10 min, ~6 GB RSS)
$V tools/runtime_profile/build_foundation.py
$V tools/runtime_profile/profile.py --tiers foundation --reps 3

# after a change
$V tools/runtime_profile/profile.py --tiers tiny,small --reps 3
$V tools/runtime_profile/compare.py --last 2
```

Useful flags: `--only a,b,c` and `--skip a,b,c` to restrict cases, `--label`
to tag a run, `--threads N` to pin the thread count, `--reps N`.

`compare.py --rtol` sets the tolerance a value may move before it counts as
`CHANGED` (default `1e-9`). Reductions in this library are summed in a fixed
index order and are meant to be bit-identical across thread counts, so a
genuine algorithmic change should show `same`, and anything else deserves an
explanation.

## Reading the timings

**Timings are warm.** Each case gets one untimed warmup call before the timed
repetitions. This matters: the live-frontier cache is built on first use, so
without a warmup whichever case ran first paid for all the others —
`top_k_20` measured 3167 ms and `top_k_100` 174 ms purely from ordering. With
the warmup they are 231 ms and 225 ms.

`peak_rss_bytes` is the peak resident-memory *growth* during the call, sampled
at 5 ms. It catches transient DP scratch that is freed before the call
returns, which is exactly what the trellis buffers are. The `file_bytes` row
carries the on-disk model size instead.

## Caveats

- Peak RSS is sampled, so a very short allocation spike can be missed. Treat
  small numbers as noise and large ones as real.
- `simulate`, `calibrate`, `pseq_analysis` and `copy` return objects with no
  stable serialisation; they are timed but marked `opaque` and not compared.
- `sample`/`simulate` are seeded, so their values *are* comparable.
- The machine is shared with whatever else is running; compare runs taken
  under similar conditions, and prefer the ratio over the absolute.
