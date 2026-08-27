# Threads and memory

## Parallelism

```python
from BEAM import set_threads, threads, effective_threads

set_threads(1)        # hard cap, PROCESS-WIDE
threads()             # the cap in force, or 0 for the runtime's default
effective_threads()   # what a parallel region would actually start
model.threads = 1          # the same setting, spelled as a property
model.effective_threads
```

A library that silently grabs every core is unusable inside a job scheduler or alongside another parallel stage, so this is a **hard cap** rather than a hint: every parallel region in the BeamModel implementation runs with at most `n` threads. `0` restores the runtime's own choice, negatives are treated as `0`, and a build without OpenMP ignores the setting and always reports 1.

!!! warning "The cap is process-wide, not per model"
    `model.threads = 1` changes it for every model in the process. The property exists because that is where a caller looks for it; `set_threads` is the spelling that says what it does.

**Results do not depend on it.** Every reduction in the implementation is either order-independent or summed in a fixed index order, so a one-thread run and a sixteen-thread run agree bit for bit — asserted in the tests for both the batch scorer and a trellis-reducing analytic. That is what makes pinning threads to 1 for a runtime comparison a fair thing to do rather than a distortion.

### Memory budget

```python
from BEAM import set_memory_budget, memory_budget

set_memory_budget(2 << 30)   # 2 GB
memory_budget()              # bytes; the default is 4 GB, 0 restores it
```

Only [`spectra_by_length`](#spectra_by_length) is large enough to care: one length's histogram grids run to 1.6 GB at order 4 with 512 bins, so running one length per core would ask for twenty-five gigabytes. It divides this budget by what one length needs and runs that many at a time.

This caps **concurrency only**. A single length always runs whatever its size, so lowering the budget can never turn a working call into a failing one, only a slower one. It is a resource knob and nothing else: **results do not depend on it**, pinned by a C unit test at a budget of one byte and by a Python test that a full by-length spectrum is bit-identical at a budget of 1.

### Measured effect of the parallel implementation

Wall clock, before and after, on the workstation. Order 4 / order 3 where both were measured.

| operation | order 4 | order 3 |
|---|---|---|
| finalize | 12.75 -> 4.31 s | 3.55 -> 0.51 s |
| entropy | 11.88 -> 4.61 s | 3.60 -> 0.24 s |
| path_count | 17.97 -> 3.67 s | 4.29 -> 0.18 s |
| pooled spectrum | 315 -> 54 s | |
| `top_k_sequences(100)` | never finished -> 3.96 s | |
| `sample(10_000)` | 18.0 -> 8.1 s | |
| save / load | 2.75 -> 0.79 s | 14.36 -> 5.47 s |

Re-measured on a busier machine: build + finalize at order 3 is 0.41 s, and `top_k_sequences(100)` at order 4 is 6.3 s. Treat the table as the shape of the change, not as a benchmark.

!!! warning "`sample` draws different residues than it used to"
    Each length now has its own generator, so a given seed produces a different sequence of residues than before the parallel implementation. **Per-seed reproducibility is preserved and tested** — the same seed gives the same draws today and tomorrow — but the old single-stream draw order is gone, and any result that pinned specific sampled strings to a seed will not reproduce.

!!! note "`surprisal_by_depth` moved by ~2e-12 relative"
    It now sums in fixed 256-element chunks in index order, chosen so the answer does not depend on the thread count. That changes the last couple of digits against the old serial value.

A float32 copy of the tables was tried and is **slower** (0.64x); it is not an avenue worth reopening.

## Thread safety

Scoring and analytics are read-only and safe to call concurrently on one model. `add` and `subtract` are **not**: both release the GIL and then re-finalize, which frees and reallocates the log-probability tables that a concurrent reader is dereferencing. Do not mutate a model that another thread is scoring against — build shard models in parallel and combine them from one thread instead.
