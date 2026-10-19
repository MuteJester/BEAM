---
title: Threads and memory
description: Limit the threads BEAM uses, set its memory budget, use models safely from several threads, and run BEAM in worker processes.
sidebar:
  order: 21
---

## Threads

```python
import BEAM

BEAM.set_threads(4)          # at most 4 threads, for the whole process
BEAM.threads()               # the limit in force (0: the system's choice)
BEAM.effective_threads()     # what a parallel step would start right now
```

BEAM uses all available cores by default, or `OMP_NUM_THREADS` if it is set. `set_threads` is
a hard limit, which is what you want inside a job scheduler or next to another parallel step. It
applies to every model in the process; `set_threads(0)` restores the default. Set it before
starting work, not while another thread is computing.

**Results do not depend on the number of threads.** Every sum is taken in a fixed order, so one
thread and sixteen give the same answer, bit for bit.

## Memory budget

```python
BEAM.set_memory_budget(8 << 30)   # 8 GiB
BEAM.memory_budget()              # bytes; the default is 4 GiB, 0 restores it
```

Before a large computation, BEAM checks what it needs against the budget and refuses with
`BudgetExceeded` if it would not fit; the message gives the amount needed. Within the budget,
the budget only changes how much runs at once, never the answer. Lengths computed in parallel
share the budget, so more threads do not multiply memory use.

After its first analysis a model keeps a cache of up to a quarter of the budget
(`model.live_cache_bytes`); `model.drop_caches()` releases it without changing any result.

## Using a model from several threads

Scoring and all analyses only read the model and are safe to run concurrently on one model.
Methods that change the model in place (`add`, `subtract`, `set_alpha`, `compact`,
`drop_caches`) are not: do not change a model while another thread is using it. Build separate
models in parallel and combine them from one thread.

## Worker processes

Start worker processes with **spawn**, for example
`multiprocessing.get_context("spawn").Pool(...)`, and call `BEAM.set_threads(T)` in each worker
so that workers × `T` fits the machine. Models are not pickled: pass each worker the model's
file path and load it there with `BeamModel.from_file`.

```python
import multiprocessing as mp
import BEAM
from BEAM import BeamModel

def score(args):
    path, sequences = args
    BEAM.set_threads(2)
    return BeamModel.from_file(path).log_pseq_batch(sequences)

if __name__ == "__main__":             # spawned workers re-import this file
    with mp.get_context("spawn").Pool(4) as pool:
        results = pool.map(score, [("donor_A.beam", batch) for batch in batches])
```
