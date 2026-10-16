---
title: Threads and memory
description: Limit the threads BEAM uses, set its memory budget, and use models safely from several threads.
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

BEAM uses all available cores by default. `set_threads` is a hard limit, which is what you want
inside a job scheduler or next to another parallel step. It applies to every model in the
process.

**Results do not depend on the number of threads.** Every sum is taken in a fixed order, so one
thread and sixteen give the same answer, bit for bit.

## Memory budget

```python
BEAM.set_memory_budget(8 << 30)   # 8 GiB
BEAM.memory_budget()              # bytes; the default is 4 GiB, 0 restores it
```

Before a large computation, BEAM checks what it needs against the budget and refuses with
`BudgetExceeded` if it would not fit; the message gives the amount needed. Within the budget,
the budget only changes how much runs at once, never the answer.

## Using a model from several threads

Scoring and all analyses only read the model and are safe to run concurrently on one model.
Methods that change counts in place (`add`, `subtract`) are not: do not change a model while
another thread is using it. Build separate models in parallel and combine them from one thread.
