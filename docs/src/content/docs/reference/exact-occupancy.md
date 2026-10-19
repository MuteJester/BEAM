---
title: ExactOccupancy
description: Expected distinct sequences, frequencies, overlaps and private sequences at any sampling depth, each with a guaranteed error bound.
sidebar:
  order: 2
---

```python
from BEAM.analytics.exact_occupancy import ExactOccupancy

occupancy = ExactOccupancy(model, rel_tol=1e-10)
```

Sampling quantities of a model, computed over every possible sequence without simulation, each
with a guaranteed bound on its relative error. A depth is a number of independent draws (cells)
from the model; it need not be an integer, but must be finite and non-negative. See
[How deeply should I sequence?](/BEAM/guides/sequencing-depth/) and
[How many sequences will two samples share?](/BEAM/guides/overlap/).

Creating it runs one sweep of the model. The first query takes a minute or two on a large model
while the integrals it needs are prepared; they are kept and reused, so later queries at any
depth are fast. `rel_tol` is the relative error each value is planned to; `bound` reports what
was reached.

## Expected numbers of sequences

Every method returns a `BoundedValue`, a list of them, or (for the cohort summaries) a dict of them.

| Method | Expected number of… |
|---|---|
| `richness(n)` | distinct sequences in `n` draws. |
| `richness_many(ns)` | the same at several depths, computed together. |
| `richness_difference(n1, n2)` | new distinct sequences gained going from `n2` to `n1` draws. |
| `frequency(n, j)` | sequences drawn exactly `j` times in `n` draws (`j` ≥ 1). |
| `poisson_richness_many(ns)` | distinct sequences when each sequence's count is Poisson with mean `n` × its probability. |
| `sharing(d1, d2)` | sequences present in both of two independent samples of depths `d1` and `d2`. |
| `private(depths)` | sequences present in exactly one sample of a cohort with these depths. |
| `equal_depth_cohort(depth, n_repertoires)` | a dict: `"seen"`, distinct sequences across `n_repertoires` samples of one depth, and `"private"`, those in exactly one sample. |
| `private_at_depth(depths, *, total_depth)` | for each sample of a cohort whose depths sum to `total_depth`, a dict of lists in input order: `"depth"`, `"distinct"` (its expected distinct sequences) and `"private"` (those no other sample has). |

## Other quantities

| Method | Gives |
|---|---|
| `richness_sd(n)`, `richness_variance(n)` | the standard deviation and variance of the number of distinct sequences in `n` draws, for a whole number `n`. |
| `novelty_many(ns)` | the chance that draw `n` (`n` ≥ 1) is a sequence not drawn before. |
| `sampling_deficit_many(ns)` | the Shannon entropy a sample of depth `n` is expected to lose: its expected entropy is the model's minus this (Poisson sampling, `n` > 0). |

## BoundedValue

| Field | Meaning |
|---|---|
| `value` | The number. |
| `bound` | Guaranteed bound on the relative error: `value` differs from the true value by at most `bound` times the true value. |
| `method` | How it was computed, for example `series` or `contour q=0.8 h=0.03436`. |

`float(bounded_value)` raises `TypeError` on purpose, so a bound is never dropped by accident.
See [Computed, not simulated](/BEAM/explanations/computed-not-simulated/).

## Saving the prepared integrals

`save_lines(path)` writes the prepared integrals to a file, and `load_lines(path)` adds them to a
new `ExactOccupancy` of the same model, which skips the slow first query. `save_lines` merges
into an existing file under a lock, so several processes can share one file; `load_lines`
refuses a file made for a different model.
