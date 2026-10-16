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

Sampling quantities of a model, computed exactly with a rigorous bound on the relative error.
A depth is a number of independent draws (cells) from the model; it need not be an integer.
See [How deeply should I sequence?](/BEAM/guides/sequencing-depth/) and
[How many sequences will two samples share?](/BEAM/guides/overlap/).

The first query takes a minute or two on a large model while the integrals it needs are
prepared; they are kept, so later queries at any depth are fast. `rel_tol` is the relative
error each value is planned to.

## Methods

Every method returns a `BoundedValue` (or a list of them).

| Method | Expected number of… |
|---|---|
| `richness(n)` | distinct sequences in `n` draws. |
| `richness_many(ns)` | the same at several depths, computed together. |
| `richness_difference(n1, n2)` | new distinct sequences gained going from `n2` to `n1` draws. |
| `frequency(n, j)` | sequences drawn exactly `j` times in `n` draws (`j` ≥ 1). |
| `sharing(d1, d2)` | sequences present in both of two independent samples of depths `d1` and `d2`. |
| `private(depths)` | sequences present in exactly one sample of a cohort with these depths. |
| `equal_depth_cohort(depth, n)` | sequences seen at all, and seen in exactly one sample, across `n` samples of one depth. |
| `novelty_many(ns)` | the chance that draw `n` is a sequence not drawn before. |

## BoundedValue

| Field | Meaning |
|---|---|
| `value` | The number. |
| `bound` | Guaranteed bound on the relative error: the true value lies within `value × (1 ± bound)`. |
| `method` | How it was computed (series or contour integral). |

`float(bounded_value)` raises `TypeError` on purpose, so a bound is never dropped by accident.
See [What "exact" means](/BEAM/explanations/what-exact-means/).

## Saving the prepared integrals

`save_lines(path)` writes the prepared integrals to a file, and `load_lines(path)` adds them to a
new `ExactOccupancy` of the same model, which skips the slow first query.
