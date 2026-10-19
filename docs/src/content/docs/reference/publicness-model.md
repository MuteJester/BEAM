---
title: PublicnessModel
description: How many repertoires of a cohort contain a sequence of a given probability, as a mean, variance or full distribution.
sidebar:
  order: 3
---

```python
from BEAM import PublicnessModel

cohort = PublicnessModel(depths)
```

A cohort, described by the sampling depth of each repertoire, turns a sequence probability
into the distribution of how many repertoires contain the sequence: a repertoire of depth $d$
contains a sequence of probability $p$ with probability $1 - (1 - p)^d$. See
[How public is a sequence?](/BEAM/guides/publicness/).

## Constructor

| Parameter | Meaning |
|---|---|
| `depths` | One depth per repertoire: the number of draws (cells or templates) sampled from it; finite and non-negative. |
| `depth_bins` | Repertoires of similar depth are grouped into at most this many groups (default 64), each represented by the geometric mean of its depths, to keep the calculation fast. Grouping changes the answer slightly: `depth_spread` reports the widest ratio of depths within a group (1.0 means none happened), and `depth_bins=len(set(depths))` keeps every depth separate. |

## Methods

| Method | Returns |
|---|---|
| `moments(p)` | Mean and variance of the number of carriers, in closed form. `p` may be one probability or an array. |
| `pmf(p)` | The probability of exactly 0, 1, …, R carriers, for one probability, by numerical inversion of the generating function, accurate to double-precision round-off. |
| `bin_mass(p, edges)` | The probability of a number of carriers in each bin `[edges[i], edges[i+1])`; an integer `edges` makes that many equal-width bins. |
| `expected_counts(probabilities, multiplicities, *, levels=None)` | For a whole spectrum of sequence probabilities (each probability and the number of sequences that have it), the expected number of sequences at each publicness level: one bin per number of carriers by default, `levels=k` for `k` equal-width bins, or a list of edges. |
| `expected_counts_by_group(probabilities, multiplicities, groups, *, n_groups=None, levels=None)` | The same, split by an integer group per probability. |

| Member | Meaning |
|---|---|
| `n_repertoires` | Number of repertoires in the cohort. |
| `n_depth_groups`, `depth_groups()` | How many depth groups are used, and their representative depths with how many repertoires each stands for. |
| `depth_spread` | Widest ratio of depths within a group; 1.0 means no grouping. |
