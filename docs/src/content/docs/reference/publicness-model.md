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
into the distribution of how many repertoires contain the sequence. See
[How public is a sequence?](/BEAM/guides/publicness/).

## Constructor

| Parameter | Meaning |
|---|---|
| `depths` | One depth (number of cells sampled) per repertoire. |
| `depth_bins` | Repertoires of similar depth are grouped into at most this many groups (default 64), each represented by one depth, to keep the calculation fast. Grouping changes the answer slightly: `depth_spread` reports the widest ratio of depths within a group (1.0 means none happened), and `depth_bins=len(set(depths))` keeps every depth separate. |

## Methods

| Method | Returns |
|---|---|
| `moments(p)` | Mean and variance of the number of carriers, in closed form. `p` may be one probability or an array. |
| `pmf(p)` | The probability of exactly 0, 1, …, R carriers, for one probability, by numerical inversion of the generating function (accurate to about $10^{-9}$, tails below that set to zero). |
| `bin_mass(p, edges)` | The probability of a number of carriers in each bin `[edges[i], edges[i+1])`. |
| `expected_counts(probabilities, multiplicities)` | For a whole spectrum of sequence probabilities, the expected number of sequences at each publicness level. |

| Member | Meaning |
|---|---|
| `n_repertoires` | Number of repertoires in the cohort. |
| `depth_spread` | Widest ratio of depths within a group; 1.0 means no grouping. |
| `depth_groups()` | The representative depths and how many repertoires each stands for. |
