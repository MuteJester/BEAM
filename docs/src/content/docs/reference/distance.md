---
title: Repertoire distance
description: compare(a, b), the local-window distance between two models, computed over every possible sequence and split into length and pattern terms.
sidebar:
  order: 4
---

```python
from BEAM.analytics.distance import compare

result = compare(a, b, width=None, top_k=100)
```

A distance between two models, computed over every possible sequence without sampling, whose
square splits into a part due to different sequence lengths and a part due to different local
patterns at each length and position. See
[How different are two repertoires?](/BEAM/guides/compare-repertoires/).

At each length $L$, with length probabilities $\pi_L$ in `a` and $\rho_L$ in `b`, the length
term is $\tfrac12(\sqrt{\pi_L} - \sqrt{\rho_L})^2$. Each word of each window adds
$\sqrt{\pi_L \rho_L}\,(\sqrt{p} - \sqrt{q})^2 / (2 n_L)$ to the pattern part, where $p$ and $q$
are the word's probabilities given the length in `a` and `b`, and $n_L$ is the number of
windows at that length. The distance lies between 0 (identical models) and 1.

The models must share the alphabet and order (`ModelMismatch` otherwise); their maximum lengths
may differ. `width` is the length of the local windows compared, from 1 to `order + 1` (the
default, `order + 1`). `top_k` keeps at most that many of the largest pattern terms; the
distance itself is always complete.

## Result

| Field | Meaning |
|---|---|
| `distance` | The distance; 0 for identical models. |
| `squared_distance` | Its square, which is what the parts add up to. |
| `length_squared_distance` | The part due to different length distributions. |
| `pattern_squared_distance` | The part due to different local patterns. |
| `lengths` | Per length: both models' length probabilities, the length term and the pattern term at that length. |
| `windows` | Per length and window position: the pattern term of that window. |
| `patterns` | The `top_k` strongest single patterns, each with `length`, `start` (from 0), `word`, `conditional_p` and `conditional_q` (its probability given the length in `a` and `b`), `contribution`, and `direction` (`"p"` if it is commoner in `a`, `"q"` if in `b`). |
| `reported_pattern_squared_distance` | The sum of the returned patterns' terms. |
| `remainder` | The pattern part not covered by the returned `patterns`. |
| `coverage`, `total_coverage` | Share of the pattern part the returned patterns explain, and share of the whole square explained by the length part plus the returned patterns (`None` when there is nothing to explain). |

`length_squared_distance + pattern_squared_distance` equals `squared_distance`, and the
returned patterns plus `remainder` equal the pattern part, both to floating-point precision.

## Grouping overlapping patterns

`result.group_overlaps(min_overlap=None)` joins overlapping returned patterns into longer
consensus motifs, for reading; it does not change the distance. Patterns join only at the same
length and in the same direction, overlapping by at least `min_overlap` letters (default
`width − 1`) with no conflicting letter. Each group has `length`, `start`, `consensus`,
`direction`, `members`, `contribution` (the sum of its members) and `probabilities(a, b)`, the
consensus's probability in each model.
