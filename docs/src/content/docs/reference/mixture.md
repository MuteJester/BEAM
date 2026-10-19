---
title: Mixture
description: Probability mixtures of several models, the object diversity across a group of repertoires is defined on.
sidebar:
  order: 5
---

```python
from BEAM import Mixture

group = Mixture([donor_a, donor_b, donor_c])          # equal weights
group = Mixture([donor_a, donor_b], weights=[0.7, 0.3])
```

The distribution $\bar P(s) = \sum_i w_i P_i(s)$: pick member $i$ with probability $w_i$, then a
sequence from that member. This is different from pooling counts with `merged`, which can join
stretches of sequence from different members into sequences no member writes; see
[Merging counts or mixing models](/BEAM/explanations/which-model/#merging-counts-or-mixing-models).

All members must share alphabet, order and maximum length (otherwise `IncompatibleMembers`).
Weights must be non-negative and sum to 1; they are refused, not rescaled, when they do not.

## Diversity

| Method | Returns | How it is computed |
|---|---|---|
| `hill_number(q)`, `hill_numbers(orders)` | Effective number of sequences of the mixture at order `q`. | Directly, over every sequence, for `q = 0` and every integer `q ≥ 2`. Other orders raise `UnsupportedOrder`; at `q = 1` use `entropy_bounds()`. |
| `alpha_diversity(q)` | The diversity of a typical member: the members' Hill numbers averaged by the power mean of order $1 - q$ (harmonic at $q = 2$). | Directly. |
| `beta_diversity(q)` | `.beta`, `.gamma`, `.alpha`: how many effectively distinct members the group has, from 1 to the number of members. | Directly, at the same orders. At `q = 1`, `.lower` and `.upper` give guaranteed bounds and `.beta` is `None`; `samples=` adds a sampled `.estimate`. |
| `entropy_bounds()` | Guaranteed lower and upper bounds on the Shannon entropy ($q = 1$), in nats: the weighted mean of the members' entropies, and that plus the entropy of the weights. | Directly. |
| `entropy_estimate(n=10000, seed=None)` | A sampled estimate of the Shannon entropy, with its standard error. | By sampling, and labelled as such. |
| `support_size()` | `.value`: the number of distinct sequences at least one member can produce. `support_size(exact=False)` returns only its bounds `.lower` and `.upper`, at no cost. | Directly, for up to 64 members. |

Why $q = 1$ is treated differently: see
[Computed, not simulated](/BEAM/explanations/computed-not-simulated/#bounds-instead-of-an-approximation).

## Scoring and drawing

| Method | Returns |
|---|---|
| `pseq(seq_or_list, log=False)`, `log_pseq(seq)` | Probability of a sequence (or of each in a list) under the mixture. |
| `sample(n, *, seed=None)` | `n` sequences drawn from the mixture. |
| `member_weights(length)` | The members' weights among sequences of one length, $w_i \pi_i(L) / \bar\pi(L)$. |

## Overlap between members

| Method | Returns |
|---|---|
| `gram_matrix()`, `log_gram_matrix()` | $G_{ij} = \sum_s P_i(s) P_j(s)$: the chance that one draw from member $i$ and one from member $j$ are the same sequence. |
| `bhattacharyya_coefficient(i, j)`, `bhattacharyya_matrix()` | $\sum_s \sqrt{P_i(s) P_j(s)}$, from 0 (no shared sequence) to 1 (identical members). |
| `hellinger_distance(i, j)` | $\sqrt{1 - \text{BC}_{ij}}$, a distance between two members. |
