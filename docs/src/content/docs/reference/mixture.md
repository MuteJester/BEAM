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

The distribution $\bar P(s) = \sum_i w_i P_i(s)$: pick a member with probability $w_i$, then a
sequence from that member. This is different from pooling counts with `merged`: pooling can
join stretches of sequence from different members into sequences no member produces, so a
pooled model is broader than the mixture.

All members must share alphabet, order and maximum length.

| Method | Returns | Exactness |
|---|---|---|
| `hill_number(q)` | Effective number of sequences of the mixture at order `q`. | Exact for `q = 0` and integer `q ≥ 2`; `q = 1` is refused (`UnsupportedOrder`). |
| `beta_diversity(q).beta` | How many effectively distinct members the group has, between 1 and the number of members. | Exact at the same orders. At `q = 1` it returns `beta = None` with exact lower and upper limits instead. |
| `entropy_bounds()` | Exact lower and upper bounds on the Shannon entropy ($q = 1$). | Exact bounds. |
| `entropy_estimate(n, seed=)` | A sampled estimate of the Shannon entropy, with its standard error. | Labelled as sampled. |
| `support_size()` | Number of distinct sequences any member can produce. | Exact. |

Why $q = 1$ is treated differently: see [Computed, not simulated](/BEAM/explanations/computed-not-simulated/#bounds-instead-of-an-approximation).
