---
title: Diversity at equal depth
description: Hill-number profiles of several repertoires, each resampled to a common sequencing depth first.
sidebar:
  order: 7
---

```python
from BEAM import depth_matched_hill_profiles

r = depth_matched_hill_profiles(
    [seqs_a, seqs_b],                # distinct sequences of each repertoire
    [counts_a, counts_b],            # integer template (or cell) counts
    n_bootstraps=20, seed=1,
)
r.depth                     # the common depth: the smallest total count
r.profiles                  # repertoire x resample x Hill order
r.resampling_band(5, 95)    # the spread over resamples
```

A diversity profile depends on how deeply a repertoire was sequenced, so comparing two
repertoires of different depths compares their depths as much as their diversity. This function
draws the smallest total count from each repertoire, with replacement and in proportion to its
counts, builds a model from each draw, and returns the Hill numbers of each model. See
[How diverse is my sample, really?](/BEAM/guides/diversity/).

```python
depth_matched_hill_profiles(repertoires, abundances=None, *, n_bootstraps=20, orders=None,
                            seed=None, weighting="abundance", n_workers=None, **config)
```

| Parameter | Meaning |
|---|---|
| `abundances` | Positive integer counts, one per sequence; `None` counts each sequence once. |
| `orders` | Hill orders to compute; default 0 to 3 in steps of 0.1. |
| `weighting` | `"abundance"` builds each resample's model from its counts; `"presence"` counts each distinct sequence drawn once. |
| `seed` | The same seed gives the same draws; `None` draws a fresh seed and records it in `r.seed`. |
| `n_workers` | How many resamples are fitted at once; the result does not depend on it. |
| `**config` | Passed to `BeamModel` (`order=`, `alphabet=`, …). |

List each sequence once and give counts, not frequencies; sequences the model cannot take
(outside its alphabet or longer than `max_length`) must be filtered out first. The common depth
depends on every repertoire passed, so compare profiles from one call.

`resampling_band(lo=5, hi=95)` returns the `lower`, `median` and `upper` profile of each
repertoire over its resamples. It describes how much the result moves between resamples; it is
not a confidence interval for the repertoire's true diversity.
