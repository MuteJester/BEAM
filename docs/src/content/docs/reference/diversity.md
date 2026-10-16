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
draws each repertoire down to the smallest total count, builds a model from each draw, and
returns the Hill numbers of each model. Any extra keyword (`order=`, `alphabet=`, …) is passed to
`BeamModel`.

The same seed gives the same draws. `resampling_band` describes how much the result moves
between resamples; it is not a confidence interval for the repertoire's true diversity.
