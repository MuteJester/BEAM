---
title: Repertoire distance
description: compare(a, b), the exact local-window distance between two models, split into length and pattern terms.
sidebar:
  order: 4
---

```python
from BEAM.analytics.distance import compare

result = compare(a, b, width=None, top_k=100)
```

An exact distance between two models, whose square splits into a part due to different
sequence lengths and a part due to different local patterns at each length and position. See
[How different are two repertoires?](/BEAM/guides/compare-repertoires/).

The models must share the alphabet and order; their maximum lengths may differ. `width` is the
length of the local windows compared, from 1 to `order + 1` (the default, `order + 1`). `top_k`
keeps that many of the largest pattern terms; the distance itself is always complete.

## Result

| Field | Meaning |
|---|---|
| `distance` | The distance; 0 for identical models. |
| `squared_distance` | Its square, which is what the parts add up to. |
| `length_squared_distance` | The part due to different length distributions. |
| `pattern_squared_distance` | The part due to different local patterns. |
| `lengths` | Per length: both models' length probabilities, the length term and the pattern term at that length. |
| `windows` | Per length and window position: the pattern term of that window. |
| `patterns` | The `top_k` strongest single patterns: length, start position, the pattern, its probability (given the length) in each model, and its contribution. |
| `remainder` | The pattern part not covered by the returned `patterns`. |
| `coverage` | Share of the pattern part the returned patterns explain. |

`result.group_overlaps()` groups overlapping returned patterns into longer consensus motifs, for
reading; it does not change the distance.

`length_squared_distance + pattern_squared_distance` equals `squared_distance`, and the
returned patterns plus `remainder` equal the pattern part, both to floating-point precision.
