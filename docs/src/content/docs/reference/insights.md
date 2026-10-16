---
title: Insights and logos
description: Sequence logos of a model, overall or restricted to common or rare sequences, as arrays ready to plot.
sidebar:
  order: 9
---

```python
logo = model.insights.logo(15)                       # all CDR3s of length 15
common = model.insights.logo(15, pseq_above=1e-8)    # only the common ones
rare = model.insights.logo(15, pseq_below=1e-8)      # only the rare ones

logo.consensus_sequence
logo.as_dict()            # {residue: heights}, ready for logomaker
logo.fraction             # share of length-15 sequences this logo describes
```

For each position of a given length, a logo gives the distribution of residues and its
information content, in bits (`height_bits`) or nats (`height_nats`). It is computed over every
sequence of that length, so no small-sample correction applies. Information is measured
against a uniform background.

`pseq_above` and `pseq_below` restrict a logo to sequences above or below a probability; the two
fractions add up to one. `surprisal_above` / `surprisal_below` do the same in nats.
`model.insights.logos(...)` returns several logos at once.
