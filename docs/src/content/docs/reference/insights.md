---
title: Insights and logos
description: Sequence logos of a model, overall or restricted to common or rare sequences, as arrays ready to plot.
sidebar:
  order: 9
---

```python
logo = model.insights.logo(15)                       # all sequences of length 15
common = model.insights.logo(15, pseq_above=1e-8)    # only the common ones
rare = model.insights.logo(15, pseq_below=1e-8)      # only the rare ones

logo.consensus_sequence
logo.fraction             # share of the probability at length 15 this logo describes
pd.DataFrame(logo.as_dict())                         # {residue: heights}, ready for logomaker
```

For each position of a given length, a logo gives the distribution of residues and its
information content (`information_bits`, `information_nats`), with the stack heights to draw
it from in `height_bits` or `height_nats` (one row per position, one column per residue). It is
computed over every sequence of that length, so no small-sample correction applies. Information
is measured against a uniform background; for nucleotide models, which have five residues
including `N`, pass `reference_residues=4` for the usual two-bit axis.

## Common and rare sequences

`pseq_above` and `pseq_below` restrict a logo to sequences above or below a probability; the two
`fraction`s, each a share of the probability at that length, add up to one.
`surprisal_above` and `surprisal_below` take the threshold as a surprisal $-\ln P(s)$ in nats
instead, so `surprisal_above` selects the rare sequences, like `pseq_below`. Give at most one
condition.

These conditional logos are computed on a fine grid of probabilities: `resolution_error`
reports the effect of the grid on a logo, `max_resolution_error=` refuses a logo whose grid
error would exceed it, and `bins=` or `spacing=` set the grid.

`model.insights.logos(15, edges=[...])` returns several logos at once, one for each band between
consecutive surprisal edges (in nats), commonest first. `model.insights.cost(15, ...)` returns
the states, edges and memory a logo needs, checked against the
[memory budget](/BEAM/reference/runtime/).

| Member | Meaning |
|---|---|
| `consensus_sequence`, `consensus_probability` | The most probable residue at each position, and its probability there. |
| `information_bits`, `information_nats` | Information content per position. |
| `height_bits`, `height_nats` | Stack heights per position and residue. |
| `fraction` | Share of the probability at this length the logo describes. |
| `n_support` | Per position, how many residues carry any probability. |
| `as_dict(unit="bits")` | `{residue: heights}`. |
