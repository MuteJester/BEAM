---
title: Repertoire likelihood
description: The probability of a whole repertoire under a model, its factorisation into interpretable pieces, and the observed clone-size spectrum against the model's expectation.
sidebar:
  order: 8
---

```python
from BEAM import repertoire_likelihood, pair_likelihood, abundance_spectrum

r = repertoire_likelihood(model, sequences, counts)
r.log_likelihood                      # ln P(repertoire | model)
for piece in r.pieces:
    print(piece.name, piece.log_prob, piece.exact)
```

A repertoire of $N$ cells is $N$ independent draws from the model, so

$$
\ln P(R \mid M) = \ln N! - \sum_s \ln c_s! + \sum_s c_s \ln P(s)
$$

where $c_s$ counts the cells carrying sequence $s$. The first two terms do not depend on the
model, so they cancel when two models score the same repertoire. Build the model with
`abundances=` and pass the same kind of counts, so that $P(s)$ is the chance that one cell
carries $s$; counts must be positive integers, and repeated sequences are merged.

## Pieces

The total factors, given $N$, into a chain of pieces, each answering one question, available as
`r.pieces` or by name with `r.piece(name)`:

| Piece | Asks |
|---|---|
| `richness` | how many distinct sequences there are; also gives `observed` against `expected`. |
| `identity` | which sequences they are. |
| `profile` | how the counts are distributed (how many sequences are seen once, twice, …); also gives `observed` against `expected`. |
| `assignment` | which sequence has which count. |

Each piece's `exact` flag says whether it is computed directly (`True`) or by a saddlepoint
approximation (`False`), with `error` its size in nats. The pieces add up to `log_likelihood`.
A piece that cannot be computed, for example where the model makes an observed abundance pattern
nearly impossible, raises `LikelihoodPieceError`; the total likelihood is unaffected.

```python
repertoire_likelihood(model, sequences, counts, *, bins=8192, profile_edges="auto",
                      atoms=None, atoms_coarse=None)
```

When scoring many repertoires against one model, compute the model's probability spectrum once
and pass it: `atoms=model.pseq_analysis().atoms(8192)` and
`atoms_coarse=model.pseq_analysis().atoms(4096)`.

## Two repertoires

`pair_likelihood(model, sequences_1, counts_1, sequences_2, counts_2)` scores two repertoires
under one model; `log_likelihood` is the sum of the two totals. Its pieces (`richness`,
`sharing`, `identity`, `profile`, `assignment`) add up to `poissonized_log_likelihood`, with
each repertoire's total left free, and `sharing` compares the number of sequences the two share
(`n_shared`) with the model's expectation.

## The clone-size spectrum

`abundance_spectrum(model, sequences, counts)` gives, for each clone size $k$, the observed
number of sequences seen $k$ times against the model's expected number and its standard
deviation, computed directly from the model, and their $z$-score (`nan` where fewer than 10
sequences are expected).

The likelihood is `-inf` only if the model cannot represent one of the sequences at all: a
character outside its alphabet, or a length beyond its maximum.
