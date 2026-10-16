---
title: Repertoire likelihood
description: The probability of a whole repertoire under a model, and its factorisation into interpretable pieces.
sidebar:
  order: 8
---

```python
from BEAM import repertoire_likelihood, pair_likelihood

r = repertoire_likelihood(model, sequences, counts)
r.log_likelihood                      # ln P(repertoire | model), exact
for piece in r.pieces:
    print(piece.name, piece.log_prob, piece.exact)
```

A repertoire of $N$ cells is $N$ independent draws from the model, so

$$
\ln P(R \mid M) = \ln N! - \sum_s \ln c_s! + \sum_s c_s \ln P(s)
$$

where $c_s$ counts the cells carrying sequence $s$. The first two terms do not depend on the
model, so they cancel when two models score the same repertoire.

The total factors into a chain of pieces, each answering one question given $N$: how many
distinct sequences there are, which ones, how their counts are distributed, and the counts
themselves. Each piece reports whether it is exact, and its observed against expected value.

`pair_likelihood(model, seqs_a, counts_a, seqs_b, counts_b, atoms=...)` does the same for two
repertoires under one model, including how many sequences they share.

The result is `-inf` only if the model cannot represent one of the sequences at all: a
character outside its alphabet, or a length beyond its maximum.
