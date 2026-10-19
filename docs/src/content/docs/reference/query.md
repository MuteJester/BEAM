---
title: BeamQuery
description: Answers computed over every sequence of a length, such as the probability that a sequence of length 14 contains a motif.
sidebar:
  order: 6
---

```python
from BEAM import BeamQuery

q = BeamQuery(model)
c = q.condition(14).at(4, "G").contains("CASS")
q.probability(c)          # P(condition | length 14)
```

A **condition** describes a set of sequences of one length. Each builder returns a new
condition, so a partly built one can be reused. Every answer is computed over all sequences of
that length, without sampling.

| Builder | Selects sequences where |
|---|---|
| `at(i, "S")` | position `i` is `S`. |
| `oneof(i, "FWY")` | position `i` is one of the residues. |
| `exclude(i, "C")`, `exclude(i, ["C", "W"])` | position `i` holds none of them (a string is one residue; pass a list for several). |
| `run(i, "CASS")` | the residues starting at `i` are exactly `CASS` (a prefix at `i = 0`). |
| `contains("QYF")` | the motif occurs somewhere. |
| `count("FWY", 2, 4)` | between 2 and 4 positions, inclusive, hold one of the residues. |
| `a & b`, `a \| b`, `~a` | both, either, not. |

Position restrictions (`at`, `oneof`, `exclude`, `run`) cost nothing extra. Motifs and counts
(`contains`, `count`) and `|`, `~` multiply the cost by the size of the pattern's automaton;
`q.cost(c)` returns the states, edges and bytes needed, in advance.

| Method | Returns |
|---|---|
| `probability(c)`, `log_probability(c)` | $P(C \mid L)$; `-inf` in log for an impossible condition. |
| `marginals(c)` | Per position, the distribution of residues given the condition. |
| `pair_joint(c, i, j)` | The joint distribution of two positions given the condition. |
| `entropy(c)`, `position_entropy(c)` | Entropy of the whole sequence, or per position, given the condition, in nats. |
| `conditional_entropy(c, i, j)` | $H(X_i \mid X_j, C)$: what remains uncertain at position $i$ once position $j$ is known. |
| `mutual_information(c)` | Mutual information between every pair of positions, within the condition. |
| `total_correlation(c)` | $\sum_i H(X_i) - H(X)$: everything the positions share, in nats. |
| `position_information(c)` | How much each position tells you about whether the condition holds. |
| `position_divergence(c)` | Per position, how far the condition shifts the residue distribution (KL divergence, in nats). |
| `surprisal_variance(c)` | Variance of $-\ln P(s)$ given the condition. |
| `expect(table, c)`, `expect_count("FWY", c)` | Mean and variance of a per-position score, or of a residue count. |
| `insertions(seq)`, `deletions(seq)` | Every sequence one insertion or deletion away, most probable first, with its log-odds against `seq` and its share among the neighbours. |

Conditioning on an impossible condition raises `ImpossibleCondition`; a malformed question (a
position outside the length, a residue outside the alphabet, a length the model does not
produce) raises `UnsupportedQuery`; one over the memory budget raises `QueryTooLarge`.
