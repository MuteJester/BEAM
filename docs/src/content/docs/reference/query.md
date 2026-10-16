---
title: BeamQuery
description: Exact answers to questions about sets of sequences, such as the probability that a CDR3 of length 14 contains a motif.
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
condition, so a partly built one can be reused.

| Builder | Selects sequences where |
|---|---|
| `at(i, "S")` | position `i` is `S`. |
| `oneof(i, "FWY")` | position `i` is one of the residues. |
| `exclude(i, "C")` | position `i` is not `C`. |
| `run(i, "CASS")` | the residues starting at `i` are exactly `CASS` (a prefix at `i = 0`). |
| `contains("QYF")` | the motif occurs somewhere. |
| `count("FWY", 2, 4)` | between 2 and 4 positions hold one of the residues. |
| `a & b`, `a \| b`, `~a` | both, either, not. |

Position restrictions (`at`, `oneof`, `exclude`, `run`) cost nothing extra. Motifs and counts
(`contains`, `count`) and `|`, `~` multiply the cost by the size of the pattern's automaton;
`q.cost(c)` tells you in advance.

| Method | Returns |
|---|---|
| `probability(c)`, `log_probability(c)` | $P(C \mid L)$; `-inf` in log for an impossible condition. |
| `marginals(c)` | Per position, the distribution of residues given the condition. |
| `pair_joint(c, i, j)` | The joint distribution of two positions given the condition. |
| `entropy(c)`, `position_entropy(c)` | Entropy of the whole sequence, or per position, given the condition. |
| `mutual_information(c)` | Mutual information between every pair of positions, within the condition. |
| `position_information(c)` | How much each position tells you about whether the condition holds. |
| `expect(table, c)`, `expect_count("FWY", c)` | Mean and variance of a per-position score, or of a residue count. |
| `insertions(seq)`, `deletions(seq)` | Every sequence one insertion or deletion away, with its probability. |

Conditioning on an impossible condition raises `ImpossibleCondition`; a question with no exact
answer raises `UnsupportedQuery`; one over the memory budget raises `QueryTooLarge`.
