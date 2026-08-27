# How BEAM works

The conceptual half of what was one API page: the worked example, why two
anchors, the DAG, and how a probability is assembled. The API reference
lives under `docs/api/python/`.

## The worked example

Every number on this page comes from the model built below, so each snippet can be pasted and checked. It is a synthetic CDR3-shaped corpus, not a real repertoire, and the numbers are here to be reproducible rather than to be quoted.

```python
import random
from BEAM import BeamModel

AA = "ACDEFGHIKLMNPQRSTVWY"
J_MOTIFS = ("EQYF", "EQFF", "NTIYF", "ETQYF", "YGYTF")
rng = random.Random(11)

sequences = []
for _ in range(1500):
    length = rng.randint(11, 18)
    j = rng.choice([m for m in J_MOTIFS if len(m) + 5 <= length])
    fill = length - 4 - len(j)
    sequences.append("CASS" + "".join(rng.choice(AA) for _ in range(fill)) + j)

model = BeamModel(sequences, order=3)
```

## Why two anchors

A single left-to-right chain learns the V-germline start well and has forgotten it by the time it reaches the J motif, so it emits terminations that no J gene produces. Anchoring one expert at each terminus recovers germline structure without ever being told which genes exist, because the V germline **is** the front-anchored statistic and the J motif **is** the back-anchored statistic.

The effect is easy to see. Swapping the terminal residue of a plausible CDR3 for one no J gene produces costs millions of fold:

```python
model.pseq("CASSLGQGAEQYF")   # 2.824e-09
model.pseq("CASSLGQGAEQYW")   # 1.143e-15
```

A left-to-right model has no strong opinion at that position. The C-anchored expert has a very strong one.

## The DAG

For a fixed length `L` the model is a layered directed acyclic graph.

- A **node** is a position `i` plus the last `K` residues emitted, sentinel-padded at the start, so there are `21**K` nodes per layer. `K = max(order, 3)`, not `order`: the mid expert is always order 3, so a narrower state could not supply its context. This is what `n_states` reports.
- An **edge** emits the next residue and carries a weight fixed at build time, depending only on `(node, residue, i, L)`.
- A **path** through all `L` layers is in bijection with a length-`L` sequence.

The C-anchored expert does not break acyclicity. It reads residues to the right of position `i`, but its index `r = L-1-i` is a constant once `L` is fixed, so the dependence is absorbed into the edge weight at build time. Concretely the backward score is paid with a **lag**: emitting position `i` settles the backward score of position `i - K`, which only now has its complete right-hand context. That lag is exactly why a node carries `K` residues.

Because the score factorises over edges of a topologically ordered DAG, every analytical quantity below is an exact dynamic program with no sampling.

## How a probability is assembled

```
ln P(s) = ln pi(L) + sum over residues of ( 0.5 * forward + 0.5 * backward ) - ln Z(L)
```

- `pi(L)` is the observed length distribution.
- Each per-residue term is itself a log-linear mix of three experts (see [Configuration](#configuration)).
- `Z(L)` is the total weight of every possible length-`L` path, computed once per length by a forward sweep. It is needed because a geometric product of two experts is not normalised on its own.

`sum over all 20**L sequences of P(s|L)` is exactly 1; the test suite verifies this by brute-force enumeration at short lengths, including on degenerate corpora of a single sequence.

