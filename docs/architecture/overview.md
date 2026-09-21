# How BEAM works

The conceptual half of what was one API page: the worked example, why two
anchors, the DAG, the alphabet, and how a probability is assembled. The API
reference lives under `docs/api/python/`.

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

- A **node** is a position `i` plus the last `K` residues emitted, sentinel-padded at the start, so there are `A**K` nodes per layer, where `A` is the alphabet's radix: its residues plus one sentinel, so 21 for amino acids and 6 for nucleotides. `K = order`: a state holds exactly the residues one context needs. This is what `n_states` reports. It used to be `max(order, 3)`, because a second expert of fixed order 3 had a context a narrower state could not supply; with one table per anchor there is nothing to be wider than.
- An **edge** emits the next residue and carries a weight fixed at build time, depending only on `(node, residue, i, L)`.
- A **path** through all `L` layers is in bijection with a length-`L` sequence.

The C-anchored expert does not break acyclicity. It reads residues to the right of position `i`, but its index `r = L-1-i` is a constant once `L` is fixed, so the dependence is absorbed into the edge weight at build time. Concretely the backward score is paid with a **lag**: emitting position `i` settles the backward score of position `i - K`, which only now has its complete right-hand context. That lag is exactly why a node carries `K` residues.

Because the score factorises over edges of a topologically ordered DAG, every analytical quantity below is an exact dynamic program with no sampling.

## The alphabet

Everything on this page holds over either alphabet. What changes is the radix
`A` — the residues plus one sentinel that pads contexts at the termini and is
never emitted — and the row width `R`.

| | symbols | `R` | `A` | states at the default order |
|---|---|---|---|---|
| `aa` | `ACDEFGHIKLMNPQRSTVWY` | 20 | 21 | `21**4` = 194,481 |
| `nt` | `ACGTN` | 5 | 6 | `6**6` = 46,656 |

An amino acid is a compressed codon: 21 values where three bases need
`6**3 = 216`. That is why the nucleotide default is order 6 rather than the
order 12 that would match the amino-acid model's context *span* — 12 is not
buildable, and 6 is two codons and cheaper than the amino-acid model already
is.

The dimensions are read from the model at run time, not from the
preprocessor, in every file but one. `src/native/dp/edge_weights.c` is
compiled once per alphabet through `edge_weights_kernels.h`, because it is
the file every dynamic program reaches its inner loop through and a runtime
radix there cost 31–51%. That file's header comment records the four ways of
avoiding the duplication that were measured and did not work, and why OpenMP
is the reason.

## How a probability is assembled

```
ln P(s) = ln pi(L) + sum over residues of ( 0.5 * forward + 0.5 * backward ) - ln Z(L)
```

- `pi(L)` is the observed length distribution.
- Each per-residue term is one table lookup per anchor, at exact depth: `i` from the N-terminus and `L-1-i` from the C-terminus. Those two depths sum to `L-1`, which is why neither table needs a length axis of its own.
- `Z(L)` is the total weight of every possible length-`L` path, computed once per length by a forward sweep. It is needed because a geometric mean of two experts is not normalised on its own, and with equal anchor weights it has a closed form: `Z(L)` is the **Bhattacharyya coefficient** between the N-anchored and C-anchored chains, so `Z(L) <= 1` with equality exactly when the two readings agree, and `-log Z(L)` measures how much they disagree at that length.

`sum over all R**L sequences of P(s|L)` is exactly 1, for an alphabet of `R` residues; the test suite verifies this by brute-force enumeration at short lengths — over all 8,000 length-3 peptides and all 125 length-3 nucleotide sequences — including on degenerate corpora of a single sequence.

