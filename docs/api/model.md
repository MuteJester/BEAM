---
tags:
  - BeamModel
description: Complete API reference for BeamModel (Ensemble of Dual Graph Experts), a bidirectional anchored DAG over CDR3 amino-acid sequences with exact analytics.
search:
  boost: 2
---

# BeamModel

**Ensemble of Dual Graph Experts.**

A probability model over CDR3 amino-acid sequences in which **every residue is scored twice**: once by an N-anchored expert that indexes it by its distance from the front of the sequence, and once by a C-anchored expert that indexes it by its distance from the back. The two readings are combined geometrically and normalised per length by an exact forward dynamic program.

BeamModel consumes **only** the amino-acid sequence and its observed abundance. It never sees a V or J gene call, a donor identity, or a sharing statistic.

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

## Constructor

```python
BeamModel(
    sequences,
    *,
    abundances=None,
    order=4,
    depth_cap=15,
    coefficients=(0.60, 0.25, 0.15),
    kappa=None,
    kappa_wide=None,
    kappa_len=None,
    weights=(0.5, 0.5),
    max_length=27,
    strict=False,
    sparse=None,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `list[str]` | Training CDR3 strings. Anything empty, longer than `max_length`, or containing a non-amino-acid character is skipped; see `n_skipped`. |
| `abundances` | `list[float]` | Per-sequence observed counts (default: all 1). A non-positive or non-finite weight skips the sequence. |
| `order` | `int` | Context order of the wide expert, **2 to 4**. Order 4 is the measured optimum. |
| `depth_cap` | `int` | Depth at which the wide expert stops resolving position exactly, **1 to 26**. Clamped to `max_length - 1`. |
| `coefficients` | `tuple` | Log-linear mixing weights for the wide, mid and length-conditioned experts. **Must sum to 1.** |
| `kappa` | `tuple \| None` | Hierarchical backoff strengths for orders 1-3. Defaults to `(20, 20, 20)`, or all zero under `strict=True`. |
| `kappa_wide`, `kappa_len` | `float \| None` | Backoff strength for the wide and length-conditioned experts. Defaults to 20 and 200, or zero under `strict=True`. |
| `weights` | `tuple` | Weight on the N-anchored and C-anchored readings. Both must be non-negative. |
| `max_length` | `int` | Longest sequence the model represents, **2 to 27**. The floor is 2, not 1, because `depth_cap` must be at least 1 and strictly below `max_length`. |
| `strict` | `bool` | Closed-world scoring. Removes the order-0 add-one and probability floor; defaults all backoff strengths to zero. |
| `sparse` | `bool \| None` | Storage selection. `None` automatically uses sparse storage exactly when `strict=True` and all five backoff strengths are zero; `False` forces dense storage. |

!!! warning "The coefficient sum is not a free parameter"
    It is a pure cross-length calibration knob. Moving it off 1 leaves *within-length* ranking untouched while wrecking *global* ranking: at sum 1.15 average precision falls from 0.847 to 0.830, and at 1.45 to 0.739. A sum other than 1 is rejected at construction rather than silently mis-calibrating.

### Configuration

Each direction is a log-linear product of three abundance-counted, hierarchically smoothed tables:

| expert | context | depth resolution | coefficient |
|---|---|---|---|
| wide | previous `order` residues | fine, capped at `depth_cap` | 0.60 |
| mid | previous 3 residues | full | 0.25 |
| length-conditioned | previous 2 residues + total length | full | 0.15 |

The wide table carries sharp motif memory; the length-conditioned table is the only one that knows `L` explicitly, which is what lets a 12-mer and a 20-mer place their junction differently.

`strict=True` makes an unobserved transition genuinely impossible rather than
merely improbable: the order-0 base is a raw MLE and zero probabilities are
not replaced by a floor. Its smoothing defaults are all zero, so the usual
strict model has exact observed-transition support. Supplying a positive
backoff strength explicitly is supported, but can carry mass into an
otherwise unobserved row.

Sparse storage is a representation of that exact closed-world model, not a
different estimator. It stores occupied 20-residue rows and preserves the
same counts and probabilities as dense strict storage. The default
`sparse=None` enables it only when `strict=True` and `kappa`, `kappa_wide`, and
`kappa_len` are all zero. `sparse=False` forces dense storage for equivalence
checks and reproducibility. Explicit `sparse=True` with open-world scoring or
positive backoff raises `ValueError`, because omitting empty rows in either
case would change the model.

`model.config` returns the configuration actually built, including the
resolved `strict` and `sparse` booleans. This is what to inspect after
construction — `depth_cap` in particular may have been clamped.

### There is no pruning

Earlier drafts carried a `prune_threshold`. It has been **removed**: the support-size and cost arguments it existed to serve are addressed instead by reading [`path_count`](#path_count) with its caveat and by sizing spectra deliberately. Passing `prune_threshold=` is a `TypeError`.

## Building without holding the corpus

`BeamModel(sequences, ...)` needs every sequence in memory at once. Three entry points do not.

### BeamModel.builder

```python
BeamModel.builder(**config) -> Builder
```

An `Builder` accumulates counts batch by batch and finalizes once. Counts are additive and `finalize` recomputes the tables from scratch, so the result is **byte-identical** to a single call over the concatenated batches — a property the test suite asserts rather than assumes.

```python
b = BeamModel.builder(order=2, max_length=20)
b.observe(sequences[:700])
b.observe(sequences[700:])
streamed = b.build()
```

| method | description |
|---|---|
| `observe(sequences, abundances=None)` | Accumulate one batch. Does **not** finalize. Returns `self`, so calls chain. |
| `build()` | Finalize and hand the counts over as an `BeamModel`. The builder is spent afterwards. |
| `snapshot()` | A finalized **copy** of the counts so far, without consuming them. The builder keeps accumulating and the snapshot is unaffected by everything that follows. |
| `n_observed`, `n_skipped`, `total_weight` | Running totals. |
| `n_nonzero_cells` | Count cells occupied so far. |
| `memory_bytes` | Approximate resident bytes of the tables accumulated so far. |

`snapshot()` is what a convergence curve needs: ingest, measure, keep ingesting. It costs a full table build and a second copy of the counts, so take one per checkpoint you mean to measure, not per batch.

### BeamModel.from_sequence_file

```python
BeamModel.from_sequence_file(path, *, sep="\t", sequence_column=0,
                        abundance_column=1, has_header=False,
                        chunk_size=1_000_000, **config) -> BeamModel
```

Trains from the `sequence<TAB>abundance` file the graph builders in this library take. A line with only one field is read as a weight of 1, so a plain list of sequences works too; pass `abundance_column=None` to ignore the weight column entirely.

### BeamModel.from_parquet and BeamModel.from_dataframe

```python
BeamModel.from_parquet(path, *, sequence_column="cdr3_aa", abundance_column=None,
                  batch_size=1_000_000, **config) -> BeamModel
BeamModel.from_dataframe(data, *, sequence_column="cdr3_aa",
                    abundance_column=None, **config) -> BeamModel
```

`from_parquet` reads one record batch at a time (needs `pyarrow`), so a corpus far larger than memory trains in constant space. `from_dataframe` is duck-typed on `.columns` / `.column_names` / `dict`, so a pandas DataFrame, an Arrow table and a plain dict of lists all work without this library acquiring a DataFrame dependency to recognise one.

## Scoring

### log_pseq

```python
log_pseq(sequence) -> float
```
Natural log of the sequence probability. Returns `-inf`, rather than raising, for a sequence the model cannot produce: empty, longer than `max_length`, containing a non-amino-acid character, or of a length never observed in training.

### log_pseq_batch

```python
log_pseq_batch(sequences) -> np.ndarray
```
Vectorised form, parallel where OpenMP is available. Same `-inf` convention.

### pseq

```python
pseq(sequence, *, log=False) -> float | np.ndarray
```
The probability on the linear scale, for one sequence or many. A single string returns a `float`; anything else returns an array.

```python
model.pseq("CASSLGQGAEQYF")                       # 2.824e-09
model.pseq(["CASSLGQGAEQYF", "CASSLGQGAEQYW"], log=True)
```

!!! note "The default is the LINEAR probability"
    which is the opposite of `FlashBackGraph.pseq`, whose default is `log=True`. BeamModel named the log form `log_pseq` from the start and that is what its own callers use, so flipping this default would silently rescale every existing script rather than break it. Code ported from a graph model should carry its `log=True` across explicitly.

### Streaming and file-backed scoring

Scoring a corpus should not require a Python list of every sequence in it.

```python
log_pseq_chunks(sequences, *, chunk_size=1_000_000)   # generator of arrays
log_pseq_file(path, *, sep="\t", column=0, has_header=False,
              chunk_size=1_000_000, out=None, dtype=np.float64) -> np.ndarray
log_pseq_parquet(path, *, column="cdr3_aa", batch_size=1_000_000,
                 out=None, dtype=np.float64) -> np.ndarray
```

`log_pseq_chunks` is the primitive: it scores an arbitrary iterable and yields one array per chunk, never holding more than `chunk_size` sequences. The other two are it, pointed at a file.

`out` takes a preallocated array or `numpy` memmap and fills it in place, which is the pattern a corpus-scale scan wants:

```python
import numpy as np
scores = np.lib.format.open_memmap("pseq.npy", mode="w+",
                                   dtype=np.float32, shape=(n_rows,))
model.log_pseq_parquet("corpus.parquet", out=scores)
```

`out` must be exactly the right length; a mismatch raises rather than silently filling a prefix. Row order is the file's, so the result lines up positionally with any other column of the same file — which is how two aligned release parquets have to be read.


### explain

```python
explain(sequence) -> dict
```
Per-residue decomposition: each residue's N-anchored and C-anchored **natural-log** probability, before the mixing weights are applied. This is what makes the model legible.

Unlike `log_pseq`, this **raises** `ValueError` for a sequence it cannot encode, naming the rule that was hit — the empty string, over `max_length`, or a character outside the twenty amino acids. Use `log_pseq` when a verdict rather than a decomposition is what is wanted.

Returned keys: `sequence`, `residues`, `forward`, `backward`, `depth_from_n`, `depth_from_c`, `log_pseq`, `length_is_producible`.

```python
import math
e = model.explain("CASSLGQGAEQYF")
for i, (r, f, b) in enumerate(zip(e["residues"], e["forward"], e["backward"])):
    print(i, r, round(-f / math.log(2), 2), round(-b / math.log(2), 2))
```

Converted to bits of surprisal, so larger means more surprised:

| pos | res | N-anchored (bits) | C-anchored (bits) |
|---|---|---|---|
| 0 | C | 0.00 | 0.01 |
| 3 | S | 0.00 | 2.47 |
| 8 | A | 6.70 | 4.50 |
| 11 | Y | 0.84 | 0.71 |
| 12 | F | 0.09 | 0.00 |

The N-anchored expert is confident over the conserved start, the C-anchored expert over the J motif, and both are surprised across the junction, which is the part that is actually random.

The decomposition **reconstructs the score exactly**, so it can be trusted as an explanation rather than an illustration:

```python
wf, wb = model.config["weights"]
raw = sum(wf * f + wb * b for f, b in zip(e["forward"], e["backward"]))
L = len(e["sequence"])
raw + model.log_length_prior(L) - model.log_partition(L)   # == e["log_pseq"]
```

!!! note "One exception, and it is flagged"
    A sequence of a length the model never observed has `log_pseq == -inf` while its per-residue terms stay finite: the length prior, not any residue, is what rules it out. `length_is_producible` says which case you are in, and the identity above holds only when it is `True`.

## Per-length quantities

```python
log_partition(length) -> float      # ln Z(L)
log_length_prior(length) -> float   # ln pi(L)
length_pmf() -> np.ndarray          # the length prior as a vector, indexed by length
```

`length` must be between 0 and `max_length`; anything else raises `ValueError` rather than returning `-inf`. A length **inside** that range that the model never saw returns `-inf`, which is the real answer.

## Analytics

All exact, all dynamic programs over the DAG, none sampled.

### path_count

```python
path_count(length=None) -> float
```
Distinct sequences the model can produce; `None` pools over every length.

!!! danger "Read this before quoting a support size"
    In the default open-world model this is exactly `20 ** length`, because hierarchical backoff leaves every cell strictly positive and so nothing is dead. **That number describes the smoother, not the grammar.** For the corpus above the support is `2.76e23`.

    Under `strict=True` with zero backoff, dead transitions remain dead and
    `path_count` instead reports the exact support of the observed-transition
    DAG. Sparse and dense strict storage return the same value.

    It is kept because it is the exact `q=0` corner of the Hill family and because it is what the counting spectrum integrates to, so the two have to agree. `effective_diversity()` (`D1`) is the number to quote if you are quoting one — here `1.04e09` against that `2.76e23`.

### The Mellin transform

```python
log_mellin(q) -> float      # ln sum_s P(s)**q
power_sum(q) -> float       # the same on the linear scale
cgf(t) -> float             # ln E[exp(t ln P)], equal to log_mellin(1 + t)
```
The generator everything else here is built from. `q=0` gives the log support size, `q=1` gives 0, and the derivatives with respect to `q` are the cumulants of `ln P`.

### Diversity

```python
entropy(*, base=math.e) -> float
hill_number(alpha) -> float
hill_numbers(orders) -> np.ndarray
hill_curve(orders=None) -> dict   # {"orders", "hill"}
diversity_profile() -> dict       # D0, D1, D2
effective_diversity() -> float    # D1
```
`alpha=0` gives the support size, `alpha=1` the exponential of the entropy, `alpha=2` the inverse collision probability. `D0` inherits the caveat on `path_count`; `D1` is the number to quote.

### Surprisal moments

```python
surprisal_moments(*, q=1.0, order=4) -> np.ndarray   # raw moments of -ln P, order+1 values
surprisal_cumulants(*, q=1.0, order=4) -> np.ndarray # mean, variance, ... (order values)
surprisal_stats(*, q=1.0) -> dict   # mean, variance, std, skewness, excess_kurtosis
```
Exact, by a jet dynamic program that carries the first `order` powers of the accumulated path weight alongside the mass. At `q=1` the mean surprisal **is** the entropy, which the test suite checks against the separate entropy DP and against an exhaustive sum over every short sequence. `order` is capped at 16.

### Extremes

```python
surprisal_range() -> (float, float)
dynamic_range(*, base=10.0) -> float
```
The smallest and largest surprisal any producible sequence attains, by exact min-plus and max-plus sweeps rather than by sampling. The most probable sequence from `top_k_sequences` sits exactly at the low end, which is asserted in the tests.

### Per-length decomposition

```python
length_marginals() -> dict
# keys: length, counting, generated, mean_surprisal, variance
```
`counting` sums to the support size and `generated` to 1, which is what makes this a decomposition rather than a summary. `generated` **is** `length_pmf()`. The length-weighted mean of `mean_surprisal` reconstructs the entropy.

### length_moments

```python
length_moments(*, q=1.0, order=4) -> dict
# keys: length, share, moments   (moments has shape (max_length+1, order+1))
```
The `q`-tilted surprisal moments of every length, **unmixed**. `share[L]` is that length's share of the `q`-tilted law — at `q=1` the length prior itself — and `moments[L, k]` is `E_q[(-ln P)**k | length L]`.

This is `surprisal_moments` with the cross-length mixture left undone, and the two are related by exactly that mixture:

```python
lm = model.length_moments(q=1.0, order=4)
mu = model.surprisal_moments(q=1.0, order=4)
float((lm["share"] * lm["moments"][:, 2]).sum())   # == mu[2]
```

The C writes each length's **share** into slot 0 rather than the constant 1 a raw jet would put there, precisely so that identity is checkable from the output alone against a separate DP. The tests assert it at `q` = 0, 0.5, 1 and 2.

It is also the numerically stable form: it never multiplies by `M(q)`, so unlike `pseq_analysis().length_derivatives` it cannot overflow below `q = 1`.

### Where the surprisal sits

```python
surprisal_by_depth(*, q=1.0) -> dict
# keys: depth, by_depth, length_constant, total
```
The expected contribution of each trellis layer, plus the per-length normalisation `E[ln Z(L) - ln pi(L)]` reported separately as `length_constant` because it belongs to the choice of length rather than to any position. `total` is their sum and equals the mean surprisal exactly.

The sink term — the backward scores of the final `max(order, 3)` residues, which never get a full right-hand context during the walk — is attributed to the **last** layer, so `by_depth[-1]` carries more than one position's worth of cost.

### Topology

```python
is_dag -> bool               # always True, by construction
topology(length) -> dict
# keys: length, n_nodes, n_edges, nodes_by_layer, max_layer_nodes,
#       mean_layer_nodes, density
```
Live nodes and edges of the trellis at one length, where live means both forward-reachable and able to reach a valid ending. `length` must be at least 1 and must be a length the model actually generates; there is no pooled-over-all-lengths trellis, so `topology(0)` raises.

### Most probable sequences

```python
top_k_sequences(k=20, *, length=None) -> list[tuple[str, float]]
```
Exact, **not a beam**. The search is guided by the backward values, which are an exact completion weight rather than a heuristic, so the `n`-th result really is the `n`-th most probable sequence. Checked against exhaustive enumeration in the tests. `k=0` returns `[]`; a negative `k` raises.

### spectrum

```python
spectrum(*, measure="generated", length=None, bins=1024, spacing=None) -> dict
# keys: surprisal, weights, spacing, total, measure
```
The exact surprisal spectrum by forward grid transport.

- `measure="counting"` weights every supported sequence 1, so the spectrum integrates to the support size.
- `measure="generated"` weights each sequence by its own probability, so it integrates to 1.

**"Exact" means no sampling, not "every bin is exact."** Mass is split linearly between adjacent bins at *every* layer, so a sequence's weight diffuses over a few bins rather than landing on the two adjacent to its own surprisal. What the splitting preserves exactly is the total and the **first moment**: the mean surprisal read off the grid matches `length_marginals()["mean_surprisal"]` to machine precision, while the variance comes out inflated by at most `(L + 1) * spacing**2 / 6` — one split per layer plus the sink — which shrinks as the *square* of the bin width. Read shape at grid resolution, and take moments from `surprisal_moments` when they matter.

!!! note "The bin floor is higher than it looks"
    With `spacing=None` the transport reserves `max_length + 9` bins of headroom for that linear splitting, so `bins` must be at least `max_length + 10` — 37 for a default model. Pass an explicit `spacing` to use a smaller grid.

!!! note "Cost"
    Cost scales with the number of live states per layer, and without pruning every state is live at every layer: 194,481 of them at order 4, 9,261 at order 2 or 3. A pooled spectrum over a full-range order-4 model needs several GB and minutes. Ask for one length at a time, or build at a lower `order`, when that matters.

### spectrum_pair

```python
spectrum_pair(*, length=None, bins=1024, spacing=None) -> dict
# keys: surprisal, counting, generated, spacing
```
Both measures through **one** transport. About 1.5x faster than two `spectrum` calls, and bit-identical to them — `max|delta|` is exactly 0, which the tests assert rather than approximate.

Speed is the smaller reason it exists. Linear splitting spreads each sequence over adjacent bins, and `generated[b] / counting[b]` is the exact mass-weighted mean probability of the sequences in bin `b` **only if both measures were split by the same fractions**. Two separate calls agree today because they recompute the same arithmetic from the same inputs; nothing holds them together, and [`atoms`](#atoms) depends on it entirely. One transport makes it structural.

### spectra_by_length

```python
spectra_by_length(*, measure="generated", max_length=None,
                  bins=1024, spacing=None) -> dict
# keys: surprisal, weights (shape (max_length+1, bins)), spacing, measure
```
Every length's spectrum on **one shared grid**. Rows for lengths the model does not generate are zero; summing the rows reproduces the pooled spectrum; each row is bit-identical to the corresponding single-length `spectrum` call.

!!! note "What the fusion buys, and what it does not"
    BeamModel's per-length trellises are genuinely **separate** dynamic programs with no shared work, so fusing them saves no arithmetic. Measured, it is **1.26x at order 3 and parity at order 4** with 512 bins, where one length's scratch is large enough that the memory budget allows only two at a time and OpenMP does not nest, so it falls back to serial-outer / parallel-inner.

    The value at order 4 is correctness and convenience, not speed: one grid, sized from **one** `accumulator_max` instead of 27, so the rows are directly comparable and summable.

Concurrency is capped by `set_memory_budget`, not by core count, because one length's scratch runs to 1.6 GB at order 4 with 512 bins.

### accumulator_max

```python
accumulator_max() -> float
```
The largest value a spectrum grid has to represent, in nats. This bounds two different things that are not the same: the DP's running accumulator, which grows monotonically along the path, and the final surprisal, which is that accumulator plus the per-length constant `ln Z(L) - ln pi(L) - tail`. The constant can be negative, in which case a grid sized to the surprisal indexes past its own top; it is not negative in general, so a grid sized only to the accumulator would let mass fall off the end. Size grids with this, which covers both.

## pseq_analysis: spectra, atoms and discovery

```python
pseq_analysis() -> PseqAnalysis
```

The accessor the graph models in this library expose, under the same name and with the same method names, so spectrum and discovery code written against `graph.pseq_analysis()` reads across to BeamModel unchanged. The object **caches** its reconstructions, which is the whole reason it exists rather than a set of free functions: a transport is the expensive step and a figure typically wants the same one several times.

```python
a = model.pseq_analysis()
```

```python
a.histogram(bins=2048, measure="generated", length=None) -> Histogram
a.histogram_pair(bins=2048)          -> {"counting": ..., "generated": ...}
a.histograms_by_length(bins=2048, measure="generated", max_length=None)
a.length_marginals()                 -> {length: {counting, generated, ...}}
a.length_derivatives(q, order=0)     -> {length: jet}
a.derivatives(q, order=4)            -> np.ndarray
a.log_mellin(q); a.mellin(q); a.cgf(t); a.moments(order=4)
a.atoms(bins=2048, length=None)      -> Atoms
```

An `Histogram` carries `surprisal`, `weights`, `grid_spacing`, `max_rounding_error`, `true_max_surprisal`, `total_mass`, `normalized_weights`, `mean`, `variance`, `std`, and the `cdf` / `pdf` / `quantile` methods. Every spectrum this object builds is placed on a grid whose spacing was computed **before** any of them was transported, so histograms at different lengths and measures share one axis by construction rather than by both happening to auto-size the same way.

`histogram_pair` goes through [`spectrum_pair`](#spectrum_pair) and `histograms_by_length` through [`spectra_by_length`](#spectra_by_length), both one C call rather than a loop. The fused results are bit-identical to the calls they replace, and they populate the same cache, so a later single-measure request at the same resolution costs nothing.

```python
pair = a.histogram_pair(1024)
pair["counting"].total_mass          # 2.759e+23, the support size
pair["generated"].total_mass         # 1.0
```

### atoms

```python
a.atoms(bins=2048, length=None) -> Atoms
```

One `(probability, multiplicity)` pair per occupied grid bin: how probable a sequence in that bin is, and how many sequences are there. This is the form every rarefaction-style calculation consumes, including `PublicnessModel.expected_counts`.

!!! danger "The probability is `generated / counting`, not `exp(-surprisal)`"
    Both spectra come out of **one** transport ([`spectrum_pair`](#spectrum_pair)), so they are the same weights carrying different mass and their **ratio** is the mass-weighted mean probability of the sequences in the bin. With that definition `sum(multiplicity * probability)` is the generated mass exactly. Taking the two from one sweep is what makes the shared splitting fractions structural rather than a coincidence of two calls doing the same arithmetic.

    Reading `exp(-surprisal)` off the bin's left edge instead biases every atom by up to `exp(spacing)`. On the worked model at 128 bins that route sums to **4.20** instead of 1; at 1024 bins it is still 1.014. Any discovery curve built on it inherits that bias, and normalising it away afterwards hides it rather than fixing it.

```python
atoms = a.atoms(1024)
atoms.mass                    # 1.0 exactly
atoms.support_size            # 2.759e+23
p, c = atoms.as_arrays()      # the pair PublicnessModel takes
```

| method | quantity |
|---|---|
| `expected_richness(n)` | `R(n) = sum_s [1 - (1 - P(s))**n]`, distinct sequences after `n` draws |
| `novelty(n)` | `U(n) = sum_s P(s) (1 - P(s))**(n-1)`, the chance draw `n` is new |
| `coverage(n)` | probability mass carried by the sequences seen in `n` draws |
| `expected_frequency(n, r)` | distinct sequences drawn exactly `r` times; `r=1,2` are Chao's `f1`, `f2` |
| `discovery_curve(draw_counts)` | all three over many depths, from one spectrum |

`R(1)` is exactly the generated mass, `U(n)` is exactly `R(n) - R(n-1)`, and `R(n)` rises monotonically to the support size. All three are asserted, and the whole family is checked against a brute-force sum over every one of the 168,000 sequences a deliberately tiny model can produce.

```python
atoms.expected_richness(1e11)     # 3.129e+10 sequences from 1e11 draws
atoms.novelty(1e11)               # 0.264, the chance draw 1e11 is new
curve = atoms.discovery_curve(np.logspace(0, 82, 400))
```

The point of `atoms` is that a 400-point discovery curve costs **one** spectrum and 400 numpy passes, not 400 spectrum rebuilds.

### length_derivatives

```python
a.length_derivatives(q, order=0) -> {length: np.ndarray}
```

For length `L`, element `r` is `sum_{len(s)=L} P(s)**q * ln(P(s))**r` — the unnormalised jet the graph models return, so pipeline code reads across unchanged.

**Exact at every `(q, order)`.** [`length_moments`](#length_moments) supplies each length's share of the `q`-tilted law and its conditional surprisal moments; the jet is that share times `M(q)`, signed by `(-1)**r`. Summed over lengths, order 0 reproduces `M(q)` and, at `q=1`, order 1 gives minus the entropy — which is what makes this a decomposition of the global jet rather than a set of plausible per-length numbers.

`M(q)` is astronomically large for `q` below about 1, so the jet overflows to `inf` there for the same reason `derivatives` does. `length_moments` is the same content in a form that never multiplies by `M(q)`; use it when the shape rather than the scale is what is wanted.

### There is no edge thresholding

`diversity_under_edge_thresholds` exists on the graph models and raises `NotImplementedError` here. Open-world BeamModel has no low-likelihood wiring to threshold away: hierarchical backoff leaves every table cell strictly positive, so such a sweep would either be a no-op or describe a model the smoother does not define. Strict BeamModel already has an exact zero-probability boundary rather than an arbitrary threshold. `spectrum` and `atoms` answer where the mass and sequences sit under either regime.

## The distribution of the surprisal

Three routes to the same object, at three prices.

```python
surprisal_distribution() -> dict   # length, weight, mean, variance, std
surprisal_pdf(x) -> float | np.ndarray
surprisal_cdf(x) -> float | np.ndarray
```
A mixture over lengths whose weights are `pi(L)` and whose per-component means and variances come from the exact jet DP. The Gaussian shape *within* a component is the only approximation. One pass per length, and the cheap answer.

```python
cgf_jet(t) -> dict                 # K, K1, K2 — from ONE trellis sweep
saddlepoint(x, *, tol=1e-9, max_iter=80) -> dict
# keys: x, pdf, cdf, sf, t, converged, range
```
Lugannani-Rice. The cumulant generating function and its derivatives are **exact**, from the jet DP; only the inversion from those to a smooth density is approximate. Outside `surprisal_range()` the density is exactly zero and that is what is returned, rather than an extrapolation.

The Newton solve is clamped to `BeamModel.MAX_TILT` (25). A point so far into a tail that the clamped tilt cannot reach it is reported at the boundary with `converged` false, rather than answered from a tilt that does not solve the saddlepoint equation. Each solve is a handful of full trellis passes, so this is seconds per point on a large model.

!!! note "Scalar in, scalar out"
    `saddlepoint`, `surprisal_pdf`, `surprisal_cdf` and `Calibration.pvalue` share one convention: a scalar argument returns Python floats, and anything with a shape returns arrays of that shape. A one-element list is a container and stays one.

## Length-conditioned anomaly scoring

```python
calibrate(*, bins=512) -> Calibration
```
Builds the **exact** length-conditioned null for the surprisal. For each length the model can produce, the distribution of `-ln P` among sequences of exactly that length is reconstructed by dynamic programming over the DAG. No simulation, so no sampling noise, no minimum-count rule, and nothing to tune: a length is covered if it has support.

The null is the model's own generated law restricted to a length, so a p-value here is exactly "how surprising is this sequence for its length, under this model", with no free parameter standing between the question and the answer. `bins` is a resolution knob and the only argument; it obeys the same floor as [`spectrum`](#spectrum).

```python
cal = model.calibrate(bins=512)
cal.lengths                      # tuple of covered lengths
cal.resolution_floor             # smallest tail probability the grid resolves
cal.pvalue(13, 20.0)             # P(U >= 20 nats at length 13)
cal.surprisal, cal.spacing, cal.survival   # the grid and the per-length tails
```

`pvalue` returns 1.0 for a length the calibration does not cover, which is the conservative answer: an uncovered length is not evidence of anomaly. A p-value at or below `resolution_floor` is at the limit of what the reconstruction can distinguish from zero and should be reported as such rather than quoted precisely.

### SCALE

```python
cal.median(length=None); cal.iqr(length=None)
cal.quantile(length, probability)
cal.scale(length, surprisal)
model.scale_score(sequence, calibration) -> float | np.ndarray
```

`scale_score(s)` is `(-ln P(s) - median_L) / IQR_L`: the same length-normalised anomaly score `FlashBackGraph.scale_score` computes, with the reference taken from the **exact** per-length law rather than measured off a few hundred thousand simulated draws. There is no `n_sim`, no seed and no minimum-count rule, so there is nothing to tune and no sampling noise in the reference.

```python
cal = model.calibrate(bins=512)
model.scale_score("CASSLGQGAEQYF", cal)   #  1.17
model.scale_score("CASSLGQGAEQYW", cal)   #  6.31 — no J gene ends in W
```

Sequences drawn from the model are its null by definition, so their scores have median 0 and interquartile range 1; the tests assert both, and assert that the median of the score is flat across lengths where the median of the raw surprisal spans more than a nat.

Quantiles use the **mid-distribution** convention: `1 - survival` is `P(U < u)`, strictly below the atom, and inverting it puts every quantile half a bin too high — a systematic bias on the median and so on every score. Splitting each atom's mass across its own step removes it, and a test asserts the remaining error shrinks as the grid is refined.

A length the calibration does not cover falls back to the `pi(L)`-weighted mixture over the lengths it does cover, rather than to a flat null or a crash.

## Generation

```python
sample(n, *, length=None, seed=None) -> list[str]
simulate(n, *, length=None, seed=None) -> SimulationResult
```
Draws are **exactly** proportional to sequence probability, by walking the trellis against its backward values. Empirical frequencies converge to the model's own probabilities, which the test suite checks directly. `simulate` packages the same draws with each one's log probability and residue count, in the container the other models return.

`n=0` returns an empty list; a negative `n` raises rather than reaching the C layer as a request for four billion draws.

## Counts are additive

```python
add(other) -> None
subtract(other) -> None
```
A corpus can be counted in shards and combined, and holding a cohort out of a foundation model is a subtraction rather than a recount:

```python
foundation = BeamModel.from_file("foundation.beam")
foundation.subtract(BeamModel(held_out_sequences, abundances=held_out_counts))
```

Both models must share `order`, `max_length` and `depth_cap`. The smoothing parameters need not match: counts do not depend on them, and this model's own smoothing is what gets reapplied. `other` must be a *different* `BeamModel`; passing the model itself is refused, because the C layer aliases source and destination.

!!! warning "`n_observed` is not decremented by `subtract`"
    `add` adjusts `n_observed` and `n_skipped`; `subtract` does not. After a subtraction those two — and `len(model)` — count everything ever added rather than what is currently in the tables. **`total_weight` is adjusted on both sides and is the number to trust.**

!!! warning "A subtraction that empties the tables is not recoverable"
    The counts are gone before the re-finalize is attempted, so a failure leaves the model unusable with nothing to roll back to. Build a fresh model rather than subtracting a cohort out of itself.

### Bayesian posterior

```python
posterior(prior, kappa=100.0) -> None      # in place
with_prior(prior, kappa=100.0) -> BeamModel     # a new model
```

A Dirichlet-multinomial update. Every row of every count table is one categorical distribution — over the twenty residues for the three expert tables, over lengths for the length histogram — and this adds `kappa * p_prior[row]` to each of this model's rows, where `p_prior` is that row of the prior **normalised**. A row then reads

```
p_post = (c_own + kappa * p_prior) / (n_own + kappa)
```

before BeamModel's own hierarchical backoff, which is the convention `FlashBackGraph.posterior` uses for graph edges. This is the personalisation primitive: a foundation model as the prior, one donor's own sequences as the counts, and `kappa` as how many pseudo-observations of the foundation that donor's data has to argue against.

```python
foundation = BeamModel(sequences[:1200], order=3)
donor      = BeamModel(sequences[1200:], order=3)
personal   = donor.with_prior(foundation, kappa=200.0)
```

`kappa = 0` is a no-op on the counts. A row the prior never observed is left alone: an empty row carries no belief, and spreading `kappa` uniformly over it would invent one.

!!! note "The sequence counters do not move"
    `total_weight` and `n_observed` report what was **observed**, and a pseudo-count is not an observation, so a leave-one-out corpus size does not shift when the smoothing strength does.

!!! warning "A large kappa does not make this the prior"
    Raising `kappa` pulls the individual toward the prior and then stops. Three things survive any `kappa`: rows the prior never observed, this model's own backoff reapplied to the blended counts, and its own `Z(L)` and `pi(L)`. On the worked corpus the gap to the prior falls by more than 10x by `kappa = 100` and then floors; the tests assert both the closing and the floor.

### The non-destructive forms

`add` and `subtract` mutate in place, and the counts are the only copy there is. Three methods return a new model instead:

```python
copy() -> BeamModel                                   # an independent clone
merged(other) -> BeamModel                            # self + other, mutating neither
without(other, *, abundances=None) -> BeamModel       # self - other, leaving self intact
```

`without` is the leave-one-cohort-out primitive. `other` may be another `BeamModel`, or the cohort's sequences themselves, in which case a configuration-matched model is built from them first:

```python
foundation = BeamModel.from_file("foundation.beam")
held_out = foundation.without(cohort_sequences, abundances=cohort_counts)
foundation.total_weight     # unchanged
```

Removing a cohort by **subtracting its counts** is not the same as rebuilding without its sequences, and the difference is the point: a sequence the held-out cohort shares with the rest of the corpus keeps the rest of the corpus's evidence instead of vanishing along with it.

`copy` is built by adding this model's counts into a fresh empty one, so it shares no storage. It is unavailable after `compact()`, which released the counts it would duplicate.

## Cross-model summary names

A handful of accessors carry the spelling the graph models in this library use, so code that summarises several models side by side does not have to special-case BeamModel. Each is a real quantity, not a stub.

| accessor | quantity |
|---|---|
| `variant` | `"beam"` |
| `n_sequences` | total abundance ingested; an alias of `total_weight`, and the number a corpus-integrity check compares against a release's summed abundance column |
| `observed_lengths` | the lengths the model produces, ascending |
| `length_distribution` | `{length: pi(L)}` — the observed length distribution, **normalised**. BeamModel keeps the prior rather than the raw per-length totals, so a count would be a reconstruction. |
| `diversity_profile()` | now also carries `entropy_nats` alongside `D0`, `D1`, `D2` |
| `pgen_diagnostics()` | `is_proper`, `total_mass`, `length_prior_mass`, `max_abs_deviation`, `measure` |

`pgen_diagnostics` **measures** properness rather than asserting it: `sum_s P(s) == 1` is evaluated through the Mellin transform at `q=1`, which is a full sweep over the trellis, and `sum_L pi(L) == 1` through the length prior. A model that normalised only per length, or whose length prior had leaked, fails one of them. The reported `measure` is `"Pseq"`: BeamModel has no recombination model, so nothing in it is a Pgen.


## Persistence and memory

```python
save(path); BeamModel.from_file(path)
compact()                # drop counts, keep the finalized tables
memory_bytes             # approximate resident bytes
n_observed, n_skipped, total_weight
n_states, n_cells, n_nonzero_cells
```

`save` writes **counts**, not the derived tables, so a loaded model can still be subtracted from and re-finalized under different smoothing. Sparse models are written as sorted occupied-row records; dense files retain their flat count blocks, and the reader remains compatible with the previous dense format. Files carry a CRC-32C over row metadata as well as values, and a corrupt one is refused.

An order-4 dense model over the full 27-residue range holds 147.8 million conceptual table cells per copy, so about 2.2 GB with counts and 1.1 GB after `compact()`. Order 3 is about five times smaller (29.3 million cells, 440 MB and 220 MB), and order 2 is the same size as order 3 because the DAG state can never hold fewer than three residues.

A strict zero-backoff model uses sparse storage by default. Its resident size
tracks occupied rows rather than all 147.8 million cells; `n_cells` still
reports the conceptual dense shape, while `n_nonzero_cells` reports actual
occupancy and `memory_bytes` reports the representation's resident storage.
At the default dimensions the six uint32 row directories consume about 28.2
MiB per table set (56.4 MiB while counts and finalized probabilities coexist);
everything beyond that fixed addressing layer is proportional to occupied
20-value rows. A one-sequence strict model therefore uses about 59 MB rather
than allocating the 2.4 GB dense count-plus-probability payload.
Pass `sparse=False` to build the bit-identical dense form for a comparison.
`compact()` makes `subtract`, `add`, `save`, `n_nonzero_cells` and re-finalize
unavailable, and each of those raises rather than returning a plausible wrong
answer.

## Parallelism

```python
from BEAM import set_threads, threads, effective_threads

set_threads(1)        # hard cap, PROCESS-WIDE
threads()             # the cap in force, or 0 for the runtime's default
effective_threads()   # what a parallel region would actually start
model.threads = 1          # the same setting, spelled as a property
model.effective_threads
```

A library that silently grabs every core is unusable inside a job scheduler or alongside another parallel stage, so this is a **hard cap** rather than a hint: every parallel region in the BeamModel implementation runs with at most `n` threads. `0` restores the runtime's own choice, negatives are treated as `0`, and a build without OpenMP ignores the setting and always reports 1.

!!! warning "The cap is process-wide, not per model"
    `model.threads = 1` changes it for every model in the process. The property exists because that is where a caller looks for it; `set_threads` is the spelling that says what it does.

**Results do not depend on it.** Every reduction in the implementation is either order-independent or summed in a fixed index order, so a one-thread run and a sixteen-thread run agree bit for bit — asserted in the tests for both the batch scorer and a trellis-reducing analytic. That is what makes pinning threads to 1 for a runtime comparison a fair thing to do rather than a distortion.

### Memory budget

```python
from BEAM import set_memory_budget, memory_budget

set_memory_budget(2 << 30)   # 2 GB
memory_budget()              # bytes; the default is 4 GB, 0 restores it
```

Only [`spectra_by_length`](#spectra_by_length) is large enough to care: one length's histogram grids run to 1.6 GB at order 4 with 512 bins, so running one length per core would ask for twenty-five gigabytes. It divides this budget by what one length needs and runs that many at a time.

This caps **concurrency only**. A single length always runs whatever its size, so lowering the budget can never turn a working call into a failing one, only a slower one. It is a resource knob and nothing else: **results do not depend on it**, pinned by a C unit test at a budget of one byte and by a Python test that a full by-length spectrum is bit-identical at a budget of 1.

### Measured effect of the parallel implementation

Wall clock, before and after, on the workstation. Order 4 / order 3 where both were measured.

| operation | order 4 | order 3 |
|---|---|---|
| finalize | 12.75 -> 4.31 s | 3.55 -> 0.51 s |
| entropy | 11.88 -> 4.61 s | 3.60 -> 0.24 s |
| path_count | 17.97 -> 3.67 s | 4.29 -> 0.18 s |
| pooled spectrum | 315 -> 54 s | |
| `top_k_sequences(100)` | never finished -> 3.96 s | |
| `sample(10_000)` | 18.0 -> 8.1 s | |
| save / load | 2.75 -> 0.79 s | 14.36 -> 5.47 s |

Re-measured on a busier machine: build + finalize at order 3 is 0.41 s, and `top_k_sequences(100)` at order 4 is 6.3 s. Treat the table as the shape of the change, not as a benchmark.

!!! warning "`sample` draws different residues than it used to"
    Each length now has its own generator, so a given seed produces a different sequence of residues than before the parallel implementation. **Per-seed reproducibility is preserved and tested** — the same seed gives the same draws today and tomorrow — but the old single-stream draw order is gone, and any result that pinned specific sampled strings to a seed will not reproduce.

!!! note "`surprisal_by_depth` moved by ~2e-12 relative"
    It now sums in fixed 256-element chunks in index order, chosen so the answer does not depend on the thread count. That changes the last couple of digits against the old serial value.

A float32 copy of the tables was tried and is **slower** (0.64x); it is not an avenue worth reopening.

## Thread safety

Scoring and analytics are read-only and safe to call concurrently on one model. `add` and `subtract` are **not**: both release the GIL and then re-finalize, which frees and reallocates the log-probability tables that a concurrent reader is dereferencing. Do not mutate a model that another thread is scoring against — build shard models in parallel and combine them from one thread instead.

## Relationship to FlashBack

BeamModel is a **separate model**, not a FlashBack variant. It keeps the geometric fact the FlashBack peel exposed on amino acids — that distance from each terminus is the informative coordinate — and drops the token fusion, which measurement showed costs about 900x the parameters for about 1% of the information. Nothing in the [FlashBackGraph](flashback-graph.md) or [FlashBackGrammar](flashback-grammar.md) implementation is shared or changed.

The C symbols are `BeamModel` / `beam_model_*` rather than `beam_*`, because `beam_v_prob` and friends already refer to *graph* edges.
