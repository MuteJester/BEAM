# Scoring detail, per-length quantities, spectra, distributions, calibration

## Scoring

### log_pseq

```python
log_pseq(sequence) -> float
```
Natural log of the sequence probability. Returns `-inf`, rather than raising, for a sequence the model cannot produce: empty, longer than `max_length`, containing a character outside the model's alphabet, or of a length never observed in training.

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
log_pseq_parquet(path, *, column=None, batch_size=1_000_000,
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

Unlike `log_pseq`, this **raises** `ValueError` for a sequence it cannot encode, naming the rule that was hit — the empty string, over `max_length`, or a character the model's alphabet does not accept. Use `log_pseq` when a verdict rather than a decomposition is what is wanted.

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
    Under the hierarchical prior every sequence up to `max_length` is possible, so this is exactly `sum_L R ** L` over the lengths `1..max_length`: `1.413e35` for the corpus above, whatever the corpus. **That number describes the alphabet and the length range, not the repertoire.**

    It is kept because it is the exact `q=0` corner of the Hill family and because it is what the counting spectrum integrates to, so the two have to agree. `effective_diversity()` (`D1`) is the number to quote if you are quoting one -- here `4.66e08`.

### The Mellin transform

```python
log_mellin(q) -> float      # ln sum_s P(s)**q
power_sum(q) -> float       # the same on the linear scale
cgf(t) -> float             # ln E[exp(t ln P)], equal to log_mellin(1 + t)
```
The generator everything else here is built from. `q=0` gives the log support size, `q=1` gives 0, and the derivatives with respect to `q` are the cumulants of `ln P`.

#### Many orders in one pass

```python
log_mellin_many(orders) -> np.ndarray   # log_mellin at every order
```

**Bit-identical** to `[log_mellin(q) for q in orders]` — compared with `==` in the tests, not within a tolerance — on both alphabets, in both worlds, at orders either side of 1, at 0.999 and 1.001, and at negative orders; and unchanged by the thread count, the memory budget, or which other orders share the call.

What it saves is traversals. Calling `log_mellin` once per order walks every length's trellis once per order and rebuilds the same layer of edge weights each time. This walks each trellis once and carries every order through it. Each order keeps its own per-layer rescaling, because each has its own dynamic range, and that is what keeps the bits: a shared rescale would be cheaper and would move every order's last digits but one. On a donor-scale model (180,000 sequences, order 4, 16 threads), 29 orders take 0.22 s against 2.65 s one at a time.

A sweep carrying `n` orders holds `n` mass vectors where one order holds one, and lengths running concurrently divide the [memory budget](runtime.md#memory-budget) as they always have. Orders are fused as widely as leaves room for one sweep per thread — 31 orders at amino-acid order 4 is 128 MB a sweep — and split into blocks when they do not fit. It refuses only where a sweep of a single order would not fit either.

Orders must be finite; `nan` and `inf` raise `BeamValueError`.

#### Complex orders

```python
log_mellin_complex(orders) -> np.ndarray   # complex128: ln|M(q)| + i arg M(q)
```

`M(q) = sum_s P(s)**q` at complex `q`, exactly. `P(s)**q` factors along the sequence at any complex `q`, so the power sum is a dynamic program everywhere in the plane, not only on the real line. `np.exp(result)` is `M(q)`; `M(conj q) = conj M(q)`.

**What it is for.** A sum over every sequence of a function of `P(s)` that does not factor along the sequence, such as expected richness `R(n) = sum_s 1 - e^{-n P(s)}`, is a one-dimensional contour integral of `M` (Mellin-Barnes). For `-1 < c < 0`:

```
R(n) = -(1/2pi) ∫ Gamma(c+it) n^-(c+it) M(-(c+it)) dt
```

- **What each part is:**
  - `Gamma` is the gamma function, and its Mellin transform gives `1 - e^{-x}`.
  - `n^-(c+it)` carries the depth.
  - `M` evaluated on the line `-(c+it)` carries the model.
- **Cost:** `M` does not depend on `n`, so one set of values on the line serves every depth.
- **Accuracy:** a trapezoid rule on `c = -1/2` with step `0.1` and `|t| <= 40` (801 values of `M`) was tested on a million exact probabilities from an enumerated order-4 model. It reproduced `R(n)` from `n = 10` to `1e12` to between `3e-12` and `3e-7` of the direct sum. That is with no spectrum and no grid, where a binned spectrum at 16,384 bins is off by about `1e-4`.

**Precision.** It is exact to rounding in the moduli. Sequences of different probability carry different phases and cancel, so `|M(q)|` can sit far below `M(Re q)`, the sum of the moduli. The relative error of the result grows by that ratio, while the absolute error stays at rounding relative to `M(Re q)`. Against enumeration the worst measured error is `6e-14` of the moduli.

**Determinism.** As for `log_mellin_many`:
- The same bits for an order whatever the thread count, the memory budget or the company it keeps.
- One traversal per length for all the orders.
- At a real order it agrees with `log_mellin` to rounding, not bit for bit, because the phase is carried.

**Memory and cost.** Each mass plane is twice as wide as a real order's, and the budget is divided the same way. A complex order currently costs about 1.5 times a real one, and fusing orders saves little. The per-edge `exp`, `cos` and `sin` are taken per order, which is the obvious thing to optimise for orders sharing a real part.

Orders must be finite, and real input is accepted.

#### Along a vertical line

```python
log_mellin_line(re, im_start, im_step, n) -> np.ndarray   # complex128
```

`log_mellin_complex` at the orders `re + 1j*(im_start + k*im_step)`, `k = 0 .. n-1`. That is how a Mellin-Barnes contour is sampled, and the line shares two things that make it several times faster:

- **Moduli.** Every order has the same real part, so each edge's modulus factor `exp(re*w - shift)` is one number for all of them, taken once per edge.
- **Rotations.** Order `k`'s rotation `exp(i(im_start + k*im_step)w)` is order `k-1`'s times `exp(i*im_step*w)`. It is recomputed directly at every multiple of 32, so the chain of multiplies is at most 31 long. One cosine and one sine per edge replace one per edge per order.

**Exact to rounding, not to the bit, against `log_mellin_complex`**, since the recurrence reaches the same rotation by different operations. Measured differences are within `1e-12` of `M(Re q)`, and the tests hold them there. What IS bit-for-bit is everything the line determines: order `k`'s value depends on the line and on `k` alone. The number of orders, the thread count and the memory budget change none of its bits, even when the budget splits the line into stretches that start between the recurrence's anchors, because such a stretch rebuilds its first rotation from the anchor below.

Use it for contours and keep `log_mellin_complex` as the reference.

### Diversity

```python
entropy(*, base=math.e) -> float
hill_number(alpha) -> float
hill_numbers(orders) -> np.ndarray
hill_profile(orders) -> np.ndarray   # the same profile, one pass per length
hill_curve(orders=None) -> dict   # {"orders", "hill"}
diversity_profile() -> dict       # D0, D1, D2
effective_diversity() -> float    # D1
```
`alpha=0` gives the support size, `alpha=1` the exponential of the entropy, `alpha=2` the inverse collision probability. `D0` inherits the caveat on `path_count`; `D1` is the number to quote.

`hill_profile` is the fast way to a whole profile. Every order but the two corners is `exp(log_mellin(q) / (1 - q))`, with all of them from one `log_mellin_many` call; `q = 0` is `path_count()` and `q` within `1e-12` of 1 is `exp(entropy())`, the routes `hill_number` itself takes. On a donor-scale model the default 31 orders take 0.42 s against 4.25 s for `hill_numbers`.

!!! note "`hill_profile` agrees with `hill_numbers` to machine precision, NOT bit for bit"
    `hill_number` reaches `sum P**q` through a different tilted sweep, one that rescales each layer by a different maximum and so rounds differently. Over nine models spanning both alphabets and both worlds, the two values of `ln sum P**q` differ by at most 7.1e-15, and on the grid 0 to 3 in steps of 0.1 the Hill numbers differ by at most 1.1e-14 relative. The two corners take `hill_number`'s own routes and are bit-identical.

    **Near `q = 1` the relative difference grows as `1 / |1 - q|`**, because `D_q = exp(ln M / (1 - q))` magnifies any rounding in `ln M` by that factor on both routes: 1.3e-12 was measured at `q = 1.001` on a model whose support was every sequence up to its maximum length. Neither route is the more accurate one there.

    Keep `hill_numbers` wherever a profile has to reproduce one already published from it to the last bit.

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

The sink term — the backward scores of the final `min(order, length)` residues, which never get a full right-hand context during the walk — is attributed to the **last** layer, so `by_depth[-1]` carries more than one position's worth of cost.

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
    Cost scales with the number of live states per layer, and without pruning every state is live at every layer: for amino acids 194,481 of them at order 4, 9,261 at order 3 and 441 at order 2, and for nucleotides 46,656 at order 6. A pooled spectrum over a full-range order-4 model needs several GB and minutes. Ask for one length at a time, or build at a lower `order`, when that matters.

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

The graph models in this library carry a `diversity_under_edge_thresholds` sweep. BeamModel does **not** have it at all — not even as a method that raises, which is what this page used to claim.

There is nothing for it to do. The hierarchical prior leaves every table cell strictly positive, so a threshold sweep would describe a cut the model does not make. `spectrum` and `atoms` answer where the mass and the sequences sit.

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
cal.scale(length, surprisal, *, method="pvalue")
model.scale_score(sequence, calibration, *, method="pvalue") -> float | np.ndarray
```

By default `scale_score(s)` is `-log10 P(U >= -ln P(s))` under the exact per-length null. Because the reference is the exact law rather than a few hundred thousand simulated draws, **a threshold here is a false positive rate**: 2 means 1% at every length. There is no `n_sim`, no seed and no minimum-count rule.

`method="robust"` returns the historical `(-ln P(s) - median_L) / IQR_L` instead — the same score `FlashBackGraph.scale_score` computes. It is the one this page used to document as the default, and its cutoffs are a different rate at every length, which is precisely why it is no longer the default.

```python
cal = model.calibrate(bins=512)
model.scale_score("CASSALNIWEQYF", cal)                   # 0.0270
model.scale_score("CASSALNIWEQYF", cal, method="robust")  # -1.0141
model.scale_score("CASSALNIWEQYW", cal)                   # 4.0151 -- possible,
                                                          # but far in the tail
```

`inf` is now reached only by a sequence the model cannot represent at all -- a character outside the alphabet, or a length past `max_length` -- whose surprisal is infinite. Every other sequence, the corrupted one above included, has a finite surprisal and a finite score.

Under `method="robust"`, sequences drawn from the model are its null by definition, so their scores have median 0 and interquartile range 1; the tests assert both, and assert that the median of the score is flat across lengths where the median of the raw surprisal spans more than a nat. The default `pvalue` method is normalised differently — a threshold on it is a false positive rate, not a z-like deviation.

Quantiles use the **mid-distribution** convention: `1 - survival` is `P(U < u)`, strictly below the atom, and inverting it puts every quantile half a bin too high — a systematic bias on the median and so on every score. Splitting each atom's mass across its own step removes it, and a test asserts the remaining error shrinks as the grid is refined.

A length the calibration does not cover falls back to the `pi(L)`-weighted mixture over the lengths it does cover, rather than to a flat null or a crash.
