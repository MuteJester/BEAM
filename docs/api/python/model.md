# Construction, scoring, generation, persistence

## Constructor

```python
BeamModel(
    sequences,
    *,
    abundances=None,
    alphabet="aa",
    order=None,
    depth_cap=None,
    coefficients=(0.60, 0.25, 0.15),
    kappa=None,
    kappa_wide=None,
    kappa_len=None,
    weights=(0.5, 0.5),
    max_length=None,
    strict=True,
    sparse=None,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `list[str]` | Training CDR3 strings. Anything empty, longer than `max_length`, or containing a character outside the model's alphabet is skipped; see `n_skipped`. |
| `abundances` | `list[float]` | Per-sequence observed counts (default: all 1). A non-positive or non-finite weight skips the sequence. |
| `alphabet` | `str` | `"aa"` for the twenty amino acids or `"nt"` for the five nucleotide residues `ACGTN`. Decides what the model accepts, the width of every table, and the legal range of `order`. See [Alphabets](#alphabets). |
| `order` | `int \| None` | Context order of the wide expert, from 2 up to the alphabet's ceiling: **4** for `aa`, **9** for `nt`. `None` takes the alphabet's default, 4 and 6 respectively. |
| `depth_cap` | `int \| None` | Depth at which the wide expert stops resolving position exactly. Clamped to `max_length - 1`. `None` takes the alphabet's default: 15 for `aa`, and every depth for `nt`, where the table is small enough that the cap buys nothing. |
| `coefficients` | `tuple` | Log-linear mixing weights for the wide, mid and length-conditioned experts. **Must sum to 1.** |
| `kappa` | `tuple \| None` | Hierarchical backoff strengths for orders 1-3. All zero by default; `(20, 20, 20)` under `strict=False`. |
| `kappa_wide`, `kappa_len` | `float \| None` | Backoff strength for the wide and length-conditioned experts. Zero by default; 20 and 200 under `strict=False`. |
| `weights` | `tuple` | Weight on the N-anchored and C-anchored readings. Both must be non-negative. |
| `max_length` | `int \| None` | Longest sequence the model represents, **2 to 255**. `None` takes the alphabet's default: 27 for `aa`, 81 for `nt`, three bases to the residue. The floor is 2, not 1, because `depth_cap` must be at least 1 and strictly below `max_length`. |
| `strict` | `bool` | Closed-world scoring, and the **default**. Removes the order-0 add-one and probability floor, and defaults all backoff strengths to zero, so the model holds only what the training sequences spell. `strict=False` selects the open world. |
| `sparse` | `bool \| None` | Storage selection. `None` automatically uses sparse storage exactly when `strict` is true and all five backoff strengths are zero, which is the default; `False` forces dense storage. |

### Alphabets

The model is the same whichever alphabet it is built over — two anchored experts, log-linear mixing, hierarchical backoff, exact dynamic programs. Only the symbols change, and with them the size of every table.

| | symbols | residues `R` | radix `A` | default `order` | max `order` | default `max_length` | default `depth_cap` |
|---|---|---|---|---|---|---|---|
| `aa` | `ACDEFGHIKLMNPQRSTVWY` | 20 | 21 | 4 | 4 | 27 | 15 |
| `nt` | `ACGTN` | 5 | 6 | 6 | 9 | 81 | every depth |

The radix is the residues plus one sentinel that pads contexts at the termini and is never emitted, and it is the base every context index is packed in. The order ceiling is per-alphabet because the real bound is on `A**order`: `21**9` does not fit the 32 bits a context count lives in, while `6**9` fits comfortably.

**Why order 6 for nucleotides.** An amino acid is a compressed codon — 21 values where three bases need `6**3 = 216`. Matching the amino-acid model's context *span* would need order 12, which is not buildable. Order 6 is two codons and is *cheaper* than the amino-acid model already is: 46,656 states against 194,481. Order 9 is three codons and remains available, at roughly 114x the finalize work.

**Case and ambiguity.** Nucleotide input is case-folded, so `acgt` and `ACGT` are one sequence. `N` is a residue rather than a wildcard: a read carrying one is data the model can represent, it is scoreable, and generation may emit it. The other IUPAC codes are refused — `R` means "A or G", which the model has nothing to say about. Amino-acid input is **not** folded, because case carries no meaning for a base but folding residues would silently widen what is accepted.

**There is no auto-detection.** `ACGT` is a perfectly good peptide, so any heuristic would quietly build the wrong model from short or low-complexity input. The alphabet is always named.

```python
BeamModel(seqs)                                   # aa
BeamModel(seqs, alphabet="nt")                    # order 6, max_length 81
BeamModel.from_parquet(p, alphabet="nt")          # column defaults to "cdr3"
model.alphabet                                    # -> "aa" | "nt"
model.n_residues                                  # -> 20 | 5
```

Models of different alphabets cannot be combined: `add`, `subtract` and `posterior` refuse them, because two models can agree on order, length and depth and still have rows of different widths.

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
BeamModel.from_parquet(path, *, sequence_column=None, abundance_column=None,
                  batch_size=1_000_000, **config) -> BeamModel
BeamModel.from_dataframe(data, *, sequence_column=None,
                    abundance_column=None, **config) -> BeamModel
```

`from_parquet` reads one record batch at a time (needs `pyarrow`), so a corpus far larger than memory trains in constant space. `from_dataframe` is duck-typed on `.columns` / `.column_names` / `dict`, so a pandas DataFrame, an Arrow table and a plain dict of lists all work without this library acquiring a DataFrame dependency to recognise one.

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

A Dirichlet-multinomial update. Every row of every count table is one categorical distribution — over the alphabet's residues for the three expert tables, over lengths for the length histogram — and this adds `kappa * p_prior[row]` to each of this model's rows, where `p_prior` is that row of the prior **normalised**. A row then reads

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
