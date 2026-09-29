# Construction, scoring, generation, persistence

## Constructor

```python
BeamModel(
    sequences,
    *,
    abundances=None,
    alphabet="aa",
    order=None,
    weights=(0.5, 0.5),
    max_length=None,
    alpha=None,
    alpha_length=None,
    sparse=None,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `list[str]` | Training CDR3 strings. Anything empty, longer than `max_length`, or containing a character outside the model's alphabet is skipped; see `n_skipped`. |
| `abundances` | `list[float]` | Per-sequence observed counts (default: all 1). A non-positive or non-finite weight skips the sequence. |
| `alphabet` | `str` | `"aa"` for the twenty amino acids or `"nt"` for the five nucleotide residues `ACGTN`. Decides what the model accepts, the width of every table, and the legal range of `order`. See [Alphabets](#alphabets). |
| `order` | `int \| None` | Context order, from 2 up to the alphabet's ceiling: **4** for `aa`, **9** for `nt`. `None` takes the alphabet's default, 4 and 6 respectively. |
| `weights` | `tuple` | Weight on the N-anchored and C-anchored readings. Both must be non-negative. |
| `max_length` | `int \| None` | Longest sequence the model represents, **2 to 255**. `None` takes the alphabet's default: 27 for `aa`, 81 for `nt`, three bases to the residue. The floor is 2, not 1: a model of one residue has no settled position and no anchored one. |
| `alpha` | `float \| sequence \| None` | Strengths of the hierarchical prior, one per context level `0..order`, or one number for all. `None` (the default) or `"auto"` estimates them from the model's own counts; numbers fix them. See [The hierarchical prior](#the-hierarchical-prior). |
| `alpha_length` | `float \| None` | Strength of the length prior. Estimated the same way unless given. |
| `sparse` | `bool \| None` | Storage. `None` or `True` stores only the rows something was observed at and resolves the rest through their parents; `False` materialises every row. The same model bit for bit; dense costs gigabytes at order 4. |

### Alphabets

The model is the same whichever alphabet it is built over — one anchored table per direction, geometric mixing, exact dynamic programs. Only the symbols change, and with them the size of every table.

| | symbols | residues `R` | radix `A` | default `order` | max `order` | default `max_length` |
|---|---|---|---|---|---|---|
| `aa` | `ACDEFGHIKLMNPQRSTVWY` | 20 | 21 | 4 | 4 | 27 |
| `nt` | `ACGTN` | 5 | 6 | 6 | 9 | 81 |

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

!!! warning "Table size is bounded by the memory budget"
    A configuration is refused with `MemoryError` if its tables would exceed
    `BEAM.memory_budget()`, which defaults to 4 GiB. The message says what the
    configuration needs and what the budget is.

    This is a guard against a failure mode the library used to have rather than
    a limit for its own sake: `calloc` on Linux grants an absurd size lazily, so
    a model asking for hundreds of gigabytes did not fail where it asked — it
    failed later, when the tables were first written, with the OOM killer
    choosing which process on the machine died. High orders reach that quickly
    on the nucleotide alphabet, where a dense wide table is
    `max_length * 6**order * 5` doubles per direction: 1.7 GB at order 7 and
    62 GB at order 9.

    Raise it with `BEAM.set_memory_budget(bytes)` if you have the memory, or
    keep the sparse default, which stores only observed rows.

!!! warning "The anchor weights should sum to 1, and are not forced to"
    The sum is a pure cross-length calibration knob. Off 1, `log W` scales with the length, which leaves *within-length* ranking untouched and wrecks *cross-length* comparison.

    It is not refused, deliberately. `(1, 0)` and `(0, 1)` score one anchor's chain on its own, and that is the only way to separate a partition function below 1 into the two causes that can put it there: the two chains disagreeing about a sequence. A refusal here would make the model's defining property unmeasurable.

### Configuration

Each direction is one abundance-counted table:

| anchor | context | depth |
|---|---|---|
| N-anchored | previous `order` residues | `i`, exact |
| C-anchored | next `order` residues | `L-1-i`, exact |

Neither table is indexed by length and neither needs to be. The two depths at a position sum to `L-1`, so the pair determines the length exactly, at every position and every length. Length awareness lives in the *combination*; length pooling lives in the *estimates*, since each row is learned from every sequence that carried that context at that distance from that terminus, whatever its length.

### The hierarchical prior

Every row, `(depth, the order residues nearest the scored one)`, is the posterior mean of a Dirichlet centred on its **parent**, the same depth with one residue less of context:

```
P_k(x) = ( n_k(x) + alpha[k] * P_{k-1}(x) ) / ( n_k + alpha[k] )
```

- `k`: the level, how many context residues the row keeps, `0..order`. Level 0 is the depth alone.
- `n_k(x)`: how often `x` followed this level-`k` context at this depth; `n_k` its row total. These are the model's own counts summed over the residues level `k` drops, so nothing extra is stored.
- `P_{k-1}(x)`: the parent's answer. Below level 0 it is the uniform `1/R`.
- `alpha[k]`: how many pseudo-observations the parent is worth.

Three consequences:

- **Every sequence up to `max_length` is possible.** A row nothing was observed at is exactly its parent, so an unseen context predicts what its shorter context predicts: not impossible, and not uniform.
- **Junk stays improbable.** An unseen context falls back through contexts that were seen, so residues rare at that position stay rare.
- **The data wins where it exists.** A row's weight on its own counts is `n_k / (n_k + alpha[k])`: nothing when unobserved, nearly all of it after a few thousand observations.

The length prior is the same rule over lengths, centred on the even distribution over `1..max_length`:

```
P(L) = ( n_L + alpha_length / max_length ) / ( N + alpha_length )
```

**The strengths are estimated from the data unless you name them.** At every finalize each level's strength is the one that maximises the *evidence* of that level's counts -- the probability of the counts with the Dirichlet integrated out, exact and in closed form:

```
log p(counts | alpha) = sum_rows [ lnG(alpha) - lnG(n + alpha)
                                   + sum_{x: n_x > 0} ( lnG(n_x + alpha m_x) - lnG(alpha m_x) ) ]
```

- `m_x`: the row's parent prediction, built with the strengths already fitted for the levels below. The levels are fitted bottom-up, pooling both reading directions and every position.
- A row its parent already predicts wants a large `alpha`; one its parent gets wrong wants a small one. The more data a model sees, the more its counts decide.
- A weak hyperprior, `log alpha ~ Normal(log 10, (ln 100)^2)`, is added to the evidence. It matters only when the counts say nothing -- one training sequence gives every row a single observation, whose evidence is the same at every strength -- and then the strengths fall back towards 10.

`model.config["alpha"]` reports the estimates and `alpha_auto` that they are estimates. On half of Emerson donor Keck0001's reads (order 4) the evidence chose `(5.45, 14.6, 25, 61.6, 135)` and a length strength of 4.5, and predicted the other half at -18.41 nats per read and another donor, Keck0002, at -18.76 -- against -18.73 and -19.39 at a fixed 10.

To choose them for a specific prediction instead, `BEAM.fit_alpha(sequences, abundances, **config)` maximises held-out likelihood (the repertoire's own reads by thinning, or another repertoire), and `model.set_alpha(alpha, alpha_length)` re-applies new ones -- numbers, or `"auto"` -- to a built model from its counts.

Sparse storage keeps each level's observed rows; a lookup that misses walks to the nearest observed ancestor. Dense storage writes that ancestor into every unobserved slot. They are the same numbers.

`model.config` returns the configuration actually built: `order`, `max_length`, `weights`, `alpha`, `alpha_length`, `alpha_auto`, `alpha_length_auto`, `sparse` and the alphabet.

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

## Tempering

```python
tempered(t) -> BeamModel
```

A new model holding `P_t(s) = P(s)**t / sum_s P(s)**t`. `t > 1` concentrates the distribution on its likeliest sequences and `t < 1` flattens it toward the rare ones.

The tempered law already drives [`surprisal_moments`](analytics.md#surprisal-moments), the spectrum and [`saddlepoint`](analytics.md), where it is a measure a sweep carries rather than a model. This makes it a model, so everything the ordinary one answers it answers too.

```python
hot = model.tempered(2.0)
hot.entropy() < model.entropy()          # True: tempering concentrates
hot.path_count() == model.path_count()   # True: the support does not move
```

`t` must be finite and strictly positive. At `t = 0` every finite log weight becomes zero and every `-inf` becomes NaN; a negative `t` turns an unspellable edge into the likeliest one. Both are refused by name rather than clamped.

Three quantities move, and the third is the one worth stating:

```
tables   ->  t * tables
Z_t(L)   =   the same forward sweep, on the scaled tables
pi_t(L)  =   pi(L)**t * Z_t(L) / Z(L)**t,  normalised over L
```

`pi_t` is **not** `pi(L)**t` normalised. It picks up the within-length power sum, and getting that wrong still leaves a normalised distribution and plausible numbers. It is exactly the share [`length_moments`](analytics.md#length_moments) reports at `q = t`, and the test suite pins it against that.

**Exact.** The result satisfies `log_pseq_t(s) == t * log_pseq(s) - log_mellin(t)` identically rather than to within a tolerance, and its whole Mellin transform follows from the original's:

```
log_mellin_t(q) == log_mellin(t*q) - q*log_mellin(t)
```

so the entropy is `log_mellin(t) - t * m'(t)` with `m'(t) = -E_t[-ln P]`, the mean surprisal under the very tilt being materialised.

!!! warning "The result arrives with its counts released"
    As after [`compact`](#persistence-and-memory), and for a reason that is not
    about memory. A tempered model's distribution is not a function of its
    counts, so `add`, `subtract`, `posterior` and `copy` would each silently
    rebuild the UNTEMPERED tables and report a plausible wrong number. They
    refuse instead.

    It is also why tempering does not yet compose by chaining: `tempered` on a
    tempered model is refused, not approximated. Carrying the exponent in the
    configuration is what would lift that, and it belongs with the file format.

`tempered` is non-destructive: the model it is called on is untouched, as with `merged` and `without`.

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

Both models must share `order` and `max_length`. The prior's strengths need not match: counts do not depend on them, and this model's own are what gets reapplied when it re-finalizes. `other` must be a *different* `BeamModel`; passing the model itself is refused, because the C layer aliases source and destination.

!!! warning "`n_observed` is not decremented by `subtract`"
    `add` adjusts `n_observed` and `n_skipped`; `subtract` does not. After a subtraction those two — and `len(model)` — count everything ever added rather than what is currently in the tables. **`total_weight` is adjusted on both sides and is the number to trust.**

!!! warning "A subtraction that empties the tables is not recoverable"
    The counts are gone before the re-finalize is attempted, so a failure leaves the model unusable with nothing to roll back to. Build a fresh model rather than subtracting a cohort out of itself.

### Bayesian posterior

```python
posterior(prior, kappa=100.0) -> None      # in place
with_prior(prior, kappa=100.0) -> BeamModel     # a new model
```

A Dirichlet-multinomial update. Every row of every count table is one categorical distribution — over the alphabet's residues for the two anchored tables, over lengths for the length histogram — and this adds `kappa * p_prior[row]` to each of this model's rows, where `p_prior` is that row of the prior **normalised**. A row then reads

```
p_post = (c_own + kappa * p_prior) / (n_own + kappa)
```

before BeamModel's own smoothing, which is the convention `FlashBackGraph.posterior` uses for graph edges. This is the personalisation primitive: a foundation model as the prior, one donor's own sequences as the counts, and `kappa` as how many pseudo-observations of the foundation that donor's data has to argue against.

```python
foundation = BeamModel(sequences[:1200], order=3)
donor      = BeamModel(sequences[1200:], order=3)
personal   = donor.with_prior(foundation, kappa=200.0)
```

`kappa = 0` is a no-op on the counts. A row the prior never observed is left alone: an empty row carries no belief, and spreading `kappa` uniformly over it would invent one.

!!! note "The sequence counters do not move"
    `total_weight` and `n_observed` report what was **observed**, and a pseudo-count is not an observation, so a leave-one-out corpus size does not shift when the smoothing strength does.

!!! warning "A large kappa does not make this the prior"
    Raising `kappa` pulls the individual toward the prior and then stops. Three things survive any `kappa`: rows the prior never observed, this model's own smoothing reapplied to the blended counts, and its own `Z(L)` and `pi(L)`. On the worked corpus the gap to the prior falls by more than 10x by `kappa = 100` and then floors; the tests assert both the closing and the floor.

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

`save` writes **counts**, not the derived tables, so a loaded model can still be subtracted from and re-finalized under different smoothing. Sparse models are written as sorted occupied-row records; dense files retain their flat count blocks. Files carry a CRC-32C over row metadata as well as values, and a corrupt one is refused.

The current format is **7**, and **formats 1 to 5 are refused, not read**. They were written with three experts per direction and a depth index clamped at 15, which merged every depth at or past the cap — so which depth an observation came from was discarded at training time and no transformation recovers it. The refusal names the format and says why, rather than reading the payload at the wrong extents and blaming the file for a checksum failure. Format **6** *is* read: it is this model, written before provenance existed, so the only thing missing is a build time and `metadata()` reports that as `None`.

An order-4 dense model over the full 27-residue range holds 210.0 million conceptual table cells per copy, so about 3.4 GB with counts and 1.7 GB after `compact()`. Each step down in order divides that by the radix: order 3 is 10.0 million cells (160 MB and 80 MB) and order 2 is 476 thousand (8 MB and 4 MB). Order 2 is no longer the same size as order 3 — it used to be, because the DAG state could not hold fewer than three residues while a second expert of fixed order 3 existed. A state now holds exactly the `order` residues one context needs.

Dense storage at order 4 is close to the 4 GiB default memory budget, which counts the count tables and the finalized log tables together. A dense configuration that does not fit is refused at construction rather than when the tables are first written.

A model uses sparse storage by default. Its resident size
tracks occupied rows rather than all 210.0 million cells; `n_cells` still
reports the conceptual dense shape, while `n_nonzero_cells` reports actual
occupancy and `memory_bytes` reports the representation's resident storage.

Row addressing is the one part that is not proportional to what was observed,
and each generation resolves it differently. While a model is still being
observed, **counts** sit on a uint32 directory over every dense row -- four
bytes each, so 40.06 MiB for both directions at the default dimensions,
whatever the corpus -- because a row can still appear. When observation stops
they are **frozen**, and the frozen form keeps no directory at all.

**Finalized probabilities** are built once, so when at most one row in eight
is occupied they are sealed into a bitmap-and-rank directory spending 1.25
bits per row: 1.56 MiB for the same directory, a 25.6x reduction that returns
the same rows and the same doubles. Everything beyond the addressing layer is
proportional to occupied 20-value rows.

Finalized rows are then de-duplicated. A context seen once normalises to one
residue at exactly `0.0` and nineteen at `-inf`, and only twenty such rows can
exist, so a repertoire model is largely copies of a few: measured on real
repertoires, 73-79% of a donor's finalized rows are byte-identical to another
and about half are one-hot. When at least a quarter repeat, the table keeps one
copy of each distinct row and the directory names it. Rows are pooled only when
all their bytes match, so every caller receives the same doubles; the cost is
one extra load on a sealed table, and none on a direct one, which folds the
pool into the directory it already has.

Which form a table gets is chosen per table at finalize, from its own
occupancy and its own repetition, and `memory_bytes` reflects the choice. Per-repertoire models fall
well under the threshold and are sealed; a corpus large enough to fill most of
the row space -- 57.90% for the 927-million-sequence foundation -- keeps the
direct directory, where it is faster. A one-sequence model therefore
uses 1.57 MiB, and the same after `compact()` -- at that size the sealed
directory is nearly all of it and there are no counts worth releasing --
rather than allocating the 3.13 GiB dense count-plus-probability payload. On a 197,000-sequence repertoire
the two together take a compacted model from 125 MB to 34 MB, at a measured
cost of about 12% on scoring and 10% on a mixture inner product.
Pass `sparse=False` to build the bit-identical dense form for a comparison.
`compact()` makes `subtract`, `add`, `save`, `n_nonzero_cells` and re-finalize
unavailable, and each of those raises rather than returning a plausible wrong
answer.


## What a model reports about itself

```python
meta = model.metadata()                      # lengths included
meta = model.metadata(lengths=False)         # the summary alone
meta = model.metadata(support=True)          # add the per-length path count
```

`metadata()` answers the questions a finished model cannot be asked any other way: when it was built, what building it cost, how large it is, and how much of what it could have learned it did learn. It is a plain dictionary, so it serialises.

Three facts are **stored in the file**, because they are not recoverable from the tables: `created_unix`, and the two build durations. Everything else is **computed on demand** from the tables, so it cannot fall out of step with them.

| key | what it is |
|---|---|
| `created`, `created_unix` | when the model was first built — an ISO-8601 string and the raw epoch seconds. A loaded model reports when it was **built**, not when it was loaded. |
| `beam_version`, `variant`, `finalized` | which code wrote it, and whether it is scoreable yet |
| `config` | the build configuration as given, including `order`, `max_length`, `alpha`, `alpha_length` and the alphabet |
| `build` | `observe_seconds`, `finalize_seconds`, `total_seconds`, and the corpus as ingested: `n_observed`, `n_skipped`, `n_sequences`, `total_weight` |
| `size` | `n_states`, `n_cells`, `n_nonzero_cells`, `occupancy`, `memory_bytes`, `compacted` |
| `lengths` | one entry per observed length; see below |

The two durations **aggregate**. A model built by streaming a hundred chunks reports what the whole build cost, not what the last chunk cost, and one re-finalized after an `add` or `subtract` reports every finalize it has had. They are wall time rather than CPU time, because the counting pass is parallel and a CPU figure would depend on the thread count.

Loading does **not** add to them. A `.beam` file stores counts rather than finalized tables — that is what keeps a loaded model expandable — so every load rebuilds the tables, and charging that would make the reported build time rise each time the model was opened, permanently if it were saved again. A finalize the caller asks for still counts; a finalize the format needs does not.

On a **compacted** model, `n_nonzero_cells` and `occupancy` are `None` and `compacted` is `True`. `compact()` releases the raw counts those two inspect, and a large model is normally deployed compacted, so this reports what it cannot know rather than raising or substituting a figure that would read as a real occupancy. The per-length breakdown is unaffected — it comes from the finalized tables, which compaction keeps.

### Per length: how much of the dictionary was found

For each length `L` the model observed:

| key | what it is |
|---|---|
| `share` | `pi(L)`, this length's share of the length prior. These sum to 1. |
| `rows_possible` | `2 * Σ_{d<L} R**min(d, order)` — every (direction, depth, context) a length-`L` sequence can actually reach |
| `rows_observed`, `row_coverage` | how many of those the corpus actually reached, and the fraction. `None` on a compacted model — see below |
| `cells_possible`, `cells_observed`, `cell_coverage` | the same at the level of individual (context, residue) cells |
| `log_partition` | `log Z(L)`, the Bhattacharyya coefficient between the two chains at this length |
| `path_count`, `log10_spellable_fraction` | only with `support=True`: how many length-`L` sequences the model can spell at all, and that as a fraction of `n_residues**L` on a log scale |

`row_coverage` is the direct answer to "how much of the spelling dictionary does this model hold for this length". Because depths accumulate, `rows_observed` can only rise with `L` while `row_coverage` need not — the denominator rises too.

The denominator is the **reachable** ceiling, not every context the alphabet can spell. At depth `d` a context has only `min(d, order)` free slots; the rest are the sentinel that pads that direction's anchor, so `R**min(d, order)` rows exist at that depth and the others cannot be addressed by any sequence. Depth 0 has exactly one row, the all-sentinel context.

This is checked rather than argued: enumerate every sequence of a length over the whole alphabet and the cells the model holds *are* the reachable set, so `cells_possible` must equal `n_nonzero_cells` and coverage must be exactly 1. `test_the_possible_cell_count_is_what_brute_force_occupies` asserts that at lengths 2, 3 and 4, and fails if the cell addressing in `indexing.h` ever changes.

These count **observations**, read from the model's counts, so `cells_observed` summed over the lengths equals `n_nonzero_cells`. Reading them off the finalized tables instead would saturate: the prior leaves every cell positive, so a two-sequence model would report that it had discovered the entire dictionary.

**Do not sum `cells_observed` or `cells_possible` over lengths.** A cell is addressed by `(direction, depth, context)` with no length in the address, so a depth-3 cell is read by every length from 4 up and a sum counts it once per length. On a five-length model the sum is 1,907,884 where the model holds 429,248. The union is the **longest** length's figure, which equals `n_nonzero_cells`.

A **compacted** model cannot answer: `compact()` releases the counts, and the prior has already blended the evidence into the tables, so the four coverage keys are `None`.

`path_count` is a dynamic program per length rather than a lookup, which is why it is opt-in rather than always present.

### Saved files are reproducible, apart from the clock

Every other byte of a `.beam` file is a function of the sequences, the weights and the config, so building the same model twice writes the same file. The provenance fields are the one exception by construction, and the trailing CRC covers them and so moves with them. `BEAM.file_provenance_span()` returns `(offset, bytes)` for that window, so a byte-for-byte comparison of two saved models can skip exactly it — and nothing else — without hardcoding an offset that a later field would move.
