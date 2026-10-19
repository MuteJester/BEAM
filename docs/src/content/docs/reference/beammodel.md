---
title: BeamModel
description: The model class, its constructor and its methods, grouped by what you want to do.
sidebar:
  order: 1
---

```python
from BEAM import BeamModel
```

A probability distribution over amino-acid or nucleotide sequences, built from their counts. See
[Building a model](/BEAM/guides/build-a-model/) for a walk-through.

## Constructor

```python
BeamModel(sequences, *, abundances=None, alphabet="aa", order=None,
          weights=(0.5, 0.5), max_length=None, alpha=None, alpha_length=None,
          sparse=None)
```

| Parameter | Meaning |
|---|---|
| `sequences` | The training sequences. Empty strings, strings longer than `max_length` and strings with characters outside the alphabet are skipped and counted in `n_skipped`. |
| `abundances` | How many cells (templates, reads) carry each sequence. Default: 1 each. Non-positive or non-finite values skip the sequence. |
| `alphabet` | `"aa"` (the 20 amino acids, upper case) or `"nt"` (`A C G T N`, either case). Always named; never detected. |
| `order` | How many neighbouring residues each expert looks at: 2 to 4 for `aa`, 2 to 9 for `nt`. Default 4 and 6. |
| `weights` | $(w_F, w_B)$: how much the start-anchored and end-anchored experts count in each residue's factor $q_F^{w_F} q_B^{w_B}$, where $q_F$ and $q_B$ are the two experts' probabilities of the residue. The default $(½, ½)$ is their geometric mean. Keep them summing to 1; `(1, 0)` or `(0, 1)` uses one expert alone. |
| `max_length` | Longest sequence represented, 2 to 255. Default 27 for `aa`, 81 for `nt`. `"data"` takes the longest training sequence the alphabet accepts (not with `from_sequence_file`, `from_parquet` or `builder`, which read the data once). |
| `alpha`, `alpha_length` | Strengths of the [hierarchical prior](/BEAM/explanations/how-beam-works/#sequences-the-model-has-never-seen). Default: chosen from the data. |
| `sparse` | Store only observed contexts (default). `False` stores every context: the same model, bit for bit, at gigabytes of memory for order 4. |

## Building from files

| Method | Reads |
|---|---|
| `BeamModel.from_dataframe(data, sequence_column=None, abundance_column=None, **config)` | A pandas DataFrame, an Arrow table or a dict of lists. |
| `BeamModel.from_sequence_file(path, sep="\t", sequence_column=0, abundance_column=1, has_header=False, **config)` | A delimited text file, without loading it into memory. |
| `BeamModel.from_parquet(path, sequence_column=None, abundance_column=None, **config)` | A parquet file, one batch at a time (needs `pyarrow`). |
| `BeamModel.builder(**config)` | A `Builder` that accumulates counts batch by batch, for corpora of any size. |
| `BeamModel.from_file(path)` | A model saved with `save`. |

`from_dataframe` and `from_parquet` read the column **`cdr3_aa`** when no `sequence_column` is
given (`cdr3` for nucleotides), and count every row once when no `abundance_column` is given. To
compare with the reference model, name `junction_aa` explicitly: AIRR's `cdr3_aa` leaves out the
conserved start and end residues.

A `Builder` takes batches with `b.observe(sequences, abundances=None)`; `b.build()` returns the
model and closes the builder, and `b.snapshot()` returns the model so far and keeps it open.

## Scoring

| Method | Returns |
|---|---|
| `pseq(seq_or_list, log=False)` | Probability (or natural log) of one sequence or of each in a list. |
| `log_pseq(seq)` | Natural log of one probability; `-inf` if the sequence cannot be represented. |
| `log_pseq_batch(seqs)` | Log probabilities of a list, as a NumPy array; parallel where available. |
| `log_pseq_arrow(array)` | Log probabilities of a pyarrow string column, read straight from its buffers; a null scores NaN. |
| `log_pseq_chunks(iterable, chunk_size=1_000_000)` | Log probabilities of an iterable, yielded one array per chunk. |
| `log_pseq_file(path, sep="\t", column=0, has_header=False, out=None)`, `log_pseq_parquet(path, column=None, out=None)` | Every sequence of a delimited file or a parquet column (default `cdr3_aa`, `cdr3` for nucleotides), in file order. `out=` fills a preallocated array or memory map. |
| `explain(seq)` | For each residue, the log probability from each expert (`forward`, `backward`) and their weighted combination (`combined` = $w_F$·`forward` + $w_B$·`backward`), with `log_length_prior` and `log_partition`; together they rebuild `log_pseq`, which is also returned. Raises `ValueError` naming why a sequence cannot be scored. See [Which residues drive a score](/BEAM/guides/score/#which-residues-drive-a-score). |
| `log_pseq_by_position(seq)` | The score split per position; the sum is the score. |
| `calibrate(*, bins=512)`, `scale_score(seq_or_list, calibration)` | A score comparable across lengths: $-\log_{10}$ of the share of same-length sequences at least as surprising, computed from the model's own distribution of surprisal at each length. A score of 2 means the rarest 1% at any length. |
| `seq in model` | Whether `seq` has a non-zero probability. |

## Generating

| Method | Returns |
|---|---|
| `sample(n, *, length=None, seed=None)` | `n` sequences drawn in proportion to their probability. Without a `seed` the draw uses seed 0, so repeated calls return the same sequences; pass different seeds for independent draws. |
| `simulate(n, *, length=None, seed=None)` | The same draws with their log probabilities (`.sequences`, `.log_probs`). |
| `top_k_sequences(k=20, *, length=None)` | The `k` most probable sequences with their log probabilities. |
| `tempered(t)` | A new model with $P_t(s) \propto P(s)^t$, for $t > 0$: sharper for $t > 1$, flatter for $t < 1$. It scores, samples and computes like any model, but cannot be saved, copied or combined. |

## Lengths

| Member | Returns |
|---|---|
| `length_distribution` | `{length: probability}` over the lengths the model produces. |
| `length_pmf()` | The same as an array indexed by length. |
| `log_length_prior(L)` | $\ln \pi(L)$, the log probability of length $L$. |
| `log_partition(L)` | $\ln Z(L)$, the normaliser of length $L$ (see [How BEAM works](/BEAM/explanations/how-beam-works/#assembling-a-probability)). |
| `observed_lengths` | Lengths the model can produce: every length up to `max_length`, since the length prior gives each some probability. |

## Summaries of the whole distribution

All computed directly over every sequence the model can produce, without simulation.

| Method | Returns |
|---|---|
| `entropy(*, base=math.e)` | Shannon entropy of the sequence distribution, in nats; `base=2` for bits. |
| `hill_number(q)`, `hill_numbers(orders)`, `diversity_profile()` | Effective numbers of sequences (Hill numbers) at order `q`. |
| `hill_profile(orders)` | Hill numbers at many orders at once, faster than `hill_numbers`. |
| `path_count(length=None)` | Number of distinct sequences the model can produce. Every sequence up to `max_length` has some probability, so this is $R^L$ at length $L$ ($R$ = 20 amino acids or 5 nucleotides), summed over lengths by default. For diversity, use Hill numbers. |
| `positional_marginals(length, *, clamp=None)` | An array with one row per position and one column per residue: $P(x_i = a \mid L)$, the probability that position $i$ holds residue $a$ at length $L$; the basis of a sequence logo. `clamp={position: residue}` conditions on residues pinned at some positions. |
| `log_clamp_probability(length, clamp)` | $\ln P(\text{clamp} \mid L)$: the log probability that a length-$L$ sequence has the residues pinned by `clamp`. |
| `surprisal_stats()` | Mean, variance, standard deviation, skewness and excess kurtosis of the surprisal $-\ln P(s)$ over the model's sequences. |
| `surprisal_range()`, `dynamic_range(*, base=10)` | The smallest and largest surprisal of any sequence, and their span in orders of magnitude. |
| `pseq_analysis()` | The model's probability spectrum: `.atoms(bins)` gives (probability, number of sequences) pairs, the input to [publicness spectra](/BEAM/guides/publicness/). |
| `insights` | Sequence logos; see [Insights and logos](/BEAM/reference/insights/). |

## Combining and updating

| Method | Effect |
|---|---|
| `a.merged(b)` | A new model with both models' counts: one pooled repertoire. |
| `add(other)`, `subtract(other)` | Add or remove another model's counts, in place. |
| `without(other, *, abundances=None)` | A new model with `other`'s counts removed; `other` is a model or a list of sequences. |
| `with_prior(prior, kappa=100)` | A new model updated towards `prior`'s probabilities; `posterior(prior, kappa=100)` does the same in place. |
| `copy()` | An independent copy. |

Combined models need the same alphabet, `order` and `max_length` (`ModelMismatch` otherwise);
the result keeps the first model's weights and prior settings. To mix *probabilities* rather
than pool counts, use [`Mixture`](/BEAM/reference/mixture/); see
[Merging counts or mixing models](/BEAM/explanations/which-model/#merging-counts-or-mixing-models).

## Saving, memory and introspection

| Member | Meaning |
|---|---|
| `save(path)` | Writes counts and configuration; `.beam` is added when the path has no suffix. |
| `compact()` | Drops the raw counts to save memory. The model still scores, samples and computes, but can no longer be saved, copied, combined or given new prior strengths. |
| `memory_bytes` | Approximate memory of the model's tables. |
| `config` | The configuration it was built with, including the prior strengths chosen. |
| `alphabet`, `n_residues` | `"aa"` or `"nt"`, and its number of residues (20 or 5). |
| `n_observed` (also `len(model)`), `n_skipped` | Sequence records accepted and rejected while building. |
| `total_weight` (alias `n_sequences`) | Total abundance counted. |
| `metadata()` | Everything the model records about itself, including when it was built. |
| `set_alpha(alpha=None, alpha_length=None)` | Rebuilds the prior with new strengths (numbers, or `"auto"`); an argument left out keeps its setting. |
