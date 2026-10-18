---
title: BeamModel
description: The model class, its constructor and its methods, grouped by what you want to do.
sidebar:
  order: 1
---

```python
from BEAM import BeamModel
```

A probability distribution over CDR3 sequences, built from counts. See
[Building a model](/BEAM/guides/build-a-model/) for a walk-through.

## Constructor

```python
BeamModel(sequences, *, abundances=None, alphabet="aa", order=None,
          weights=(0.5, 0.5), max_length=None, alpha=None, alpha_length=None,
          sparse=None)
```

| Parameter | Meaning |
|---|---|
| `sequences` | CDR3 strings. Empty strings, strings longer than `max_length` and strings with characters outside the alphabet are skipped and counted in `n_skipped`. |
| `abundances` | How many cells (templates, reads) carry each sequence. Default: 1 each. Non-positive or non-finite values skip the sequence. |
| `alphabet` | `"aa"` (the 20 amino acids) or `"nt"` (`A C G T N`). Always named; never detected. |
| `order` | How many neighbouring residues each expert looks at. Default 4 for `aa` (also the maximum), 6 for `nt` (maximum 9). |
| `weights` | Weights of the start-anchored and end-anchored experts. Keep them summing to 1. |
| `max_length` | Longest sequence represented, 2 to 255. Default 27 for `aa`, 81 for `nt`. `"data"` takes the longest training sequence the alphabet accepts (not available to the streaming readers). |
| `alpha`, `alpha_length` | Strengths of the [hierarchical prior](/BEAM/explanations/how-beam-works/#sequences-the-model-has-never-seen). Default: estimated from the data. |
| `sparse` | Store only observed contexts (default). `False` stores every context: the same model, at gigabytes of memory. |

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

## Scoring

| Method | Returns |
|---|---|
| `pseq(seq_or_list, log=False)` | Probability (or natural log) of one sequence or of each in a list. |
| `log_pseq(seq)` | Natural log of one probability; `-inf` if the sequence cannot be represented. |
| `log_pseq_batch(seqs)` | Log probabilities of a list, as a NumPy array; parallel where available. |
| `log_pseq_chunks(iterable, chunk_size=)` | Log probabilities of an iterable, one array per chunk. |
| `log_pseq_file(path, ...)`, `log_pseq_parquet(path, column=)` | Scores every sequence in a file. |
| `explain(seq)` | For each residue, the log probability from each expert (`forward`, `backward`) and their weighted combination (`combined`, $\ln\psi_i$), with `log_length_prior` and `log_partition`; together they reconstruct the score. Raises `ValueError` naming why a sequence cannot be scored. See [Which residues drive a score](/BEAM/guides/score/#which-residues-drive-a-score). |
| `log_pseq_by_position(seq)` | The score split per position; the sum is the score. |

## Generating

| Method | Returns |
|---|---|
| `sample(n, length=None, seed=None)` | `n` sequences drawn exactly in proportion to their probability. Without a `seed` the draw uses seed 0, so repeated calls return the same sequences; pass different seeds for independent draws. |
| `simulate(n, length=None, seed=None)` | The same draws with their log probabilities (`.sequences`, `.log_probs`). |
| `top_k_sequences(k=20, length=None)` | The `k` most probable sequences with their log probabilities. |
| `tempered(t)` | A new model with $P_t(s) \propto P(s)^t$: sharper for $t > 1$, flatter for $t < 1$. |

## Lengths

| Member | Returns |
|---|---|
| `length_distribution` | `{length: probability}` over the lengths the model produces. |
| `length_pmf()` | The same as an array indexed by length. |
| `log_length_prior(L)` | $\ln \pi(L)$. |
| `observed_lengths` | Lengths the model can produce. |

## Summaries of the whole distribution

All exact, computed over every sequence the model can produce.

| Method | Returns |
|---|---|
| `entropy(base=e)` | Shannon entropy of the sequence distribution. |
| `hill_number(q)`, `hill_numbers(orders)`, `diversity_profile()` | Effective numbers of sequences (Hill numbers). |
| `path_count(length=None)` | Number of distinct sequences the model can produce. Because every sequence up to the maximum length has some probability, this is the size of the whole sequence space (for example $20^L$ at length $L$), not a measure of the repertoire. |
| `positional_marginals(length)` | $P(x_i = a \mid L)$ for every position and residue: the basis of a sequence logo. |
| `surprisal_stats()` | Mean, variance, standard deviation, skewness and excess kurtosis of $-\ln P$. |

## Combining and updating

| Method | Effect |
|---|---|
| `a.merged(b)` | A new model with both models' counts: one pooled repertoire. |
| `add(other)`, `subtract(other)` | Add or remove another model's counts, in place. |
| `without(other)` | A new model with `other`'s contribution removed. |
| `with_prior(prior, kappa=100)` | A new model updated towards `prior`'s probabilities. |
| `copy()` | An independent copy. |

To mix *probabilities* rather than pool counts, use [`Mixture`](/BEAM/reference/mixture/).

## Saving, memory and introspection

| Member | Meaning |
|---|---|
| `save(path)` | Writes counts and configuration; `.beam` is added when the path has no suffix. |
| `compact()` | Drops the raw counts to save memory. The model still scores, samples and computes, but can no longer be saved, copied, combined or given new prior strengths. |
| `memory_bytes` | Approximate memory of the model's tables. |
| `config` | The configuration it was built with, including the prior strengths chosen. |
| `n_observed`, `n_skipped` | Sequence records accepted and rejected while building. |
| `total_weight` (alias `n_sequences`) | Total abundance counted. |
| `metadata()` | Everything the model records about itself, including when it was built. |
| `set_alpha(alpha, alpha_length)` | Rebuilds the prior with new strengths (numbers, or `"auto"`). |
