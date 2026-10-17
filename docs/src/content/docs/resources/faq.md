---
title: FAQ
description: Answers to common questions about installing and using BEAM.
sidebar:
  order: 1
---

## `pip install beam-model` tries to compile and fails

Ready-built packages exist for Linux (x86-64, ARM), macOS (Apple Silicon) and Windows (x86-64)
with Python 3.11 to 3.15. On another platform or Python version pip falls back to building from
source, which needs a C compiler with OpenMP; see
[Installation](/BEAM/start/installation/#from-source).

## Should I use `junction_aa` or `cdr3_aa`?

`junction_aa`, if you want to compare with the reference model: it includes the conserved
cysteine at the start and the phenylalanine or tryptophan at the end, as the reference model's
sequences do. See [Key ideas](/BEAM/start/key-ideas/).

## My model gives probability 0 to a sequence

The sequence is longer than the model's maximum length (27 amino acids by default) or contains
a character outside its alphabet. `model.explain(sequence)` says which.

## Can I build a model from nucleotide sequences?

Yes: `BeamModel(sequences, alphabet="nt")`. The alphabet is `A C G T N`, and the default maximum
length is 81 bases.

## How much memory do I need?

A model of one repertoire takes from a few megabytes to a few hundred. The reference model takes
about 1 GB. Large computations check their needs against a budget first; see
[Threads and memory](/BEAM/reference/runtime/).

## Are the results reproducible?

Yes. Exact quantities do not depend on random numbers or on the number of threads, bounded
values agree within their stated bounds, and sampling with the same `seed` gives the same
sequences (without a `seed`, sampling uses seed 0).

## Is BEAM a model of V(D)J recombination?

No. BEAM describes the distribution of CDR3 sequences as they occur in the repertoires it was
built from, after selection, not the recombination process that produced them. Its
probabilities are frequencies in a population, not generation probabilities.
