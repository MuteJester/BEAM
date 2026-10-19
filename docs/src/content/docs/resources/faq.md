---
title: FAQ
description: Answers to common questions about installing and using BEAM.
sidebar:
  order: 1
---

## `pip install beam-model` tries to compile and fails

Ready-built packages exist for Linux (x86-64, ARM), macOS (Apple Silicon, macOS 14 or newer) and
Windows (x86-64) with Python 3.11 to 3.15. On another platform or Python version pip falls back
to building from source, which needs a C compiler with OpenMP; see
[Installation](/BEAM/start/installation/#from-source).

## Can I use BEAM for sequences other than TCRβ CDR3s?

Yes. BEAM learns from any set of amino-acid or nucleotide sequences up to 255 residues long, so
you can build models of BCR CDR3s, TCRα chains or other sequence families from your own data
([Building a model](/BEAM/guides/build-a-model/)). For nucleotides use
`BeamModel(sequences, alphabet="nt")`: the alphabet is `A C G T N`, and the default maximum
length is 81 bases. The published reference model describes human TCRβ CDR3s.

## Should I use `junction_aa` or `cdr3_aa`?

`junction_aa`, if you want to compare with the reference model: it includes the conserved
cysteine at the start and the phenylalanine or tryptophan at the end, as the reference model's
sequences do. See [Key ideas](/BEAM/start/key-ideas/).

## My model gives probability 0 to a sequence

The sequence is empty, longer than the model's maximum length (27 amino acids by default) or
contains a character outside its alphabet. `model.explain(sequence)` says which. Every other
sequence, including one never seen, has a non-zero probability.

## How much memory do I need?

A model of one repertoire takes from a few megabytes to a few hundred. The reference model takes
about 1 GB. Large computations check their needs against a budget first; see
[Threads and memory](/BEAM/reference/runtime/).

## Why does the first depth calculation take a minute or two?

The first `ExactOccupancy` query on a large model prepares the integrals that every later query
reuses, so later queries at any depth are fast. `save_lines` and `load_lines` keep them between
sessions; see [ExactOccupancy](/BEAM/reference/exact-occupancy/#saving-the-prepared-integrals).

## Can I run BEAM in several processes?

Yes: start the workers with `spawn`, load the model from its file in each worker, and set
`BEAM.set_threads` so that workers × threads fits the machine. See
[Worker processes](/BEAM/reference/runtime/#worker-processes).

## Are the results reproducible?

Yes. Quantities computed over every sequence do not depend on random numbers or on the number of
threads, bounded values agree within their stated bounds, and sampling with the same `seed`
gives the same sequences (without a `seed`, sampling uses seed 0). Linux, macOS and Windows give
the same answers to within $10^{-10}$, checked on every change to BEAM.

## Is BEAM a model of V(D)J recombination?

No. BEAM describes the distribution of sequences as they occur in the repertoires it was built
from, after selection, not the recombination process that produced them. Its probabilities are
frequencies in a population, not generation probabilities.
