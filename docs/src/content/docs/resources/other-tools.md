---
title: BEAM and other tools
description: How BEAM relates to OLGA, SoNIA, TCRdist, the Repertoire Dissimilarity Index and gene-usage methods, and when to use which.
sidebar:
  order: 2
---

BEAM answers questions about **observed** CDR3β repertoires with exact calculations. Other tools
answer related but different questions. This page summarises the comparisons made in the BEAM
paper; see the paper for the full benchmarks.

## Generation and selection models: OLGA, SoNIA

**OLGA** computes the probability that V(D)J recombination *generates* a sequence, before
thymic and peripheral selection. BEAM describes the distribution of sequences as they are
*observed* in repertoires, after selection, expansion and sampling. They answer different
questions: OLGA how likely a sequence is to be made; BEAM how likely it is to be found.

**SoNIA** adds a selection model on top of a generation model. In the paper, a SoNIA model
refitted to the same corpus and BEAM described held-out repertoires similarly well for overall
amino-acid composition; BEAM was closer for sequence length, and SoNIA for positional
composition.

## Repertoire comparison: TCRdist, RDI, gene usage

**TCRdist** compares receptors by sequence similarity and, with optimal transport, compares
whole repertoires; it also uses V genes. In the paper, clustering 69 mouse repertoires by cell
population gave the same concordance with BEAM's distance as with TCRdist.

The **Repertoire Dissimilarity Index (RDI)** compares gene-segment usage. BEAM and RDI agreed
that naive–memory differences are larger in CD8 than in CD4 T cells.

**Gene-usage methods** were stronger in some within-study comparisons, and empirical positional
composition was competitive. BEAM uses the CDR3 amino-acid sequence alone, so it applies when
gene calls are missing or not comparable between studies, and its distance names the sequence
patterns behind a difference.

## What BEAM adds

- **Exact sampling calculations**: expected distinct sequences, overlaps, publicness and
  sampling bias at any depth, with guaranteed error bounds instead of simulation.
- **Probabilities for every sequence**, including ones never observed.
- **One model for many questions**: scoring, generation, depth, publicness and comparison from
  the same model.

## Using them together

BEAM works on CDR3 sequences, so it combines naturally with tools that use V and J genes: use a
gene-aware method for clonotype-level similarity, and BEAM for sequence probabilities, sampling
expectations and composition-level comparisons.
