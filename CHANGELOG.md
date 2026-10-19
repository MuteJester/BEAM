# Changelog

All notable changes to BEAM are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and BEAM uses
[Semantic Versioning](https://semver.org/): see [`RELEASING.md`](RELEASING.md)
for what counts as a breaking change, including changes to the model file
format.

## [Unreleased]

## [0.9.0] - 2026-10-19

The first public release, made while the manuscript *A computable probability
landscape for T cell receptor repertoire sampling and comparison* is under
review; version 1.0.0 will accompany its publication.

### Added
- `BeamModel`: a probability model of amino-acid or nucleotide sequences, such
  as T cell receptor CDR3s, combining a start-anchored and an end-anchored
  expert, normalised over every sequence of each length, with a hierarchical
  Dirichlet prior whose strengths are chosen from the data.
- Scoring (`pseq`, `log_pseq`, batched, streaming and Arrow input),
  `explain` for each residue's contribution, sampling, saving and loading.
- Sampling quantities computed without simulation, each with a guaranteed
  error bound (`ExactOccupancy`): expected distinct sequences at any
  sequencing depth and their standard deviation (`richness_sd`,
  `richness_variance`), overlap between samples, and sequences private to one
  repertoire of a cohort.
- `PublicnessModel`: the distribution of how many repertoires in a cohort
  contain a sequence.
- The local-window repertoire distance (`BEAM.analytics.distance.compare`),
  split into length and local-pattern terms that add up to it.
- Probability mixtures of repertoires (`Mixture`), depth-matched diversity
  profiles, repertoire likelihoods, sequence logos and a query engine for
  conditional probabilities over all sequences of a length.
- `BEAM.load_pretrained`: published models, downloaded once and verified by
  SHA-256. The reference model of human TCRβ CDR3s is published with the
  paper.
- Wheels for Linux (x86-64, ARM), macOS (Apple Silicon) and Windows
  (x86-64), Python 3.11 to 3.15.
- Documentation: https://mutejester.github.io/BEAM/
