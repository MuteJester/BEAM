# Changelog

All notable changes to BEAM are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and BEAM uses
[Semantic Versioning](https://semver.org/): see [`RELEASING.md`](RELEASING.md)
for what counts as a breaking change, including changes to the model file
format.

## [1.0.0] - Unreleased

The first public release, accompanying the manuscript *A computable
probability landscape for T cell receptor repertoire sampling and comparison*.

### Added
- `BeamModel`: a probability model of CDR3 sequences combining an N-anchored
  and a C-anchored expert, normalised exactly per length, over amino acids or
  nucleotides, with a hierarchical Dirichlet prior whose strengths are
  estimated from the data.
- Exact scoring (`pseq`, `log_pseq`, batched, streaming and Arrow input),
  exact sampling, saving and loading.
- Exact occupancy quantities with rigorous error bounds (`ExactOccupancy`):
  expected richness at any sequencing depth, overlap between samples, and
  sequences private to one repertoire of a cohort.
- `PublicnessModel`: the distribution of how many repertoires in a cohort
  contain a sequence.
- The local-window repertoire distance (`BEAM.analytics.distance.compare`),
  split exactly into length and local-pattern terms.
- Probability mixtures of repertoires (`Mixture`), depth-matched diversity
  profiles, repertoire likelihoods, sequence logos and an exact query engine.
- `BEAM.load_pretrained`: published models, downloaded once and verified by
  SHA-256.
- Wheels for Linux (x86-64, ARM), macOS (Apple Silicon) and Windows
  (x86-64), Python 3.11 to 3.15.
