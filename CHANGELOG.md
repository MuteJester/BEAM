# Changelog

All notable changes to BEAM are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and BEAM uses
[Semantic Versioning](https://semver.org/): see [`RELEASING.md`](RELEASING.md)
for what counts as a breaking change, including changes to the model file
format.

## [Unreleased]

Nucleotide models now hold reads of up to 512 bases. Format-9 model files
still load, and every default call returns identical numbers; the exceptions
are listed under Changed.

### Added
- Nucleotide models of up to 512 residues, enough for reads of about 400
  bases. Amino-acid models keep their limit of 255.
- `max_max_length` in the alphabet descriptor, so the Python layer reads each
  alphabet's limit from C instead of restating it. This is not a public path:
  `BEAM.model.alphabets` is not exported and the recorded API surface does not
  cover it. The limits themselves are documented, and a refusal now names the
  alphabet and the range it is quoting.
- `BEAM.BeamPerformanceWarning`, raised before a build that costs far more
  than it looks. Finalizing runs one partition-function computation for every
  length from 1 to `max_length`, not only the observed ones, so the cost grows
  with the square of `max_length`: about 24 times the nucleotide default at
  400. The cost is unchanged and only announced.
- `BEAM.constants.BEAM_LENGTH_BUFFER`, the size of the per-length buffers,
  kept separate from the validation limit.

### Changed
- **`max_length='data'` on a nucleotide corpus now stops at 512 instead of
  255.** A build that used to skip sequences longer than 255 silently now keeps
  them, sizes the model larger and emits a `BeamPerformanceWarning`, so the
  same call on a corpus with 400-base reads returns a different model.
  Amino-acid corpora are unaffected.
- **C API break.** `BeamModelConfig.max_length` is now `uint32_t`. The size of
  the struct and the offset of every field after it moved: `sizeof` went from
  128 to 136 bytes, `max_length` from offset 5 to 8, and `alpha` from offset 24
  to 32. A C caller built against 0.9.0's header and linked against this
  release finds `alpha` eight bytes from where it looks and reads its prior
  strength out of padding. Rebuild C code against the headers it links; see
  "The C API" in [`RELEASING.md`](RELEASING.md).
- Models longer than 255 are written as format 10. Builds before this release
  refuse them with a "written by a newer version of BEAM, upgrade" message
  rather than a corruption error. Models of 255 or fewer are still written as
  format 9, byte for byte as before, and this release reads both.
- `max_length` is validated per alphabet and per order, because the real limit
  is `max_length` times the alphabet size to the power of the order against a
  32-bit row index. Both configuration errors now name the alphabet whose
  limit they quote, for `order` as well as `max_length`; the order limits are
  further apart (4 for amino acids, 9 for nucleotides) than the length limits.

### Fixed
- The lifecycle guards now run in an order in which each refusal can be acted
  on: a row count no budget can fix is no longer answered with an offer to
  raise the budget. The row bound was also one rank block too loose.
- The lower bound on `max_length` in `config.py` was 2 where the C library
  accepts 1, so Python refused a length that C allowed.
- `BEAM_MAX_LENGTH_CEILING` in `constants.py` was 255 while the C header said
  512. It had no live reader.

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
