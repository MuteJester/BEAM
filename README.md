# BEAM

**B**idirectional **E**xpert-**A**nchored **M**odel: a probability model over
CDR3 sequences in which every residue is scored twice.

One expert is **N-anchored**, indexing a residue by its distance from the front
of the sequence. The other is **C-anchored**, indexing it by its distance from
the back. The two readings are combined log-linearly and normalised per length
by an exact forward dynamic program over a layered DAG. Nothing is sampled and
nothing is pruned: every probability the model reports is exact given its
tables.

## Install

```bash
pip install -e .
```

Requires a C compiler with OpenMP and NumPy. No other runtime dependencies.

## Use

```python
from BEAM import BeamModel

model = BeamModel(sequences, abundances=abundances, order=4, max_length=27)
model.log_pseq("CASSLGYEQYF")        # exact log probability
model.save("foundation")             # -> foundation.beam
```

### Amino acids or nucleotides

The same model either way — two anchored experts, one table each, exact
dynamic programs. Only the symbols change, and with them the size of every
table.

```python
BeamModel(seqs)                        # amino acids: ACDEFGHIKLMNPQRSTVWY
BeamModel(seqs, alphabet="nt")         # nucleotides: ACGTN
```

`order` and `max_length` default to what the alphabet needs:
order 4 over 27 residues, or order 6 over 81 bases, three bases to the
residue. An amino acid is a compressed codon, so matching the amino-acid
model's context *span* would need order 12 and is not buildable; order 6 is
two codons and is cheaper than the amino-acid model already is.

Nucleotide input is case-folded, so `acgt` and `ACGT` are one sequence. `N`
is a residue rather than a wildcard — a read carrying one is data the model
can represent — while the other IUPAC ambiguity codes are refused, since `R`
means "A or G" and the model has nothing to say about that. Amino-acid input
is not folded: case carries no meaning for a base, but folding residues would
silently widen what is accepted.

There is no auto-detection. `ACGT` is a perfectly good peptide, so a
heuristic would quietly build the wrong model from short or low-complexity
input; the alphabet is always named.

### Combining models: counts or probabilities

Two different things, and choosing the wrong one is the most consequential
mistake this library makes easy.

```python
a.merged(b)              # ADD THE COUNTS: the model you would get by
                         # training on the pooled corpus
Mixture([a, b])          # MIX THE PROBABILITIES: pick a donor, then a
                         # sequence from that donor
```

The pooled model has a strictly larger support, because a pooled path may
splice an edge learned from one donor onto an edge learned from another and
spell a sequence no single donor spells. Pooling 64 donors at a fixed
30,000-sequence budget raises `D0` by 24.3% while the number of transitions
rises 0.57%.

`Mixture` is the object beta diversity between individuals is defined on, and
`D0(union) <= D0(merged)` always. See [`docs/api/python/mixture.md`](docs/api/python/mixture.md),
which also says which Hill orders are exact (`q = 0` and integer `q >= 2`),
which are bounded (`q = 1`, refused rather than estimated), and what one inner
product costs.

### Diversity at a common sequencing depth

```python
from BEAM import depth_matched_hill_profiles

r = depth_matched_hill_profiles([seqs_a, seqs_b], [templates_a, templates_b],
                                seed=1)
r.profiles                     # (2 repertoires, 20 resamples, 31 orders)
r.resampling_band(5, 95)       # a resampling band, not a confidence interval
```

Each repertoire is resampled to the smallest template total in the call, and
every resample is a freshly fitted model. The depth belongs to the call, so
profiles are comparable only within one. See
[`docs/api/python/diversity.md`](docs/api/python/diversity.md).

### Closed world by default

The model holds what the sequences you gave it actually spell. A sequence no
path supports scores `-inf`, and `path_count()` is the number of sequences the
data can spell rather than the size of the whole lattice. Storage is sparse,
so only occupied rows exist.

```python
model = BeamModel(seqs, abundances=ab)          # observed support only
model.log_pseq("CWWWWWWWF")                     # -inf, if nothing spells it
```

Pass `strict=False` for the open world instead. Hierarchical backoff then
leaves every table cell positive, so every sequence over the alphabet receives
a graded probability and the support becomes all of sequence space:

```python
open_world = BeamModel(seqs, abundances=ab, strict=False)
```

That is a different model, and a useful one: it is what lets BEAM rank
sequences the corpus never saw. It is no longer what you get without asking,
because a support of `sum_L R**L`, for an alphabet of `R` residues, describes
the smoother rather than the data, and it should not arrive by default.

## Testing

```bash
make test                 # C unit tests
python -m pytest tests    # Python suite
```

## Layout

| Path | Contents |
|---|---|
| `include/beam/` | Public C headers |
| `src/native/model/` | Model state: config, lifecycle, training, smoothing, finalize |
| `src/native/dp/` | The dynamic programs over the trellis |
| `src/native/math/` | Named probability operations: Mellin, moments, spectrum, diversity |
| `src/native/mixture/` | Probability mixtures of several models: inner products, support unions |
| `src/native/algorithms/` | Sampling and exact top-k |
| `src/native/occupancy/` | Poisson-binomial sharing model (`PublicnessModel`) |
| `src/native/storage/` | Sparse row storage |
| `src/native/io/` | Model persistence |
| `src/native/core/` | Errors, logging, alphabet, runtime, RNG, CRC32C |
| `src/native/bindings/python/` | The `_cbeam` extension, split by domain |
| `src/BEAM/` | Python package: `model/`, `analytics/`, `math/`, `io/`, `publicness/`, `query/`, `mixture/`, `insights/`, `diversity/` |
| `build_support/` | Explicit source manifests read by both `Makefile` and `setup.py` |
| `tests/python/` | Python suite, grouped as `model/`, `math/`, `analytics/`, `publicness/`, `query/`, `mixture/`, `insights/`, `diversity/` |
| `tests/native/` | C unit tests |
| `tests/reference/` | An independent reimplementation, used to check the model against |
| `docs/` | `architecture/`, `api/python/` |

## A note on the file format

Saved models carry the magic `0x4245414D` (`"BEAM"`). This model shipped as
EDGE before it was split into its own repository, under a different magic, and
files written then are **not loadable** even though the format is otherwise
identical. That is deliberate: the foundation models are being rebuilt, and a
file that loads under a name the project no longer uses is worse than one that
refuses with a reason. Such files are detected and refused by name, so the
failure says "written by EDGE, rebuild it" rather than looking like corruption.
