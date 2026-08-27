# BEAM

**B**idirectional **E**xpert-**A**nchored **M**odel: a probability model over
CDR3 amino-acid sequences in which every residue is scored twice.

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
because a support of `sum_L 20**L` describes the smoother rather than the
data, and it should not arrive by default.

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
| `src/native/algorithms/` | Sampling and exact top-k |
| `src/native/occupancy/` | Poisson-binomial sharing model (`PublicnessModel`) |
| `src/native/storage/` | Sparse row storage |
| `src/native/io/` | Model persistence |
| `src/native/core/` | Errors, logging, alphabet, runtime, RNG, CRC32C |
| `src/native/bindings/python/` | The `_cbeam` extension, split by domain |
| `src/BEAM/` | Python package: `model/`, `analytics/`, `math/`, `io/`, `publicness/` |
| `build_support/` | Explicit source manifests read by both `Makefile` and `setup.py` |
| `tests/python/` | Python suite, grouped as `model/`, `math/`, `analytics/`, `publicness/` |
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
