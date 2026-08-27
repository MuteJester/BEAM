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

### Open world or closed

By default the model smooths, so every sequence over the alphabet gets a finite
probability and `path_count()` counts the whole lattice. Pass `strict=True` to
turn smoothing off: unseen sequences then score `-inf`, and `path_count()`
becomes the number of sequences the training data can actually spell.

```python
open_world = BeamModel(seqs, abundances=ab)                 # every path positive
closed     = BeamModel(seqs, abundances=ab, strict=True)    # observed support only
```

## Testing

```bash
make test                 # C unit tests
python -m pytest tests/   # Python suite
```

## Layout

| Path | Contents |
|---|---|
| `include/beam/` | Public C headers |
| `lib/model/` | The model: build, score, sample, analytics, sparse storage, I/O |
| `lib/occupancy/` | Poisson-binomial sharing model (`PublicnessModel`) |
| `lib/core/` | Shared support: error handling, RNG, CRC32C |
| `src/BEAM/` | Python package and its C extension |
| `docs/api/model.md` | Full API reference |

## A note on the file format

Saved models carry the magic `0x4245414D` (`"BEAM"`). This model shipped as
EDGE before it was split into its own repository, under a different magic, and
files written then are **not loadable** even though the format is otherwise
identical. That is deliberate: the foundation models are being rebuilt, and a
file that loads under a name the project no longer uses is worse than one that
refuses with a reason. Such files are detected and refused by name, so the
failure says "written by EDGE, rebuild it" rather than looking like corruption.
