---
title: Model files
description: What a saved .beam file contains, how it is checked, and which versions read it.
sidebar:
  order: 22
---

```python
from BEAM import BeamModel

model.save("donor_A")                       # writes donor_A.beam
model = BeamModel.from_file("donor_A.beam")
```

A `.beam` file stores a model's **counts** and its configuration (alphabet, order, maximum
length, expert weights and prior strengths), so a loaded model gives exactly the same
probabilities and can still be combined with other models. By default only observed contexts
are stored. `save` adds `.beam` only when the path has no suffix; `from_file` reads any path and
recognises the file by its contents.

Each file also records when the model was built (`model.metadata()["created"]`) and ends in a
CRC-32C checksum of its contents: a damaged or truncated file is refused with `ModelFileError`
rather than read wrongly. A sound file too large for the
[memory budget](/BEAM/reference/runtime/#memory-budget) raises `BudgetExceeded` instead.

Save before `compact()`: a compacted model has no counts left to write (`NotFinalized`).

Building the same model twice writes the same file, apart from the build times and the final
checksum. `BEAM.file_provenance_span()` returns the `(offset, length)` of the build times,
`(32, 24)` in the current format, for byte-for-byte comparisons.

## Compatibility

BEAM 1.0 reads and writes file **format 9** (`BEAM.pretrained.MODEL_FORMAT`). The file format is
part of the API: a release that writes a new format still reads the previous one, or it is a new
major version. A file from a pre-release development build is refused with a message naming its
format; rebuild it from its sequences.
