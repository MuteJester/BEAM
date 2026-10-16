---
title: Model files
description: What a saved .beam file contains, how it is checked, and which versions read it.
sidebar:
  order: 22
---

```python
model.save("donor_A")                       # writes donor_A.beam
model = BeamModel.from_file("donor_A.beam")
```

A `.beam` file stores a model's **counts** and its configuration (alphabet, order, maximum
length, expert weights and prior strengths), so a loaded model gives exactly the same
probabilities and can still be combined with other models. Only observed contexts are stored.

Each file also records when the model was built (`model.metadata()["created"]`) and carries
checksums over its contents: a damaged or truncated file is refused with `ModelFileError`
rather than read wrongly.

## Compatibility

BEAM 1.0 reads and writes file **format 9**. Later releases keep reading format 9 files; a
release that could not would be a new major version. A file from a pre-release development
build is refused with a message naming its format; rebuild it from its sequences.
