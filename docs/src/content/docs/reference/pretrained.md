---
title: Published models
description: List, download and load the published BEAM models, and where they are stored.
sidebar:
  order: 23
---

```python
import BEAM

BEAM.pretrained_models()                     # what is available
model = BEAM.load_pretrained("foundation")   # a ready BeamModel
path = BEAM.pretrained_path("foundation")    # the local file, downloading it if needed
```

| Name | Description |
|---|---|
| `foundation` | Built from 38,538 human TCRβ repertoires from 447 published studies; 926,909,993 distinct CDR3β sequences. About 90 MB to download, about 1 GB in memory. |

## Download and checking

A model is downloaded once and kept in a cache directory. Its SHA-256 checksum is checked
before it is first loaded and on every later load; a damaged file is downloaded again, and a
file that does not match is never loaded. Once cached, loading needs no network.

The cache is `$BEAM_CACHE_DIR` if set, otherwise `$XDG_CACHE_HOME/beam-model` if that is set,
otherwise `~/.cache/beam-model` on Linux and `~/Library/Caches/beam-model` on macOS.

## Where the models come from

Published models are released separately from the code, archived on Zenodo with a DOI and
mirrored on GitHub. Each release has a model card describing how the model was built, what it
is for and its limitations. `BEAM.pretrained_models()` lists the name, size, release, DOI and
description of each.
