---
title: Published models
description: List, download and load the published BEAM models, and where they are stored.
sidebar:
  order: 23
---

```python
import BEAM

BEAM.pretrained_models()                     # what is available
model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")   # a ready BeamModel
path = BEAM.pretrained_path("human_tcrb_cdr3_aa_foundation")    # the local file, downloading it if needed
```

| Name | Description |
|---|---|
| `human_tcrb_cdr3_aa_foundation` | Built from 38,538 human TCRβ repertoires from 447 published studies; 926,909,993 distinct CDR3β sequences, written in junction form (from the conserved C to the closing F or W), up to 27 residues; order 4. About 90 MB to download, about 1 GB in memory once loaded. |

`load_pretrained(name, *, progress=True)` and `pretrained_path(name, *, progress=True)` print
download progress; `progress=False` silences it. `pretrained_models()` returns, for each model,
its `name`, `release`, `file`, `bytes`, `sha256`, `format`, `description`, `urls`, `doi`,
`citation`, `license` and `built_with`.

## Download and checking

A model is downloaded once and kept in a cache directory. Its SHA-256 checksum is checked
before it is first loaded and on every later load; a damaged file is downloaded again, and a
file that does not match is never loaded. Once cached, loading needs no network. Several
processes can download at once safely: each file is checked, then moved into place.

The cache is `$BEAM_CACHE_DIR` if set, otherwise `$XDG_CACHE_HOME/beam-model` if that is set,
otherwise `~/.cache/beam-model` on Linux, `~/Library/Caches/beam-model` on macOS and
`%LOCALAPPDATA%\beam-model` on Windows (`BEAM.pretrained.cache_dir()` returns it). Files are
kept at `<cache>/<name>/<release>/<file>`: on a machine without network, copying a file there is
enough, and it is checked against its SHA-256 before use.

An unknown name raises `KeyError` listing the available models; a model in a newer file format
raises `ValueError` asking you to upgrade BEAM (`pip install -U beam-model`); if no mirror
delivers a matching file, `OSError` lists each address's error.

## Where the models come from

Published models are released separately from the code, as GitHub releases tagged `models-vN`
and archived on Zenodo with a DOI. Each release has a model card describing how the model was
built and what it is for.

`BEAM_PRETRAINED_REGISTRY` points BEAM at a registry file of your own, for a mirror or for your
own models:

```json
{"schema": 1, "models": {"my_model": {"release": "v1", "file": "my_model.beam", "bytes": 123456,
  "sha256": "…", "format": 9, "description": "…", "urls": ["https://…/my_model.beam"]}}}
```
