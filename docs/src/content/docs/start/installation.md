---
title: Installation
description: Install BEAM from PyPI on Linux or macOS, under WSL on Windows, or from source.
sidebar:
  order: 1
---

## With pip

```bash
pip install beam-model
```

The package is called `beam-model`; in Python it is imported as `BEAM`. Ready-built packages exist
for:

| Platform | Architectures | Python |
|---|---|---|
| Linux | x86-64, ARM (aarch64) | 3.11 to 3.15 |
| macOS | Apple Silicon (arm64) | 3.11 to 3.15 |

NumPy and SciPy are installed with it. Nothing else is needed.

Check that it works:

```bash
python -c "import BEAM; print(BEAM.__version__)"
```

## On Windows

BEAM's core is written in C for Linux and macOS compilers. On Windows, install it inside
[WSL](https://learn.microsoft.com/windows/wsl/install) (Windows Subsystem for Linux): open an
Ubuntu terminal there and use `pip install beam-model` as above.

## From source

To use the newest code, or to contribute, build BEAM from its repository. You need a C compiler
with OpenMP: GCC or Clang on Linux; on macOS, Apple's compiler plus `libomp`
(`brew install libomp`).

```bash
git clone https://github.com/MuteJester/BEAM.git
cd BEAM
pip install -e ".[dev]"
```

Without `libomp` on macOS, BEAM still builds and gives the same results, single-threaded.

## Memory

A model built from one repertoire takes from a few megabytes to a few hundred. The reference
model built from 38,538 repertoires takes about 1 GB once loaded, so a machine with 4 GB of
memory or more is comfortable.
