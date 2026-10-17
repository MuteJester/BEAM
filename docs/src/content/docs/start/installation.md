---
title: Installation
description: Install BEAM from PyPI on Linux, macOS or Windows, or from source.
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
| macOS 14 or newer | Apple Silicon (arm64) | 3.11 to 3.15 |
| Windows | x86-64 | 3.11 to 3.15 |

NumPy and SciPy are installed with it. Nothing else is needed.

Check that it works:

```bash
python -c "import BEAM; print(BEAM.__version__)"
```

## From source

To use the newest code, or to contribute, build BEAM from its repository. You need a C compiler
with OpenMP: GCC or Clang on Linux; on macOS, Apple's compiler plus `libomp`
(`brew install libomp`); on Windows, the
[Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) plus
[LLVM](https://releases.llvm.org/), whose `clang-cl` compiles BEAM and whose `libomp` provides
multithreading. If LLVM is not on `PATH`, set `BEAM_LLVM_PREFIX` to its folder.

```bash
git clone https://github.com/MuteJester/BEAM.git
cd BEAM
pip install -e ".[dev]"
```

Without `libomp` on macOS or Windows, BEAM still builds and gives the same results,
single-threaded.

## Memory

A model built from one repertoire takes from a few megabytes to a few hundred. The reference
model built from 38,538 repertoires takes about 1 GB once loaded, so a machine with 4 GB of
memory or more is comfortable.
