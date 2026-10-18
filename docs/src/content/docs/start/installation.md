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

Check that it works, and how many processor cores it will use:

```bash
python -c "import BEAM; print(BEAM.__version__, BEAM.effective_threads(), 'threads')"
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

## Speed and threads

BEAM uses every processor core by default; `BEAM.set_threads(n)` limits it, for example on a
shared machine or inside a job scheduler. The answers do not depend on the number of threads:
one thread and sixteen give the same result, bit for bit.

On Linux x86-64, BEAM's heaviest sums over all sequences use the processor's AVX2 or AVX-512
instructions where it has them. The Windows packages do not use these yet, so the same work can
take longer there: in our tests, the same set of calculations took about twice as long on
Windows as on Linux, on GitHub's 4-core machines for each. The answers agree: Windows and Linux
gave the same results to within 3 × 10⁻¹¹ (relative).

## Memory

A model built from one repertoire takes from a few megabytes to a few hundred. The reference
model built from 38,538 repertoires takes about 1 GB once loaded, and briefly about 2 GB while
loading, so a machine with 4 GB of memory or more is comfortable for scoring, sampling and most
analyses.

The heaviest calculations over all sequences need more. BEAM checks what such a calculation needs
before it starts, against a memory budget of 4 GiB by default, and refuses with a message giving
the amount needed rather than running out of memory. The finest publicness spectrum in these
pages, for example, needed about 20 GB. To allow more, raise the budget with
`BEAM.set_memory_budget`; see [Threads and memory](/BEAM/reference/runtime/).
