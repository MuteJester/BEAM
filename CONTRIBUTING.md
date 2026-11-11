# Contributing to BEAM

Bug reports, questions and pull requests are welcome.

- **Questions and ideas:** [Discussions](https://github.com/MuteJester/BEAM/discussions).
- **Bugs:** [Issues](https://github.com/MuteJester/BEAM/issues), with a
  minimal example, your BEAM version (`python -c "import BEAM; print(BEAM.__version__)"`),
  Python version and operating system.
- **Changes:** fork, branch from `main`, and open a pull request. CI must
  pass, and new behaviour needs tests.

## Development setup

You need a C compiler with OpenMP (GCC or Clang; on macOS,
`brew install libomp`; on Windows, LLVM's clang-cl with the Microsoft C++
Build Tools) and Python 3.11 or newer.

```bash
git clone https://github.com/MuteJester/BEAM.git
cd BEAM
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

`pip` fetches the build requirements itself. If you call `setup.py` directly
(for example `python setup.py build_ext --inplace`), your environment needs
`setuptools>=77`.

## Two build systems, one set of sources

- `pip install -e .` / `setup.py` build the Python extension `BEAM._cbeam`.
- `make` builds `build/libbeam.a` and the C unit tests, from a separate
  object tree. **It does not rebuild the extension Python imports.**

Both read the explicit source lists in `build_support/`; add a new `.c` file
there, not to a glob.

`build_ext` decides whether to rebuild by comparing modification times, and
an edit made within the same second as the previous build can be skipped
silently. When you compare results before and after a C change, rebuild with
`python setup.py build_ext --inplace --force` and check the `.so` actually
changed.

## Tests

```bash
python -m pytest tests            # Python suite (1,224 tests)
make test                         # C suites, every one of them (390 tests)
python -m pytest verification/tests   # the tooling: the gate, the runner, the harness
python tools/release/check_readme.py README.md   # the README's examples
python verification/check.py      # the verification gate
```

`make test` runs `tools/qa/run c`, which reports every suite instead of
stopping at the first failure, caps the threads each one may claim, and writes
a JSON record per test. `tools/qa/run c --suite NAME` runs one of them and
`make list-tests` names every C test without running it.

`verification/tests` is the second kind of test in this repository: it does
not test BEAM, it tests the things that test BEAM -- that the gate can still
see a deleted test, that the recorder refuses to write a short inventory, that
the C harness reports the outcome a body actually reached. It is deliberately
outside `testpaths`, because everything under `tests/` must be able to run
from a copy of that directory alone, so it needs naming here to be run at all.

The verification gate compares the public API, the C symbols, both test
inventories, and every recorded number (as exact hex floats) against
`verification/baseline/`. A deliberate change to any of them is recorded with
`python verification/check.py --record TAG` and explained in a new section of
`verification/baseline/RERECORDED.md`; the recorder refuses if the
independent check against brute-force enumeration fails.

## Layout

| Path | Contents |
|---|---|
| `include/beam/` | Public C headers |
| `src/native/model/` | Model state: config, lifecycle, training, smoothing, finalize |
| `src/native/dp/` | The dynamic programs over the trellis |
| `src/native/math/` | Named probability operations: Mellin, moments, spectrum, diversity |
| `src/native/mixture/` | Probability mixtures of several models |
| `src/native/algorithms/` | Sampling, exact top-k, fingerprints, repertoire distance |
| `src/native/occupancy/` | Poisson-binomial sharing model (`PublicnessModel`) |
| `src/native/storage/` | Sparse row storage |
| `src/native/io/` | Model persistence |
| `src/native/core/` | Errors, alphabet, runtime, RNG, CRC32C |
| `src/native/bindings/python/` | The `_cbeam` extension, split by domain |
| `src/BEAM/` | The Python package |
| `src/BEAM/pretrained/` | The published-model registry and loader |
| `build_support/` | Source manifests read by both `Makefile` and `setup.py` |
| `tests/suites/`, `tests/native/` | Python and C test suites |
| `tests/harness/` | The C harness: outcomes, counters, per-test timing, JSON |
| `tests/_engineering/` | Shared engineering: the corpora and the built fixtures |
| `tests/oracles/` | The independent references the implementation is checked against |
| `tools/qa/run` | The runner: every suite, every failure, a thread cap |
| `verification/` | The verification gate and its baselines |
| `tools/release/` | Release tooling: smoke test, README check, changelog, model publishing |
| `docs/` | The documentation website (Astro Starlight); `docs/examples/` are its recorded examples |

## The model file format

Saved models start with the magic number `0x4245414D` (`"BEAM"`) and a format
number, and store counts plus the full configuration, so a loaded model can
be added to and re-finalized. BEAM writes **format 9** for a model of at most 255 residues and **format 10** for a longer one, and reads both. Any
later format must keep reading format 9 files, or be released as a new major
version (see [`RELEASING.md`](RELEASING.md)).

## Releases

Maintainers: see [`RELEASING.md`](RELEASING.md).
