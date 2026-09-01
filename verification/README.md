# Verification

Evidence for the reorganization, tracked in the repository rather than left in
an ignored directory. It was in `.private/` until a reviewer pointed out that
`git clean -fdx` would have destroyed the only copies.

## Run anything that builds a model under `verification/bounded`

```bash
verification/bounded -m 12 -t 900 -- python -m pytest tests -q
PYTHONPATH=src verification/bounded -- python verification/check.py
```

Not optional, and not only for sweeps. BEAM can be asked for a configuration
whose tables run to hundreds of gigabytes, and `calloc` on Linux grants that
lazily: the process does not fail where it asked, it fails much later when the
pages are first written, by which point the OOM killer is picking a victim
from the whole machine. During this work that victim was twice the developer's
editor -- once from a review agent sweeping configurations, once from a
one-line sanity check typed by hand.

`beam_model_new` now refuses a configuration larger than
`beam_model_memory_budget()` before allocating, which closes the common case.
`bounded` is the belt to that braces: it sets a hard address-space ceiling, so
anything the guard does not anticipate fails inside the process that asked
rather than on the machine around it. Use it for one-liners too. That is where
it went wrong both times.

## Running the gate

```bash
make clean && make test          # 66 C tests
python -m pytest tests -q        # Python suite
python verification/check.py     # the contract below
```

`check.py` compares, against `baseline/`:

| Check | What it catches |
|---|---|
| `_cbeam` callables | an entry point renamed or dropped |
| C public symbols | a symbol declared in `include/beam/` going missing |
| Python public API | a moved default, a renamed keyword, a method that stopped being public |
| pytest inventory | a test that stopped being collected -- a subset of a passing suite also passes |
| C test inventory | the same for `RUN_TEST` cases |
| numeric | 78 quantities, order 2/3/4 x open/strict, as hex floats |
| saved-model bytes | the sha256 of a written model |
| thread independence | results at 1, 4 and 16 threads |

The numeric check is exact except for the publicness keys, which are compared
within 1e-9 relative. Those run their forward transform through numpy's FFT,
whose last bits depend on the numpy build; demanding bit-identity of another
library's FFT is not a property of BEAM, and doing so made this gate fail on a
reviewer's machine while passing here.

The 12 MB `.beam` fixture is deliberately not tracked. `check.py` rebuilds it
from `baseline/corpus.json` and compares the sha256 recorded in
`baseline/numeric.json`, so the bytes are still pinned without carrying them.

## What else is here

- `FOUNDATION_EQUIVALENCE.md` -- the BEAM foundation model against the one
  LZGraphs produced as EDGE: identical across all 701,713,108 payload bytes.
- `REORGANIZATION.md` -- phase-by-phase record, including everything
  deliberately left undone and why.
- `baseline/RERECORDED.md` -- why two baseline files were re-recorded, with
  the Phase 0 versions kept beside them.
