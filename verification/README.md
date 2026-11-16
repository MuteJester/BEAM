# Verification

Evidence for the reorganization, tracked in the repository rather than left in
an ignored directory. It was in `.private/` until a reviewer pointed out that
`git clean -fdx` would have destroyed the only copies.

## Which interpreter

Run the gate as `.venv/bin/python verification/check.py`, and re-record any
baseline the same way.

Not a style preference. The test inventory compares what the running
interpreter collects against a list recorded from another, and
`tests/suites/io/test_parquet_streaming.py` opens with
`pytest.importorskip("pyarrow")` at module level. An interpreter without
pyarrow collects nothing from that file, `--collect-only` reports "636 tests
collected" instead of 642 with no error and no skip count, and the six cases
look deleted. Re-recording from there would delete them.

The gate now refuses rather than reporting that as missing tests: a file the
baseline knows about, still on disk, that collected nothing here, is an
environment fault and is named as one. A file that is genuinely gone still
reads as missing tests, which is the distinction that matters.

## Run anything that builds a model under `verification/bounded`

```bash
verification/bounded -m 12 -t 900 -- .venv/bin/python -m pytest tests -q
PYTHONPATH=src verification/bounded -- .venv/bin/python verification/check.py
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

## Looking for data races

```bash
make tsan                      # clang + libomp + Archer, into build-tsan/
tools/qa/run c --tsan          # 33 suites, Archer loaded, TSAN_OPTIONS set
```

Nothing in CI runs this; it needs a clang toolchain
(`sudo apt install clang-21 libomp-21-dev`) and it is a developer's tool. Run
it after anything that changes a parallel region.

Run the binaries in `build-tsan/` any other way and the answer is noise.
`libomp` carries no sanitiser annotations, so without Archer loaded as an OMPT
tool ThreadSanitizer has no happens-before edge between any two OpenMP
threads: every implicit barrier is invisible and every region whose result
somebody later reads is reported as a race. On one suite at 8 threads that is
783 reports with Archer off and 0 with it on; under GCC and `libgomp`, where
there is no Archer at all, it was 1186. `--tsan` is the mode that gets all
three settings right -- the tool, `ignore_noninstrumented_modules=1`, and no
`ulimit -v`, which TSan refuses to start under.

Four suites are not instrumented, all four about the allocator, which is the
one thing TSan replaces wholesale. `verification/tests/test_tsan.py` derives
that set from the sources rather than listing it, and its last test is the one
that matters: it runs a suite with Archer off and checks the reports appear,
because a clean run is only evidence if a dirty one is reachable.

## Observing the teams that actually ran

```bash
make teamwidth                                       # build/teamwidth.so
python -m pytest verification/tests/test_teamwidth.py -q
```

Every other thread check here is static. `thread counts` proves each of the 93
parallel regions *names* a thread count; `thread plan wiring` proves each
GIL-releasing region *resolves* a plan. Neither can see what the runtime
granted, and that is precisely where the nesting defect lived -- `thread
counts` was OK in the same gate run that the defect was live in, because the
clause was present and correct and only its value on a worker thread was wrong.

`teamwidth.c` interposes the two libgomp entry points BEAM creates teams
through and records, for every region entry, the count requested and the width
that actually ran. One `entropy()` call creates **949 teams**, which is why
reading 93 pragmas could never have answered the question. It observes the
shipped extension through `LD_PRELOAD`, with no rebuild, so it measures the
same binary the gate hashes.

Both numbers are recorded because the mutation says they are not
interchangeable. With the nested-region guard removed from `beam_threads()`, in
the **default** environment, BEAM asked for 16 threads against a plan of 8 and
every team that ran was still the correct width -- libgomp declines to widen a
nested region when nesting is off. A check on the width that ran passes on that
mutant; only the requested count catches it. The runtime's default policy was
the only thing between the plan and sixteen-way oversubscription, which is luck
rather than design, so the requested channel is the primary assertion.

Two of the tests exist to stop the file lying to you. One reads the built
extension's imports and fails if BEAM ever grows a third way to create a team,
since the observer would then be blind *and* quiet. The other requires the
measurement to have reached a nested region, because a dead path and a
well-behaved one report the same zero.

## Running the gate

```bash
make clean && make test          # 439 C tests
.venv/bin/python -m pytest tests -q     # Python suite
.venv/bin/python verification/check.py    # the contract below
.venv/bin/python -m pytest verification/tests -q   # the tests guarding this gate
```

The last line is the suite that guards the checks themselves -- the comparison
logic, the record selector, the inventory readers, the GIL-region and
thread-count scans, and the race detector's settings. It lives outside `tests/` because CI copies `tests/`
elsewhere and runs it against the installed package, so a test that imports
this repository's tooling cannot live under it. It runs in CI as
`Verification tooling`; until that job existed it ran only by hand.

`check.py` compares, against `baseline/`:

| Check | What it catches |
|---|---|
| `_cbeam` callables | an entry point renamed or dropped |
| C public symbols | a symbol declared in `include/beam/` going missing |
| Python public API | a moved default, a renamed keyword, a method that stopped being public |
| pytest inventory | a test that stopped being collected -- a subset of a passing suite also passes |
| C test inventory | the same for `RUN_TEST` cases |
| thread plan wiring | a region that drops the GIL and resolves no thread plan, an unbalanced plan macro, or a path that escapes one |
| thread counts | a parallel region naming no thread count -- so it runs at one thread per core whatever the plan resolved -- or naming one outside the permitted forms |
| numeric | 78 quantities, order 2/3/4 x default/uneven prior, as hex floats |
| saved-model bytes | the sha256 of a written model |
| thread independence | results at 1, 4 and 16 threads |
| exact | 416 quantities against brute-force enumeration of the whole support |

Every row above `exact` pins REPRODUCIBILITY: each quantity is compared with a
recording of ITSELF, so a quantity that has been wrong since the day it was
written passes for ever. Three defects lived inside quantities `numeric`
already pinned -- `positional_marginals` returning rows summing to 5.9e-264,
the tilted sweeps overflowing at any negative `q`, and a zero anchor weight
filling the edge weights with NaNs -- and the baseline never moved for any of
them.

`exact.py` is the answer to that. It builds models small enough to enumerate
the whole support, scores every sequence once, and checks each exported
functional against a direct sum over the resulting vector. Nothing in it reads
a baseline file. Its tolerances are measured rather than guessed:
`BEAM_EXACT_VERBOSE=1` prints how much of each one is actually consumed (worst
is currently 5.7%).

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
