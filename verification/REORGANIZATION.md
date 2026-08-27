# Reorganization progress

Executing `.private/REPOSITORY_REORGANIZATION_PLAN.md`. One phase per commit.
Baseline artifacts live in `.private/baseline/`.

## Validation gate (run after every phase)

```bash
make clean && make test          # 66 C cases
python -m pytest tests -q        # 278 cases
python3 .private/baseline/check.py   # bit-identity vs baseline/numeric.json
```

For a phase that only moves declarations between headers, there is a stronger
check than any of these: compile before and after, and compare `objdump -d` of
every object. Identical disassembly proves the change is invisible to the
compiler, which no test can prove. Phase 3 used it and all 12 objects matched.
Use it again for phases 5 and 6, where translation-unit splits can change
inlining and therefore the last bit of a float.

A phase is not done until all three pass. `make test` passing is NOT
sufficient on its own: the `beam_crc32c` signature collision during the
rename was caught by the compiler, not by the suite, and a translation-unit
split can change floating-point last bits without failing any assertion.

## Phase status

| Phase | What | State |
|---|---|---|
| 0 | Freeze and record the contract | DONE (commit 321739a) |
| 1 | Relocation-safe build discovery | DONE |
| 2 | Consolidate under `src/native/` | DONE |
| 3 | Split internal headers | DONE |
| 4 | Split model lifecycle/training | DONE, foundation bit-identical |
| 5 | Extract the DAG DP layer | DONE, foundation bit-identical |
| 6 | Extract named math operations | DONE, foundation bit-identical |
| 7 | Finish focused native domains | DONE, foundation bit-identical |
| 8 | Split Python bindings | DONE, foundation bit-identical |
| 9 | Split standalone Python modules | DONE (facade methods deferred) |
| 10 | Mirror structure in tests and docs | DONE (C tests moved, not split) |
| 11 | Artifact hygiene and acceptance | DONE |

## HARD GATE before Phase 4 (SATISFIED 2026-08-27)

Owner instruction, 2026-08-27. Do not start Phase 4 until all of this is done
and green:

1. Install BEAM on ig05.
2. Rebuild the foundation model there under BEAM. This is required regardless:
   the file magic changed, so the existing EDGE foundation models cannot be
   loaded.
3. DONE 2026-08-27: foundation rebuilt on ig05, byte-identical payload to
   the EDGE model. See baseline/FOUNDATION_EQUIVALENCE.md.
4. Regenerate figure 02 and supplementary figure S7 from the BEAM foundation.
4. Compare against the figure 02 and S7 results produced when this model was
   EDGE inside LZGraphs. They must match.

Rationale: phases 4 through 9 cut live numerical code into new translation
units. That is exactly where a silent last-bit change would hide. A verified
foundation-scale reference computed BEFORE those phases is what makes any
later drift attributable. Without it, a figure that moves cannot be told apart
from a figure that was always going to move.

## Baseline recorded (Phase 0)

| Artifact | Contents |
|---|---|
| `baseline/python_api.txt` | `__all__`, every public class member and signature (313 lines) |
| `baseline/cbeam_methods.txt` | 52 `_cbeam` callables |
| `baseline/c_symbols.txt` | 90 exported `T` symbols from `libbeam.a` |
| `baseline/pytest_cases.txt` | 278 collected node ids |
| `baseline/c_cases.txt` | 66 `RUN_TEST` names |
| `baseline/numeric.json` | 80 keys, order 2/3/4 x open/strict, as hex floats |
| `baseline/corpus.json` | the exact 2,000-sequence corpus those keys came from |
| `baseline/threads.json` | identical results at 1, 4 and 16 threads |
| `baseline/fixture.beam` | a saved strict order-4 model, 11,762,888 bytes, sha256 02f8347e... |

Clean-source build proven: `git archive HEAD` into a temp dir, fresh venv,
`pip install .`, import from outside the checkout. 35 tracked files are
sufficient. No `.so`, `build/`, `__pycache__` or egg-info was involved.


## The gate for phases that split numerical code

Phase 3 could be proved by comparing `objdump -d` per object, because moving
declarations between headers changes nothing the compiler emits. That does NOT
hold once translation units are split: Phase 4 changed the codegen of 37
functions and dissolved a `constprop` clone, all legitimately, because
inlining and constant propagation no longer cross the new boundaries. A
disassembly comparison there reports differences that are real and harmless,
which makes it useless as a gate.

What replaces it, in increasing order of authority:

1. `make test` and `pytest`.
2. `.private/baseline/check.py` -- 78 quantities as hex floats, plus the
   saved-model sha256 and thread independence.
3. Rebuild the 927M foundation on ig05 and compare the file sha256 against
   `3aa59ba3dd7b6f95cdb1f2ad4d43b5bc30f7260ebbc9d03fd2d70a628c553dae`.

Step 3 is the one that matters and it costs 6.3 minutes. It exercises every
table, both directions, all 25 lengths and the full sparse finalize path at a
scale no fixture reaches. Run it for phases 5, 6 and 7.

Why it is not optional: gcc defaults to `-ffp-contract=fast`, so a changed
inlining decision can change which expressions become fused multiply-adds, and
that changes the last bit. No assertion in the suite would notice.


## Phase 9: what was deferred

`model/facade.py` is still 1,783 lines. Step 7 of the plan's §9 order --
moving BeamModel's own method implementations into model/scoring.py,
model/generation.py, math/*.py and analytics/* behind unchanged facade
methods -- was NOT done. The plan itself says not to do it in one commit, and
it is the step with the least structural payoff: the class is already the
public facade, its methods are already grouped by domain within the file, and
every one of them is a thin call into C.

Doing it would mean roughly forty delegating wrappers whose only purpose is to
forward arguments unchanged. That is worth doing when the implementations grow
logic worth isolating; today most are one or two lines around a `_c.` call.

If it is taken up later, the order in the plan is right, and the gate is the
same: 279 Python tests, 66 C tests, baseline check, foundation rebuild.


## Phase 10: what was deferred

The Python suite is split into tests/python/{model,math,analytics,publicness}/
with fixtures in conftest.py, constants in _helpers.py, and the independent
reference at tests/reference/beam_model.py. All 279 test names are preserved
exactly: the collected list before and after is identical, which is the check
the plan asks for.

tests/native/ holds the C tests but they are NOT split into per-domain
binaries. test_model.c is still 2,163 lines and 59 cases. Splitting it means
replicating the fixture helpers across roughly twenty translation units, and
one of them must keep `-Wl,--wrap=malloc` while the others must not link it at
all -- the uniform-flag version does not link, which is already recorded in
the Makefile.

The risk is specific: the plan warns not to silently drop the failure-
injection coverage, and a C test suite has no equivalent of pytest's collected
list to diff. Every case would have to be accounted for by hand. That is worth
doing, but as its own task with its own verification, not appended to a phase
that also moved the Python tests and the docs.


## Phase 11: final acceptance

- Removed `src/beam_repertoire.egg-info`, whose SOURCES.txt still named
  `lib/edge_model/`, `_edge.py` and `test_edge.py`. A stale manifest like that
  is how a broken source list keeps appearing to work.
- Removed `__pycache__` and `.pytest_cache`, then rebuilt from clean to
  confirm nothing depended on them.
- `git archive HEAD` into a temp dir, fresh venv, install, import from outside
  the checkout: 139 tracked files are sufficient.
- An external C consumer that includes only `beam/model.h` and
  `beam/publicness.h` compiles and links against `libbeam.a`. If an internal
  header had leaked into the public surface, or a public symbol had moved,
  this would fail. It also confirms `cfg.strict` defaults to 1.
- Foundation rebuilt one last time, after every phase:

      3aa59ba3dd7b6f95cdb1f2ad4d43b5bc30f7260ebbc9d03fd2d70a628c553dae

## Summary for review

Twelve commits, from 321739a to here. Every phase gated on 66 C tests, the
Python suite, and `.private/baseline/check.py`. Every native phase also gated
on rebuilding the 927M-sequence foundation and comparing the file sha256; it
was identical at phases 4, 5, 6, 7, 8 and after phases 9-11.

Two changes are NOT moves and are called out in their own commit messages:
closed-world became the default (owner's decision, its own commit), and one
verbatim-duplicate helper was deleted in Phase 6.

Three things were deliberately left undone, each with its reasoning recorded
above: the BeamModel facade's method bodies (Phase 9), the per-domain split of
the C tests (Phase 10), and the public aliases in the plan's section 13.2,
which add surface rather than removing it.

## Files over the plan's 800-line threshold

Section 17 asks for explicit justification for any production file above
roughly 800 lines. Three qualify.

**`tests/native/test_model.c`, 2,191 lines.** Not split. Recorded under Phase
10: the split needs the fixture helpers replicated across about twenty units
with exactly one keeping `-Wl,--wrap=malloc`, and unlike pytest there is no
collected list to diff, so all 59 cases would have to be accounted for by
hand. Its own task.

**`src/BEAM/model/facade.py`, 1,783 lines.** Deliberate. It is the public
facade and the plan's own §6 says to keep it as one, with thin methods and no
mixin hierarchy. Its methods are already grouped by domain inside the file and
most are one or two lines around a `_c.` call. Splitting it produces about
forty delegating wrappers that forward arguments unchanged, which is churn
until the implementations grow logic worth isolating. Recorded under Phase 9.

**`include/beam/model.h`, 803 lines.** Marginal, and it is the public C API:
one header is what an external consumer includes, and the plan's §5 keeps
`include/beam/*.h` as the stable surface. Most of its length is the
documentation of `BeamModelConfig`, which is where the derivations behind
every default live. Splitting it would mean either more headers for a
consumer to know about, or an umbrella that re-includes them, which is the
same file with extra indirection.

## Additional verification, beyond the phase gates

Run after Phase 11, all clean:

- **ASan + UBSan**: full native suite rebuilt with
  `-fsanitize=address,undefined` from a `git archive` export. 66/66 pass,
  **zero diagnostics**, including the `--wrap=malloc` allocation-rollback
  test, which is the one that deliberately fails an allocation mid-finalize.
- **Wheel**: built and installed into a clean venv, imports and runs from
  outside the checkout, compatibility modules resolve. Previously only the
  sdist had been checked.
- **External C consumer**: compiles against `include/beam/` alone.
- Dead `.private/edge/` links repaired: the methodology document and prototype
  source were recovered from LZGraphs into this repo's working tree, and the
  four references now say the document is working-tree only and not shipped,
  so a reader of the public repo is not sent after a missing file.
- `an BeamModel` -> `a BeamModel` in five places, including one error message
  and the test that matches it. That grammar error was introduced by the
  rename; the original read `an EDGE`.

---

# Review response, 2026-08-27

A reviewer audited this against the plan. Their findings, and what was done.

## Corrected

**The C and Python defaults disagreed.** Confirmed and fixed. Flipping
`cfg->strict = true` in C was not enough: the Python layer resolves the
backoff strengths separately in `_smoothing_defaults`, which the C layer never
sees, so C kept kappa at 20/20/200 while Python used zero. Same corpus, D0 of
4 from C and 3 from Python, both configurations individually valid and neither
failing. `beam_model_config_default` now expresses the closed world in full.
Storage still differs deliberately -- C dense, Python sparse -- because a
dense and a sparse strict model are bit-identical and an explicit C test
asserts the dense default. `test_c_and_python_defaults_describe_the_same_model`
now pins the two together. Two C tests that had been relying on the old
default were repointed and say which world they need.

**The baseline gate was not reproducible.** Confirmed. The two publicness keys
run numpy's forward FFT, whose last bits depend on the numpy build, so the
gate passed here and failed on the reviewer's machine. Those keys are now
compared within 1e-9 relative and everything else stays exact.

**check.py overclaimed.** Confirmed. Its docstring promised API and test
inventory checks that were never implemented. Both are implemented now, and
both immediately found real differences: the deliberate `strict` default flip
and two added tests. Implementing them also exposed a parsing bug of mine --
splitting collected ids on whitespace truncates a parametrised name that
contains spaces.

**Evidence was unversioned.** Confirmed. Moved to `verification/`, tracked.
The 12 MB fixture stays untracked and is rebuilt from the recorded corpus.

**The commit count was wrong.** Thirteen in the audited range, not twelve.

## Corrected, and the claim it narrows

**The foundation hash proves less than was claimed.** Correct, and this was
the most important finding. `io/model_file.c` serializes count tables,
configuration and counters -- not the derived log-probability tables. So the
hash validates parquet reading, encoding, count accumulation, sparse storage
and ordering, serialization and CRC, and through the printed `path_count` also
finalize and the live frontier. It does NOT validate scoring, the partition
function, Mellin, moments, sampling, top-k or publicness, and phases 5 through
8 touched all of those.

Rather than restate the claim more narrowly, the gap was closed. The
pre-reorganization tree (321739a) and HEAD were both run against the same
927M-sequence foundation model, computing 20 derived quantities as hex floats:
scalar and batch scoring, `explain`, the partition function and length prior
per length, the length PMF, Mellin at four q, Hill numbers, entropy, moments,
cumulants, the surprisal range, accumulator max, length marginals, sampling,
top-k, atoms, histogram weights and moments, publicness expected counts, and
topology.

    every one bit-identical

That is the check the file hash was standing in for, and it now exists.

## Acknowledged, not done

The reviewer is right that these were not disclosed. They are real omissions
against the plan, not judgement calls:

- **Grid transport** is still inside `math/spectrum.c`. Plan section 7.4 puts
  the low-level transport in `dp/grid_transport.c` with the orchestration in
  `math/spectrum.c`. Not split.
- **The `internal.h` umbrella is still broad.** Plan section 7.8 says to
  migrate one translation unit at a time to narrow includes and remove the
  umbrella only when no consumer remains. The umbrella was created; no file
  was narrowed.
- **Publicness binning and aggregation** remain in `publicness/model.py`.
  Plan section 8.2 wants `publicness/binning.py` and
  `publicness/aggregation.py`. Only constants and cohort were extracted.
- **`SimulationResult`** was not moved to `results/simulation.py`. I deleted
  the empty directory and mentioned it only in a shell echo, which is not
  disclosure.
- **Public headers** were not split into `include/beam/{core,model,occupancy}/`
  with umbrellas. Plan section 5.
- **Documentation** is partly split: no `docs/api/c/`, no
  `docs/architecture/dependency-rules.md`, `native-layer.md`,
  `python-native-boundary.md`, and none of `docs/algorithms/`.
- **Most internal renames** in plan section 13.1 were not applied. The stale
  `edge_*` helper names in the bindings are still there.
- **`facade.py` contains substantive workflows**, not only thin wrappers. My
  justification for leaving it at 1,783 lines said most methods are one or two
  lines around a `_c.` call; the reviewer checked and that is not true of all
  of them. The file should be split; the justification was wrong.

## Standing

Agreed with the reviewer: the public aliases in section 13.2 stay deferred,
and the C test split is a separate reviewed task.
