# Why the API and test-inventory baselines were re-recorded

Recorded at Phase 0, re-recorded 2026-08-27 after the review. The Phase 0
files are kept beside them as `*.phase0.txt` so the change is auditable
rather than erased, which is the whole risk of re-recording a baseline.

Exactly three differences, all intended, all traceable to a commit:

1. `BeamModel.__init__`: `strict: bool = False` -> `True`
2. `Builder.__init__`: the same
   Both from "Make closed-world support the default" (fe9b77c), which is your
   decision and has its own commit. `Builder` had to move with `BeamModel` or
   the streamed and one-shot paths build different models from identical
   arguments.
3. Two tests added, neither replacing anything:
   - `test_pickle_path_survives_the_module_split` (Phase 9), which pins
     PublicnessModel.__module__ so old pickles keep loading.
   - `test_c_and_python_defaults_describe_the_same_model` (post-review), which
     pins the C and Python defaults together after they were found to
     disagree.

Nothing was removed from either file. `diff python_api.phase0.txt
python_api.txt` and the same for the case lists shows only the above.

# Why the saved-model bytes were re-recorded

Re-recorded 2026-08-31, for the format-3 header that records a model's
alphabet.

    fixture_bytes   11762888 -> 11762896   (+8)
    fixture_sha256  02f8347e... -> c772a1c7...

Eight bytes, and they are the two words appended to the header: the alphabet
and a reserved one. The change was not taken on trust. The same fixture was
saved by the commit before and the commit after, and compared byte for byte:

  - bytes 0..135 identical except the format word, 2 -> 3;
  - the appended tail reads 0000000000000000 -- alphabet 0, which is amino
    acid, and reserved 0;
  - the PAYLOAD is byte-identical;
  - the trailing CRC differs, as it must, being computed over the above.

So no count, no probability and no table moved. `numeric` and `thread
independence` were green across the same change, which is the other half of
the argument: had anything about the model shifted, those would have caught it
and this file would not be the only baseline to move.

The four surface checks -- `_cbeam callables`, `C public symbols`, `Python
public API`, `pytest inventory` -- are still failing and are still NOT
re-recorded here. They carry the in-progress PRISM work as well as the
nucleotide work, and re-recording them now would fuse two changes into one
unauditable diff. They are re-recorded once, deliberately, when the branch is
finished.

# Why the four surface baselines were re-recorded

Re-recorded 2026-09-01, once, at the end of the nucleotide-alphabet branch.
They were deliberately left failing through every step of it: they carry the
in-progress PRISM work as well, and re-recording them per commit would have
fused two unrelated changes into one diff that nobody could audit. The
previous files are kept beside them as `*.prealphabet.txt`.

Every line that moved was attributed by rebuilding the surfaces at four
commits and diffing pairwise, rather than by reading names and guessing:

| baseline | pre-branch | nt: descriptor | PRISM | nt: the rest |
|---|---|---|---|---|
| `cbeam_methods.txt` | 52 | — | +2 | +2 |
| `c_symbols.txt` | 103 | +2 | +3 | +4 |
| `python_api.txt` | 351 | — | +12 −1 | +6 −4 |
| `pytest_cases.txt` | 295 | — | +15 | +24 |
| `c_cases.txt` | 66 | +15 | — | +23 −3 |

`c_symbols.txt` also grows from 90 recorded entries to 112. The old file was a
subset: file splits since it was written gave internal helpers external
linkage, which `check.py` reported as a standing note rather than a failure.
Recording the full symbol table restores the file's original meaning, and the
note is now empty — so any future note means a genuinely new internal symbol
escaped, which is what it was for.

## The removals, which are what to scrutinise

Additions are self-evidently the new work. Three things were REMOVED, and none
of them is a loss of coverage:

1. `test_dense_v1_file_still_loads` → `test_legacy_format_files_still_load`.
   A rename with more in it: the old test synthesised a format-1 file, the new
   one synthesises both format 1 and format 2 and checks each loads as amino
   acid. Coverage strictly increased.
2. `test_the_descriptor_table_matches_the_legacy_pair` and
   `test_the_descriptor_decode_matches_the_legacy_decode`. Added and removed
   within this branch, so they appear in neither file. They existed to pin the
   new decode table against the one it replaced, on all 256 bytes, until the
   switch-over happened. The legacy tables are gone; what they checked is now
   covered by `test_every_byte_maps_exactly_as_specified`, which hardcodes its
   expectations independently.
3. `test_scale_score_is_centred_on_the_model_s_own_output`. **Not from this
   branch.** Removed on main by 54c3569, "Redefine SCALE as a p-value", which
   made a centred score meaningless and replaced it with seven tests including
   `test_scale_score_is_a_false_positive_rate`. The baseline had been stale
   against main since that commit.

Two more disappeared without ever reaching a baseline, and are recorded here
because both had a stated lifetime:
`test_construction_refuses_an_alphabet_the_kernels_cannot_read` guarded the
refusal that kept nucleotide models unbuildable until the kernels could read
the alphabet, and went when the refusal did;
`test_saving_a_nucleotide_model_is_refused_for_now` did the same for the file
format.

## A ghost the re-record nearly baked in

`c_symbols.txt` was about to gain `beam_repair_factor` and
`beam_repair_factor_batch`, which no source in the tree defines. `ar rcs` ADDS
to an existing archive rather than replacing it, so `build/libbeam.a` had been
carrying a `repair.o` from some earlier state of the tree across every
incremental rebuild, exporting symbols from a file that no longer exists. It
was invisible to everything except the archive's symbol table — which is
exactly what this baseline is.

The Makefile now removes the archive before creating it, and the surfaces were
re-dumped from a clean build. The lesson is narrow but worth keeping: a
baseline taken from build output has to be taken from a build that cannot
carry anything the sources do not.

## Addendum, same day: two inventories re-recorded again

An independent review of the finished branch found a segfault, and the tests
written to pin it moved these two files a second time. Six cases added, none
removed:

    c_cases.txt      test_every_nucleotide_order_finalizes_with_backoff
                     test_the_partition_normalises_with_backoff_above_order_four
    pytest_cases.txt test_the_open_world_builds_at_the_default_order
                     test_every_order_survives_backoff[4,5,6,7]
                     test_a_smoothed_nucleotide_model_is_still_a_distribution

and one more from the same review, closing its remaining findings:

    c_cases.txt      test_the_widest_table_still_fits_a_uint32_row_index

and two more from a second review round, which checked the fix rather than
the branch and found the coverage behind it still thin:

    pytest_cases.txt test_every_order_survives_backoff[8]
                     test_every_order_survives_backoff[9]

The C loop over orders was widened from 2..6 to 2..9 at the same time. The
point of both: the original defect hid behind a test that asserted order 9 was
VALID without ever building one, and the tests written to close it stopped at
6 and 7 — the same shape of gap, one layer down.

The defect: the wide expert's backoff base was indexed at `order - 1` into
arrays holding the mid hierarchy's orders 0..3. Amino acids cap at order 4, so
that index was always in range; a nucleotide model at its default order 6 read
two slots past the end of a stack array. `BeamModel(seqs, alphabet="nt",
strict=False)` — the documented open world at the documented default — crashed
the process.

Nothing in the suite had built a nucleotide model with backoff switched on:
every one was order 3, or the closed-world default where all five strengths
are zero and the branch is skipped. The gap was in the tests, not in the gate,
which is why every baseline was green over a segfaulting configuration.

## Addendum: two backoff cases replaced by one ceiling case

`test_every_order_survives_backoff[8]` and `[9]` were removed and
`test_the_ceiling_order_builds` added. Not a reduction in coverage — a
correction of how it was bought.

Backoff forces dense storage, and a dense nucleotide wide table is
`(depth_cap + 1) * 6**order * 5` doubles per direction. Those two cases were
allocating 1.9 GB and 11.3 GB inside the test suite. Orders 5, 6 and 7 take
the clamped backoff branch through identical code, so what 8 and 9 added was
coverage of the CONFIGURATION rather than of the arithmetic. The replacement
builds order 9 — the alphabet's ceiling — in the closed world, where storage
is sparse and the same model costs 0.3 GB, and it still builds rather than
merely asserting the config is legal, which was the point.

# Why the surface baselines moved again: the positional-marginal API

Re-recorded 2026-09-02, for `positional_marginals`, `log_clamp_probability`
and `expected_counts`. Purely additive, and that is checkable rather than
asserted: `diff` against the previous files shows **no removed line in any of
the four**.

| baseline | added | removed |
|---|---|---|
| `cbeam_methods.txt` | `expected_counts`, `positional_marginals` | none |
| `c_symbols.txt` | `beam_model_expected_counts`, `beam_model_expected_counts_free`, `beam_model_positional_marginals` | none |
| `python_api.txt` | the three `BeamModel` methods | none |
| `pytest_cases.txt` | 37 cases, all in `tests/python/analytics/test_positional_marginals.py` | none |

The four checks that would have caught a change to existing behaviour were
green THROUGHOUT, and were never re-recorded: `numeric` (78 quantities as
exact hex floats over order 2/3/4 x open/strict), `saved-model bytes`,
`thread independence`, and `C test inventory`. That is the evidence that this
is an addition rather than an edit. It was checked independently as well, by
hashing `log_pseq` over 4,500 amino-acid and 1,000 nucleotide sequences across
four models before and after the change: the dumps are byte-identical, and
`pgen_diagnostics()` still reports `is_proper: True`, `total_mass: 1.0`,
`max_abs_deviation: 0.0` on every strict model.

`pytest_cases.txt` is written as the previous file's lines in their recorded
order with the new ones appended, rather than as a fresh collection dump.
Pytest's collection order is not stable between runs, so a fresh dump reorders
~200 unchanged lines and buries the 36 that are actually new. `check.py`
compares this file as a SET, so ordering carries no meaning and the only thing
at stake is whether the diff can be read.

# Why five baselines were re-recorded for the query engine

Re-recorded 2026-09-06, at the end of the `probabilistic-queries` branch. The
superseded files are kept beside them as `*.prequery.txt`, so
`diff c_symbols.prequery.txt c_symbols.txt` and its siblings show exactly what
moved.

Re-recorded ONCE, at the end, and not per commit. The branch left all five
failing through thirteen commits deliberately: re-recording as it went would
have fused unrelated changes into diffs nobody could audit.

    cbeam_methods.txt    58 ->  67   (+9,  -0)
    c_symbols.txt       115 -> 162   (+47, -0)
    c_cases.txt         104 -> 198   (+94, -0)
    pytest_cases.txt    380 -> 412   (+32, -0)
    python_api.txt      409 -> 486   (+34, -1)

**Nothing was removed from any of them.** The single line the Python API lost
is `__all__:` itself, replaced by the same line with two names added.

## The additions, by source

1. **47 C symbols**, all `beam_query_*`, from the query engine under
   `src/native/query/`. 24 are declared in `include/beam/query.h` and are
   public API; the rest are cross-file helpers declared only in
   `beam/query_internal.h`, which the check reports as a note rather than a
   failure, exactly as it does for the existing `beam_layer_weights` and
   `beam_betas`.

2. **9 `_cbeam` callables**, `query_*`, one per public entry point of that
   engine.

3. **34 Python API lines**: the `BeamQuery` and `Condition` classes, plus their
   two names in `__all__`. `BEAM.query` is exported at top level as
   `PrismScorer` is.

4. **94 C test cases** across five new files -- `test_query.c`,
   `test_query_automaton.c`, `test_query_moments.c`,
   `test_query_information.c`, `test_query_indel.c` -- and one added to
   `test_model.c`.

5. **32 pytest cases**: 19 in `tests/python/query/test_query.py`, and 13 in
   `tests/python/analytics/test_rules.py`, which is a separate module
   committed on this branch.

## What did NOT move, which is the point

`numeric` (78 hex-float quantities, order 2/3/4 x open/strict), the
saved-model bytes, and thread independence were green at every commit,
including the two that changed shipped numerics:

  - the query engine's forward sweep moving to log space, and
  - `beam_model_positional_marginals` and `beam_model_expected_counts` moving
    with it, which fixed a real defect in shipped code -- at
    `beam_model_config_default()`, a length-60 nucleotide sweep returned six of
    sixty rows summing to 5.9e-264 instead of 1, three of them PINNED
    positions whose row must be one-hot.

That a correctness fix to shipped numerics moved no recorded quantity is worth
stating plainly. It is not luck: the recorded models do not span enough
dynamic range to underflow, and where the linear sweep did not underflow the
log-space one reproduces it bit for bit. Had anything else shifted, `numeric`
would have caught it, and it is the check that must never be re-recorded
casually.

## One symbol was removed before recording rather than recorded

The first pass of this re-record listed `beam_query_layer_max_product` among
the additions. Its declaration had been deleted with its two siblings when the
per-layer rescale was removed, but the definition was left behind -- dead code
with external linkage, reachable by nobody. It was deleted rather than
recorded. The symbol check found it, which is the check doing its job on the
day it was updated.

## Amended 2026-09-06, same branch, for the third review's fixes

Three of the five moved again, all additive, all in the same session as the
re-record above; the `*.prequery.txt` copies still record the pre-branch state,
which is the comparison that matters.

    python_api.txt      486 -> 487   (+1)   BeamQuery.total_correlation
    c_cases.txt         198 -> 199   (+1)   multiplicity partitions the edits
    pytest_cases.txt    412 -> 417   (+5)   multiplicity, the length-prior
                                            flag, total correlation, the
                                            cheap conditional entropy, and
                                            the non-finite functional refusal

`numeric`, the saved-model bytes, thread independence, `_cbeam` callables and
the C public symbols were all green across these, which is the expected
signature of a change that adds capability and fixes documentation without
moving an existing number.

Two moved once more, in the same session, for the position-versus-property
information the same review recommended:

    python_api.txt      487 -> 489   (+2)   BeamQuery.position_information
                                            and .position_divergence
    pytest_cases.txt    417 -> 421   (+4)   enumeration, the bounds, the
                                            pinned-position case, and the
                                            constant condition

Both are derived quantities -- two marginal sweeps and arithmetic -- so no C
symbol, no `_cbeam` entry and no C test case moved with them.

## Amended again 2026-09-06, for two defects found in SHIPPED code

    c_cases.txt         199 -> 201   (+2)

Two regression tests, both for defects a review found in code that predates
this branch and that all eight checks passed over:

  - `test_a_negative_tilt_does_not_overflow_the_sweeps` -- the tilted forward
    sweeps rescaled by `q * wmax` with `wmax` the largest UNTILTED weight, so
    a negative `q` overflowed every layer. `log_mellin(-5)` returned `-inf`
    and `hill_number(-5)` returned `0.0`, both with BEAM_OK, and `q = -1.5`
    returned a finite WRONG number.
  - `test_a_zero_forward_weight_leaves_no_nan_edge_weights` -- `0 * -inf` in
    the forward mix, which the backward half already guarded against.
    Measured before the fix: 608,573 NaN edge weights of 1,666,980.

`numeric` did not move for either, and could not have: the tilted fix returns
`q * wmax` unchanged for every `q >= 0`, which is every quantity the baseline
records, and the weight guard changes nothing for a finite mix. That is the
point worth keeping -- both defects lived in quantities the baseline pins, and
the baseline pins them against a recording of themselves.

## Amended 2026-09-06, for the memory-budget guards

    pytest_cases.txt    421 -> 440   (+20, -1)

Superseded file kept as `pytest_cases.prebudget.txt`.

Fifteen of eighteen files in `dp/`, `math/` and `algorithms/` allocated
per-length scratch with no memory-budget check at all, and every one of them
is parallel over lengths, so the real peak was threads x per-sweep. `calloc`
grants an absurd request lazily on Linux and fails on first write, so an
unguarded sweep does not fail -- it takes the machine down.

The one removed case is a rename, not a deletion:

  - `test_spectra_by_length_does_not_depend_on_the_memory_budget` became
    `test_a_reduced_budget_changes_the_schedule_and_not_the_answer` plus
    `test_a_budget_below_one_sweep_is_refused`. It asserted that a budget of
    one byte still returned success, pinning the old promise that the budget
    was "a cap on CONCURRENCY, never on what a single length is allowed to
    allocate". That promise was never kept -- `positional.c` and the whole
    query layer refused above the budget from the same setter -- so the two
    halves of the codebase disagreed about what the number meant. The
    refusing half was kept, and the header, `runtime.c` and `runtime.py` now
    say so.

The twenty added cases are `tests/python/analytics/test_memory_budget.py`
(fourteen entry points refusing, one that must NOT be budgeted, the reported
figure checked against an independent calculation, and answers pinned
identical under a reduced budget) plus the spectrum pair above.

`numeric` did not move, and that is the load-bearing observation: the guards
refuse before allocating and the concurrency bounds only change how many
lengths run at once, so no recorded quantity could shift. `exact` also held
at 280 quantities. Two defects were found on the way and are fixed here
rather than recorded: `path_count` and `path_count_total` returned 0.0 to
Python on refusal, which is a legitimate answer and made a refusal look like
"this model generates nothing"; and `spectrum` reported a refused accumulator
sweep as "the accumulator bound is zero, so the model generates nothing to
bin", sending the reader to their corpus for a problem in their settings.

## Amended 2026-09-06, for the error-reporting defects

    pytest_cases.txt    440 -> 448   (+8, -0)

Superseded file kept as `pytest_cases.prediag.txt`. Purely additive: eight
cases in the new `tests/python/model/test_error_reporting.py`, nothing renamed
and nothing removed.

Two audits of the error surface found failures being reported in forms a
caller could not distinguish from a real answer, or carrying a diagnosis that
belonged to an unrelated earlier call. The two worst were verified by
reproduction:

  - saving to a full disk raised "beam_model_path_count: length 0 is not one
    this model generates". `beam_model_save` ended in a bare
    `return ... BEAM_ERR_IO_WRITE`, and the binding read the thread-local
    message buffer, which nothing had cleared. The diagnosis was specific,
    actionable and about a different call.
  - a sound model file that merely exceeded the memory budget was reported as
    "not a model file, truncated, or corrupt". All nine of `beam_model_load`'s
    failure returns set no error, so the binding hard-coded one sentence --
    and `beam_model_new` / `beam_model_finalize`, which DO set precise ones,
    had theirs discarded. The remedy was `set_memory_budget`; the reader was
    told to rebuild the file.

Also fixed: `path_count_total` returned a silent 0.0 on an unfinalized model
while `path_count` twelve lines above set an error on the identical condition
(this defeated the binding guard added in the previous commit -- its premise,
that the per-length calls clear the error, is false on a branch that makes
none); worker failures dropped inside OpenMP loops in the path-count and
accumulator drivers, which summed zeros into a silently short total; a
`malloc` failure in the accumulator sweep that set no error at all;
`order=258` truncating to a LEGAL order 2 and building a working model that
scored wrongly; two bindings sizing an allocation from an unvalidated length,
so a nonsense length raised an empty MemoryError from the allocator instead of
the C layer's precise refusal; the three budget messages that still read
"0.0 GB"; and seven bare-code returns in `occupancy/` and `io/`.

One behaviour change beyond the defects, and it was NOT intended: making
`BEAM_ERR_IO_CORRUPT` and `BEAM_ERR_IO_VERSION` reachable exposed that they
mapped to ValueError, while every `beam_model_load` failure had always been a
hard-coded OSError. Three tests caught it. The mapping was changed to OSError
rather than the contract, since the codes were previously unreachable and the
OSError contract is what users and tests depend on.

`numeric` did not move and `exact` held at 280 quantities: every change here
is to how a failure is REPORTED, not to any computed value.

## Amended 2026-09-06, for the error-contract pin test

    pytest_cases.txt    448 -> 470   (+22, -0)

Superseded file kept as `pytest_cases.precontract.txt`. Purely additive:
`tests/python/test_error_contract.py`.

The error system is being restructured to give failures a real Python
exception hierarchy, and the danger is a failure silently changing class --
every `except ValueError` in a user's pipeline stops catching and nothing says
so. That already happened once: making BEAM_ERR_IO_CORRUPT and
BEAM_ERR_IO_VERSION reachable converted three `beam_model_load` failures from
OSError to ValueError, because the switch mapping them had never run.

This file generalises the three tests that caught it. It pins
(ENTRY POINT -> class), deliberately not (code -> class): one code does not
imply one class, and assuming it does is how that regression happened.
BEAM_ERR_NO_LIVE_PATHS reaches Python as a ValueError from
`positional_marginals`, as an `ImpossibleCondition` from `BeamQuery`, and as a
plain -inf return from `log_clamp_probability` -- all three deliberate, all
three pinned.

Verified to bite: re-introducing the IO_CORRUPT regression fails exactly the
three rows that describe it.

Two uncomfortable facts are pinned as facts rather than left as claims: a
`beam_model_load` failure is NOT always an OSError (a budget refusal is a
MemoryError, so the invariant asserted in an earlier commit message was
already false), and several `PyErr_NoMemory()` sites raise with an empty
message and no error code at all, so a future `except BeamMemoryError` will
not catch them.

Also in this commit: the two remaining refusal messages that still hardcoded
gigabytes, in `query/engine.c` and `math/positional.c`. The previous commit
fixed three such sites by name without grepping for the shape, so these two
survived reading "0.0 GB".

## Amended 2026-09-06, for the Python exception hierarchy

    python_api.txt      489 -> 879   (+390)

Superseded file kept as `python_api.prehierarchy.txt`. Purely additive: 15
exception classes, each contributing exactly 26 lines (a `class` header plus
the 25 inherited BaseException members the snapshot records). 15 x 26 = 390,
which is the whole delta -- it matched the figure predicted before the change
was made, which is the cross-check that nothing unintended was exported.

Before this, `BEAM.__all__` exported no exception at all. The query package
had the library's only hierarchy and it was not reachable from the root: a
caller had to know to write `from BEAM.query.errors import QueryTooLarge`.
Everything else raised bare built-ins, so the only way to tell a memory-budget
refusal from a genuine out-of-memory was to match on the text of the message --
which is exactly what `tests/python/analytics/test_memory_budget.py` had to do.

Every class keeps the built-in it replaces as a second base, so this is
additive rather than breaking: `except ValueError` catches everything it caught
before. That is verified rather than asserted -- the full suite passes
unchanged at 470, and `tests/python/test_error_contract.py` pins the observed
class per entry point and is untouched by this commit.

The built-in is chosen PER CLASS, not uniformly. Collapsing everything onto
`BeamValueError` would have been tidier and would have broken the eight
`pytest.raises(TypeError)` sites that guard entry points where the argument's
type is wrong rather than its value.

`python_api.txt` is the only baseline that moved. `numeric`, `exact`,
`pytest_cases`, `c_symbols` and the rest are untouched: no computed value
changed and no test was added or renamed.

## Amended 2026-09-07, for the budget-refusal leaks

    pytest_cases.txt    470 -> 475   (+5, -0)

Superseded file kept as `pytest_cases.preleak.txt`. Purely additive: five
parametrised cases of `test_a_refusal_can_be_repeated_without_growing`.

A warning-and-defensive audit found SEVEN memory leaks on the memory-budget
refusal path, all introduced by the commit that added the up-front driver
refusals, all reachable from Python, and all proven under ASan+LSan rather than
inferred. The pattern is worth recording because it is a process failure, not
a typo:

  - the three sites written BY HAND (surprisal.c, extrema.c, path_count.c)
    free their per-length arrays before returning;
  - the five generated with a UNIFORM SCRIPT (mellin.c, tilt.c, marginals.c,
    moments.c x2, diversity.c x2) do not.

The script inserted an identical refusal at each site. The cleanup obligation
differs at every one of them -- mellin.c holds two arrays, tilt.c holds five,
marginals.c holds one -- and a template cannot see that. The sites were chosen
for scripting precisely BECAUSE they looked alike, which is the trap: they were
alike at the point of insertion and different everywhere around them.
`topology.c:28` already carried a comment warning about this exact hazard.

Why nothing caught it: a full ASan+UBSan+LSan run over all 201 C tests is
completely clean, because no test lowers the budget far enough to execute a
refusal RETURN. `budget.h` predicted the blind spot in so many words --
"removing a sweep's own guard does not fail the suite where a driver refuses
first". The five new cases are the missing shape: they call each refusing
driver fifty times under a starved budget, which makes those returns execute.
Under a plain run they pin that the refusal is deterministic; under LSan they
are the leak test.

Also in this commit, and the reason no baseline but this one moved: the
`const char **edge_borrow_strings` prototype (5 warnings, and a hard error on
GCC 14 / Clang 16) and a one-line read clamp in `indexing.h` (18 warnings, one
line inlined 18 times). Both builds now compile with ZERO warnings, down from
23. `numeric` and `exact` are unchanged, which is the evidence that the clamp
never fires: if it did, a context index would move and every recorded quantity
with it.

## Amended 2026-09-07, after review, before merge

    pytest_cases.txt    475 -> 477   (+3, -1)

Superseded file kept as `pytest_cases.prereview.txt`. A pre-merge review found
four defects, three of them in claims rather than in code. The one removal is
deliberate, not a deletion:

  - REMOVED `[surprisal_cumulants]`. It was redundant. At a 2 kB budget it
    refuses inside `beam_mellin_terms`, which `[log_mellin]` already covers,
    so it exercised no return of its own.
  - ADDED `[cgf_jet]`, which reaches `tilt.c`; `[length_marginals]`, which
    reaches `marginals.c`; and `[length_derivatives]`, which reaches
    `moments.c`. None of those three returns was executed by any test before.

The finding behind it: the list's own comment claimed to cover "the returns
that leaked", and that was never measured. At 2 kB BOTH moment drivers refuse
inside `beam_mellin_terms` -- they call it first -- so three of the seven
fixed returns were exercised and four were not. The list is now a table of
(entry point, budget, expected refusal), and the test asserts WHICH refusal
fired. The 1.8 MB tier is what separates them: a Mellin jet fits there and a
moment jet does not.

Three more corrections from the same review, none affecting a recorded value:

  - `make test` was NOT at zero warnings, though the previous commit said so:
    `beam_model_crc32c` was declared in the error-macro header this branch
    renamed, `test_alphabet.c` did not follow, and the call fell back to an
    implicit int -- a hard error on GCC 14, which is the exact argument that
    commit used to justify the `const char **` fix. The declaration now has
    its own home in `beam/io_internal.h`; a checksum in an errors header is
    what let it go missing.
  - `BeamTypeError` was exported and documented while nothing could raise it.
    It is now raised by the shared Python validator, which is the only place
    that can: the C layer has no code for a type error because a value is
    converted before it reaches C.
  - `include/beam/errors.h` claimed BEAM_ERR_NULL_ARG was "unreachable under
    its own name". There are 56 BEAM_ENULL sites and 13 direct ones. The true
    statement is "unreachable FROM PYTHON", verified by instrumentation.

`numeric` and `exact` unchanged.

# Why the API and inventory baselines were re-recorded for the mixture package

Re-recorded 2026-09-07, adding `BEAM.mixture`. The previous files are kept
beside them as `*.premixture.txt`.

Five of the nine checks moved. **Four did not, and that is the load-bearing
half of this note**: `numeric` (78 keys), the saved-model bytes, `thread
independence` and `exact against enumeration` (280 quantities) all passed
unchanged. Nothing this change touches alters a number the library already
produced, and if any of those four had moved it would have been a defect in
the change rather than something to re-record.

What moved, and why each is additive only:

1. `cbeam_methods.txt`: +3, -0
   `mixture_check_shape`, `mixture_inner_product`, `mixture_support_union`.

2. `c_symbols.txt`: +5, -0
   The five entry points `include/beam/mixture.h` declares. `check.py` flagged
   them as "newly external", which is its heuristic for a symbol that gained
   external linkage without being public API. Here they ARE public API: they
   are declared in an installed header, which is the test that heuristic
   applies, so the note is expected rather than a warning.

3. `python_api.txt`: +247, -1
   The one removal is the `__all__` line itself, which is a single line listing
   every export and therefore always "changes" rather than "grows". The 247
   additions are `Mixture` and the four frozen result types; the dataclasses
   contribute the bulk, each bringing `__eq__`, `__init__`, `__repr__` and the
   rest of the generated surface. No pre-existing signature moved.

4. `pytest_cases.txt`: 477 -> 589, -0
   Every one of the 112 added cases is under `tests/python/mixture/`. Checked
   rather than assumed:

       diff <(sort pytest_cases.premixture.txt) <(sort pytest_cases.txt) \
         | grep '^>' | grep -v 'mixture/'

   returns nothing.

5. `c_cases.txt`: +15, -0
   Fourteen from the new `tests/native/test_mixture.c`, plus
   `test_accumulator_max_ignores_an_earlier_threads_error` in `test_model.c`.
   That last one is not part of the mixture package: it pins a defect the
   mixture work uncovered in `math/extrema.c`, where
   `beam_model_accumulator_max` read `beam_last_error()` back from an OpenMP
   worker that had never cleared it, and so reported "a per-length sweep
   failed" for a sweep that succeeded. It was intermittent -- it surfaced as
   eight unrelated spectrum tests failing in a full-suite run and passing when
   the same command was re-run. `accumulator_max_at` now clears on entry, the
   way its sibling `beam_model_path_count` always did. The fix changes no
   computed value, which is why `numeric` and `exact` did not move.

# Why the C baselines were re-recorded for the rescale invariant

Re-recorded 2026-09-07, adding the per-layer rescale guard. Previous files kept
as `*.prerescale.txt`.

Two of the nine checks moved, and only two:

  c_cases.txt    +5  -0   the new tests/native/test_rescale.c
  c_symbols.txt  +3  -0   beam_rescale_flushes, _flush_reset, _note_flush

Those three are INTERNAL, declared in src/native/internal/beam/dp.h and in no
installed header, so `check.py` reports them under its "newly external" note
rather than as public API -- which is correct, and is why "C public symbols"
still passes.

`numeric`, `exact against enumeration`, the saved-model bytes and thread
independence are ALL UNMOVED, which is the point of the change rather than an
incidental fact about it. The guard replaces `v / nmax` with a function whose
body is `v / nmax`, so every sweep in the library had to come back bit for bit
identical; had `numeric` moved by a single bit, the guard would have been
computing the quotient some other way and would itself have been the
regression. `test_the_guard_divides_exactly_as_the_operator_does` pins that
directly, by `memcmp` rather than by comparison.

# Why the C symbol baseline was re-recorded for the dead-code deletion

Re-recorded 2026-09-07, removing the logging subsystem. Previous file kept as
`c_symbols.predeletion.txt`.

ONE check moved, and it is the one that is supposed to be hard to move:

  c_symbols.txt   -3   beam_log_set, beam_log_level, beam_log_emit

`check.py` fails outright on a symbol going missing, rather than noting it the
way it notes a new one, and that asymmetry is right: adding public surface is
additive, removing it breaks callers. This removal is deliberate. The whole
subsystem -- 35 lines of implementation and 76 of declarations -- had ZERO call
sites in the library. Nothing ever called `BEAM_INFO`, `BEAM_WARN` or any of
the other macros, so no message could ever be emitted and installing a callback
did nothing. The one apparent user, `BEAM_LOGZ_FREE` in `dp/partition.c`, only
shares the prefix; it frees the log-partition scratch.

Eight error codes went with it, and they moved NO baseline at all. Nothing in
the library raised BEAM_ERR_INVALID_VARIANT, LENGTH_MISMATCH, HAS_CYCLES,
NO_GENE_DATA, GENE_NOT_FOUND, MISSING_EDGE, CONVERGENCE or OVERFLOW; their only
references anywhere were the two tables in `bindings/python/common.c`. That
`tests/python/test_error_contract.py` still passes unchanged, pinning the class
raised by each of 22 entry points, is the evidence that no reachable path lost
its error: a code that had been produced would have taken a row with it.

`numeric`, `exact against enumeration`, the saved-model bytes and thread
independence are unmoved, as they must be -- nothing here touches a
computation.

The retired NUMBERS are not reused; see the note in `include/beam/errors.h` for
why, and for the fact that the next free value is 24 rather than 5.

# Why the baselines moved for the mixture scale work

Re-recorded 2026-09-07, splitting the 64-member ceiling and adding a Gram
driver that parallelises over pairs. Previous files kept as `*.prescale.txt`.

  cbeam_methods.txt   +1   -0   mixture_gram
  c_symbols.txt       +1   -0   beam_mixture_gram
  python_api.txt      +1   -0   Mixture.log_gram_matrix
  pytest_cases.txt   +12   -0   589 -> 601, all under tests/python/mixture/
  c_cases.txt         +2   -1

The one REMOVAL is `test_too_many_members_are_refused`, and it is a removal
because what it asserted stopped being true on purpose. It required
`beam_mixture_check_shape` to refuse 65 members. That gate runs for every entry
point, so enforcing the union sweep's `uint64` bitmask there refused a
cohort-scale mixture AT CONSTRUCTION and took q >= 2, the alpha diversity and
the q = 1 bounds with it -- none of which involve a bitmask. Two tests replace
it: `test_the_union_ceiling_binds_only_the_union`, which requires the gate to
pass at 65 while the union alone refuses, and
`test_the_gram_matrix_matches_the_pairs_one_at_a_time`.

`numeric`, `exact against enumeration`, the saved-model bytes and thread
independence are unmoved. The Gram driver runs the same sweep as
`beam_mixture_log_inner_product`, one pair per thread instead of one at a time,
and a C test compares every entry against that sweep run alone by `memcmp`
rather than by tolerance.

A collection error caught here is worth recording: the new tests were first
`tests/python/mixture/test_scale.py`, which collides with
`tests/python/analytics/test_scale.py` because the suite has no `__init__.py`
and pytest imports by basename. The mixture group alone passed; the full suite
would not collect. Renamed to `test_many_members.py`. Test basenames must stay
unique across the whole tree.

---

## 2026-09-07, the occupancy-selected row directory

Re-recorded `c_symbols.txt` and `c_cases.txt` on branch `memory-optimization`.
The previous files are kept beside them as `c_symbols.prerank.txt` and
`c_cases.prerank.txt`.

**Nothing numeric moved.** `numeric` (78 hex-float keys over order 2/3/4 x
open/strict), `saved-model bytes`, `thread independence` and `exact against
enumeration` (280 quantities) all matched without being touched, which is the
point: this work changes how a row is FOUND, never what it contains.

Two differences, both intended:

1. `beam_sparse_rows` removed, `beam_sparse_seal_rank` added.

   `beam_sparse_rows` listed a table's occupied rows. Exactly one caller ever
   used it -- `beam_layer_max`, on one of the six finalized tables -- and it
   now reads the directory slice for the depth bucket it wants instead, which
   reaches the same rows in the same order. With its last reader gone the
   accessor was deleted and finalized tables stopped carrying the slot -> row
   map it returned (16.32 MiB on the foundation).

   `beam_sparse_seal_rank` replaces a finalized table's uint32-per-dense-row
   directory with a bitmap-and-rank one when at most one row in eight is
   occupied. Both are declared only under `src/native/internal/`; neither is
   public API.

2. Fifteen tests added, none replacing anything, in three new files:

   - `test_layer_max.c` (5) pins `beam_layer_max` against its DEFINITION --
     every state, no shortcut -- rather than against whichever row source it
     happens to use, so the source can change again without the test moving.
   - `test_live_alias.c` (4) covers `live[i]` now aliasing `reached[i]` where
     the two lists are equal: that the alias is taken exactly then, that the
     byte accounting stops charging for it, and that dropping the cache frees
     each list once. Checked under ASan/UBSan via `verification/bounded-rss`.
   - `test_rank_directory.c` (6) requires a sealed directory to return the
     same NULL or the same row pointer as the direct one for EVERY dense row
     of every table, that finalize applies the threshold per table, and that a
     table numbered out of order is refused rather than silently mis-sealed.

---

## 2026-09-07, finalized row interning

Re-recorded `c_symbols.txt` and `c_cases.txt` again on `memory-optimization`;
the previous pair is kept as `*.preintern.txt`.

**Nothing numeric moved**, on the same four gates as the entry above.
Interning shares storage between rows that were ALREADY equal to the byte, so
every caller receives the same doubles it received before.

- `beam_sparse_intern_rows` added; internal, declared only under
  `src/native/internal/`.
- Five tests added in `tests/native/test_interning.c`. The load-bearing one is
  `test_a_pooled_row_is_still_log_of_its_counts`: it recomputes every
  finalized row from the counts it came from and compares bit patterns,
  through the rank directory and the shared pool both. The before/after form
  used elsewhere loses its force here, because finalize now interns on its own
  and there is no un-interned "before" left to compare against.

---

## 2026-09-07, the frozen count representation

Re-recorded again; previous pair kept as `*.prefrozen.txt`. Five
`beam_frozen_*` symbols and four tests added, all internal.

This commit adds the representation and its tests but does not yet route the
model through it, so no numeric gate could move: nothing in a model's
lifecycle calls it. It is committed separately because the exactness argument
is worth reviewing on its own, before the lifecycle changes that will use it.

---

## 2026-09-07, routing the model through frozen counts

Re-recorded once more; previous pair kept as `*.prewire.txt`. Five model-level
`beam_model_*frozen*`/`*_counts` symbols and two tests added, all internal.

**`saved-model bytes` is the gate that matters here** and it did not move.
`write_count_table` now reads the frozen counts directly instead of thawing
them, so it produces those bytes through a different path; the baseline says
it produces the same ones. `test_saving_a_frozen_model_writes_the_same_bytes`
says so a second way, by comparing a frozen model's file against an
identically built model whose counts were thawed by hand.

The other new test is the one that guards the integration rather than the
encoding. A consumer that forgets to thaw does not crash -- `beam_count_row`
returns NULL for every row and the caller's `if (!values) continue;` skips it,
so a save would write an empty model and an add would add nothing, silently.
`test_every_lifecycle_operation_survives_frozen_counts` runs observe,
re-finalize, add, subtract and posterior on a frozen model and on a control
that was thawed by hand, and requires identical scores and identical nonzero
cell counts.

---

## 2026-09-07, format 4

`saved-model bytes` was re-recorded, and this is the one entry in this file
where that is the POINT rather than a side effect. The check's own docstring
says a new format is an intentional saved-byte change needing an explicit
compatibility fixture rather than a silent overwrite, so:

- the fixture that must keep loading is `tests/fixtures/format3_{sparse,dense}.beam`,
  committed in 3233ef8, written by the last build that emitted format 3 and
  never regenerated since;
- `tests/python/model/test_format_compatibility.py` loads both and compares
  scores as hex floats, plus total weight and nonzero cell count, and asserts
  each fixture's own byte size so a regenerating script fails rather than
  quietly re-records;
- the previous `numeric.json` is kept as `numeric.preformat4.json`.

The baseline fixture went from 11,762,896 bytes to 813,854 -- 14.5x -- for a
model whose every reported value is unchanged. `numeric`, `thread
independence` and `exact against enumeration` did not move, which is what
says the file got smaller rather than the model getting different.

`pytest_cases.txt` gains the four compatibility tests; the previous list is
kept as `pytest_cases.preformat4.txt`.

---

## 2026-09-08, dictionary-indexed log cells

`beam_sparse_pack_cells` added, `beam_sparse_row_final` removed -- the kernels
take a `BeamLogRow` now, because a packed table has no array of contiguous
rows to hand back a pointer into. Five tests added. Previous pair kept as
`*.prepack.txt`.

`numeric`, `saved-model bytes`, `thread independence` and the 280 enumeration
quantities did not move. They could not: a dictionary holds the same bit
patterns, and what changed is which of them a lookup walks through.

This one carries a real runtime cost and the commit message records it rather
than burying it here.

---

## 2026-09-08, streaming parquet reads

`pytest_cases.txt` gains six tests in `tests/python/io/test_parquet_streaming.py`;
the previous list is kept as `pytest_cases.preparquet.txt`. Nothing else moved,
and nothing else could: the change is which buffers pyarrow keeps while
handing BEAM the same rows in the same order. The two foundation artifacts
built either side of it are byte-identical.

---

## 2026-09-08, exact-capacity reserve at finalize

`beam_sparse_reserve` added, four tests with it; previous pair kept as
`*.prereserve.txt`. Nothing numeric moved and nothing could: the rows, the
slots and the bytes are the same, and the only difference is that the payload
holding them was allocated once at its final size instead of being doubled
into twenty-one times.

`test_a_reserved_table_holds_what_a_grown_one_holds` is the one that says so,
by filling the same rows both ways and comparing every dense row's presence,
every cell's bits, and every directory slot -- slots included because the
finalized directory hands out slot k to the k-th row added, and sealing later
depends on exactly that.

---

## 2026-09-08, finalize divides the memory budget

Three tests added in `tests/native/test_finalize_budget.c`; the previous case
list is kept as `c_cases.prebudget2.txt`. No symbol changed --
`beam_logz_sweep_bytes` is a static inline in an internal header.

Nothing numeric moved, and `test_a_narrow_budget_finalizes_to_the_same_bits`
is the statement of why it could not: how many lengths run at once is a
performance choice, so a model finalized under a budget admitting two
concurrent sweeps and one admitting sixteen must produce the same log_z and
the same scores, bit for bit. That is the property that makes lowering the
budget safe rather than merely possible.

---

## 2026-09-08, format-4 fixtures recorded

`pytest_cases.txt` gains the format-4 compatibility cases and loses the
format-3-specific test names, which are now parametrised over both formats;
the previous list is kept as `pytest_cases.prefmt5.txt`. No library code
changed in that commit -- it records bytes, before the encoding they were
written by is replaced.

---

## 2026-09-08, format 5: the frozen counts re-encoded

`saved-model bytes` re-recorded, and again that is the point rather than a
side effect. The compatibility fixtures that had to exist first are
`tests/fixtures/format4_*.beam` from 6781dda, alongside the format-3 pair, and
`test_format_compatibility.py` is parametrised over both.

The baseline fixture went 813,854 -> 519,329 bytes for a model whose every
reported value is unchanged. `numeric`, `thread independence` and the 280
enumeration quantities did not move.

Worth recording because it nearly went the other way: the first version
identified rows with a presence bitmap only, which costs one bit per DENSE row
however few are occupied. That is 18x smaller than a list of ids on the
foundation and several times LARGER on a small model -- this fixture grew 40%,
to 1,142,777 bytes, and that growth is the only thing that said so. The form
is now chosen per table from its own occupancy and the file records which it
holds.

---

## 2026-09-08, backward tables laid out by residue

`beam_sparse_plane_transpose` added, five tests with it; previous pair kept as
`*.preplanes.txt`.

Nothing numeric moved and nothing could: the same doubles are at different
addresses. `test_a_transposed_table_reads_back_identically` says so on a
hand-built table -- hand-built because a per-repertoire model's backward
tables are packed and never reach this path, so a test that waited for
finalize to produce one would have been silently vacuous. It was: the first
version asserted "nothing was transposed" and failed, which is how the
vacuity was found rather than shipped.

## Amended 2026-09-09, for the memory-management reorganisation

Four baselines moved, all additive, none removed anything.

`c_symbols.txt`, `cbeam_methods.txt`, `python_api.txt`
    `beam_model_live_cache_bytes` and `beam_model_drop_caches`, and their
    Python members. The per-length live cache was real memory no caller could
    see or release: a model reporting 13.8 MB of tables was holding 29.2 MB,
    and on the foundation 806.7 MB of tables carried 382.1 MB of cache.
    `beam_model_memory_bytes` deliberately still reports the tables only --
    folding a droppable memoisation into it would make the number mean two
    things and change with when it was asked -- so the cache is reported
    beside it instead.

`pytest_cases.txt`
    Six tests for the above, and nothing replaced.

The check that matters did NOT move. `numeric`, `saved-model bytes`, `thread
independence` and the 280 enumeration quantities were all unchanged across a
reorganisation that moved four files' worth of code out of core/runtime.c,
model/introspection.c and dp/liveness.c into src/native/memory/. That is the
evidence the move was mechanical.

## Added 2026-09-09: c_public_symbols.txt

A new baseline, not a re-recording: the 112 symbols that a public header
declares, out of the 187 in `c_symbols.txt`.

It exists because the symbol check could not tell public from internal at
RECORD time, only at check time, and that made two opposite mistakes at once.
It failed when any of the 75 internal symbols stopped being exported -- which
is encapsulation, and 13 of them are used in a single .c file and could be
`static` today. And when the public set was instead recomputed from the
current headers, deleting a declaration became self-approving: the symbol left
both sides of the comparison and the removal vanished with it.

Both were demonstrated before and after. Making `beam_free_counts` static now
passes with a note; deleting `beam_model_radix` from include/beam/ still fails
on two checks.

`public_text_names` also compared by substring, so `beam_mix` matched inside
`beam_mixture_support_union` and `jet_sweep` matched a prose comment. Two names
were counted as public API that no header declares; the recorded set has 112
rather than 114.


# Why six baselines were re-recorded for the banded marginals

Re-recorded 2026-09-11, adding `beam_model_positional_banded` and its Python
method. The previous files are kept beside them as `*.preband.txt`, so the
change is auditable rather than erased.

Compared as SETS -- which is how `check.py` compares them, and how they must
be diffed, because GNU sort and Python order underscores differently and a
line diff of the same content reports hundreds of spurious moves.

Every difference is an ADDITION except one, listed below.

- `cbeam_methods.txt`  +1: `positional_banded`
- `c_public_symbols.txt` +1: `beam_model_positional_banded`
- `python_api.txt`     +1: `BeamModel.positional_banded`
- `pytest_cases.txt`  +16: `tests/python/analytics/test_positional_banded.py`
- `c_cases.txt`       +11: `tests/native/test_positional_banded.c`
- `c_symbols.txt`     +22, -1

`c_symbols.txt` is the one that needs explaining, because only two of its
twenty-two additions belong to this change:

  - `beam_model_positional_banded`, the new entry point;
  - `beam_spectrum_spacing`, which was `static` in `math/spectrum.c` and is
    now shared, so the banded sweep lands on the same grid the spectrum does.
    Its own docstring already argued for that -- the paired and per-length
    spectra are only comparable because they share it -- and a second copy of
    the arithmetic would agree today and drift later.

The other twenty, and the single removal, are PRE-EXISTING drift from the
memory-optimisation campaign that preceded this: the `account.c`,
`fb_sweep.c` and `compaction.c` splits gave internal helpers external
linkage, and `beam_model_crc32c` went away when `core/crc32c.c` was rewritten.
The gate had been reporting all of them as notes rather than failures --
correctly, since none is declared in `include/beam/` -- and they entered the
recorded set the moment it was re-recorded for any reason. Nothing public was
removed; `c_public_symbols.txt` lost nothing.


# Why six baselines were re-recorded for the insights module

Re-recorded 2026-09-11 for `beam_insights_logos` and the `insights`
subpackage. Previous copies kept as `*.preinsights.txt`.

Compared as SETS, which is how `check.py` compares them. **Nothing was
removed from any of the six.**

- `cbeam_methods.txt`   +2: `insights_logos`, `insights_logo_cost`
- `c_public_symbols.txt` +4: the two entry points, `beam_logo_request_default`
  and `beam_insights_logo_set_free`
- `python_api.txt`      +1: `BeamModel.insights`. Its signature was recorded
  twice in the same session: `surprisal_above` / `surprisal_below` were added
  after the first recording, so that both axes have named keywords.
- `pytest_cases.txt`   +30: `tests/python/insights/test_logo.py`
- `c_cases.txt`        +12: `tests/native/test_insights_logo.c`
- `c_symbols.txt`       +6: the four public ones above, plus two internals

The two internal symbols are worth naming, because both are `static`
functions that became shared and neither is public API:

  - `beam_surprisal_range_at`, previously `range_at` in `math/extrema.c`.
    The logo layer refuses a band outside the surprisal a length can attain,
    and must do so BEFORE allocating -- two sweeps over the state space
    against a pool measured in gigabytes.
  - `beam_positional_banded_bytes`, previously `banded_bytes` in
    `math/positional_banded.c`, so `beam_insights_logo_cost` reports the same
    figure the sweep is guarded on. `budget.h` requires a guard and its
    allocation to come from one expression; an estimator that disagreed with
    the guard would be that defect wearing a different hat.

`exact.py` grew from 280 quantities to 336: the logo's probability, entropy,
information and heights are now checked against entropy computed by hand from
the enumerated support. That is deliberate placement. `numeric.json` pins a
quantity against a recording of itself, so a logo wrong from the day it was
written would pass it forever; `exact.py` is the only check with an
independent reference.


# Why six baselines were re-recorded for depth-matched Hill profiles

Re-recorded 2026-09-15 for `beam_model_log_mellin_many`,
`BeamModel.log_mellin_many` and `BeamModel.hill_profile`, and the `diversity`
subpackage. Previous copies kept as `*.predepth.txt`.

Compared as SETS, which is how `check.py` compares them, except
`python_api.txt`, which it compares line for line. **Nothing was removed from
any of the six** -- the one line that left `python_api.txt` is the old
`__all__`, replaced by one naming three more.

- `cbeam_methods.txt`    +1: `log_mellin_many`
- `c_public_symbols.txt` +1: `beam_model_log_mellin_many`
- `c_symbols.txt`        +2: the public one above, and `jet_sweep_fan`, which
  is internal: declared only in `src/native/internal/beam/dp.h`, beside the
  `jet_sweep` whose bits it is held to.
- `python_api.txt`      +53 lines: the two `BeamModel` methods,
  `DepthMatchedProfiles`, `ResamplingBand`, `depth_matched_hill_profiles`,
  and three names in `__all__`.
- `pytest_cases.txt`    +47: `tests/python/math/test_hill_profile.py` (24) and
  `tests/python/diversity/test_depth_matched.py` (23)
- `c_cases.txt`          +5: `tests/native/test_mellin_many.c`

`numeric`, `saved-model bytes` and `thread independence` stayed green across
the change, which is the half of the argument a re-record cannot supply: the
change adds entry points and alters no existing one. Separately, `hill_number`,
`log_mellin`, `entropy` and `path_count` were compared hex for hex before and
after on nine further models, both alphabets and both worlds included -- 684
values, none moved.

`exact.py` grew from 336 quantities to 406: `log_mellin_many` at five orders,
two of them negative, and `hill_profile` at five orders including both
corners, on each of the seven amino-acid configurations. Against the
enumeration rather than against the single-order forms, which the C and Python
suites already pin them to with `==`: a fused sweep wrong the same way at every
order would pass that comparison and cannot pass this one.

# Why the C test inventory was re-recorded (log-Z combined kernel)

Re-recorded 2026-09-16. One test added, nothing removed; the prior file is
kept beside it as `c_cases.prelogz.txt` so the change is auditable.

    + test_sparse_combined_max_and_marked_rows

It pins the contract of the combined thin-frontier path in
`beam_compute_log_z`, which now asks one traversal for two things: the exact
maximum over every occupied state, and complete rows for a marked subset. The
test checks both halves against the old state-at-a-time definition -- the
maximum against `reference_layer_max`, and every marked row byte-for-byte
against `beam_state_weights`, including marked states whose forward-wide row
is absent and whose defined output is therefore an all--inf row.

The addition was not taken on trust. Across the same change `numeric` (78
pinned hex floats), `exact against enumeration` (406 quantities), `thread
independence` and `saved-model bytes` were all green, the Python suite held at
735 passed, and every one of 18 foundation-model quantities -- including
`n_nonzero_cells` 63,136,647, `path_count`, `entropy`, the 16384-bin counting
spectrum and the positional marginals -- was bit-identical to the pre-change
binary.
---

# The single-expert model, 2026-09-21

Branch `feature/single-expert-model`. Six baselines move and one does not,
and this section is written BEFORE the re-record so that what is expected to
move is on the record independently of what did.

This is a model change, not a refactor. Nothing will be bit-identical and
nothing should be. The gate below therefore cannot certify the change; it can
only certify that the change is the ONLY change, which is what re-recording
in a commit that touches nothing else is for.

## What was verified first

Re-recording to make a failure go away is the hazard this file exists to
prevent, so the preconditions were checked and are recorded here.

- **C suite: 315 passed, 0 failed.** No tolerance was loosened without the
  bound that justifies it written beside it.
- **Python suite: 675 passed, 4 skipped, 0 failed.**
- **`exact.py`: 336 quantities against brute-force enumeration, worst
  deviation 8.8% of its tolerance.** This is the check that compares against
  arithmetic rather than against a recording, so it is the one that can fail
  when a baseline cannot, and it passes.
- **The independent reference was PORTED, not deleted.**
  `tests/reference/beam_model.py` is rewritten from the specification rather
  than from the new C, because deriving it from the implementation would make
  the agreement test a tautology. Measured: the two agree to 8.9e-16 and
  agree exactly on which sequences are in the support.
- **Thread independence checked directly**, not through the baseline:
  bit-identical at 1, 4 and 16 threads. The baseline entry fails only because
  it records the old model's values.

## What moves, and why

`python_api.txt`. `BeamModel.__init__` and `Builder.__init__` lose
`depth_cap`, `coefficients`, `kappa`, `kappa_wide` and `kappa_len`. The model
has three free parameters now: the context order and the two anchor weights.

`c_symbols.txt`, `c_public_symbols.txt`. `beam_mix` loses three parameters;
`beam_depth_part`, the per-expert row accessors and the table-kind enum are
gone.

`pytest_cases.txt`. **Eighteen removed, fourteen added**, and every one is
accounted for below. The gate prints only the first six of each, so an
earlier draft of this section said "six and six" from reading its output --
which is the imprecision this file exists to prevent, and is corrected here
by diffing the recorded files rather than the report.

Ten of the eighteen are the on-disk compatibility fixtures, five under
`test_an_older_file_still_loads_with_the_same_values` and five under
`test_an_older_file_survives_a_round_trip_through_the_current_format`. They
become ten others, `test_an_older_file_is_refused_and_says_why` and
`test_the_refusal_is_not_a_checksum_complaint`, because formats 1 to 5 are
now refused: their depth index was clamped, so which depth an observation
came from was discarded at training time and no transformation recovers it.
The five fixture FILES are kept, unmodified. What they are evidence of has
changed from "this build reads them" to "this build declines to misread
them", and regenerating them would turn evidence about the past into a
recording of the present. The second of the two new tests exists because the
dangerous failure was never refusal: before the format bump an old file
carried the same magic, header size and format number as a new one, so
nothing would have rejected it, the payload extents would simply have been
wrong, and it would have surfaced as a checksum complaint -- blaming an
intact file for the reader's mistake.

Three more are `test_sparse_strict_is_bit_identical_to_dense_strict`, which
loses its `depth_cap` axis: `[2-5-None]` becomes `[2-None]`. Reparametrised,
not deleted.

The remaining five are deletions whose subject is gone:
`test_coefficients_must_sum_to_one` (replaced by
`test_single_anchor_weights_are_accepted`, the fourteenth addition),
`test_strict_defaults_kappa_to_zero_but_explicit_wins`, the two
`test_out_of_range_configuration_is_refused[...-depth_cap]` parametrisations,
and `test_incompatible_members_are_refused_at_construction[differs3]`, which
was the mixture shape-mismatch case built on `depth_cap`.

`c_cases.txt`. Six removed, four added.

  - `test_depth_cap_is_clamped`, `test_layers_beyond_the_depth_cap` --
    subject deleted.
  - `test_state_covers_the_mid_expert` becomes
    `test_the_state_is_exactly_one_context_wide`, and its assertion INVERTS:
    an order-2 model carries a 2-residue state where it used to carry 3.
  - `test_rejects_uncalibrated_coefficients` becomes
    `test_single_anchor_weights_are_accepted`. The weights should sum to 1 and
    are deliberately not refused when they do not, because scoring one anchor
    alone is the only way to separate a partition function below 1 into its
    two causes.
  - `test_sparse_requires_strict_zero_backoff` becomes
    `test_sparse_requires_strict`.
  - `test_legacy_format_files_still_load` becomes
    `test_legacy_format_files_are_refused_by_name`.

`numeric.json`. 60 of 78 keys move. **18 do not, and each is principled** --
this was checked with `BEAM_NUMERIC_ALL=1`, which lists them, because a key
that should have moved and did not is a code path that was missed:

  - `lenpmf` (6 keys): the empirical length distribution. Same corpus.
  - `D0` at orders 3 and 4 (4 keys) and `topology` at orders 3 and 4 (4 keys):
    the closed-world support is set by the BINDING expert. At order 4 that was
    the order-4 wide table both before and after; at order 3 the old wide and
    mid were both order 3. `o2/strict/D0` and `o2/*/topology` DID move,
    because the old order-3 mid expert was WIDER than an order-2 wide one and
    was the binding constraint there, and because `state_order` went 3 to 2.
  - `unseen` in strict mode (3 keys): still exactly `-inf`.
  - `o2/open/D0` (1 key): the open world's support is the full grammar.

  All six blocks have moved keys, which is the check that no code path was
  missed.

`fixture_sha256`. Different tables, and the file format is now 6.

`threads.json`. Same values at 1, 4 and 16 threads; different values.

## What does not move

`cbeam_methods.txt`. No entry point was added or removed. The wire format
between `_cbeam` and `config.py` narrowed from `OOiii(ddd)(ddd)dd(dd)pps` to
`OOii(dd)pps`, which is a change of arity and not of membership.

`header_file_refs_historical.txt`. Eight historical mentions, unchanged.

## `c_symbols.txt`, same branch, one symbol

`beam_build_hierarchy` removed. It built the backoff parent levels, and the
single-expert change deleted all five backoff strengths, so it had no caller
left. `smoothing.c` goes from 190 lines to 62: four of its five functions
were the hierarchy and its helpers, and what survives is
`beam_to_log_inplace`.

Recorded separately from the re-record above, and after it, because it is a
different act. The first was "this model produces different numbers"; this is
"this function no longer exists". Folding them together would have hidden a
deletion inside a diff nobody would read line by line, which is the failure
this file exists to prevent.

Nothing else moved: the C suite stays at 315 passing and the Python suite at
675, both run against a clean rebuild.

One comment in `finalize.c` went stale with it. It justified a sparse-storage
condition by saying that if the rule ever relaxed, `beam_build_hierarchy`
reads counts across tables and would find them gone. There are no longer
several tables to read across, and no hierarchy to read them. The condition
is still written out rather than assumed, for the reason that survives: the
config check guards construction, and a file could carry a header this code
would otherwise trust.

## The storage constants, re-measured

The implementation plan flagged three storage constants as tuned against the
old three-table geometry and therefore possibly stale. Measured on the new
model, with a 200,000-sequence corpus drawn from the pooled release table at
order 4, `max_length` 22, closed world:

```
table        dense rows     occupied      occ%   sealed      pool      dict   repeat
forward         4,278,582      254,357    5.945%     yes    56,775    26,356    12.2x
backward        4,278,582      268,909    6.285%     yes    56,022    28,012    11.2x
```

**All three gates fire exactly as their own comments say they should**, and
none needs changing:

  - The rank-seal gate fires below 12.5% occupancy. Measured 5.9% and 6.3%,
    so a per-repertoire model still seals, which is what the comment says the
    threshold is for.
  - The interning gate fires when at least a quarter of rows repeat. Measured
    77.7%, against the 73.1 to 79.1% the comment records for a donor.
  - `BEAM_PACK_MAX_DICT` is not a tuned constant at all: 65,536 is what a
    two-byte index can name. Measured dictionaries are 26,356 and 28,012, well
    inside it. Packing cuts the payload by 66%, from 2,518,504 bytes to
    853,722.

**One earlier note in this branch was wrong and is corrected.** I recorded in
`test_packed_cells.c` that the packing repetition factor had fallen to 1.46
against a gate assuming 2, and called it the constant being stale. It is the
FIXTURE. That test builds from residues drawn uniformly at random with
weights 1 to 60, so contexts almost never recur and `log(count/total)` takes a
different value nearly every time; uniform noise is the worst case for a
dictionary. Real repertoire data has shared motifs and packs 11 times over.

The test now asserts the break-even rather than a margin -- packing pays when
`dict * 4 < cells * 3`, since a cell costs two bytes and a distinct value
eight -- which holds on the noise fixture and has enormous room on data.

The rank-seal timing trade (scoring faster, Hill sweep slower) has NOT been
re-measured. The constant still selects the right representation, which is
what the occupancy figures above establish; whether the crossover point moved
is a benchmark, and it would change only how much memory is returned, not
which answers come out.

# Why six baselines were re-recorded for model metadata

Re-recorded 2026-09-21, tag `premeta`, for `BeamModel.metadata()` and the
format-7 provenance it reads. The previous files are kept as `*.premeta.*`.

## What moved, from a diff of the files rather than from the gate's output

    python_api.txt       1 line changed, 2 added, 0 gone
    pytest_cases.txt     11 added, 0 gone
    c_cases.txt          2 added, 0 gone
    cbeam_methods.txt    3 added, 0 gone
    c_symbols.txt        3 added, 0 gone
    c_public_symbols.txt 3 added, 0 gone
    numeric.json         1 key changed: fixture_sha256
    threads.json         byte-identical

The gate prints six added pytest cases and the file gained eleven. Both are
right and the file is the one to quote: the gate compares deduplicated test
NAMES, and five of the eleven are names that already existed elsewhere in the
suite. This section is written from `diff`, which is why it disagrees.

Added, and none of them replaces anything:

  - `metadata(self, *, lengths=True, support=False)` on BeamModel, and the
    module-level `file_provenance_span()`.
  - Eleven cases in `tests/python/model/test_metadata.py`.
  - Two C cases: `test_provenance_is_recorded_and_aggregated` and
    `test_depth_occupancy_sums_to_the_nonzero_cells`.
  - Three public C symbols, and the three `_cbeam` callables over them:
    `beam_model_provenance`, `beam_model_depth_occupancy`,
    `beam_model_file_provenance_span`.

`c_symbols.txt`, `c_public_symbols.txt` and `cbeam_methods.txt` are not
written by `--record`, by design: adding a public symbol should be a
deliberate edit rather than a side effect of a recording run. They were edited
by hand to add exactly the three names above.

## The digest moved, the file did not

    fixture_bytes   233050 -> 233050   (unchanged)
    fixture_sha256  9661451a... -> cba4c711...

The saved file is the same SIZE because provenance was written into bytes the
header already had: three `int64` fields in the space formats 1 to 5 spent on
the expert mixing coefficients, which the single-expert model left dead. The
static assertion that the header is 144 bytes is unchanged and still fires.

The digest moved because what is hashed moved, not because the bytes did. A
`.beam` file used to be a pure function of the corpus, and `fixture_sha256`
hashed the whole of it. Three of those bytes now read a clock, so hashing the
whole file would compare a timestamp against a recording of itself and fail
every run from here on. That is a genuine loss and it is stated rather than
absorbed: the check now hashes everything EXCEPT the provenance window and the
trailing CRC that covers it.

Two things keep that from being a quiet weakening:

  - The window is not hardcoded anywhere. `beam_model_file_provenance_span`
    derives it with `offsetof` from the first and last provenance field, with
    static assertions that the three stay contiguous, and both callers --
    `check.py` and `test_saving_a_frozen_model_writes_the_same_bytes` -- ask
    for it. A field added between them moves the window instead of leaving a
    stale offset aimed at payload.
  - The C test asserts the arithmetic of its own skip list: the window is 24
    bytes, and the number of bytes it compared is exactly the file size minus
    that window minus the four-byte CRC. A skip list that grew to swallow
    payload fails there rather than passing quietly.
    `test_provenance_is_the_only_thing_two_identical_builds_disagree_on`
    checks the same property from Python and decodes the window to confirm it
    is aimed at the timestamps and not at dead bytes.

So the reach that was lost is 28 bytes of a 233,050-byte file, all three
fields of it unreadable as anything but a clock.

## What did NOT move, which is the argument that nothing broke

`numeric.json` holds 80 lines. The 78 the gate compares -- every count,
probability, entropy, Hill number and partition function in it -- are
bit-identical across this change, and `fixture_bytes` is unchanged too, so
`fixture_sha256` is the only line that moved;
`threads.json` is byte-identical; `exact` still passes 336 quantities against
brute-force enumeration at 8.8% of its worst tolerance. Metadata only reads
the model -- it computes occupancy from the tables and returns stored
timestamps -- so no count, no probability and no table could move, and these
three are what say so rather than the claim.

## Addendum, same day, tag `prova`: two defects the suite did not catch

Both were found by probing `metadata()` on a loaded and on a compacted model
rather than by the tests, which is worth recording as plainly as the fixes.

    pytest_cases.txt   2 added, 0 gone
    everything else    unchanged

`python_api.txt`, `c_cases.txt`, `numeric.json` and `threads.json` came back
byte-identical, so their `*.prova.*` snapshots were deleted rather than
committed: a snapshot that records no change records nothing.
`c_cases.txt` is unchanged because the new C assertions went INSIDE
`test_provenance_is_recorded_and_aggregated` rather than adding a case, and
`python_api.txt` because `metadata()`'s signature did not move -- only what
its `size` dictionary contains.

### The build timer rose every time the model was opened

A `.beam` file stores counts rather than finalized tables, which is what keeps
a loaded model expandable, so every load rebuilds the tables. That finalize
was being charged to the model. The reported build time therefore grew on
each load, and grew PERMANENTLY if the model was then saved: the foundation
model would accrue a load's finalize per session for ever.

It also contradicted the requirement, which is that the timer aggregate each
time sequences are ADDED. A load adds none.

The load's own finalize is now excluded and a caller's re-finalize still
counts. The round-trip test previously asserted `finalize_seconds >= before`
with a comment explaining why growth was acceptable -- the comment
rationalised the behaviour instead of checking it -- and now asserts equality.
`test_a_caller_s_own_refinalize_still_counts` is the counterweight: without
it, the cheapest way to make the round trip exact would be to stop timing
finalizes at all. The C test gained the same round trip next to the code.

### metadata() raised on a compacted model

`compact()` releases the raw counts, `n_nonzero_cells` inspects them, so
`metadata()` raised on exactly the models most worth asking about: a large
model is deployed compacted. It now reports those two as `None` with
`compacted: True` beside them, rather than raising or substituting a figure
that would read as a real occupancy. The per-length breakdown is unaffected
and asserted equal across `compact()`, since it comes from the finalized
tables that compaction keeps.

`numeric`, `saved-model bytes`, `thread independence` and `exact` are all
unchanged across both fixes, which is what says neither touched the model.

## Addendum, tag `presweep`: what a three-way audit of the branch found

Three agents swept the tree for leftovers of the single-expert change, for
gaps in the metadata work, and for documentation that no longer matches the
code. Every claim quoted below was re-verified here before it was acted on;
several of their other claims were right about the defect and wrong about its
size, which is why the numbers in this section come from `diff` and from the
gate rather than from the reports.

    pytest_cases.txt   4 added, 0 gone
    everything else    unchanged

`c_cases.txt` did not move even though two C test functions were deleted --
neither was ever registered with `RUN_TEST`, so neither was ever in the
baseline. That is the evidence they were dead rather than merely unused.

### Two defects in the metadata feature

**`add` and `subtract` did not carry the build time.** The three corpus
counters moved and the clock did not, so a model assembled from shards
reported `n_observed` rising while `observe_seconds` stayed put, and
`Builder.snapshot()` -- which goes through `add` -- reported 0.0 for a corpus
it had genuinely spent time counting. That is the requirement this feature
exists to meet: aggregate each time sequences are added. `add` now adds the
other model's `observe_nanos` and `subtract` removes it, floored, exactly as
the comment beside the counters already argued for them. `created_unix`
deliberately does not move: absorbing counts is not a re-creation.

**A format-6 file was refused with a false diagnosis.** The metadata commit
widened the "three experts per direction and a capped depth index ... retrain
the model" refusal from formats 1-5 to 1-6. Format 6 is not an older model --
it is this one, written before provenance existed -- so every clause was
false, and for a foundation-sized corpus an expensive thing to be told.

### The fix for that second one was itself wrong, and a fixture caught it

Accepting format 6 looked safe: diffing `3806d4c..46e3468` shows the header
layout byte-identical, provenance occupying the 24 bytes format 6 wrote as
explicit zeros. That diff was the wrong thing to check. The reader gates the
frozen-counts payload on `h.format >= BEAM_FORMAT`, and bumping that constant
to 7 silently stopped it matching format 6, so a format-6 file -- whose counts
ARE frozen -- was read as plain sparse records and reported as truncated. The
gate is now against `BEAM_FORMAT_V6`, which is the version the payload
actually changed at.

`tests/fixtures/format6_sparse.beam` is what found it. It was written by a
worktree at `3806d4c`, the last commit that emitted format 6 -- NOT by
patching a format-7 file with this build, which is the one thing
`test_format_compatibility.py` says a fixture here must never be, and which
would have reproduced the bug rather than exposing it. `format6.json` records
what that build read back, and the test asserts this build reproduces every
`log_pseq`, the path count and the entropy bit for bit.

### Dead code the sweep turned up

- `test_depth_cap_is_clamped` and `test_layers_beyond_the_depth_cap`: defined,
  never registered, both emitting `-Wunused-function`. There is no depth cap.
- Three `uint32_t kind_unused` declarations, left from when `count_table` took
  an expert-kind out-parameter.
- `tests/native/test_alphabet.c`: a refusal case stood without its `++n`, so
  the loop never reached it. That was load-bearing by accident -- sparse AND
  strict is now a VALID config, so had it ever run it would have failed.
- The build is now warning-free.

### The gate lost 45 quantities, and they were duplicates

    exact against enumeration   336 -> 291

`exact.py` built a seventh configuration tagged "no-backoff" to put a badly
conditioned model in the grid. The keyword that switched backoff off no longer
exists, so `kw = {} if kap is None else dict()` had both branches empty and
the seventh config had become a bit-for-bit copy of the sixth -- measuring its
quantities twice under a name for a regime nothing was testing. Removing it
costs nothing: 336 = 7 x 45 + 21 and 291 = 6 x 45 + 21, so exactly one
configuration's worth left and the six that remain are unchanged, which the
unmoved `numeric` and `saved-model bytes` confirm.

## Addendum, tag `prehill`: the bootstrap Hill profile ported onto one expert

`feature/bootstrap-hill-profile` (94799e6) was written against the THREE-EXPERT
model and never merged, so two live figure scripts --
`donor_hill_separability/_remote/pair_separability.py` and
`structure_profiles.py` -- called `BEAM.depth_matched_hill_profiles` and
`BeamModel.hill_profile`, neither of which existed on this branch. They would
have failed on ig05 after the foundation rebuild.

    python_api.txt    53 signatures added, __all__ modified
    c_cases.txt       1 added        pytest_cases.txt   2 added
    numeric.json      byte-identical        threads.json  byte-identical

`numeric` and `threads` not moving is the claim worth making: the port added
an API and changed no answer.

### What was taken, and what was deliberately left

The branch had three strands. The API came across whole -- `BEAM.diversity`,
`DepthMatchedProfiles`, `ResamplingBand`, `hill_profile`, `log_mellin_many` --
and so did `jet.c` and `mellin.c`, which carry the `log_mellin_many` fusion
measured at 11.7x on ig05. All 47 of the branch's own tests pass unmodified
against the single-expert model.

Two conflicts were real. In `table_access.h` the branch side was the
three-expert `beam_lp_wide_row` / `_mid_` / `_len_`; its contribution was not
those functions but `BEAM_INLINE` over plain `static inline`, because at -O2
GCC declined to inline them and that was measured at 3.75% of cycles inside
the edge-weight kernel. The single accessor now carries it, since the reason
survives the model change.

In `edge_weights_kernels.h` the branch's work was specific to three experts --
hoisting invariant backward mid and length rows out of the per-residue loop --
and there are no such rows now, so HEAD was taken whole. No caller anywhere
still references `_with_fw`, `beam_lp_mid_row` or `n_ctx_mid`; the
optimisation was self-contained in that header.

The `beam_compute_log_z` fusion (69.36s -> 55.97s) was NOT ported. It was
measured against a kernel that no longer exists, and re-deriving it for the
single-expert shape is a separate piece of work with its own measurement.

## Addendum, tag `preceiling`: the coverage denominator was counting the unreachable

    pytest_cases.txt   4 added, 1 gone (a rename plus a parametrisation)
    everything else    unchanged

`metadata()` reported `rows_possible` as `2 * L * radix ** order` -- every
context the alphabet can spell, at every depth. Most of those rows cannot be
addressed by any sequence. At depth `d` only `min(d, order)` context slots are
free; the rest are the sentinel that pads that direction's anchor, so
`R ** min(d, order)` rows exist there and the remainder are structurally
unreachable rather than merely unobserved.

Measured on 60,000 distinct 6-mers, which saturate the shallow depths:

    depth   rows observed   old denominator   reachable
      0           1             194,481            1
      1          20             194,481           20
      2         400             194,481          400
      3       7,994             194,481        8,000

So a model that had found everything reachable at depth 0 was reported at
1/194,481 of it. The denominator is now `2 * sum_d R ** min(d, order)`.

### Checked against brute force, not argued

Enumerate EVERY sequence of a length over the whole alphabet and the cells the
model then holds ARE the reachable set, with nothing sampled and nothing
assumed:

    L   sequences   n_nonzero_cells   cells_possible   coverage
    2         400             840              840       1.0000
    3       8,000          16,840           16,840       1.0000
    4     160,000         336,840          336,840       1.0000

A model built from every sequence of a length now reads as 100%. Under the old
denominator length 4 read as 1.08%, claiming 1,555,848 possible rows where
16,842 exist.

`test_the_possible_cell_count_is_what_brute_force_occupies` is that table, and
it fails if the addressing in `indexing.h` moves.

### Where the check came from

`.private/manuscript/figures/.../parameter_discovery` already had this
reasoning and its own brute-force verifier, written when the figure was built.
It was the only place that knew, and it ran only when the figure ran. The
check now lives in the test suite; the figure asserts agreement with
`metadata()` instead of keeping a second, separately-verified opinion.

That figure was also reworked for the single expert. Its ceiling lost the
wide/mid/len split and the `depth_cap` clamp, and its union over lengths
collapsed: a cell is addressed by (direction, depth, context) with no length
in the address, so a shorter length's depth set is a strict subset of a longer
one's and the union is simply the longest length's ceiling. The old
`max(wide) + max(mid) + sum(len)` existed because one of the three tables
carried the length; none does now.

`numeric`, `threads`, `python_api` and `c_cases` are unchanged: this moves a
reported denominator, not an answer, a signature or a test binary.

## Addendum, tag `presum`: one test recorded late

    pytest_cases.txt   1 added, 0 gone
    everything else    unchanged

`test_cells_observed_may_not_be_summed_over_lengths` was committed with the
storage re-measurements and its inventory line was not recorded with it, so
the gate failed on the next run. It pins that per-length `cells_observed` must
not be summed -- a cell carries no length in its address, so a depth-3 cell is
read by every length from 4 up and the sum counts it once per length -- and
that the frozen and mutable count paths agree, which is what the sparse
foundation model will exercise.

`numeric`, `threads`, `python_api` and `c_cases` are all unchanged.

## Addendum, tag `prelayer`: log Z(L) split across its layers

    python_api.txt   2 added      c_cases.txt      1 added
    pytest_cases.txt 10 added     c_symbols/public/cbeam  1 each
    numeric.json     byte-identical    threads.json  byte-identical

`numeric` and `threads` not moving is the claim: this adds a reading of the
trellis and changes no answer.

`log P(s) = log W(s) + log pi(L) - log Z(L)`, and both trailing terms belong to
the sequence's LENGTH rather than to any position in it. Anything reading a
per-position cell -- a K x L matrix over a panel of per-donor models -- has to
place them, and spreading them evenly over L is arithmetically valid and
analytically arbitrary: any weights summing to one give the same row total and
a different matrix.

The two terms are NOT equally placeable, and separating them is the point.

`log Z(L)` has positional structure. It is a forward sum over the trellis, a
forward sum factorises by layer, and `partition.c` was already accumulating
the per-layer factors into a scalar and discarding the breakdown. Writing `M_i`
for the mass once residue i has been emitted, `Z(L) = prod_i (M_i / M_{i-1})`.
`beam_model_log_partition_by_layer` returns `log M_i`, so the caller differences
it and the split is the model's own.

It TELESCOPES, which is the property worth having here: the last entry is
`log Z(L)` bit for bit, so the identity holds by construction rather than to
within a tolerance. Re-summing the differences costs about one ULP over L
terms, which is why the API says to read the end point rather than re-add.

It is also far from even. On the test fixture one layer carries more than half
of `-log Z` while four carry essentially none; the uniform split would have put
the same share on each. `test_the_split_is_not_the_uniform_one` fails if that
ever stops being true, so the construction cannot silently degenerate into the
arbitrary split it replaces.

`log pi(L)` has no positional structure at all -- one number off the observed
length histogram, with no decomposition to find. `log_pseq_by_position` places
it on the FINAL position, for the same reason the sink goes there: the length
is settled once the sequence ends. That is a convention and is documented as
the one arbitrary choice in the vector.

COST is one add inside a loop that already rescales and collects every
surviving state, plus one forward sweep per call where `log_partition` reads a
value cached at finalize.

# Why five inventories were re-recorded for tempering

Re-recorded 2026-09-29, for `BeamModel.tempered` -- the tempered law
`P_t(s) proportional to P(s)^t` materialised as a model rather than carried
inside a sweep.

Five files, and every one of them gained lines without losing any. That is
the whole claim, and it is checkable rather than asserted:

    cbeam_methods.txt        +1  -0     temper
    c_public_symbols.txt     +1  -0     beam_model_temper
    c_symbols.txt            +1  -0     the same, newly external
    python_api.txt           +1  -0     def tempered(self, t: 'float') -> 'BeamModel'
    pytest_cases.txt        +14  -0     tests/python/model/test_tempering.py

`git diff verification/baseline/` shows no `-` line in any of them.

Nothing NUMERIC moved, which is the part worth stating plainly. `numeric`
(78 keys, order 2/3/4 x open/strict), `exact against enumeration` (351
quantities), `saved-model bytes` and `thread independence` all passed
UNCHANGED against the recordings made before this work. Tempering adds an
operation and alters no existing one: `beam_model_temper` is only ever
reached through `tempered`, it is the only caller of the scaling it performs,
and it runs on a COPY, so no model that existed before can observe it.

The new C symbol is public rather than internal because the header declares
it, which is deliberate: it is an operation on a model, of the same standing
as `beam_model_add` and `beam_model_compact` beside it, and the note the gate
prints about a symbol newly external is the honest description of a function
that was written for this purpose rather than exposed by a file split.

The fourteen cases are listed here so a later reader can tell an added test
from a renamed one:

    test_tempering_at_one_is_the_identity
    test_the_defining_identity
    test_the_tempered_mellin_transform_is_the_original_reparametrised
    test_the_tempered_model_is_a_distribution
    test_the_length_prior_is_the_tilted_share
    test_tempering_reweights_the_support_without_widening_it
    test_the_entropy_follows_the_mellin_derivative
    test_raising_the_exponent_lowers_the_entropy
    test_within_one_length_the_ranking_survives
    test_what_tempering_refuses
    test_the_result_has_no_counts_left
    test_the_original_is_untouched
    test_the_identity_holds_for_every_storage_and_world[cfg0]
    test_the_identity_holds_for_every_storage_and_world[cfg1]

None of them replaces anything. Each was also shown to FAIL against a
deliberately broken implementation before being trusted: dropping `Z_t(L)`
from the length prior fails 7, never recomputing `Z_t` fails 8, scaling only
the N-anchored table fails 6, skipping the sparse dictionary branch fails 5,
and leaving the length prior unnormalised fails 7. A test battery that has
never failed pins nothing, and this one was made to.

## What this re-record deliberately did NOT absorb

`pytest_cases.txt` was recorded at 778 cases: the 764 the previous baseline
held, plus the 14 above. The suite as it stands in this working tree collects
**785**, and the seven-case difference is `tests/python/likelihood/
test_lattice.py`.

Those seven were left out on purpose. `tests/python/likelihood/` and
`src/BEAM/likelihood/` are UNTRACKED -- `git ls-files` returns nothing for
either -- so they are uncommitted work in the tree rather than part of the
repository. Recording them would pin, in a tracked baseline, tests that no
fresh clone and no `git clean` can produce, and the gate would then fail for
everyone else with those seven under `missing`.

So the gate's `pytest inventory` row FAILS on this branch, and it failed the
same way before this work: the previous baseline's 764 contains no
`test_lattice` case either, while the files were already present and already
collectable. The row is a statement about the untracked likelihood work, not
about tempering. Committing that work, or removing it, is what resolves the
row; re-recording around it would only hide which of the two happened.

# Why every baseline was re-recorded for the hierarchical prior

Re-recorded 2026-09-29, tag `preprior`: the closed world (`strict=True`) and
the open world (`strict=False`, a Laplace add-one) were REMOVED and replaced by
one model, a hierarchical Bayesian prior. Every row of either table is now the
posterior mean of a Dirichlet centred on its parent, the same depth with one
residue less of context,

    P_k(x) = (n_k(x) + alpha[k] * P_{k-1}(x)) / (n_k + alpha[k]),

down to the depth alone and then the uniform 1/R, and the length prior is the
same rule over lengths 1..max_length. Every string up to `max_length` is
possible; an unobserved context predicts what its shorter context predicts.
This is a change of MODEL, so every number moved, and every file below was
re-recorded from one gate run (`check.py --record preprior`) with the previous
copy kept beside it.

## The independent witness passed before anything was recorded

`exact.py` was moved to the new support first -- every length 1..max_length,
168,420 amino-acid strings per configuration, orders 2/3/4 crossed with the
default prior and an uneven one (strengths five orders of magnitude apart
between levels) -- and passed all 351 quantities, worst 4.4% of its tolerance.
The recorder refuses to write while it fails. Beyond the gate, the finalized
tables match a plain-Python evaluation of the rule to ~1e-14, and
`tests/reference/beam_model.py` (rewritten for the prior, with its own Z(L)
sweep) matches whole-model scores to ~1e-13.

## What moved

    numeric.json      all 78 keys, relabelled o{2,3,4}/{default,uneven}
                      from o{2,3,4}/{open,strict}; D0 is now sum_L 20^L
                      exactly (3.449e19 at max_length 15); fixture sha256
                      cba4c711.. -> 64c3b695.., 233,050 -> 233,082 bytes
                      (format 8: the header carries the prior's strengths)
    threads.json      re-recorded; identical at 1, 4 and 16 threads
    python_api.txt    1135 -> 1234 lines: `strict=` removed from BeamModel
                      and Builder; `alpha=`, `alpha_length=`, `set_alpha`,
                      `fit_alpha`/`AlphaFit` added, and the likelihood
                      package's public API (below)
    pytest_cases.txt  778 -> 817: +142 -103, the migration of every test
                      that asserted a closed-world fact, the new prior and
                      format tests, and tests/python/likelihood
    c_cases.txt       325 -> 331: +32 -26, the same migration in C
    cbeam_methods     +set_prior
    c_public_symbols  +beam_model_set_prior, -beam_model_is_legacy_file
    c_symbols         the same, and -beam_to_log_inplace (the add-one's
                      log conversion, dead with it)

`beam_model_is_legacy_file` is removed on purpose: only the current file
format (8) is read, and a file from an earlier development build -- EDGE's
included -- is refused with its format number and "Rebuild the model from its
sequences". There are no readers for formats 1-7 left to test.

## What this re-record absorbs that the previous one declined

The tempering section above left `tests/python/likelihood/` out of
`pytest_cases.txt` because it was untracked. It is recorded now: the
likelihood package is part of this branch's work and is to be committed with
it. If it is not, this row fails for exactly that reason, which is the
honest place for it to fail.

# Why the default-prior baselines were re-recorded for estimated strengths

Re-recorded 2026-09-29, tag `preauto`. A model built without `alpha=` no longer
uses a fixed strength of 10: finalize estimates each level's strength, and the
length prior's, as the maximum of the evidence of the model's own counts (the
Dirichlet-multinomial marginal likelihood, exact) under a weak log-normal
hyperprior centred on 10. So every quantity of a DEFAULT model moved, and
nothing else did.

    numeric.json      33 of 78 keys: exactly the o{2,3,4}/default ones, whose
                      strengths are now estimated. The o{2,3,4}/uneven keys
                      name their strengths and are bit-identical. The fixture
                      grows 233,082 -> 233,090 bytes: format 9, whose header
                      records which strengths were estimated.
    threads.json      re-recorded; identical at 1, 4 and 16 threads, which is
                      the property the estimate was written to keep (its sums
                      run in fixed chunks, whatever the thread count).
    python_api.txt    +1 -1: fit_alpha's search bound, 1e6 -> 1e12, the same
                      range the estimate searches.
    pytest_cases.txt  +6 -1: the evidence, thread-independence, round-trip
                      and "auto" tests; the default-is-ten test is replaced
                      by the one saying the default is estimated.
    c_cases.txt       unchanged.

`exact.py` passed before recording (351 quantities, worst 4.0%). The estimate
itself is checked in `test_hierarchical_prior.py` against the evidence
recomputed in plain Python with `math.lgamma`, level by level.

# Why the baselines moved for complex Mellin orders

Re-recorded 2026-09-30, tag `precomplex`. The power sum M(q) = sum_s P(s)^q is
now computed at COMPLEX q, and a module computes occupancy quantities exactly
from it, so the public surface grew and nothing that existed moved.

    cbeam_methods.txt      +2 -0  log_mellin_complex, log_mellin_line
    c_public_symbols.txt   +2 -0  beam_model_log_mellin_complex,
                                  beam_model_log_mellin_line
    c_symbols.txt          +4 -0  the two above, and the internal sweeps
                                  jet_sweep_cfan and jet_sweep_line
    python_api.txt         +2 -0  BeamModel.log_mellin_complex,
                                  BeamModel.log_mellin_line
    pytest_cases.txt      +58 -0  test_mellin_complex.py (19),
                                  test_mellin_line.py (13),
                                  analytics/test_exact_occupancy.py (26)
    c_cases.txt            +6 -0  tests/native/test_mellin_complex.c
    numeric.json           unchanged (re-written by the recorder, byte-identical)
    threads.json           unchanged (likewise)

`exact.py` passed before recording, now at 411 quantities (+60: complex
orders and the vertical line, each against the enumerated sum as
M / M(Re q)), worst 5.8% of its tolerance. The per-edge `exp` of
`log_mellin_complex` was made shared across orders with an equal real part
and shift, and held bit-identical to the build before it on 44 orders.

# Why the test inventory moved for batched exact occupancy

Re-recorded 2026-09-30, tag `prebatch`. `ExactOccupancy` gained batch
methods, persisted contour lines and a planner that builds one line at a time;
no C changed, so only the test inventory moved.

    pytest_cases.txt      +20 -0  analytics/test_exact_occupancy.py, ten tests
                                  on both alphabets: novelty (and its underflow
                                  and unresolved cases), private_at_depth, the
                                  vectorised richness, the line round trip,
                                  merging saves, the longer of two lines, the
                                  refusal of another model's file, and a batch
                                  building no more lines than one depth at a
                                  time
    python_api.txt         unchanged (re-written by the recorder, byte-identical)
    c_cases.txt            unchanged (likewise)
    numeric.json           unchanged (likewise)
    threads.json           unchanged (likewise)

The gate passed on everything else before recording, `exact.py` at 411
quantities, worst 5.8% of its tolerance. The full suite: 899 passed, 1
skipped.

# Why the test inventory moved for logged contour lines

Re-recorded 2026-09-30, tag `preline4`. `ExactOccupancy` logs one INFO
record per contour line it computes or extends -- its cost and how many of
the values waiting on it it resolved -- so a long batch can be followed.

    pytest_cases.txt       +2 -0  test_every_line_computed_is_logged, both
                                  alphabets
    python_api.txt         unchanged (re-written by the recorder, byte-identical)
    c_cases.txt            unchanged (likewise)
    numeric.json           unchanged (likewise)
    threads.json           unchanged (likewise)

Recorded with the four-orders-at-a-time line sweep of the next commit already
built, which moved none of these: `numeric`, `thread independence` and
`exact.py` (411 quantities, worst 5.8% of its tolerance, as before) passed
unchanged. The full suite: 901 passed, 1 skipped; the C suite 337 passed.

# Why the test inventory moved for the sampled Hill numbers

Re-recorded 2026-09-30, tag `predeficit`. `ExactOccupancy` gained
`sampling_deficit_many` -- S(N) = sum_s P D(N P), which makes a Poisson
sample's expected Shannon entropy H - S(N) and so its Hill D1 exact -- and
`poisson_richness_many`, its D0. No C changed.

    pytest_cases.txt      +17 -0  analytics/test_exact_occupancy.py: the
                                  deficit kernel against a 30-digit
                                  reference; S(N) at six depths and deep in
                                  N; Poisson richness; each on both alphabets
                                  (the deep-N case skips the amino-acid model,
                                  whose direct sum is too slow there)
    python_api.txt         unchanged (re-written by the recorder, byte-identical)
    c_cases.txt            unchanged (likewise)
    numeric.json           unchanged (likewise)
    threads.json           unchanged (likewise)

The gate passed on everything else before recording, `exact.py` at 411
quantities, worst 5.8% of its tolerance. The full suite: 917 passed, 2
skipped.

# Why the baselines moved for scoring straight from Arrow buffers

Re-recorded 2026-09-30, tag `prepacked`. Scoring a parquet column spent most
of its time making a Python string per sequence and a Python float per score.
`beam_model_log_pseq_packed` scores sequences packed end to end in one buffer
-- a string column's bytes and offsets, as Arrow holds them -- and
`BeamModel.log_pseq_arrow` hands it a pyarrow column directly;
`log_pseq_parquet` now uses it. Every score is bit-identical to
`log_pseq_batch`. Nothing that existed moved.

    cbeam_methods.txt      +1 -0  log_pseq_packed                 (by hand)
    c_public_symbols.txt   +1 -0  beam_model_log_pseq_packed      (by hand)
    c_symbols.txt          +2 -0  the above, and beam_encode_n,
                                  the length-bounded encoder      (by hand)
    python_api.txt         +1 -0  BeamModel.log_pseq_arrow
    pytest_cases.txt       +6 -0  io/test_arrow_scoring.py: bit-identity on
                                  string and large_string columns, a slice
                                  starting mid-buffer, a chunked column,
                                  nulls as NaN, log_pseq_parquet
    c_cases.txt            +1 -0  test_packed_matches_scalar_and_refuses_bad_offsets
    numeric.json           unchanged (re-written by the recorder, byte-identical)
    threads.json           unchanged (likewise)

The gate passed on everything else before recording and on everything after,
`exact.py` at 411 quantities, worst 5.8% of its tolerance. The full suites:
Python 923 passed, 2 skipped; C 338 passed.
