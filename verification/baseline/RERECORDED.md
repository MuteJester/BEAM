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
