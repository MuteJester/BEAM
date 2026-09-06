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
