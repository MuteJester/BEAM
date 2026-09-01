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
