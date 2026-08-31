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
