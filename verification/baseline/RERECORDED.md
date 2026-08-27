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
