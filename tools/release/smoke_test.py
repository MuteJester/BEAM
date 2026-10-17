"""Check that an installed BEAM works, not merely that it imports.

Run against every wheel the release builds, on every platform, before
anything is published. Uses only BEAM and NumPy, and no files or network.

The central check is the defining property of the model: its probabilities
are normalised exactly, so summed over EVERY sequence it can represent they
come to 1. A miscompiled dynamic program, a wrong SIMD path or a broken
OpenMP reduction all show up here as a sum that is not 1.
"""
import itertools
import math
import os
import sys
import tempfile

import numpy as np

import BEAM
from BEAM import BeamModel

TRAINING = [
    "CASS", "CASL", "CSSL", "CAWS", "CASF", "CSRL", "CATS", "CASQ",
    "CAS", "CSL", "CAW", "CSA", "CASY", "CSVL", "CAIS", "CAST",
]
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
MAX_LENGTH = 4


def main():
    print(f"BEAM {BEAM.__version__} on Python {sys.version.split()[0]}, "
          f"NumPy {np.__version__}, {sys.platform}")

    # A wheel built without OpenMP still gives the right answers, slowly; only
    # this catches it. OMP_NUM_THREADS, when set, is a deliberate limit.
    cores, threads = os.cpu_count() or 1, BEAM.effective_threads()
    print(f"{threads} OpenMP threads on {cores} cores")
    if "OMP_NUM_THREADS" not in os.environ:
        assert cores == 1 or threads > 1, "built without OpenMP: one thread on several cores"

    model = BeamModel(TRAINING, order=2, max_length=MAX_LENGTH)

    every = ["".join(t) for n in range(1, MAX_LENGTH + 1)
             for t in itertools.product(ALPHABET, repeat=n)]
    logp = np.asarray(model.log_pseq_batch(every), dtype=float)
    finite = logp[np.isfinite(logp)]
    total = math.fsum(np.exp(finite))
    print(f"sum of P over all {len(every):,} sequences up to length "
          f"{MAX_LENGTH}: {total:.15f}")
    assert abs(total - 1.0) < 1e-9, f"probabilities sum to {total}, not 1"

    draws = model.sample(200, seed=1)
    assert len(draws) == 200 and all(1 <= len(s) <= MAX_LENGTH for s in draws)
    assert all(np.isfinite(model.log_pseq(s)) for s in draws)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "smoke")
        model.save(path)
        files = os.listdir(d)
        loaded = BeamModel.from_file(os.path.join(d, files[0]))
    again = np.asarray(loaded.log_pseq_batch(every), dtype=float)
    assert np.array_equal(np.isfinite(logp), np.isfinite(again))
    assert np.array_equal(logp[np.isfinite(logp)], again[np.isfinite(again)]), \
        "a saved and reloaded model scores differently"

    print("smoke test passed")


if __name__ == "__main__":
    main()
