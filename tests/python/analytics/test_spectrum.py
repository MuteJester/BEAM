"""Fused spectrum entry points.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel

from _helpers import (AA, J_MOTIFS, make_corpus, ANALYSIS_BINS)

# Fused C entry points
#
# Each of these replaces a loop of separate calls. The load-bearing test for
# every one is that it is BIT-IDENTICAL to the loop it replaces: a fused DP
# that merely agrees to a tolerance is a second implementation to keep in
# step, not a shortcut.
# ════════════════════════════════════════════════════════════════════════


def test_spectrum_pair_is_bit_identical_to_two_calls(small):
    spacing = small._grid_spacing(ANALYSIS_BINS)
    pair = small.spectrum_pair(bins=ANALYSIS_BINS, spacing=spacing)
    c = small.spectrum(measure="counting", bins=ANALYSIS_BINS, spacing=spacing)
    g = small.spectrum(measure="generated", bins=ANALYSIS_BINS, spacing=spacing)
    assert np.array_equal(pair["counting"], c["weights"])
    assert np.array_equal(pair["generated"], g["weights"])
    assert pair["spacing"] == c["spacing"] == g["spacing"]


def test_spectrum_pair_at_one_length_is_bit_identical(small):
    L = max(small.observed_lengths)
    spacing = small._grid_spacing(ANALYSIS_BINS)
    pair = small.spectrum_pair(length=L, bins=ANALYSIS_BINS, spacing=spacing)
    c = small.spectrum(measure="counting", length=L, bins=ANALYSIS_BINS,
                       spacing=spacing)
    assert np.array_equal(pair["counting"], c["weights"])
    assert pair["counting"].sum() == pytest.approx(small.path_count(L),
                                                   rel=1e-9)


def test_spectrum_pair_auto_spacing_matches_the_single_call(small):
    pair = small.spectrum_pair(bins=ANALYSIS_BINS)
    solo = small.spectrum(measure="counting", bins=ANALYSIS_BINS)
    assert pair["spacing"] == solo["spacing"]
    assert np.array_equal(pair["counting"], solo["weights"])


def test_spectra_by_length_rows_are_bit_identical(small):
    spacing = small._grid_spacing(ANALYSIS_BINS)
    got = small.spectra_by_length(measure="generated", bins=ANALYSIS_BINS,
                                  spacing=spacing)
    W = got["weights"]
    assert W.shape[1] == ANALYSIS_BINS
    for L in small.observed_lengths:
        solo = small.spectrum(measure="generated", length=L,
                              bins=ANALYSIS_BINS, spacing=spacing)
        assert np.array_equal(W[L], solo["weights"])
    # Lengths the model does not generate are left at zero, not garbage.
    for L in range(W.shape[0]):
        if L not in small.observed_lengths:
            assert not W[L].any()
    # And the rows sum to the pooled spectrum, which is what one shared grid
    # is for.
    pooled = small.spectrum(measure="generated", bins=ANALYSIS_BINS,
                            spacing=spacing)["weights"]
    assert np.allclose(W.sum(axis=0), pooled, rtol=0, atol=1e-15)


def test_a_reduced_budget_changes_the_schedule_and_not_the_answer(small):
    """The budget divides across concurrent lengths, so squeezing it (while
    still leaving room for one sweep) must change how many run at once and
    nothing else."""
    from BEAM import memory_budget, set_memory_budget

    spacing = small._grid_spacing(ANALYSIS_BINS)
    before = memory_budget()
    try:
        roomy = small.spectra_by_length(bins=ANALYSIS_BINS, spacing=spacing)
        set_memory_budget(64 << 20)          # far below the default, above one sweep
        cramped = small.spectra_by_length(bins=ANALYSIS_BINS, spacing=spacing)
    finally:
        set_memory_budget(before)
    assert np.array_equal(roomy["weights"], cramped["weights"])
    assert memory_budget() == before


def test_a_budget_below_one_sweep_is_refused(small):
    """Below the size of a single sweep the budget REFUSES rather than running.

    It used to promise the opposite -- "a cap on concurrency, never on what a
    single length may allocate" -- while the positional and query paths refused
    above it from the same setter. The refusing half was kept: one sweep is
    6.6 GB on a nucleotide model at the ceiling order, and a budget that cannot
    say no to that protects nothing.
    """
    from BEAM import memory_budget, set_memory_budget

    spacing = small._grid_spacing(ANALYSIS_BINS)
    before = memory_budget()
    try:
        set_memory_budget(1)
        with pytest.raises(Exception) as exc:
            small.spectra_by_length(bins=ANALYSIS_BINS, spacing=spacing)
    finally:
        set_memory_budget(before)
    text = str(exc.value)
    assert "memory budget" in text          # names the cause
    assert "set_memory_budget" in text      # and the way out


def test_memory_budget_zero_restores_the_default():
    from BEAM import memory_budget, set_memory_budget
    before = memory_budget()
    try:
        set_memory_budget(1 << 20)
        assert memory_budget() == 1 << 20
        set_memory_budget(0)
        assert memory_budget() == 4 * (1 << 30)
    finally:
        set_memory_budget(before)


def test_histograms_by_length_match_the_single_length_route(small):
    """The analysis object now fuses; the fused rows must equal what the
    per-length calls it replaced would have produced."""
    a = small.pseq_analysis()
    fused = a.histograms_by_length(ANALYSIS_BINS, measure="generated")
    for L, h in fused.items():
        solo = small.spectrum(measure="generated", length=L,
                              bins=ANALYSIS_BINS,
                              spacing=small._grid_spacing(ANALYSIS_BINS))
        assert np.array_equal(h.weights, solo["weights"])


def test_histogram_pair_populates_the_single_measure_cache(small):
    """One fused transport has to satisfy a later single-measure request, or
    the fusion costs a sweep instead of saving one."""
    a = small.pseq_analysis()
    pair = a.histogram_pair(ANALYSIS_BINS)
    assert a.histogram(ANALYSIS_BINS, measure="counting") is pair["counting"]
    assert a.histogram(ANALYSIS_BINS, measure="generated") is pair["generated"]


# ════════════════════════════════════════════════════════════════════════
