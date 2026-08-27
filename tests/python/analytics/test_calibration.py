"""Length-conditioned calibration and SCALE.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel

from _helpers import (AA, J_MOTIFS, make_corpus, ANALYSIS_BINS, CALIBRATION_BINS)

# Length-conditioned calibration
# ═══════════════════════════════════════════════════════════════

def test_calibration_survival_is_a_valid_tail(small):
    cal = small.calibrate(bins=256)
    assert len(cal.lengths) > 0
    for L in cal.lengths:
        sf = cal.survival[L]
        assert np.all(np.diff(sf) <= 1e-12), "a survival function decreases"
        assert sf[0] == pytest.approx(1.0, rel=1e-9), "starts at 1"
        assert sf[-1] >= 0.0


def test_calibration_pvalues_are_uniform_under_its_own_null(small):
    """Draws from the model ARE the null, so their p-values must be roughly
    uniform. This is the check that the survival function is oriented
    correctly: if it were reversed the p-values would pile up at 1."""
    cal = small.calibrate(bins=1024)
    draws = small.sample(4000, seed=23)
    p = np.array([cal.pvalue(len(s), -small.log_pseq(s)) for s in draws])
    assert np.all((p >= 0) & (p <= 1 + 1e-12))
    assert 0.35 < p.mean() < 0.65, p.mean()
    assert 0.05 < np.mean(p < 0.1) < 0.2


def test_calibration_flags_an_implausible_sequence(open_small):
    """A sequence ending in a residue the corpus never terminates on must score
    as an outlier against its own length's null."""
    cal = open_small.calibrate(bins=1024)
    good = next(s for s in make_corpus(n=20, seed=31, lo=8, hi=8)[0])
    bad = good[:-1] + "W"
    pg = cal.pvalue(len(good), -open_small.log_pseq(good))
    pb = cal.pvalue(len(bad), -open_small.log_pseq(bad))
    assert pb < pg
    assert pb < 0.05



def test_calibration_unknown_length_is_conservative(small):
    cal = small.calibrate(bins=256)
    missing = max(cal.lengths) + 3
    assert cal.pvalue(missing, 50.0) == 1.0


def test_calibration_resolution_floor_is_reported(small):
    cal = small.calibrate(bins=256)
    assert 0.0 < cal.resolution_floor < 1.0


def test_calibration_takes_no_modelling_parameters(small):
    """The null is exact, so `bins` (a resolution knob) is the only argument.
    Anything else would be a modelling assumption smuggled into a measurement.
    """
    import inspect
    params = inspect.signature(BeamModel.calibrate).parameters
    assert set(params) == {"self", "bins"}


# ═══════════════════════════════════════════════════════════════
# SCALE, from the exact null
# ════════════════════════════════════════════════════════════════════════


#: Grid for the SCALE tests. Finer than ANALYSIS_BINS because a quantile is
#: resolved to a bin width while a total is not: at 384 bins the median of the
#: reconstructed null is still a few hundredths of an IQR out. Still ~2s.


@pytest.fixture(scope="module")
def calibration(small):
    return small.calibrate(bins=CALIBRATION_BINS)


def test_scale_score_is_centred_on_the_model_s_own_output(small, calibration):
    """Sequences drawn FROM the model are its null by definition, so their
    SCALE scores must have median ~0 and interquartile range ~1. That is what
    makes the score comparable across lengths."""
    cal = calibration
    drawn = small.sample(6000, seed=17)
    z = small.scale_score(drawn, cal)
    assert np.median(z) == pytest.approx(0.0, abs=0.06)
    q25, q75 = np.percentile(z, [25, 75])
    assert q75 - q25 == pytest.approx(1.0, abs=0.1)


def test_scale_score_ranks_an_implausible_sequence_above_a_drawn_one(
        small, calibration):
    cal = calibration
    L = max(small.observed_lengths)
    drawn = [s for s in small.sample(500, seed=5) if len(s) == L][:50]
    assert drawn, "the fixture must produce its longest length"
    worst = "W" * L                      # the residue the corpus never favours
    assert small.scale_score(worst, cal) > np.max(
        small.scale_score(drawn, cal))


def test_scale_score_is_length_normalised(small, calibration):
    """The whole point: raw surprisal grows with length, the score does not."""
    cal = calibration
    drawn = small.sample(6000, seed=23)
    lengths = np.array([len(s) for s in drawn])
    z = small.scale_score(drawn, cal)
    u = -small.log_pseq_batch(drawn)
    per_length = sorted(set(lengths.tolist()))
    raw = [float(np.median(u[lengths == L])) for L in per_length]
    scaled = [float(np.median(z[lengths == L])) for L in per_length]
    assert max(raw) - min(raw) > 1.0
    assert max(scaled) - min(scaled) < 0.2


def test_scale_score_matches_the_calibration_arithmetic(small, calibration):
    cal = calibration
    s = small.sample(1, seed=3)[0]
    want = (-small.log_pseq(s) - cal.median(len(s))) / cal.iqr(len(s))
    assert small.scale_score(s, cal) == pytest.approx(want, rel=1e-12)


def test_scale_score_rejects_a_foreign_calibration(small):
    with pytest.raises(TypeError, match="Calibration"):
        small.scale_score("CASSF", {"median": 1.0})


def test_calibration_quantiles_are_monotone(small, calibration):
    cal = calibration
    L = cal.lengths[len(cal.lengths) // 2]
    qs = cal.quantile(L, [0.05, 0.25, 0.5, 0.75, 0.95])
    assert all(b >= a for a, b in zip(qs, qs[1:]))
    # The median must agree with the survival function it was read from.
    assert cal.pvalue(L, cal.median(L)) == pytest.approx(0.5, abs=0.03)


def test_calibration_pools_over_lengths_for_an_uncovered_one(small, calibration):
    """An uncovered length falls back to the pi(L)-weighted mixture, not to a
    flat one and not to a crash."""
    cal = calibration
    absent = max(cal.lengths) + 1
    assert absent not in cal.survival
    assert cal.median(absent) == pytest.approx(cal.median(None), rel=1e-12)
    assert np.isfinite(cal.scale(absent, 10.0))
    sf = cal.pooled_survival
    assert sf[0] == pytest.approx(1.0, abs=1e-9)
    assert all(b <= a + 1e-12 for a, b in zip(sf, sf[1:]))


# ════════════════════════════════════════════════════════════════════════
