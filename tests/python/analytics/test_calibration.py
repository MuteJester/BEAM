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
    as an outlier against its own length's null.

    The comparative claim is the one that matters and it is exact: the
    corrupted sequence sits further into the tail than the drawn one. The
    absolute level is stated as what it measures rather than as a round
    number, because it moved and the reason is worth recording.

    Under the previous open world a single wrong terminal residue reached
    p < 0.05. The open world is now a Laplace add-one at one table rather
    than a backoff hierarchy running orders 3, 2 and 1 with strengths of 20
    and a length-conditioned 200, and add-one hands an unseen residue MORE
    mass than that hierarchy did. So one wrong terminus is a weaker outlier:
    measured p = 0.073, converged against the grid (0.07306 at 1024 bins,
    0.07312 at 65536), so this is the model's answer and not a resolution
    artefact.

    The tail itself is unaffected, which is what the third case checks. A
    wholly implausible sequence still reaches p = 0.0002, so the threshold
    below is a statement about how much ONE residue is worth, not evidence
    that the calibration has gone flat.
    """
    cal = open_small.calibrate(bins=1024)
    good = next(s for s in make_corpus(n=20, seed=31, lo=8, hi=8)[0])
    bad = good[:-1] + "W"
    pg = cal.pvalue(len(good), -open_small.log_pseq(good))
    pb = cal.pvalue(len(bad), -open_small.log_pseq(bad))
    assert pb < pg
    assert pb < 0.08
    hopeless = "W" * len(good)
    assert cal.pvalue(len(hopeless), -open_small.log_pseq(hopeless)) < 1e-3



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


def test_scale_score_is_a_false_positive_rate(small, calibration):
    """The point of the p-value form: a threshold IS a rate.

    Sequences drawn FROM the model are its null by definition, so their
    p-values are uniform and ``-log10 p`` exceeds ``t`` for exactly ``10**-t``
    of them. That is the property the robust form does NOT have, and the whole
    reason the definition changed."""
    cal = calibration
    drawn = small.sample(20000, seed=17)
    z = small.scale_score(drawn, cal)
    assert np.median(z) == pytest.approx(np.log10(2.0), abs=0.05)
    for t in (1.0, 1.5):
        assert np.mean(z >= t) == pytest.approx(10.0 ** -t, rel=0.35)


def test_scale_score_rate_is_flat_across_lengths(small, calibration):
    """The defect being fixed. On the 927M foundation the robust form's 1%
    pooled cutoff flags 5.5% of length-4 sequences and 0.02% of length-25 ones;
    the p-value form is 10% at every length by construction."""
    cal = calibration
    rates_p, rates_robust = [], []
    for L in small.observed_lengths:
        drawn = small.sample(4000, length=L, seed=100 + L)
        if len(drawn) < 500:
            continue
        u = -small.log_pseq_batch(drawn)
        rates_p.append(np.mean([cal.scale(L, float(v)) >= 1.0 for v in u]))
        rates_robust.append(np.mean(
            [cal.scale(L, float(v), method="robust") >= 1.35 for v in u]))
    assert len(rates_p) >= 3
    assert max(rates_p) == pytest.approx(0.10, abs=0.035)
    assert min(rates_p) == pytest.approx(0.10, abs=0.035)
    # The robust form's spread across lengths is the thing being fixed, so it
    # must be materially wider than the p-value form's.
    assert (max(rates_robust) - min(rates_robust)) > \
           (max(rates_p) - min(rates_p))


def test_robust_method_keeps_its_historical_definition(small, calibration):
    """Published thresholds are quoted on the robust form, so it stays exact."""
    cal = calibration
    L = max(small.observed_lengths)
    u = float(-small.log_pseq(small.sample(1, length=L, seed=3)[0]))
    expected = (u - cal.median(L)) / cal.iqr(L)
    assert cal.scale(L, u, method="robust") == pytest.approx(expected)


def test_scale_methods_rank_identically_within_a_length(small, calibration):
    """Within one length the two forms are monotone transforms of each other,
    so they cannot disagree on ordering. All of the difference is BETWEEN
    lengths."""
    # Its own grid, finer than the shared fixture's. The claim is exact --
    # the two forms are monotone transforms within a length -- so every
    # departure from 1 is a TIE in the p-value grid, and the grid has to be
    # fine enough to resolve the sample. Measured convergence on this model:
    # 0.9947 at 256 bins, 0.99927 at 1024, 0.99970 at 4096, 0.99973 at 16384.
    # At 256 the instrument is coarser than the effect it is measuring.
    cal = small.calibrate(bins=4096)
    L = max(small.observed_lengths)
    drawn = [s for s in small.sample(3000, length=L, seed=11)]
    u = -small.log_pseq_batch(drawn)
    a = np.array([cal.scale(L, float(v)) for v in u])
    b = np.array([cal.scale(L, float(v), method="robust") for v in u])
    ok = np.isfinite(a) & np.isfinite(b)
    assert ok.sum() > 100
    from scipy.stats import spearmanr  # noqa: PLC0415
    assert spearmanr(a[ok], b[ok]).statistic > 0.999


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
    """The facade must be a pass-through to Calibration.scale, both methods."""
    cal = calibration
    s = small.sample(1, seed=3)[0]
    u = -small.log_pseq(s)
    assert small.scale_score(s, cal) == pytest.approx(
        cal.scale(len(s), u), rel=1e-12)
    robust = (u - cal.median(len(s))) / cal.iqr(len(s))
    assert small.scale_score(s, cal, method="robust") == pytest.approx(
        robust, rel=1e-12)


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
