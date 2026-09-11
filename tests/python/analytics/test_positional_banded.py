"""Per-position marginals conditioned on a band of the sequence's surprisal.

The C tests in `tests/native/test_positional_banded.c` pin the arithmetic.
What is left for this layer is everything between the C answer and the
caller's hands, and the risk there is not numerical: the marginals arrive as
one flat block of doubles and are reshaped to `(n_bands, length, n_residues)`,
which any wrong permutation of the same three sizes would survive as a shape
check. So the checks here compare VALUES against exact references -- a
transposed reshape moves numbers, and that is what catches it.

The model is enumerable, as in `test_positional_marginals.py`: strict, small
enough that `top_k_sequences(k=path_count)` is the entire support, so the
reference is a direct sum over sequences rather than a second dynamic program
that could share a mistake with the first.

A strict model is also the only kind whose automatic grid resolves anything.
With backoff on, `accumulator_max` is set by the smoothing floor hundreds of
nats from any real sequence, and every useful band is narrower than one bin.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from BEAM import BeamModel
from BEAM.constants import BEAM_ALPHABET as AA


L_ENUM = 15
BINS = 2048

#: Every quantity here is O(1) and the residue axis is exact, so machine
#: precision is the honest bar. Measured worst case across this file is 3e-15.
EXACT = 1e-12


def _enumerable(order=3, alphabet="AG", n=10, seed=7):
    rng = np.random.default_rng(seed)
    seqs = ["CASS" + "".join(rng.choice(list(alphabet), 7)) + "EQYF"
            for _ in range(n)]
    return BeamModel(list(dict.fromkeys(seqs)), order=order, max_length=18)


@pytest.fixture(scope="module")
def model():
    return _enumerable()


@pytest.fixture(scope="module")
def support(model):
    """The complete length-15 support as (sequence, log probability)."""
    n = int(model.path_count(L_ENUM))
    sup = model.top_k_sequences(k=n, length=L_ENUM)
    assert len(sup) == n, "top_k did not return the whole support"
    return sup


@pytest.fixture(scope="module")
def grid(support):
    """An explicit spacing covering the support, with splitting headroom.

    Explicit rather than automatic so the test owns the grid: the reference
    below places a threshold a fixed number of BINS away from every sequence,
    which is only meaningful if the test knows how wide a bin is.
    """
    smax = max(-v for _, v in support)
    return smax * 1.05 / (BINS - 32)


def _direct(support, keep, length):
    """Marginals and mass over the subset `keep` of the support."""
    rows = [(s, math.exp(v)) for s, v in support if keep(-v)]
    z = sum(w for _, w in rows)
    out = np.zeros((length, len(AA)))
    for s, w in rows:
        for i, ch in enumerate(s):
            out[i, AA.index(ch)] += w
    return (out / z if z > 0 else out), z


# The band as a whole
# ═══════════════════════════════════════════════════════════════

def test_one_band_over_everything_is_the_pooled_marginal(model, grid):
    """The reshape, the ordering and the normalisation, in one comparison.

    A `(n_bands, length, n_residues)` block transposed to
    `(n_bands, n_residues, length)` has the same size and the same sum; only
    comparing it against a known answer position by position tells them apart.
    """
    b = model.positional_banded(L_ENUM, [0.0, 1e12], bins=BINS, spacing=grid)
    assert b.marginals.shape == (1, L_ENUM, len(AA))
    assert np.abs(b.marginals[0] - model.positional_marginals(L_ENUM)).max() < EXACT


def test_the_counting_measure_counts_the_support(model, grid):
    b = model.positional_banded(L_ENUM, [0.0, 1e12], bins=BINS, spacing=grid,
                                counting=True)
    assert b.band_mass[0] == pytest.approx(model.path_count(L_ENUM), rel=1e-9)


def test_bands_partition_the_mass(model, support, grid):
    b = model.positional_banded(L_ENUM, [0.0, 1e12], bins=BINS, spacing=grid)
    total = sum(math.exp(v) for _, v in support)
    assert b.band_mass.sum() == pytest.approx(total, rel=1e-9)
    assert b.fraction.sum() == pytest.approx(1.0)


# Against the enumerated support
# ═══════════════════════════════════════════════════════════════

def _cut_in_a_gap(support, ds, length):
    """A threshold no sequence's binned mass can straddle.

    Linear splitting spreads one sequence over up to `length + 1` bins, so a
    threshold nearer than that to a real surprisal classifies part of it the
    wrong side. Placed in a gap twice that wide, every sequence is wholly one
    side and the comparison below is exact rather than approximate.
    """
    s = sorted(-v for _, v in support)
    margin = 2 * (length + 2) * ds
    best, bal = None, None
    for k in range(1, len(s)):
        if s[k] - s[k - 1] <= 2 * margin or s[k - 1] <= margin:
            continue
        d = abs(k - len(s) // 2)
        if bal is None or d < bal:
            best, bal = 0.5 * (s[k - 1] + s[k]), d
    return best


def test_the_bands_match_a_direct_sum_over_the_support(model, support, grid):
    cut = _cut_in_a_gap(support, grid, L_ENUM)
    assert cut is not None, (
        "no gap wide enough to split exactly; this fixture's surprisals are "
        f"{sorted(round(-v, 4) for _, v in support)}")

    b = model.positional_banded(L_ENUM, [0.0, cut, 1e12], bins=BINS,
                                spacing=grid)
    lo_ref, lo_z = _direct(support, lambda s: s < cut, L_ENUM)
    hi_ref, hi_z = _direct(support, lambda s: s >= cut, L_ENUM)
    assert lo_z > 0 and hi_z > 0, "the cut put every sequence on one side"

    assert b.band_mass[0] == pytest.approx(lo_z, rel=1e-9)
    assert b.band_mass[1] == pytest.approx(hi_z, rel=1e-9)
    assert np.abs(b.marginals[0] - lo_ref).max() < EXACT
    assert np.abs(b.marginals[1] - hi_ref).max() < EXACT


def test_a_band_differs_from_the_pooled_marginal(model, support, grid):
    """Otherwise every check above would pass with the banding doing nothing."""
    cut = _cut_in_a_gap(support, grid, L_ENUM)
    b = model.positional_banded(L_ENUM, [0.0, cut, 1e12], bins=BINS,
                                spacing=grid)
    pooled = model.positional_marginals(L_ENUM)
    assert np.abs(b.marginals[0] - pooled).max() > 1e-3


def test_the_bands_recombine_into_the_pooled_marginal(model, support, grid):
    cut = _cut_in_a_gap(support, grid, L_ENUM)
    b = model.positional_banded(L_ENUM, [0.0, cut, 1e12], bins=BINS,
                                spacing=grid)
    mix = ((b.band_mass[:, None, None] * b.marginals).sum(axis=0)
           / b.band_mass.sum())
    assert np.abs(mix - model.positional_marginals(L_ENUM)).max() < EXACT


# What the container carries
# ═══════════════════════════════════════════════════════════════

def test_rows_sum_to_one_and_mass_is_reported_per_position(model, grid):
    b = model.positional_banded(L_ENUM, [0.0, 1e12], bins=BINS, spacing=grid)
    assert b.marginals.sum(axis=2) == pytest.approx(np.ones((1, L_ENUM)))
    assert b.position_mass.shape == (1, L_ENUM)


def test_resolution_error_reports_a_resolved_grid_as_resolved(model, support,
                                                              grid):
    """The diagnostic is the only thing that tells a caller the grid was fine
    enough, so it has to be near zero when it was."""
    cut = _cut_in_a_gap(support, grid, L_ENUM)
    b = model.positional_banded(L_ENUM, [0.0, cut, 1e12], bins=BINS,
                                spacing=grid)
    assert b.resolution_error < 1e-9
    assert "resolution_error" in repr(b)
    assert len(b) == 2


def test_an_empty_band_is_reported_rather_than_refused(model, support, grid):
    """One band with no mass must not deny the caller the others."""
    smax = max(-v for _, v in support)
    b = model.positional_banded(L_ENUM, [0.0, smax + 1.0, smax + 2.0],
                                bins=BINS, spacing=grid)
    assert b.band_mass[0] > 0.0
    assert b.band_mass[1] == 0.0
    assert np.all(b.marginals[1] == 0.0)
    assert np.all(np.isfinite(b.fraction))


# What it refuses, and whether the message helps
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("edges,fragment", [
    ([5.0, 3.0],        "ascend strictly"),
    ([1.0, 1.0],        "ascend strictly"),
    ([3.0],             "define no band"),
    ([0.0, math.inf],   "must be finite"),
    ([0.0, math.nan],   "must be finite"),
])
def test_a_malformed_band_is_refused_by_name(model, edges, fragment):
    with pytest.raises(ValueError, match=fragment):
        model.positional_banded(L_ENUM, edges, bins=BINS)


def test_a_length_the_model_cannot_produce_is_refused(model):
    with pytest.raises(ValueError, match="generates no sequences of length"):
        model.positional_banded(2, [0.0, 1e12], bins=BINS)


def test_descending_edges_say_what_a_probability_user_did_wrong(model):
    """Passing probabilities descending is THE mistake this API invites."""
    with pytest.raises(ValueError, match=r"-log\(p\) reverses the order"):
        model.positional_banded(L_ENUM, [30.0, 20.0, 10.0], bins=BINS)
