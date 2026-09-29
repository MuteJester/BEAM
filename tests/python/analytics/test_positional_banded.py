"""Per-position marginals conditioned on a band of the sequence's surprisal.

The C tests in `tests/native/test_positional_banded.c` pin the arithmetic.
What is left for this layer is everything between the C answer and the
caller's hands, and the risk there is not numerical: the marginals arrive as
one flat block of doubles and are reshaped to `(n_bands, length, n_residues)`,
which any wrong permutation of the same three sizes would survive as a shape
check. So the checks here compare VALUES against exact references -- a
transposed reshape moves numbers, and that is what catches it.

The model is enumerable, as in `test_positional_marginals.py`: under the
prior the support is every string over the whole alphabet, so the fixture is
nucleotide at length 6 (15,625 strings) and the reference is a direct sum
over all of them rather than a second dynamic program that could share a
mistake with the first.

The grid is explicit rather than automatic. The automatic one is sized by
`accumulator_max`, which the prior sets by the most improbable string of the
whole space -- far above anything the corpus resembles -- and a test that
places thresholds a fixed number of bins from every sequence has to know how
wide a bin is.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from BEAM import BeamModel


L_ENUM = 6
BINS = 2048
NT = "ACGTN"

#: Every quantity here is O(1) and the residue axis is exact, so machine
#: precision is the honest bar. Measured worst case across this file is 3e-15.
EXACT = 1e-12


def _enumerable(order=3, n=40, seed=7):
    """The fixture of `test_positional_marginals.py`: a conserved head and
    tail around a short random middle, over nucleotides, lengths 5 and 6."""
    rng = np.random.default_rng(seed)
    seqs = ["TG" + "".join(rng.choice(list("AG"), int(k))) + "C"
            for k in rng.integers(2, 4, n)]
    return BeamModel(list(dict.fromkeys(seqs)), alphabet="nt", order=order,
                     max_length=L_ENUM)


@pytest.fixture(scope="module")
def model():
    return _enumerable()


@pytest.fixture(scope="module")
def support(model):
    """The complete length-6 support as (residue codes, log probability)."""
    seqs = ["".join(t) for t in itertools.product(NT, repeat=L_ENUM)]
    codes = np.array([[NT.index(c) for c in s] for s in seqs], dtype=np.int64)
    lp = np.asarray(model.log_pseq_batch(seqs), dtype=np.float64)
    assert np.isfinite(lp).all(), "the prior leaves no string impossible"
    return codes, lp


@pytest.fixture(scope="module")
def grid(support):
    """An explicit spacing covering the support, with splitting headroom.

    Explicit rather than automatic so the test owns the grid: the reference
    below places a threshold a fixed number of BINS away from every sequence,
    which is only meaningful if the test knows how wide a bin is.
    """
    smax = float((-support[1]).max())
    return smax * 1.05 / (BINS - 32)


def _direct(support, keep, length):
    """Marginals and mass over the sequences whose surprisal `keep` accepts."""
    codes, lp = support
    sel = keep(-lp)
    w = np.exp(lp[sel])
    z = float(w.sum())
    out = np.stack([np.bincount(codes[sel, i], weights=w, minlength=len(NT))
                    for i in range(length)])
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
    assert b.marginals.shape == (1, L_ENUM, len(NT))
    assert np.abs(b.marginals[0] - model.positional_marginals(L_ENUM)).max() < EXACT


def test_the_counting_measure_counts_the_support(model, grid):
    """Every string of the length, so 5**6 of them -- counted from the
    alphabet here, not read back from the model."""
    b = model.positional_banded(L_ENUM, [0.0, 1e12], bins=BINS, spacing=grid,
                                counting=True)
    assert b.band_mass[0] == pytest.approx(len(NT) ** L_ENUM, rel=1e-9)
    assert b.band_mass[0] == pytest.approx(model.path_count(L_ENUM), rel=1e-9)


def test_bands_partition_the_mass(model, support, grid):
    b = model.positional_banded(L_ENUM, [0.0, 1e12], bins=BINS, spacing=grid)
    total = float(np.exp(support[1]).sum())
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
    s = np.sort(-support[1])
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
        "no gap wide enough to split exactly; this fixture's surprisals run "
        f"from {(-support[1]).min():.4f} to {(-support[1]).max():.4f}")

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
    """Otherwise every check above would pass with the banding doing nothing.

    Checked on the rare band: the cut leaves over 99.9% of the MASS below it
    (the corpus-like strings) and a quarter of the STRINGS above it, so the
    common band is close to the pooled answer by construction and the rare
    one -- junk the prior made possible -- is nothing like it."""
    cut = _cut_in_a_gap(support, grid, L_ENUM)
    b = model.positional_banded(L_ENUM, [0.0, cut, 1e12], bins=BINS,
                                spacing=grid)
    pooled = model.positional_marginals(L_ENUM)
    assert np.abs(b.marginals[1] - pooled).max() > 0.1


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
    smax = float((-support[1]).max())
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


def test_a_length_outside_the_model_is_refused(model):
    """Every length from 1 to max_length is producible now -- 2 included,
    which the corpus never had -- so the refusals left are the two ends."""
    assert model.positional_banded(2, [0.0, 1e12], bins=BINS).band_mass[0] > 0
    with pytest.raises(ValueError, match="between 0 and max_length"):
        model.positional_banded(L_ENUM + 1, [0.0, 1e12], bins=BINS)
    with pytest.raises(ValueError, match="at least 1"):
        model.positional_banded(0, [0.0, 1e12], bins=BINS)


def test_descending_edges_say_what_a_probability_user_did_wrong(model):
    """Passing probabilities descending is THE mistake this API invites."""
    with pytest.raises(ValueError, match=r"-log\(p\) reverses the order"):
        model.positional_banded(L_ENUM, [30.0, 20.0, 10.0], bins=BINS)
