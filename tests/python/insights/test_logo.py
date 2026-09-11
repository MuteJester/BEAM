"""Sequence logos, from the Python side.

`tests/native/test_insights_logo.c` pins the arithmetic against enumeration.
What is left for this layer is everything between the C answer and the
caller's hands, and the risk there is structural rather than numerical: the
arrays arrive as flat byte blocks and are reshaped to
``(length, n_residues)``, which any wrong permutation of the same two sizes
would survive as a shape check. So the checks here compare VALUES against
`positional_marginals`, which moves under a transpose.

The other half is the keyword surface. `pseq_above` and `pseq_below` are the
only place in the library where a probability meets an axis measured in
surprisal, and `-log(p)` reverses the order -- so which band each produces is
worth pinning explicitly rather than trusting to read correctly.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from BEAM import BeamModel
from BEAM.constants import BEAM_ALPHABET as AA


L_ENUM = 15
EXACT = 1e-12

#: A surprisal inside this fixture, whose length-15 sequences run from about
#: 2.3 to 4.1 nats. Bands are checked against the grid, so an edge past it
#: is refused rather than quietly clamped.
CUT = 3.0


def _enumerable(order=3, alphabet="AG", n=10, seed=7):
    rng = np.random.default_rng(seed)
    seqs = ["CASS" + "".join(rng.choice(list(alphabet), 7)) + "EQYF"
            for _ in range(n)]
    return BeamModel(list(dict.fromkeys(seqs)), order=order, max_length=18)


@pytest.fixture(scope="module")
def model():
    return _enumerable()


@pytest.fixture(scope="module")
def nt_model():
    seqs = ["ACGTA", "ACGTC", "ACGGA", "AGGTA", "ACTTA", "CGTAC", "CGGAC",
            "GGTAC", "CTTAG", "CGTAG", "TACGT", "TAGGC", "GCATG", "GCTAG",
            "TTACG", "ATCGA", "AGCTT", "CATGC", "TGCAT", "GATCA"]
    return BeamModel(seqs, order=2, max_length=8, alphabet="nt")


# The reshape, and the values that would move under one
# ═══════════════════════════════════════════════════════════════

def test_the_unconditional_logo_is_the_exact_marginal(model):
    """Pins the reshape, the ordering and the normalisation at once.

    A block reshaped to `(n_residues, length)` has the same size and the same
    sum; only comparing position by position against a known answer tells
    them apart.
    """
    g = model.insights.logo(L_ENUM)
    assert g.probability.shape == (L_ENUM, len(AA))
    assert np.abs(g.probability - model.positional_marginals(L_ENUM)).max() < EXACT
    assert g.resolution_error is None, "no grid was involved, so there is no error"


def test_the_stack_sums_to_the_information(model):
    g = model.insights.logo(L_ENUM)
    assert g.height_bits.sum(axis=1) == pytest.approx(g.information_bits)
    assert g.height_nats.sum(axis=1) == pytest.approx(g.information_nats)
    assert g.probability.sum(axis=1) == pytest.approx(np.ones(L_ENUM))


def test_information_is_the_ceiling_less_the_entropy(model):
    g = model.insights.logo(L_ENUM)
    s = model.insights.logos(L_ENUM)
    assert g.information_bits == pytest.approx(s.max_entropy_bits - g.entropy_bits)
    assert s.max_entropy_bits == pytest.approx(math.log2(20))


def test_consensus_reads_out_as_a_sequence(model):
    g = model.insights.logo(L_ENUM)
    seq = g.consensus_sequence
    assert len(seq) == L_ENUM
    assert seq.startswith("CASS"), "the fixture's conserved head"
    by_hand = "".join(AA[int(a)] for a in g.probability.argmax(axis=1))
    assert seq == by_hand


def test_as_dict_is_keyed_by_residue_and_matches_the_heights(model):
    g = model.insights.logo(L_ENUM)
    d = g.as_dict()
    assert set(d) == set(AA)
    assert d["C"] == pytest.approx(g.height_bits[:, AA.index("C")])
    assert d["C"].shape == (L_ENUM,)
    with pytest.raises(ValueError, match="bits.*nats"):
        g.as_dict(unit="decibans")


# Which band a probability threshold selects
# ═══════════════════════════════════════════════════════════════

def test_pseq_above_and_below_are_complementary(model):
    """The one place a probability meets a surprisal axis.

    `-log(p)` reverses the order, so `Pseq > t` is the LOW-surprisal band and
    `Pseq < t` the high one. Getting this backwards would return a plausible
    logo of the wrong sequences, so it is pinned by construction rather than
    by reading.
    """
    # Inside this fixture's surprisal, which runs from about 2.3 to 4.1 nats.
    # A threshold past the grid is refused, and says so.
    t = math.exp(-CUT)
    hi_p = model.insights.logo(L_ENUM, pseq_above=t, bins=2048)
    lo_p = model.insights.logo(L_ENUM, pseq_below=t, bins=2048)

    assert hi_p.lo_nats == pytest.approx(0.0)
    assert hi_p.hi_nats == pytest.approx(CUT)
    assert lo_p.lo_nats == pytest.approx(CUT)
    assert lo_p.hi_nats > CUT
    assert hi_p.mass > 0 and lo_p.mass > 0
    assert hi_p.fraction + lo_p.fraction == pytest.approx(1.0)


def test_the_bands_recombine_into_the_pooled_logo(model):
    """Mass-weighting the bands undoes the conditioning.

    The tolerance is the bands' own `resolution_error`, not a constant, and
    that is the honest bound rather than a convenient one. Each band's mass is
    reported at position 0 while its marginals are normalised per position, so
    the reconstruction cannot be better than those agree -- which is exactly
    what `resolution_error` measures. Measured here the reconstruction is a
    third of it or less.

    This fixture's support is twenty-five sequences, so its spectrum is
    twenty-five spikes and an edge near one of them makes that spike's
    membership genuinely grid-dependent. That is why the error is percent-
    scale and does not shrink much with `bins`: it is the discreteness of the
    support, not the fineness of the grid.
    """
    s = model.insights.logos(L_ENUM, edges=[-math.inf, CUT, math.inf],
                             bins=2048)
    assert len(s) == 2
    mix = sum(g.mass * g.probability for g in s) / sum(g.mass for g in s)
    worst = float(np.abs(mix - model.positional_marginals(L_ENUM)).max())
    bound = max(g.resolution_error for g in s)
    assert worst <= bound, (
        f"recombination is off by {worst:.3g}, worse than the bands' own "
        f"self-consistency of {bound:.3g} -- so it is not the grid")


def test_a_band_differs_from_the_pooled_logo(model):
    """Otherwise everything above would pass with banding doing nothing."""
    s = model.insights.logos(L_ENUM, edges=[-math.inf, CUT, math.inf],
                             bins=2048)
    pooled = model.positional_marginals(L_ENUM)
    assert max(np.abs(g.probability - pooled).max() for g in s) > 1e-3


# The set, and what it says about its own grid
# ═══════════════════════════════════════════════════════════════

def test_a_set_reports_its_grid_and_its_shortfall(model):
    s = model.insights.logos(L_ENUM, edges=[-math.inf, CUT, math.inf],
                             bins=2048)
    assert s.spacing is not None and s.spacing > 0
    assert not s.exact
    assert s.grid_shortfall < 1e-9, "this grid should hold the whole length"
    assert len(s) == 2 and s[0].lo_nats < s[1].lo_nats


def test_the_unconditional_set_has_no_grid_at_all(model):
    s = model.insights.logos(L_ENUM)
    assert s.exact
    assert s.spacing is None
    assert s[0].resolution_error is None
    assert s[0].position_mass is None
    assert s.on_grid_total == pytest.approx(s.length_total)


def test_a_single_band_over_everything_is_still_exact(model):
    s = model.insights.logos(L_ENUM, edges=[-math.inf, math.inf], bins=1024)
    assert s.exact, "no interior edge means nothing can be misclassified"
    assert np.abs(s[0].probability
                  - model.positional_marginals(L_ENUM)).max() < EXACT


# Nucleotides, where the reference alphabet matters
# ═══════════════════════════════════════════════════════════════

def test_the_nucleotide_ceiling_is_log2_of_five_unless_asked_otherwise(nt_model):
    """BEAM has five nucleotide residues because N is emitted, not a wildcard.

    Every published DNA logo tops out at 2 bits. Ours tops at log2(5) = 2.32
    unless the caller asks for the conventional axis, and a 16% height
    difference is not visible by inspection.
    """
    assert nt_model.n_residues == 5
    default = nt_model.insights.logos(5)
    conventional = nt_model.insights.logos(5, reference_residues=4)

    assert default.max_entropy_bits == pytest.approx(math.log2(5))
    assert conventional.max_entropy_bits == pytest.approx(2.0)
    assert default[0].entropy_bits == pytest.approx(conventional[0].entropy_bits), \
        "the reference moves the ceiling, never the distribution"
    assert default.residues == "ACGTN"


# What it refuses
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("kw,fragment", [
    ({"pseq_above": 2.0},                    "between 0 and 1"),
    ({"pseq_above": 0.0},                    "between 0 and 1"),
    ({"edges": [5.0, 3.0]},                  "ascend strictly"),
    ({"edges": [0.0, math.nan]},             "NaN"),
    ({"edges": [-8.0, -3.0]},                "never negative"),
    ({"edges": [3.0]},                       "define no band"),
    ({"reference_residues": 99},             "reference_residues"),
])
def test_a_malformed_request_is_refused_by_name(model, kw, fragment):
    with pytest.raises(ValueError, match=fragment):
        model.insights.logo(L_ENUM, **kw)


def test_two_conditions_at_once_are_refused(model):
    with pytest.raises(ValueError, match="at most one of"):
        model.insights.logo(L_ENUM, pseq_above=0.5, pseq_below=0.5)


def test_a_length_the_model_cannot_produce_is_refused(model):
    with pytest.raises(ValueError, match="not one this model generates"):
        model.insights.logo(2)


def test_an_empty_band_is_refused_rather_than_reported_as_conserved(model):
    """The refusal the whole layer exists for.

    Zero probabilities give zero entropy and therefore MAXIMUM information, so
    an answered empty band is a blank plot beside a claim that every position
    is perfectly conserved.
    """
    with pytest.raises((ValueError, RuntimeError)) as ex:
        model.insights.logo(L_ENUM, edges=[1e-9, 1e-6], bins=2048)
    assert "no sequence" in str(ex.value) or "no mass" in str(ex.value)


def test_asking_for_one_logo_from_several_bands_is_refused(model):
    with pytest.raises(ValueError, match="defines 3 bands"):
        model.insights.logo(L_ENUM, edges=[-math.inf, 3.0, 3.6, math.inf],
                            bins=2048)


# Cost
# ═══════════════════════════════════════════════════════════════

def test_the_cost_is_reported_before_the_sweep_and_scales_with_bins(model):
    a = model.insights.cost(L_ENUM, edges=[-math.inf, math.inf], bins=1024)
    b = model.insights.cost(L_ENUM, edges=[-math.inf, math.inf], bins=2048)
    c = model.insights.cost(L_ENUM, edges=[-math.inf, math.inf], bins=4096)
    assert a["bytes"] > 0
    assert (c["bytes"] - b["bytes"]) == pytest.approx(
        2 * (b["bytes"] - a["bytes"]), rel=1e-9)
