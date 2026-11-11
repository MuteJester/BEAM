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

import BEAM
from BEAM import BeamModel
from BEAM.constants import BEAM_ALPHABET as AA


LENGTH = 15
EXACT = 1e-12

#: A surprisal inside this fixture, whose length-15 sequences run from 4.8 to
#: 63.3 nats under the prior, with the median at 21.8: 20 nats splits the mass
#: about 40/60, so both bands are substantial. Bands are checked against the
#: grid, so an edge past it is refused rather than quietly clamped.
CUT = 20.0


def _cdr3_model(order=2, alphabet="AG", n=10, seed=7):
    """Ten CDR3-shaped sequences: a conserved CASS...EQYF frame around a
    two-letter middle.

    Order 2 rather than 3: the prior makes every string of the length
    possible, so a banded sweep runs over the whole state space, and nothing
    here depends on the width of the context -- the checks are on the reshape
    and the keywords. Order 3 costs twenty times more per banded logo.
    """
    rng = np.random.default_rng(seed)
    seqs = ["CASS" + "".join(rng.choice(list(alphabet), 7)) + "EQYF"
            for _ in range(n)]
    return BeamModel(list(dict.fromkeys(seqs)), order=order, max_length=18)


@pytest.fixture(scope="module")
def model():
    return _cdr3_model()


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
    g = model.insights.logo(LENGTH)
    assert g.probability.shape == (LENGTH, len(AA))
    assert np.abs(g.probability - model.positional_marginals(LENGTH)).max() < EXACT
    assert g.resolution_error is None, "no grid was involved, so there is no error"


def test_the_stack_sums_to_the_information(model):
    g = model.insights.logo(LENGTH)
    assert g.height_bits.sum(axis=1) == pytest.approx(g.information_bits)
    assert g.height_nats.sum(axis=1) == pytest.approx(g.information_nats)
    assert g.probability.sum(axis=1) == pytest.approx(np.ones(LENGTH))


def test_information_is_the_ceiling_less_the_entropy(model):
    g = model.insights.logo(LENGTH)
    s = model.insights.logos(LENGTH)
    assert g.information_bits == pytest.approx(s.max_entropy_bits - g.entropy_bits)
    assert s.max_entropy_bits == pytest.approx(math.log2(20))


def test_consensus_reads_out_as_a_sequence(model):
    g = model.insights.logo(LENGTH)
    seq = g.consensus_sequence
    assert len(seq) == LENGTH
    assert seq.startswith("CASS"), "the fixture's conserved head"
    by_hand = "".join(AA[int(a)] for a in g.probability.argmax(axis=1))
    assert seq == by_hand


def test_as_dict_is_keyed_by_residue_and_matches_the_heights(model):
    g = model.insights.logo(LENGTH)
    d = g.as_dict()
    assert set(d) == set(AA)
    assert d["C"] == pytest.approx(g.height_bits[:, AA.index("C")])
    assert d["C"].shape == (LENGTH,)
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
    # Inside this fixture's surprisal, which runs from 4.8 to 63.3 nats.
    # A threshold past the grid is refused, and says so.
    t = math.exp(-CUT)
    hi_p = model.insights.logo(LENGTH, pseq_above=t, bins=2048)
    lo_p = model.insights.logo(LENGTH, pseq_below=t, bins=2048)

    assert hi_p.lo_nats == pytest.approx(0.0)
    assert hi_p.hi_nats == pytest.approx(CUT)
    assert lo_p.lo_nats == pytest.approx(CUT)
    assert lo_p.hi_nats > CUT
    assert hi_p.mass > 0 and lo_p.mass > 0
    assert hi_p.fraction + lo_p.fraction == pytest.approx(1.0)


def test_surprisal_keywords_select_the_same_sequences_as_the_pseq_ones(model):
    """The reversal, pinned rather than trusted.

    Surprisal is the native axis, so `surprisal_*` needs no conversion and
    `pseq_*` does. The trap is that "above" means opposite things on the two:
    rarer is HIGHER surprisal and LOWER probability, so `surprisal_above` and
    `pseq_below` are the same request. Getting it backwards returns a
    perfectly plausible logo of the wrong sequences, which is why this
    compares the arrays rather than the band edges.
    """
    t = math.exp(-CUT)
    common_p = model.insights.logo(LENGTH, pseq_above=t, bins=2048)
    common_s = model.insights.logo(LENGTH, surprisal_below=CUT, bins=2048)
    rare_p = model.insights.logo(LENGTH, pseq_below=t, bins=2048)
    rare_s = model.insights.logo(LENGTH, surprisal_above=CUT, bins=2048)

    assert np.array_equal(common_p.probability, common_s.probability)
    assert np.array_equal(rare_p.probability, rare_s.probability)
    assert common_s.fraction + rare_s.fraction == pytest.approx(1.0)
    # And they are genuinely different bands, or the equalities above would
    # hold for a stopped clock.
    assert not np.allclose(common_s.probability, rare_s.probability)


def test_edges_in_nats_need_no_conversion(model):
    """`edges` is the native axis, so it must agree with the keyword form."""
    a = model.insights.logo(LENGTH, edges=[-math.inf, CUT], bins=2048)
    b = model.insights.logo(LENGTH, surprisal_below=CUT, bins=2048)
    assert np.array_equal(a.probability, b.probability)
    assert a.lo_nats == b.lo_nats and a.hi_nats == b.hi_nats


def test_the_bands_recombine_into_the_pooled_logo(model):
    """Mass-weighting the bands undoes the conditioning.

    The tolerance is the bands' own `resolution_error`, not a constant, and
    that is the honest bound rather than a convenient one. Each band's mass is
    reported at position 0 while its marginals are normalised per position, so
    the reconstruction cannot be better than those agree -- which is exactly
    what `resolution_error` measures. Measured here the bands' error is about
    1.5e-3 and the reconstruction a ninth of it.

    Under the prior the support is every string of the length, so the
    spectrum is dense rather than a few spikes, and an edge splits the
    linear-splitting spread of the sequences near it rather than the whole
    mass of one spike. That is why the error is per-mille rather than
    percent-scale.
    """
    s = model.insights.logos(LENGTH, edges=[-math.inf, CUT, math.inf],
                             bins=2048)
    assert len(s) == 2
    mix = sum(g.mass * g.probability for g in s) / sum(g.mass for g in s)
    worst = float(np.abs(mix - model.positional_marginals(LENGTH)).max())
    bound = max(g.resolution_error for g in s)
    assert worst <= bound, (
        f"recombination is off by {worst:.3g}, worse than the bands' own "
        f"self-consistency of {bound:.3g} -- so it is not the grid")


def test_a_band_differs_from_the_pooled_logo(model):
    """Otherwise everything above would pass with banding doing nothing."""
    s = model.insights.logos(LENGTH, edges=[-math.inf, CUT, math.inf],
                             bins=2048)
    pooled = model.positional_marginals(LENGTH)
    assert max(np.abs(g.probability - pooled).max() for g in s) > 1e-3


# The set, and what it says about its own grid
# ═══════════════════════════════════════════════════════════════

def test_a_set_reports_its_grid_and_its_shortfall(model):
    s = model.insights.logos(LENGTH, edges=[-math.inf, CUT, math.inf],
                             bins=2048)
    assert s.spacing is not None and s.spacing > 0
    assert not s.exact
    assert s.grid_shortfall < 1e-9, "this grid should hold the whole length"
    assert len(s) == 2 and s[0].lo_nats < s[1].lo_nats


def test_the_unconditional_set_has_no_grid_at_all(model):
    s = model.insights.logos(LENGTH)
    assert s.exact
    assert s.spacing is None
    assert s[0].resolution_error is None
    assert s[0].position_mass is None
    assert s.on_grid_total == pytest.approx(s.length_total)


def test_a_single_band_over_everything_is_still_exact(model):
    s = model.insights.logos(LENGTH, edges=[-math.inf, math.inf], bins=1024)
    assert s.exact, "no interior edge means nothing can be misclassified"
    assert np.abs(s[0].probability
                  - model.positional_marginals(LENGTH)).max() < EXACT


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
    ({"surprisal_above": -3.0},              "never negative"),
    ({"surprisal_below": math.inf},          "finite number of nats"),
    ({"pseq_above": 0.0},                    "between 0 and 1"),
    ({"edges": [5.0, 3.0]},                  "ascend strictly"),
    ({"edges": [0.0, math.nan]},             "NaN"),
    ({"edges": [-8.0, -3.0]},                "never negative"),
    ({"edges": [3.0]},                       "define no band"),
    ({"reference_residues": 99},             "reference_residues"),
])
def test_a_malformed_request_is_refused_by_name(model, kw, fragment):
    with pytest.raises(ValueError, match=fragment):
        model.insights.logo(LENGTH, **kw)


@pytest.mark.parametrize("kw", [
    {"pseq_above": 0.5, "pseq_below": 0.5},
    {"surprisal_above": 3.0, "pseq_below": 0.5},
    {"surprisal_below": 3.0, "edges": [0.0, 3.0]},
])
def test_two_conditions_at_once_are_refused(model, kw):
    """Mixing axes is the mistake worth catching: a caller who gives both a
    probability and a surprisal has almost certainly confused which is
    which."""
    with pytest.raises(ValueError, match="at most one of"):
        model.insights.logo(LENGTH, **kw)


def test_a_length_the_model_cannot_produce_is_refused(model):
    """Every length up to max_length is produced now -- length 2, which the
    corpus never had, included -- so the one it cannot is past max_length."""
    assert model.insights.logo(2).probability.shape == (2, len(AA))
    with pytest.raises(ValueError, match="not one this model generates"):
        model.insights.logo(int(model.config["max_length"]) + 1)


def test_an_empty_band_is_refused_rather_than_reported_as_conserved(model):
    """The refusal the whole layer exists for.

    Zero probabilities give zero entropy and therefore MAXIMUM information, so
    an answered empty band is a blank plot beside a claim that every position
    is perfectly conserved.
    """
    with pytest.raises((ValueError, RuntimeError)) as ex:
        model.insights.logo(LENGTH, edges=[1e-9, 1e-6], bins=2048)
    assert "no sequence" in str(ex.value) or "no mass" in str(ex.value)


def test_asking_for_one_logo_from_several_bands_is_refused(model):
    with pytest.raises(ValueError, match="defines 3 bands"):
        model.insights.logo(LENGTH, edges=[-math.inf, CUT, CUT + 5.0, math.inf],
                            bins=2048)


# Cost
# ═══════════════════════════════════════════════════════════════

def test_the_cost_is_reported_before_the_sweep_and_scales_with_bins(model):
    edges = [-math.inf, math.inf]
    a = model.insights.cost(LENGTH, edges=edges, bins=1024)
    b = model.insights.cost(LENGTH, edges=edges, bins=2048)
    c = model.insights.cost(LENGTH, edges=edges, bins=4096)
    assert 0 < a["bytes"] < b["bytes"] < c["bytes"]
    # And it is the guard's own figure, not a bound on it: the backward rows
    # are sized by the sweep's range pass, so growth alone would also pass an
    # estimate charging every row the full grid.
    was = BEAM.memory_budget()
    try:
        BEAM.set_memory_budget(b["bytes"])
        model.insights.logos(LENGTH, edges=edges, bins=2048)
        BEAM.set_memory_budget(b["bytes"] - 1)
        with pytest.raises(BEAM.BudgetExceeded):
            model.insights.logos(LENGTH, edges=edges, bins=2048)
    finally:
        BEAM.set_memory_budget(was)
