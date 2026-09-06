"""Rule extraction: exact conditionals, and a score that resists passengers.

The reference here is CONSTRUCTION, not a second implementation. Corpora are
built so that a rule is true by fiat -- position 9 always echoes position 4,
say -- and the test then demands that extraction finds that rule and not its
shadows. Where a numeric identity is checked it is checked exactly, against
:meth:`BeamModel.log_clamp_probability`, which is the same quantity the search
consumes.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from BEAM import BeamModel
from BEAM.analytics.rules import (CanonicalRule, Rule, canonicalise, extract,
                                  reproducible)

CFG = dict(order=4, depth_cap=15, max_length=27, strict=True, sparse=True)
#: Marginals and clamped conditionals are both exact, so agreement between a
#: rule's stored probability and a fresh clamp is a machine-precision matter.
EXACT = 1e-10


def _corpus(n, length, rng, *, echo=None, filler="ACDEFGHIKLMNPQRSTVWY"):
    """Random sequences with a fixed C...F frame and an optional planted rule.

    `echo=(i, j)` makes position `j` always carry whatever position `i` drew,
    which is a deterministic order-2 rule with nothing else attached to it.
    """
    out = set()
    while len(out) < n:
        s = ["C"] + [rng.choice(list(filler)) for _ in range(length - 2)] + ["F"]
        if echo is not None:
            i, j = echo
            s[j] = s[i]
        out.add("".join(s))
    return sorted(out)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(20260916)


def test_rejects_order_below_two():
    m = BeamModel(["CASSLGGYEQYF"], **CFG)
    with pytest.raises(ValueError, match="at least 2"):
        extract(m, max_order=1)


def test_finds_a_planted_rule(rng):
    """A deterministic echo WITHIN the model's order must surface."""
    seqs = _corpus(2500, 11, rng, echo=(3, 6), filler="ACDE")
    m = BeamModel(seqs, **CFG)
    got = extract(m, lengths=[11], max_order=2, min_p=1e-3, min_marginal=1e-3)

    echoes = [r for r in got[11]
              if r.pos == (3, 6) and r.res[0] == r.res[1]]
    assert echoes, "the planted rule was not found"
    # P(x6 = a | x3 = a) is 1 by construction, so the conditional lift is
    # 1 / P(x6 = a) ~ 4 for a 4-letter alphabet.
    assert min(r.min_cond_lift for r in echoes) > 3.0
    # and a mismatched pair at the same positions must be impossible
    assert not [r for r in got[11]
                if r.pos == (3, 6) and r.res[0] != r.res[1]]


def test_a_rule_beyond_the_model_order_is_not_representable(rng):
    """Not a defect of extraction: an order-k model cannot hold it.

    A state is (position, last `order` residues), so an emission can depend on
    at most the `order` positions before it. A dependency reaching further is
    washed out by the model itself, and extraction correctly reports the
    resulting near-independence rather than inventing the planted rule. Any
    search for long-range rules must raise the model order first.
    """
    seqs = _corpus(2500, 12, rng, echo=(3, 9), filler="ACDE")
    m = BeamModel(seqs, **CFG)          # order 4, the gap is 6
    got = extract(m, lengths=[12], max_order=2, min_p=1e-3, min_marginal=1e-3)
    echoes = [r for r in got[12]
              if r.pos == (3, 9) and r.res[0] == r.res[1]]
    # it survives as a weak pair at best, nowhere near the ~4x a real echo gives
    assert all(r.min_cond_lift < 1.5 for r in echoes)


def test_probability_matches_a_fresh_clamp(rng):
    """A rule's stored probability is the model's own clamped probability."""
    seqs = _corpus(3000, 11, rng, echo=(2, 7), filler="ACDEF")
    m = BeamModel(seqs, **CFG)
    got = extract(m, lengths=[11], max_order=3, min_p=2e-3, min_marginal=2e-3)
    assert got[11]
    for r in got[11][:20]:
        clamp = dict(zip(r.pos, r.res))
        assert r.p == pytest.approx(
            math.exp(m.log_clamp_probability(11, clamp)), abs=EXACT)


def test_passenger_position_caps_the_score(rng):
    """min_cond_lift must punish a rule carrying an uninformative member.

    Position 6 echoes 3, and position 9 is independent of both. The triple
    {3, 6, 9} therefore has a strong PAIR inside it, so lift over independence
    stays high, while the score must collapse to what position 9 contributes.
    """
    seqs = _corpus(2500, 11, rng, echo=(3, 6), filler="ACDE")
    m = BeamModel(seqs, **CFG)
    got = extract(m, lengths=[11], max_order=3, min_p=1e-3, min_marginal=1e-3)

    pair = [r for r in got[11]
            if r.order == 2 and r.pos == (3, 6) and r.res[0] == r.res[1]]
    triple = [r for r in got[11]
              if r.order == 3 and r.pos == (3, 6, 9) and r.res[0] == r.res[1]]
    assert pair and triple, "expected both the pair and its extension"

    p, t = max(pair, key=lambda r: r.p), max(triple, key=lambda r: r.p)
    # the passenger cannot make the rule stronger than its own pair
    assert t.min_cond_lift < p.min_cond_lift
    # while lift over independence is NOT protected, which is the whole point
    assert t.lift_indep > p.min_cond_lift


def test_a_planted_rule_surfaces_on_a_nucleotide_model(rng):
    """The residue letters must be the MODEL's, not the amino-acid ones.

    Extraction clamps on the letters it reports, so an alphabet hard-coded to
    amino acids does not merely mislabel a nucleotide model's residues -- it
    asks the model to clamp position 0 to `D`, which `nt` does not have, and
    the query is refused. Every letter below has to be one of ACGTN, and the
    planted echo has to survive the round trip through them.
    """
    out = set()
    while len(out) < 1200:
        s = [rng.choice(list("ACGT")) for _ in range(9)]
        s[4] = s[2]        # deterministic, and within an order-3 context
        out.add("".join(s))
    seqs = sorted(out)

    m = BeamModel(seqs, alphabet="nt", order=3, max_length=12, strict=True)
    assert m.alphabet == "nt" and m.n_residues == 5

    got = extract(m, lengths=[9], max_order=2, min_p=1e-3, min_marginal=1e-3)
    assert got[9], "the search found rules at all"
    for r in got[9]:
        assert set(r.res) <= set("ACGTN"), f"{r.res} is not nucleotide"
        assert set(r.pattern) <= set("ACGTN_")

    echoes = [r for r in got[9]
              if r.pos == (2, 4) and r.res[0] == r.res[1]]
    assert echoes, "the planted echo between positions 2 and 4 was found"
    # Deterministic and within the model's order, so knowing one position
    # fixes the other: the conditional lift approaches 1 / P(residue), which
    # for four roughly equal nucleotides is near four.
    assert max(r.min_cond_lift for r in echoes) > 2.0

    # And the stored probability is the model's own clamped probability.
    best = max(echoes, key=lambda r: r.min_cond_lift)
    clamp = dict(zip(best.pos, best.res))
    assert best.p == pytest.approx(
        math.exp(m.log_clamp_probability(9, clamp)), rel=EXACT)


def test_reproducible_drops_disagreement():
    a = {12: [Rule(12, (1, 5), ("A", "C"), 0.1, 4.0, 9.0),
              Rule(12, (2, 6), ("D", "E"), 0.1, 4.0, 9.0)]}
    b = {12: [Rule(12, (1, 5), ("A", "C"), 0.1, 4.1, 9.0),   # agrees
              Rule(12, (2, 6), ("D", "E"), 0.1, 40.0, 9.0)]}  # does not
    kept = reproducible(a, b, tol=0.25)
    assert [r.pos for r in kept[12]] == [(1, 5)]


def test_reproducible_drops_absent_rules():
    a = {12: [Rule(12, (1, 5), ("A", "C"), 0.1, 4.0, 9.0)]}
    assert reproducible(a, {12: []}) == {12: []}


def test_canonicalise_collapses_and_infers_end_anchor():
    """One rule at a fixed distance from the C terminus, three lengths."""
    rules = {L: [Rule(L, (L - 5, L - 3), ("G", "Y"), 0.1, 8.0, 50.0)]
             for L in (12, 13, 14)}
    can = canonicalise(rules, {12: 0.2, 13: 0.3, 14: 0.5})
    assert len(can) == 1
    r = can[0]
    assert r.anchor == "end"
    assert r.offsets == (4, 2) and r.res == ("G", "Y")
    assert r.breadth == 3 and r.lengths == (12, 13, 14)
    assert r.mass == pytest.approx(1.0)
    assert r.mcl_weighted == pytest.approx(8.0)


def test_canonicalise_infers_start_anchor():
    """Constant position from the N terminus, varying distance from the C."""
    rules = {L: [Rule(L, (1, 3), ("A", "S"), 0.5, 2.0, 3.0)]
             for L in (12, 13, 14)}
    can = canonicalise(rules)
    assert len(can) == 1 and can[0].anchor == "start"
    assert can[0].offsets == (1, 3)


def test_single_length_rule_is_not_assigned_an_anchor():
    """Start and end are indistinguishable from one length, so do not guess."""
    can = canonicalise({15: [Rule(15, (4,), ("L",), 0.2, 1.0, 1.0),
                             Rule(15, (4, 9), ("L", "Y"), 0.1, 3.0, 5.0)]})
    assert {r.anchor for r in can} == {"length-specific"}
    assert all(r.breadth == 1 for r in can)


def test_canonicalise_keeps_the_per_length_profile():
    """Breadth is evidence and the spread across lengths must survive."""
    rules = {L: [Rule(L, (L - 4, L - 2), ("Q", "H"), 0.1, lift, 20.0)]
             for L, lift in ((12, 5.0), (13, 10.0), (14, 20.0))}
    r = canonicalise(rules, {12: 0.5, 13: 0.3, 14: 0.2})[0]
    assert r.min_cond_lift == {12: 5.0, 13: 10.0, 14: 20.0}
    assert r.mcl_range == (5.0, 20.0)
    # mass-weighted, so the common length dominates
    assert r.mcl_weighted == pytest.approx(0.5 * 5 + 0.3 * 10 + 0.2 * 20)


def test_rule_geometry_round_trips():
    r = Rule(15, (0, 10, 14), ("C", "Q", "F"), 0.05, 3.0, 9.0)
    assert r.pattern == "C_________Q___F"
    assert r.rpos == (14, 4, 0)
    assert r.order == 3
