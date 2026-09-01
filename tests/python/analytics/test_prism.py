"""REPAIR: the one-edit repair factor and its posterior.

The claims under test are the ones the method rests on: that B is a genuine
likelihood ratio (so E[B] <= 1 makes a threshold a false-positive bound with
no calibration), that the posterior is a posterior, and that the C path
computes the same number a direct implementation does.
"""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from BEAM import BeamModel, PrismScorer

from _helpers import AA, make_corpus

ALPHA = "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture(scope="module")
def structured():
    """Motif-structured CDR3s, so an edit is genuinely improbable.

    make_corpus fills the middle with i.i.d. uniform residues, where a
    substitution produces something the model finds equally likely and no
    method can or should detect it.
    """
    rng = random.Random(23)
    vocab = ["".join(rng.choice("GTSLNRDQ") for _ in range(4)) for _ in range(40)]
    seqs = ["CASS" + "".join(rng.choice(vocab) for _ in range(rng.randint(1, 2)))
            + rng.choice(("EQYF", "EQFF", "NTIYF")) for _ in range(6000)]
    return BeamModel(seqs, abundances=None, order=3, max_length=24)


@pytest.fixture(scope="module")
def scorer(structured):
    return PrismScorer(structured)


def _reference_log_b(model, s):
    """B computed the obvious way, as the check on the C path."""
    L = len(s)
    cand, kind = [], []
    for i in range(L):
        for a in ALPHA:
            if a != s[i]:
                cand.append(s[:i] + a + s[i+1:]); kind.append(0)
    if L > 1:
        for i in range(L):
            cand.append(s[:i] + s[i+1:]); kind.append(1)
    if L < model.config["max_length"]:
        for j in range(L + 1):
            for a in ALPHA:
                cand.append(s[:j] + a + s[j:]); kind.append(2)
    lp = np.asarray(model.log_pseq_batch(cand), dtype=np.float64)
    kind = np.asarray(kind)
    parts = []
    for k, d in ((0, 19 * L), (1, 20 * L), (2, L + 1)):
        v = lp[kind == k]
        v = v[np.isfinite(v)]
        if v.size:
            mx = v.max()
            parts.append(mx + math.log(np.exp(v - mx).sum()) - math.log(d))
    if not parts:
        return -math.inf
    mx = max(parts)
    log_q = mx + math.log(sum(math.exp(p - mx) for p in parts)) - math.log(3.0)
    p0 = model.log_pseq(s)
    return math.inf if not math.isfinite(p0) else log_q - p0


def test_c_path_matches_a_direct_implementation(structured, scorer):
    """The C edits an encoded buffer in place; this builds strings. Same number."""
    seqs = structured.sample(40, seed=5)
    got = scorer.factor(seqs)
    want = np.array([_reference_log_b(structured, s) for s in seqs])
    both = np.isfinite(got) & np.isfinite(want)
    assert both.sum() > 20
    assert np.allclose(got[both], want[both], rtol=0, atol=1e-9)
    assert np.array_equal(np.isposinf(got), np.isposinf(want))


def test_B_is_an_e_value(structured, scorer):
    """THE property the whole design rests on.

    B is a likelihood ratio, so E[B] = Q(support(P)) <= 1 under the model.
    That is what makes P(B >= t) <= 1/t hold by Markov, and therefore what lets
    a threshold be a false-positive bound with no calibration table, no
    simulated null and nothing fitted. If this ever fails, every stated
    guarantee goes with it.
    """
    draws = structured.sample(4000, seed=11)
    b = scorer.factor(draws)
    b = b[np.isfinite(b)]
    assert b.size > 3000
    assert float(np.exp(b).mean()) <= 1.0 + 0.05


def test_markov_bound_holds_on_the_model_s_own_draws(structured, scorer):
    """P(B >= t) <= 1/t, checked at two thresholds."""
    draws = structured.sample(4000, seed=13)
    b = scorer.factor(draws)
    for t in (20.0, 100.0):
        rate = float(np.mean(b >= math.log(t)))
        assert rate <= 1.0 / t + 1e-9, (t, rate)


def test_posterior_is_a_posterior(structured, scorer):
    """Sums to 1 where a repair exists, and is undefined where none does.

    log_q = -inf means NO single edit reaches the model's support, which is
    the documented B = -inf case. There is no edit history to distribute over,
    so NaN is the right answer and 0 would be a lie: it would read as "no site
    is likely" when the truth is "the question does not apply".
    """
    checked = undefined = 0
    for s in structured.sample(40, seed=7):
        r = scorer.explain(s)
        p = r.posterior()
        if not math.isfinite(r.log_q):
            undefined += 1
            assert all(np.all(np.isnan(v)) for v in p.values())
            continue
        checked += 1
        tot = sum(float(np.nansum(v)) for v in p.values())
        assert tot == pytest.approx(1.0, abs=1e-9)
        assert all(np.all(v[np.isfinite(v)] >= -1e-12) for v in p.values())
    assert checked > 5, "the fixture produced no repairable sequences"


def test_deletion_posterior_is_indexed_on_gaps(scorer, structured):
    """A deletion happens BETWEEN residues, so its array is one longer."""
    s = structured.sample(1, seed=9)[0]
    r = scorer.explain(s)
    assert r.substitution.size == len(s)
    assert r.insertion.size == len(s)
    assert r.deletion.size == len(s) + 1


def test_repair_localises_a_substitution(structured, scorer):
    """Not a performance claim, a wiring claim: the posterior must point at the
    edit far more often than chance, or the site indices are wrong."""
    rng = random.Random(31)
    draws = [s for s in structured.sample(300, seed=17)
             if math.isfinite(structured.log_pseq(s))]
    hit = tot = 0
    for s in draws:
        i = rng.randrange(len(s))
        bad = s[:i] + rng.choice([c for c in ALPHA if c != s[i]]) + s[i+1:]
        k, j, _ = scorer.explain(bad).site()
        tot += 1
        hit += int(k == "substitution" and j == i)
    chance = 1.0 / np.mean([len(s) for s in draws])
    assert tot > 100
    assert hit / tot > 4 * chance, (hit / tot, chance)


def test_unspellable_but_repairable_is_infinite(structured, scorer):
    """P=0 with Q>0 is the strongest one-edit evidence there is, not an error."""
    s = structured.sample(1, seed=3)[0]
    bad = s[:-1] + ("W" if s[-1] != "W" else "P")
    if math.isfinite(structured.log_pseq(bad)):
        pytest.skip("fixture did not produce an unspellable neighbour")
    assert scorer.factor(bad) == math.inf


def test_bad_alphabet_scores_nan_rather_than_raising(structured, scorer):
    """One unencodable row in a repertoire must not lose the other million.

    The good sequence is drawn from the model rather than written by hand: a
    hand-written CDR3 is not necessarily in a small fixture's support, and this
    test is about the BAD row, not about spellability.
    """
    good = structured.sample(1, seed=2)[0]
    out = scorer.factor([good, "CASSXXTEQYF"])
    assert not math.isnan(out[0])
    assert math.isnan(out[1])


def test_flag_defaults_to_the_non_arbitrary_point(structured, scorer):
    """B = 1 is where a repair becomes the better explanation.

    It is the default because it needs no convention: no bound, no calibration,
    no clean set. Any larger threshold ALSO carries a Markov bound, which is
    checked separately; this checks that the default is the ratio's own
    crossing point and not a rate someone picked.
    """
    draws = structured.sample(500, seed=19)
    b = scorer.factor(draws)
    assert np.array_equal(scorer.flag(draws), b >= 0.0)


def test_larger_thresholds_still_carry_the_bound(structured, scorer):
    draws = structured.sample(2000, seed=19)
    assert float(np.mean(scorer.flag(draws, threshold=100.0))) <= 0.01 + 1e-9
    with pytest.raises(ValueError):
        scorer.flag(draws, threshold=0.0)


def test_scorer_rejects_a_non_model(structured):
    with pytest.raises(TypeError):
        PrismScorer(object())


# omega: how many edits the alternative allows
# ═══════════════════════════════════════════════════════════════

def test_omega_matches_the_number_of_edits_it_is_looking_for(structured, scorer):
    """omega=2 is for TWO-edit corruption, and does not sharpen one-edit.

    An earlier version of this test asserted B2 >= B1 on the grounds that a
    wider ball can only add mass. That is WRONG and it passed by luck on eight
    sequences. Q_k = T^k P, and T is an AVERAGING operator: it spreads mass
    rather than accumulating it. For a singly-corrupted sequence Q1 concentrates
    on the clean parent, and Q2 dilutes that across the parent's own
    neighbours, so B typically FALLS. Measured on the 927M foundation, mean
    log B went -1.105 at omega=1 to -2.000 at omega=2.

    What omega=2 buys is the alternative it names: two edits. That is what this
    checks, and it is the only reason to pay the ~650x cost.
    """
    rng = random.Random(53)
    clean = [s for s in structured.sample(120, seed=41)
             if math.isfinite(structured.log_pseq(s))][:60]

    def hit(s, n):
        for _ in range(n):
            i = rng.randrange(len(s))
            s = s[:i] + rng.choice([c for c in ALPHA if c != s[i]]) + s[i+1:]
        return s

    doubles = [hit(s, 2) for s in clean]
    b1 = scorer.factor(doubles, omega=1)
    b2 = scorer.factor(doubles, omega=2)
    c1 = scorer.factor(clean, omega=1)
    c2 = scorer.factor(clean, omega=2)

    def sep(err, cl):
        err = np.asarray(err, float); cl = np.asarray(cl, float)
        top = np.nanmax(np.r_[err, cl][np.isfinite(np.r_[err, cl])]) + 1.0
        err = np.where(np.isposinf(err), top, err)
        cl = np.where(np.isposinf(cl), top, cl)
        ok_e, ok_c = ~np.isnan(err), ~np.isnan(cl)
        return float(np.mean(err[ok_e][:, None] > cl[ok_c][None, :]))

    # On two-edit corruption the wider alternative must not be worse.
    assert sep(b2, c2) >= sep(b1, c1) - 0.05, (sep(b1, c1), sep(b2, c2))


def test_omega_2_keeps_the_e_value_property(structured, scorer):
    """The guarantee must survive the wider alternative, or omega is a trap.

    T maps a probability law to a probability law at any depth, so E[B] <= 1
    holds for every omega. A user who widens omega must not silently lose the
    calibration-free bound that is the whole point.
    """
    draws = structured.sample(250, seed=43)
    b = scorer.factor(draws, omega=2)
    b = b[np.isfinite(b)]
    assert b.size > 150
    assert float(np.exp(b).mean()) <= 1.0 + 0.10


def test_omega_out_of_range_is_refused(scorer, structured):
    s = structured.sample(1, seed=2)[0]
    for bad in (0, 3, 9):
        with pytest.raises(Exception):
            scorer.factor([s], omega=bad)


def test_site_arrays_are_nan_above_omega_1(scorer, structured):
    """At omega=2 a 'site' is a PAIR of edits, which is a different object.

    Returning a marginal would invite reading a pair posterior as a site
    posterior, so the arrays are NaN and say so.
    """
    s = structured.sample(1, seed=6)[0]
    r = scorer.explain(s, omega=2)
    assert np.all(np.isnan(r.substitution))
    assert np.all(np.isnan(r.deletion))
    assert not math.isnan(r.log_b) or not math.isfinite(r.log_p0)
