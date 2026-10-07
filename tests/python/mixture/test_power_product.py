"""Per-member exponents: ``ln sum_s prod_f P_f(s)^a_f``, exactly.

The plain inner product fixes every exponent at 1. The exponents that matter
for comparing two repertoires do not: the Bhattacharyya coefficient
``sum_s sqrt(P_A P_B)`` needs 1/2 apiece, and the Chernoff family needs
``(a, 1 - a)``. The quantity still factors over the trellis -- a power of a
product of edge weights is the product of the powers -- so it is one sweep
whose edge weight is the WEIGHTED sum of the members' edge weights, and the
per-length constant scales the same way.

Two references, and they are independent of each other and of the sweep:

  * at one member, ``sum_s P(s)^q`` is ``log_mellin(q)``, which math/diversity.c
    reaches by a tilted sweep that shares no code with this file. That runs at
    FULL SCALE and at fractional q, which is where an exponent defect would
    actually live;
  * at two or more, a Python dict of probabilities over an enumerable support
    and a plain sum over it, the way `_tiny` does for everything else.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from BEAM import Mixture
from BEAM.errors import BeamValueError
from _tiny import strings


# ── The brute-force reference ────────────────────────────────────

def brute_power_product(supports, exponents) -> float:
    """``sum_s prod_f P_f(s)^a_f`` by direct summation over the support.

    A member whose exponent is 0 contributes ``P(s)^0 = 1`` at every sequence,
    INCLUDING ones it cannot spell, so it is skipped rather than multiplied in.
    That is the convention the sweep must implement too, and it is the reason
    the test for it is not a formality: multiplying by a zero-power factor and
    intersecting supports are different answers.
    """
    keys: set[str] = set()
    for sup, a in zip(supports, exponents):
        if a > 0.0:
            keys = set(sup) if not keys else keys & set(sup)
    total = 0.0
    for s in sorted(keys):
        term = 1.0
        for sup, a in zip(supports, exponents):
            if a == 0.0:
                continue
            term *= sup[s] ** a
        total += term
    return total


# ── One member: against the Mellin transform ─────────────────────

@pytest.mark.parametrize("q", [0.5, 1.5, 2.0, 0.25, 3.0, 0.75])
def test_one_member_matches_log_mellin(tiny_a, q):
    """``ln sum_s P^q`` by the product sweep and by the tilted sweep."""
    mix = Mixture([tiny_a, tiny_a])
    got = mix.log_power_product(0, exponents=(q,))
    want = tiny_a.log_mellin(q)
    assert got == pytest.approx(want, rel=1e-12), (
        f"sum_s P^{q}: product sweep {got!r}, Mellin {want!r}")


@pytest.mark.parametrize("q", [0.5, 1.5, 2.5])
def test_one_member_matches_log_mellin_at_order_three(model, q):
    """Order 3, amino acids, a real corpus: a wider trellis and a sink that
    settles three residues rather than two."""
    mix = Mixture([model, model])
    assert mix.log_power_product(0, exponents=(q,)) == pytest.approx(
        model.log_mellin(q), rel=1e-11)


def test_exponent_one_is_the_normalisation(tiny_a):
    """``sum_s P(s) = 1``: the exponent path must not disturb it."""
    mix = Mixture([tiny_a, tiny_a])
    assert mix.log_power_product(0, exponents=(1.0,)) == pytest.approx(
        0.0, abs=1e-12)


# ── All exponents 1: the existing kernel, bit for bit ────────────

@pytest.mark.parametrize("idx", [(0, 1), (0, 0), (0, 1, 1)])
def test_all_ones_reproduces_the_inner_product_exactly(tiny_a, tiny_b, idx):
    """Not approximately. ``1.0 * x`` is exact in binary floating point, so
    the exponent path must land on the same bits as the path without one --
    otherwise the scaling has been applied somewhere it changes the order of
    an accumulation."""
    mix = Mixture([tiny_a, tiny_b])
    want = mix.log_inner_product(*idx)
    got = mix.log_power_product(*idx, exponents=(1.0,) * len(idx))
    assert got == want, (
        f"all-ones exponents gave {got!r} where log_inner_product gives "
        f"{want!r}; the two must agree to the last bit")


def test_exponents_default_to_one(tiny_a, tiny_b):
    mix = Mixture([tiny_a, tiny_b])
    assert mix.log_power_product(0, 1) == mix.log_inner_product(0, 1)


# ── Two or more members: against direct summation ────────────────

@pytest.mark.parametrize("exponents", [
    (0.5, 0.5),            # Bhattacharyya
    (0.25, 0.75),          # Chernoff, off centre
    (0.75, 0.25),
    (1.5, 0.5),            # sums past 1: still a well-defined sum
    (0.5, 0.0),            # a dropped member
])
def test_pair_matches_direct_summation(tiny_a, tiny_b, sup_a, sup_b,
                                       exponents):
    mix = Mixture([tiny_a, tiny_b])
    want = brute_power_product([sup_a, sup_b], exponents)
    got = mix.log_power_product(0, 1, exponents=exponents)
    assert got == pytest.approx(math.log(want), rel=1e-12), (
        f"exponents {exponents}: sweep {math.exp(got)!r}, brute {want!r}")


@pytest.mark.parametrize("exponents", [
    (1 / 3, 1 / 3, 1 / 3),
    (0.5, 0.25, 0.25),
    (0.5, 0.5, 0.0),
])
def test_triple_matches_direct_summation(tiny_a, tiny_b, tiny_c,
                                         sup_a, sup_b, sup_c, exponents):
    mix = Mixture([tiny_a, tiny_b, tiny_c])
    want = brute_power_product([sup_a, sup_b, sup_c], exponents)
    got = mix.log_power_product(0, 1, 2, exponents=exponents)
    assert got == pytest.approx(math.log(want), rel=1e-12)


def test_a_zero_exponent_drops_the_member_rather_than_zeroing_the_product(
        tiny_a, far_b):
    """``P^0 = 1`` everywhere, so ``<P_A^1, P_B^0> = sum_s P_A(s) = 1``.

    The trap this catches: treating the zero-exponent member as still
    constraining the support, which would return the mass of the intersection
    instead of 1. With `far_b` the two models share almost nothing, so the
    two answers differ by twenty orders of magnitude rather than by rounding.
    """
    mix = Mixture([tiny_a, far_b])
    assert mix.log_power_product(0, 1, exponents=(1.0, 0.0)) == pytest.approx(
        0.0, abs=1e-12)


# ── Bhattacharyya and Hellinger ──────────────────────────────────

def test_bhattacharyya_of_a_model_with_itself_is_one(tiny_a):
    """``sum_s sqrt(P P) = sum_s P = 1``. The sharpest single check there is:
    it fails for any error in the half-power of the per-length constant, of an
    edge weight, or of the sink."""
    mix = Mixture([tiny_a, tiny_a])
    assert mix.log_bhattacharyya(0, 1) == pytest.approx(0.0, abs=1e-12)
    assert mix.bhattacharyya_coefficient(0, 1) == pytest.approx(1.0, abs=1e-12)


def test_bhattacharyya_of_a_real_model_with_itself_is_one(model):
    """The same identity on an amino-acid order-3 model, where the trellis is
    wide enough that a rescaling defect has somewhere to hide."""
    mix = Mixture([model, model])
    assert mix.log_bhattacharyya(0, 1) == pytest.approx(0.0, abs=1e-11)


def test_bhattacharyya_matches_direct_summation(tiny_a, tiny_b,
                                                sup_a, sup_b):
    mix = Mixture([tiny_a, tiny_b])
    want = sum(math.sqrt(sup_a[s] * sup_b[s]) for s in sup_a)
    assert mix.bhattacharyya_coefficient(0, 1) == pytest.approx(want,
                                                                rel=1e-12)


def test_bhattacharyya_is_bounded_by_one(tiny_a, tiny_b, tiny_c):
    """``sqrt(pq) <= (p + q)/2`` termwise, so BC <= 1 with equality only when
    the two distributions are identical."""
    mix = Mixture([tiny_a, tiny_b, tiny_c])
    for i in range(3):
        for j in range(3):
            bc = mix.bhattacharyya_coefficient(i, j)
            assert 0.0 < bc <= 1.0 + 1e-12
            if i != j:
                assert bc < 1.0


def test_bhattacharyya_is_at_least_the_inner_product(tiny_a, tiny_b):
    """``sqrt(pq) >= pq`` for ``pq <= 1``, termwise."""
    mix = Mixture([tiny_a, tiny_b])
    assert mix.log_bhattacharyya(0, 1) > mix.log_inner_product(0, 1)


def test_hellinger_is_a_metric_on_three_models(tiny_a, tiny_b, far_a):
    """``H = sqrt(1 - BC)`` is a true metric, so the triangle inequality holds
    on every ordering of three distributions. An exponent applied to the wrong
    factor still produces a plausible number; it does not survive this."""
    mix = Mixture([tiny_a, tiny_b, far_a])
    d = np.array([[mix.hellinger_distance(i, j) for j in range(3)]
                  for i in range(3)])
    assert np.allclose(np.diag(d), 0.0, atol=1e-7)
    assert np.allclose(d, d.T, atol=1e-12)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                assert d[i, j] <= d[i, k] + d[k, j] + 1e-12


def test_hellinger_of_near_disjoint_models_approaches_one(far_a, far_b):
    """The two far fixtures share ~4e-20 of mass, so their BC is ~2e-10 and
    the distance is 1 to ten digits -- but the log is finite, not -inf, which
    is the point of doing this on the log scale."""
    mix = Mixture([far_a, far_b])
    lbc = mix.log_bhattacharyya(0, 1)
    assert math.isfinite(lbc)
    assert mix.hellinger_distance(0, 1) == pytest.approx(1.0, abs=1e-6)


# ── Per length ───────────────────────────────────────────────────

def test_by_length_sums_to_the_total(tiny_a, tiny_b):
    mix = Mixture([tiny_a, tiny_b])
    by = mix.log_power_product_by_length(0, 1, exponents=(0.5, 0.5))
    assert by.shape == (tiny_a.config["max_length"] + 1,)
    assert by[0] == -math.inf                 # there is no length-zero CDR3
    live = by[np.isfinite(by)]
    total = float(np.log(np.sum(np.exp(live - live.max()))) + live.max())
    assert total == pytest.approx(mix.log_bhattacharyya(0, 1), rel=1e-12)


def test_by_length_matches_direct_summation_within_each_length(
        tiny_a, tiny_b, sup_a, sup_b):
    """Each entry is the Bhattacharyya mass of ONE length, including the
    length priors -- not a conditional-on-length quantity."""
    mix = Mixture([tiny_a, tiny_b])
    by = mix.log_power_product_by_length(0, 1, exponents=(0.5, 0.5))
    for L in range(1, tiny_a.config["max_length"] + 1):
        want = sum(math.sqrt(sup_a[s] * sup_b[s])
                   for s in strings(L, "ACGTN") if len(s) == L)
        assert by[L] == pytest.approx(math.log(want), rel=1e-12)


# ── The matrix driver ────────────────────────────────────────────

def test_bhattacharyya_matrix_matches_the_pairwise_calls(tiny_a, tiny_b,
                                                         tiny_c, far_a):
    mix = Mixture([tiny_a, tiny_b, tiny_c, far_a])
    g = mix.log_bhattacharyya_matrix()
    assert g.shape == (4, 4)
    assert np.allclose(g, g.T, atol=0.0)      # mirrored, not recomputed
    for i in range(4):
        assert g[i, i] == pytest.approx(0.0, abs=1e-12)
        for j in range(i + 1, 4):
            assert g[i, j] == pytest.approx(mix.log_bhattacharyya(i, j),
                                            rel=1e-12)


def test_bhattacharyya_matrix_fills_the_cache(tiny_a, tiny_b):
    """As `log_gram_matrix` does, so the pairwise calls afterwards are free."""
    mix = Mixture([tiny_a, tiny_b])
    g = mix.log_bhattacharyya_matrix()
    assert mix.log_bhattacharyya(0, 1) == g[0, 1]


# ── The named-pair driver ────────────────────────────────────────

def test_named_pairs_match_the_matrix(tiny_a, tiny_b, tiny_c, far_a):
    mix = Mixture([tiny_a, tiny_b, tiny_c, far_a])
    g = mix.log_bhattacharyya_matrix()
    want = [(0, 1), (2, 3), (1, 1), (3, 0), (0, 0)]
    got = mix.log_bhattacharyya_pairs(want)
    assert got.shape == (5,)
    for (i, j), v in zip(want, got):
        assert v == pytest.approx(g[i, j], rel=1e-12)


def test_a_pair_naming_one_member_twice_is_one(tiny_a, tiny_b):
    """``BC(P, P) = 1``, so such a row is a check the caller can insert."""
    mix = Mixture([tiny_a, tiny_b])
    got = mix.log_bhattacharyya_pairs([(1, 1)])
    assert got[0] == pytest.approx(0.0, abs=1e-12)


def test_named_pairs_fill_the_cache(tiny_a, tiny_b):
    mix = Mixture([tiny_a, tiny_b])
    got = mix.log_bhattacharyya_pairs([(1, 0)])
    assert mix.log_bhattacharyya(0, 1) == got[0]


def test_an_empty_pair_list_returns_nothing(tiny_a, tiny_b):
    mix = Mixture([tiny_a, tiny_b])
    assert mix.log_bhattacharyya_pairs(np.empty((0, 2))).shape == (0,)


def test_a_pair_naming_a_missing_member_is_refused(tiny_a, tiny_b):
    mix = Mixture([tiny_a, tiny_b])
    with pytest.raises(BeamValueError, match="outside"):
        mix.log_bhattacharyya_pairs([(0, 5)])


def test_a_pair_list_of_the_wrong_shape_is_refused(tiny_a, tiny_b):
    mix = Mixture([tiny_a, tiny_b])
    with pytest.raises(BeamValueError, match=r"\(n, 2\)"):
        mix.log_bhattacharyya_pairs([0, 1])


# ── Refusals ─────────────────────────────────────────────────────

def test_a_negative_exponent_is_refused(tiny_a, tiny_b):
    mix = Mixture([tiny_a, tiny_b])
    with pytest.raises(BeamValueError, match="non-negative"):
        mix.log_power_product(0, 1, exponents=(1.5, -0.5))


def test_a_non_finite_exponent_is_refused(tiny_a, tiny_b):
    mix = Mixture([tiny_a, tiny_b])
    with pytest.raises(BeamValueError, match="finite"):
        mix.log_power_product(0, 1, exponents=(1.0, math.inf))
    with pytest.raises(BeamValueError, match="finite"):
        mix.log_power_product(0, 1, exponents=(1.0, math.nan))


def test_all_zero_exponents_are_refused(tiny_a, tiny_b):
    """``sum_s 1`` over the whole space is the support size, which is what
    `support_size` is for; returning it from here would be an answer to a
    different question."""
    mix = Mixture([tiny_a, tiny_b])
    with pytest.raises(BeamValueError, match="at least one"):
        mix.log_power_product(0, 1, exponents=(0.0, 0.0))


def test_a_wrong_number_of_exponents_is_refused(tiny_a, tiny_b):
    mix = Mixture([tiny_a, tiny_b])
    with pytest.raises(BeamValueError, match="one exponent per"):
        mix.log_power_product(0, 1, exponents=(0.5, 0.5, 0.5))


def test_exponents_are_positional_by_naming_not_by_member_index(tiny_a,
                                                                tiny_b):
    """``log_power_product(1, 0, exponents=(0.25, 0.75))`` raises member 1 to
    0.25 and member 0 to 0.75, so reversing both leaves the answer alone."""
    mix = Mixture([tiny_a, tiny_b])
    a = mix.log_power_product(0, 1, exponents=(0.25, 0.75))
    b = mix.log_power_product(1, 0, exponents=(0.75, 0.25))
    assert a == pytest.approx(b, rel=1e-13)


# ── The per-position decomposition ───────────────────────────────

def test_by_position_sums_to_the_total(tiny_a, tiny_b):
    """The contract: positions, sink and the non-positional constant account
    for the total and nothing is left over."""
    mix = Mixture([tiny_a, tiny_b])
    checked = 0
    for L in range(1, tiny_a.config["max_length"] + 1):
        by, sink, konst, total = mix.log_power_product_by_position(
            0, 1, length=L, exponents=(0.5, 0.5))
        if not math.isfinite(total):
            continue
        assert by.shape == (L,)
        assert float(konst + by.sum() + sink) == pytest.approx(total, abs=1e-12)
        checked += 1
    assert checked >= 3


def test_by_position_total_matches_the_by_length_entry(tiny_a, tiny_b):
    """Against a function that already existed and is already tested, bit for
    bit -- the decomposition adds accumulators and moves no arithmetic."""
    mix = Mixture([tiny_a, tiny_b])
    byL = mix.log_power_product_by_length(0, 1, exponents=(0.5, 0.5))
    for L in range(1, tiny_a.config["max_length"] + 1):
        _, _, _, total = mix.log_power_product_by_position(
            0, 1, length=L, exponents=(0.5, 0.5))
        assert total == byL[L]


def test_by_position_total_matches_direct_summation(tiny_a, tiny_b,
                                                    sup_a, sup_b):
    """The independent reference this file uses everywhere else."""
    mix = Mixture([tiny_a, tiny_b])
    for L in range(1, tiny_a.config["max_length"] + 1):
        _, _, _, total = mix.log_power_product_by_position(
            0, 1, length=L, exponents=(0.5, 0.5))
        #: `strings(L, ...)` yields every string of length UP TO L, so the
        #: length filter is not decoration -- without it this sums the wrong
        #: support and the reference, not the sweep, is what is wrong.
        want = sum(math.sqrt(sup_a[s] * sup_b[s])
                   for s in strings(L, "ACGTN") if len(s) == L)
        assert total == pytest.approx(math.log(want), rel=1e-12)


def test_the_profile_does_not_depend_on_member_order(tiny_a, tiny_b):
    """The gauge test. Swapping the members recomputes every rescaling
    maximum, so a term that depended on the rescaling would move."""
    ab = Mixture([tiny_a, tiny_b])
    ba = Mixture([tiny_b, tiny_a])
    for L in range(1, tiny_a.config["max_length"] + 1):
        p1, s1, c1, t1 = ab.log_power_product_by_position(
            0, 1, length=L, exponents=(0.5, 0.5))
        p2, s2, c2, t2 = ba.log_power_product_by_position(
            0, 1, length=L, exponents=(0.5, 0.5))
        if not (math.isfinite(t1) and math.isfinite(t2)):
            continue
        np.testing.assert_allclose(p1, p2, atol=1e-12)
        assert s1 == pytest.approx(s2, abs=1e-12)
        assert c1 == pytest.approx(c2, abs=1e-12)


def test_the_dissimilarity_profile_sums_to_a_nonnegative_distance(tiny_a,
                                                                 tiny_b):
    """The object the decomposition exists for.

        d_jk(i) = (1/2)[g_jj(i) + g_kk(i)] - g_jk(i)

    At a FIXED length the self-products are not 1 -- ``Z_jj(L) = pi_j(L)`` --
    so the distance is ``(1/2)[log pi_j + log pi_k] - log Z_jk``, non-negative
    by Cauchy-Schwarz in the form ``sum sqrt(P_j P_k) <= sqrt(pi_j pi_k)``.
    That is also where the ``sqrt(pi_j pi_k)`` weight for aggregating across
    lengths comes from.
    """
    mix = Mixture([tiny_a, tiny_b])
    half = (0.5, 0.5)
    checked = 0
    for L in range(1, tiny_a.config["max_length"] + 1):
        aa = mix.log_power_product_by_position(0, 0, length=L, exponents=half)
        bb = mix.log_power_product_by_position(1, 1, length=L, exponents=half)
        ab = mix.log_power_product_by_position(0, 1, length=L, exponents=half)
        if not all(math.isfinite(x[3]) for x in (aa, bb, ab)):
            continue
        d = 0.5 * (aa[0] + bb[0]) - ab[0]          # per position
        D = float(d.sum()
                  + 0.5 * (aa[1] + bb[1]) - ab[1]   # sink
                  + 0.5 * (aa[2] + bb[2]) - ab[2])  # const
        scalar = 0.5 * (aa[3] + bb[3]) - ab[3]
        assert D == pytest.approx(scalar, abs=1e-12)
        assert D > -1e-12
        checked += 1
    assert checked >= 3


def test_a_self_pair_has_an_identically_zero_profile(tiny_a, tiny_b):
    """``d_jj(i) = 0`` term by term, which is the consistency of the gauge
    across separate calls rather than an algebraic identity: the three sweeps
    are three independent invocations."""
    mix = Mixture([tiny_a, tiny_b])
    half = (0.5, 0.5)
    g1 = mix.log_power_product_by_position(0, 0, length=4, exponents=half)
    g2 = mix.log_power_product_by_position(0, 0, length=4, exponents=half)
    np.testing.assert_array_equal(g1[0], g2[0])
    np.testing.assert_allclose(0.5 * (g1[0] + g2[0]) - g1[0], 0.0, atol=0.0)


def test_a_length_below_one_is_refused(tiny_a, tiny_b):
    mix = Mixture([tiny_a, tiny_b])
    with pytest.raises(BeamValueError):
        mix.log_power_product_by_position(0, 1, length=0, exponents=(0.5, 0.5))
