"""Exact occupancy quantities from power sums, against enumeration.

`ExactOccupancy` computes sums over every sequence of functions of P(s) --
expected richness, occupancy counts, sharing, private counts -- from the
power sums M(q) = sum_s P(s)^q alone: exact real series for small depths and
Mellin-Barnes contours on vertical lines for the rest. No binned spectrum.

Every value comes with a RIGOROUS relative bound, and the tests hold each one
to the direct sum over an enumerated support with that bound -- not to a
tolerance chosen here. A bound that is too loose to be useful fails too.
"""
from __future__ import annotations

import itertools
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel
from BEAM.analytics.exact_occupancy import BoundedValue, ExactOccupancy


def _enumerated(alphabet):
    if alphabet == "nt":
        rng = random.Random(5)
        seqs = ["TG" + "".join(rng.choice("ACGT") for _ in range(rng.randint(0, 3)))
                + "C" for _ in range(400)]
        ab = [rng.randint(1, 40) for _ in seqs]
        m = BeamModel(seqs, abundances=ab, alphabet="nt", order=3, max_length=6)
        every = ["".join(t) for L in range(1, 7)
                 for t in itertools.product("ACGTN", repeat=L)]
    else:
        rng = random.Random(11)
        seqs = ["C" + "".join(rng.choice("ASGLT") for _ in range(rng.randint(0, 2)))
                + "F" for _ in range(300)]
        ab = [rng.randint(1, 9) for _ in seqs]
        m = BeamModel(seqs, abundances=ab, order=2, max_length=4)
        every = ["".join(t) for L in range(1, 5)
                 for t in itertools.product("ACDEFGHIKLMNPQRSTVWY", repeat=L)]
    p = np.exp(np.asarray(m.log_pseq_batch(every), dtype=np.float64))
    return m, p.astype(np.longdouble)


@pytest.fixture(scope="module", params=["nt", "aa"])
def case(request):
    m, p = _enumerated(request.param)
    return m, p, ExactOccupancy(m, rel_tol=1e-10)


def _miss(p, n):
    """(1 - p)^n in long double, safe for tiny p and huge n."""
    return np.exp(np.longdouble(n) * np.log1p(-p))


def _seen(p, n):
    """1 - (1 - p)^n, as -expm1: the direct difference loses every digit
    that 1 and (1-p)^n share, which for tiny p is all of them."""
    return -np.expm1(np.longdouble(n) * np.log1p(-p))


def _log_binom(n, j):
    """ln C(n, j) for integer j, as a sum of logs: lgamma(n+1) at n = 1e8 is
    1.7e9 and a difference of two of them is good to 1e-7 at best."""
    return (float(np.sum(np.log(np.longdouble(n) - np.arange(j, dtype=np.longdouble))))
            - math.lgamma(j + 1))


def _held(v: BoundedValue, exact: float, tol: float = 1e-10):
    """The value is within its own bound of the truth, and the bound is useful."""
    assert isinstance(v, BoundedValue)
    assert abs(v.value - exact) <= v.bound * abs(exact) + 1e-300, (v, exact)
    assert v.bound <= tol, v


DEPTHS = [0.5, 1.0, 3.0, 17.5, 1e2, 1e3, 1e4, 1e5, 1e6, 1e8, 1e10, 1e13]


def test_richness(case):
    m, p, eo = case
    for n in DEPTHS:
        exact = float(np.sum(_seen(p, n)))
        _held(eo.richness(n), exact)


def test_richness_corners(case):
    m, p, eo = case
    assert eo.richness(0.0).value == 0.0
    _held(eo.richness(1.0), float(np.sum(p)))          # R(1) is the mass
    d0 = float(p.size)
    big = eo.richness(1e40)                            # every sequence drawn
    assert big.value == pytest.approx(d0, rel=1e-10)


def test_richness_many_depths_at_once(case):
    m, p, eo = case
    ns = np.geomspace(1.0, 1e12, 40)
    vals = eo.richness_many(ns)
    for n, v in zip(ns, vals):
        _held(v, float(np.sum(_seen(p, n))))


def test_frequency(case):
    """E[# sequences drawn exactly j times in n draws] = sum C(n,j) p^j (1-p)^{n-j}."""
    m, p, eo = case
    for n in (5.0, 50.0, 1e3, 1e5, 1e8):
        for j in (1, 2, 3, 5):
            if j > n:
                continue
            logc = _log_binom(n, j)
            exact = float(np.sum(np.exp(np.longdouble(logc) + j * np.log(p)
                                        + np.longdouble(n - j) * np.log1p(-p))))
            _held(eo.frequency(n, j), exact, tol=1e-9)


def test_frequencies_add_up_to_richness(case):
    """Seen at least once = sum over j >= 1 of seen exactly j times (n small)."""
    m, p, eo = case
    n = 6.0
    total = sum(eo.frequency(n, j).value for j in range(1, 7))
    assert total == pytest.approx(eo.richness(n).value, rel=1e-10)


def test_richness_difference_keeps_its_digits(case):
    """R(n1) - R(n2) for n1 close to n2, where subtracting two R's would
    lose every digit the two share."""
    m, p, eo = case
    for n2, gap in ((1e6, 1.0), (1e9, 1e3), (1e12, 7.0)):
        exact = float(np.sum(_miss(p, n2) * _seen(p, gap)))
        _held(eo.richness_difference(n2 + gap, n2), exact, tol=1e-9)


def test_sharing(case):
    """E[# sequences in both of two independent repertoires of depths d1, d2]."""
    m, p, eo = case
    for d1, d2 in ((1e3, 1e3), (1e3, 1e6), (5e5, 2e8)):
        exact = float(np.sum(_seen(p, d1) * _seen(p, d2)))
        _held(eo.sharing(d1, d2), exact, tol=1e-9)


def test_private_in_a_cohort(case):
    """E[# sequences seen in exactly one repertoire], unequal depths."""
    m, p, eo = case
    rng = np.random.default_rng(3)
    depths = rng.integers(100, 100_000, size=25).astype(float)
    T = depths.sum()
    exact = float(np.sum([np.sum(_seen(p, d) * _miss(p, T - d)) for d in depths]))
    _held(eo.private(depths), exact, tol=1e-9)


def test_equal_depth_cohort(case):
    """N repertoires of one depth, N up to 1e10 where R(N d) - R((N-1) d)
    differ in their tenth digit."""
    m, p, eo = case
    for depth, N in ((1e3, 10), (1e4, 768), (1e3, 1e10)):
        seen_exact = float(np.sum(_seen(p, N * depth)))
        one = _seen(p, depth) * _miss(p, (N - 1) * depth)
        priv_exact = float(N * np.sum(one))
        c = eo.equal_depth_cohort(depth, N)
        _held(c["seen"], seen_exact)
        _held(c["private"], priv_exact, tol=1e-9)


def test_bounded_value_does_not_pass_for_a_float(case):
    m, p, eo = case
    v = eo.richness(10.0)
    with pytest.raises(TypeError):
        float(v)
    assert v.value > 0 and 0 <= v.bound < 1


def test_lines_are_reused(case):
    m, p, eo = case
    eo.richness(1e6)
    before = len(eo.lines)
    for n in (1.1e6, 2e6, 5e6):
        eo.richness(n)
    assert len(eo.lines) == before


def test_an_extended_line_matches_a_computed_one(case):
    """Extending a line appends the nodes a longer line would have computed."""
    m, p, eo = case
    fresh = ExactOccupancy(m, rel_tol=1e-10)
    short = fresh._line(0.5, 0.1, 5.0)
    n_short = short.t.size
    long_ = fresh._line(0.5, 0.1, 12.0)
    assert long_ is short and long_.t.size > n_short     # extended in place
    direct = m.log_mellin_line(0.5, 0.0, -0.1, long_.t.size)
    ref = m.log_mellin(0.5)
    d = np.abs(np.exp(long_.log_m - ref) - np.exp(direct - ref))
    assert d.max() <= 1e-13


def test_novelty(case):
    """U(n) = sum_s P (1-P)^(n-1): the chance draw n is a new sequence."""
    m, p, eo = case
    ns = [1.0, 2.0, 50.0, 1e3, 1e6, 1e9]
    for n, v in zip(ns, eo.novelty_many(ns)):
        exact = float(np.sum(p * _miss(p, n - 1)))
        _held(v, exact, tol=1e-9)


def test_novelty_below_the_smallest_double_is_zero_and_says_so(case):
    """Past (n-1) p_min = 745 every term of U(n) is below the smallest double:
    the value is 0, its relative error is exactly 1, and the bound says so."""
    m, p, eo = case
    for n in (800.0 / eo.p_min, 1e300):
        v = eo.novelty_many([n])[0]
        assert (v.value, v.bound, v.method) == (0.0, 1.0, "underflow")
        assert np.sum(p * _miss(p, n - 1)) <= np.exp(np.longdouble(-745))


def test_novelty_short_of_its_tolerance_still_tells_the_truth(case):
    """Deep in saturation U(n) is exponentially small and a contour cannot
    resolve it to rel_tol; whatever bound it reports must still hold."""
    m, p, eo = case
    for n in (3.0 / eo.p_min, 30.0 / eo.p_min, 300.0 / eo.p_min):
        v = eo.novelty_many([n])[0]
        assert np.isfinite(v.value) and v.bound > 0
        _held(v, float(np.sum(p * _miss(p, n - 1))), tol=math.inf)


def test_richness_many_is_vectorised_and_agrees_with_one_at_a_time(case):
    m, p, eo = case
    ns = np.unique(np.round(np.geomspace(150.0, 1e9, 300)))
    many = eo.richness_many(ns)
    for n, v in zip(ns[::37], many[::37]):
        one = eo.richness(float(n))
        assert v.value == pytest.approx(one.value, rel=max(v.bound, one.bound) * 2 + 1e-15)
    for n, v in zip(ns, many):
        _held(v, float(np.sum(_seen(p, n))))


def test_private_at_depth(case):
    """Per repertoire: R(d_i) and R(T) - R(T - d_i), for many depths."""
    m, p, eo = case
    rng = np.random.default_rng(7)
    depths = rng.integers(100, 100_000, size=400).astype(float)
    T = float(depths.sum())
    out = eo.private_at_depth(depths, total_depth=T)
    for i in range(0, depths.size, 40):
        d = depths[i]
        _held(out["distinct"][i], float(np.sum(_seen(p, d))))
        _held(out["private"][i], float(np.sum(_seen(p, d) * _miss(p, T - d))), tol=1e-9)


def test_lines_survive_a_round_trip_and_refuse_another_model(case, tmp_path):
    """Lines are expensive; a file of them serves a later process, and only
    for the model that made them."""
    m, p, eo = case
    eo.richness_many([1e3, 1e6, 1e9])
    path = tmp_path / "lines.npz"
    eo.save_lines(path)
    fresh = ExactOccupancy(m, rel_tol=1e-10)
    fresh.load_lines(path)
    assert fresh.lines == eo.lines
    a = eo.richness(1e6)
    b = fresh.richness(1e6)
    assert a.value == b.value and a.bound == b.bound
    other, _ = _enumerated("nt" if m.alphabet == "aa" else "aa")
    with pytest.raises(ValueError):
        ExactOccupancy(other, rel_tol=1e-10).load_lines(path)


def test_a_batch_builds_no_more_lines_than_one_depth_at_a_time(case):
    """Lines are the whole cost. Asked for many depths at once, the planner
    must not build a line per depth range that one line would have served."""
    m, p, eo = case
    ns = np.geomspace(1e3, 1e15, 121)
    batch = ExactOccupancy(m, rel_tol=1e-10)
    batch.richness_many(ns)
    one = ExactOccupancy(m, rel_tol=1e-10)
    for n in ns:
        one.richness_many([n])
    assert len(batch.lines) <= len(one.lines)
    assert sum(x[2] for x in batch.lines) <= sum(x[2] for x in one.lines)


def test_every_line_computed_is_logged(case, caplog):
    """A line is minutes on a large model; each one says what it cost and
    what it resolved, so a long batch can be followed."""
    import logging
    m, p, eo = case
    fresh = ExactOccupancy(m, rel_tol=1e-10)
    with caplog.at_level(logging.INFO, logger="BEAM.analytics.exact_occupancy"):
        fresh.richness_many(np.geomspace(1e3, 1e9, 20))
    msgs = [r.getMessage() for r in caplog.records]
    assert len([x for x in msgs if x.startswith("line ")]) == len(fresh.lines)
    assert all("orders" in x and "resolved" in x for x in msgs if x.startswith("line "))


def test_saving_keeps_the_lines_another_process_saved(case, tmp_path):
    """Two processes on one model each save what they computed to the same
    file; the file ends up with both, not with whichever saved last."""
    m, p, eo = case
    path = tmp_path / "lines.npz"
    a = ExactOccupancy(m, rel_tol=1e-10)
    b = ExactOccupancy(m, rel_tol=1e-10)
    a._line(0.5, 0.1, 10.0)
    b._line(0.3, 0.08, 12.0)
    a.save_lines(path)
    b.save_lines(path)
    later = ExactOccupancy(m, rel_tol=1e-10)
    later.load_lines(path)
    assert sorted(later.lines) == sorted(a.lines + [(0.3, 0.08, 151)])


def test_loading_a_longer_line_replaces_the_shorter(case, tmp_path):
    m, p, eo = case
    path = tmp_path / "lines.npz"
    short = ExactOccupancy(m, rel_tol=1e-10)
    short._line(0.5, 0.1, 10.0)
    long = ExactOccupancy(m, rel_tol=1e-10)
    long._line(0.5, 0.1, 20.0)
    long.save_lines(path)
    short.load_lines(path)
    assert short.lines == [(0.5, 0.1, 201)]


def test_saving_over_another_models_lines_refuses(case, tmp_path):
    m, p, eo = case
    path = tmp_path / "lines.npz"
    ExactOccupancy(m, rel_tol=1e-10).save_lines(path)
    other, _ = _enumerated("nt" if m.alphabet == "aa" else "aa")
    with pytest.raises(ValueError):
        ExactOccupancy(other, rel_tol=1e-10).save_lines(path)


def test_rejects_bad_depths(case):
    m, p, eo = case
    for bad in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            eo.richness(bad)
    with pytest.raises(ValueError):
        eo.frequency(10.0, 0)


# ── the sampling deficit (sampled Hill D1) ────────────────────────────────

def _deficit_one(lam):
    """D(lam) = E[ln((1+Y)/lam)], Y ~ Poisson(lam), in long double.

    Weights by the recurrence w_k = w_(k-1) lam / k outward from the mode,
    normalised by their own sum (the Poisson PMF sums to exactly 1), so no
    log-gamma of a large argument is ever formed; and each term's value as
    log1p((1 + k - lam) / lam), so ln lam is never subtracted from a sum
    near it. Tails beyond 60 standard deviations are dropped."""
    if lam > 1e6:
        # 1/(2 lam) + 1/(12 lam^2) + O(lam^-3): E[ln(1 + x)], x = (1+Y-lam)/lam,
        # expanded to x^4 -- the x^4 term is ORDER lam^-2 too, since
        # E[(Y-lam)^4] ~ 3 lam^2: -1/2 + 4/3 - 3/4 = 1/12. (Stopping at x^3
        # gives 5/6, which is wrong; mpmath confirms 1/12.)
        return 1.0 / (2.0 * lam) + 1.0 / (12.0 * lam * lam)
    ld = np.longdouble
    lam = ld(lam)
    mode = int(lam)
    span = int(60 * float(np.sqrt(lam)) + 80)
    lo, hi = max(0, mode - span), mode + span
    up = np.ones(hi - mode + 1, dtype=ld)
    for k in range(mode + 1, hi + 1):
        up[k - mode] = up[k - mode - 1] * lam / ld(k)
    down = np.ones(mode - lo + 1, dtype=ld)
    for k in range(mode - 1, lo - 1, -1):
        down[mode - k] = down[mode - k - 1] * ld(k + 1) / lam
    w = np.concatenate([down[::-1], up[1:]])
    k = np.arange(lo, hi + 1, dtype=ld)
    return float(np.sum(w * np.log1p((1 + k - lam) / lam)) / np.sum(w))


def _deficit_direct(p, n):
    """sum_s p D(n p), each distinct D(n p) summed directly (`_deficit_one`)."""
    p = np.asarray(p, dtype=np.float64)
    lam, inv = np.unique(n * p, return_inverse=True)
    d = np.array([_deficit_one(x) for x in lam])
    return float(np.sum(np.longdouble(p) * d[inv]))


def _deficit_kernel_reference(w, dps=30):
    """D~(w) to `dps` digits without integrating through a singularity:
    [0, 1/2] by the exact Gregory series (160 terms, error below 2^-150),
    [1/2, 1] as the smooth integral, subdivided where it oscillates."""
    import mpmath as mp
    from fractions import Fraction
    N = 160
    a = [Fraction(1, k + 1) for k in range(N + 1)]
    b = [Fraction(1)] + [Fraction(0)] * N
    for k in range(1, N + 1):
        b[k] = -sum(a[j] * b[k - j] for j in range(1, k + 1))
    with mp.workdps(dps):
        w = mp.mpc(w)
        ln2 = mp.log(2)
        head = mp.fsum(mp.mpf((-b[n]).numerator) / (-b[n]).denominator
                       * mp.exp((w - n) * ln2) / (n - w) for n in range(1, N + 1))
        f = lambda x: mp.exp(-x) * mp.exp(-w * mp.log1p(-mp.exp(-x) / 2)) / (ln2 + x)
        log_part = mp.quad(f, [mp.mpf(i) / 4 for i in range(49)] + [mp.inf]) / 2
        return complex(mp.gamma(w) * (1 / w + head + mp.expm1(w * ln2) / w - log_part))


def test_deficit_kernel_matches_a_high_precision_reference():
    """D~(w) = Gamma(w) [1/w + int_0^1 v^-w (1/v + 1/ln(1-v)) dv], the Mellin
    transform of the sampling deficit, held to the rounding the contour bound
    charges it: EPS (|ln Gamma(w)| + 10 (1 + |Im w|))."""
    pytest.importorskip("mpmath")       # the reference's, not the library's
    from scipy.special import loggamma
    from BEAM.analytics.exact_occupancy import EPS, _deficit_kernel
    for c in (0.05, 0.3, 0.7, 0.95):
        for t in (0.0, 0.4, 3.0, 17.0, 60.0, 150.0):
            w = complex(c, t)
            got = complex(_deficit_kernel(np.array([w]))[0])
            want = _deficit_kernel_reference(w)
            charged = EPS * (abs(loggamma(w)) + 10 * (1 + t))
            assert abs(got - want) <= charged * abs(want), (w, abs(got / want - 1))


@pytest.mark.parametrize("n", [0.3, 1.0, 7.0, 60.0, 1e3, 1e4])
def test_sampling_deficit(case, n):
    """S(N) = sum_s P D(N P); the sampled Hill D1 is exp(H - S(N)), so S is
    held to rel_tol of max(S, 1 nat): its absolute error is D1's relative."""
    m, p, eo = case
    v = eo.sampling_deficit_many([n])[0]
    exact = _deficit_direct(p.astype(np.float64), n)
    _held(v, exact, tol=1e-10 * max(1.0, 1.0 / exact))


def test_sampling_deficit_deep_in_n_is_resolved_absolutely(case):
    """Far past the model's own scale S ~ D0 / 2N is tiny; it is resolved to
    an absolute 1e-10 -- all that exp(H - S) needs -- without a pile of
    lines chasing a relative target nothing uses."""
    m, p, eo = case
    if m.alphabet == "aa":
        pytest.skip("the direct sum is slow over 168,420 sequences at these depths")
    fresh = ExactOccupancy(m, rel_tol=1e-10)
    ns = [1e8, 1e12, 1e20]
    for n, v in zip(ns, fresh.sampling_deficit_many(ns)):
        exact = _deficit_direct(p.astype(np.float64), n)
        assert abs(v.value - exact) <= v.bound * exact + 1e-300, (n, v, exact)
        assert v.bound * exact <= 1e-10, (n, v)
    assert len(fresh.lines) <= 4, fresh.lines


def test_poisson_richness(case):
    """sum_s 1 - exp(-N P): the richness of a Poisson sample of depth N."""
    m, p, eo = case
    ns = [0.5, 3.0, 1e2, 1e4, 1e6]
    for n, v in zip(ns, eo.poisson_richness_many(ns)):
        _held(v, float(np.sum(-np.expm1(-np.longdouble(n) * p))))


def test_a_value_no_line_can_resolve_stops_the_planner(case):
    """Where most of the mass lies far above 1/n, U(n) is tiny (1e-36 on the
    nucleotide model here) and no line resolves it to rel_tol. The planner
    must stop -- returning each value with the bound it has -- rather than
    build line after line that resolves nothing (3-4 minutes each on the
    foundation). Every bound it returns must still hold."""
    m, p, eo = case
    fresh = ExactOccupancy(m, rel_tol=1e-10)
    ns = np.geomspace(1e-3 / eo.p_min, 1e-1 / eo.p_min, 12)
    out = fresh.novelty_many(ns)
    for n, v in zip(ns, out):
        _held(v, float(np.sum(p * _miss(p, n - 1))), tol=math.inf)
    assert len(fresh.lines) <= 4, fresh.lines


def test_a_value_below_its_rounding_floor_builds_at_most_one_new_line(case):
    """What cost the foundation's discovery curve 9.5 minutes: with the
    richness lines already built, novelties that the rounding of those lines
    cannot resolve get ONE more line, at a lower q where the sum cancels less
    -- on the foundation such a line resolved 16 of 17 values the higher q
    could not -- and no more; and they come back within the bounds they
    report."""
    m, p, eo = case
    fresh = ExactOccupancy(m, rel_tol=1e-10)
    ns = np.geomspace(1e-3 / fresh.p_min, 1e-1 / fresh.p_min, 12)
    fresh.richness_many(ns)
    built = len(fresh.lines)
    out = fresh.novelty_many(ns)
    for n, v in zip(ns, out):
        _held(v, float(np.sum(p * _miss(p, n - 1))), tol=math.inf)
    unresolved = sum(v.bound > 1e-10 for v in out)
    if unresolved:
        assert len(fresh.lines) <= built + 1, (fresh.lines[built:], unresolved)


def test_a_fresh_batch_below_its_rounding_floor_builds_at_most_two_lines(case):
    m, p, eo = case
    fresh = ExactOccupancy(m, rel_tol=1e-10)
    ns = np.geomspace(1e-3 / fresh.p_min, 1e-1 / fresh.p_min, 12)
    out = fresh.novelty_many(ns)
    for n, v in zip(ns, out):
        _held(v, float(np.sum(p * _miss(p, n - 1))), tol=math.inf)
    if any(v.bound > 1e-10 for v in out):
        assert len(fresh.lines) <= 2, fresh.lines


class _CountingModel:
    """The model, with a record of the real orders each `log_mellin_many`
    call asked for: each call is a full sweep, eight orders at a time."""

    def __init__(self, m):
        self._m = m
        self.many = []
        self.orders = []

    def __getattr__(self, name):
        return getattr(self._m, name)

    def log_mellin_many(self, qs):
        self.many.append(int(np.asarray(qs).size))
        self.orders += [float(q) for q in np.asarray(qs).ravel()]
        return self._m.log_mellin_many(qs)


def test_series_power_sums_are_taken_only_when_a_series_needs_them(case):
    """The series route's ninety real power sums are ninety full sweeps --
    about a minute on the foundation -- and a workload that only ever uses
    the contours must not pay for them."""
    m, p, _ = case
    cm = _CountingModel(m)
    eo = ExactOccupancy(cm, rel_tol=1e-10)
    deep = 1e9
    assert not eo._series_ok(deep)
    _held(eo.richness_many([deep])[0], float(np.sum(_seen(p, deep))))
    _held(eo.novelty_many([deep])[0], float(np.sum(p * _miss(p, deep - 1))))
    ks = [float(k) for k in range(1, ExactOccupancy._K_SERIES + 1)]
    # 1 and 3 are planning and fingerprint orders, taken up front anyway
    assert not set(ks) - {1.0, 3.0} & set(cm.orders)
    small = 3.0
    _held(eo.richness(small), float(np.sum(_seen(p, small))))
    assert all(cm.orders.count(k) == 1 for k in ks)


def test_the_fingerprint_is_taken_once(case, tmp_path):
    """Every save and load checks the model's fingerprint; it is the same
    model each time, so it is computed once, not a sweep per save."""
    m, p, _ = case
    cm = _CountingModel(m)
    eo = ExactOccupancy(cm, rel_tol=1e-10)
    eo._line(0.5, 0.1, 10.0)
    path = tmp_path / "lines.npz"
    for _ in range(3):
        eo.save_lines(path)
    eo.load_lines(path)
    assert all(cm.orders.count(q) == 1 for q in ExactOccupancy._FINGERPRINT_Q)


def test_the_power_sums_every_use_needs_are_one_sweep(case, tmp_path):
    """D0, the fingerprint and the planning grid are needed by nearly every
    use, and a sweep costs the same for one order as for eight: they are
    taken together, once, not as three sweeps."""
    m, p, _ = case
    cm = _CountingModel(m)
    eo = ExactOccupancy(cm, rel_tol=1e-10)
    eo.richness_many([1e9])
    eo.save_lines(tmp_path / "lines.npz")
    assert len(cm.many) == 1 and len(set(cm.orders)) == len(cm.orders)
    assert eo.log_d0 == m.log_mellin(0.0)


def test_power_sums_saved_with_the_lines_are_not_swept_again(case, tmp_path):
    """A later process loads the real power sums an earlier one swept --
    the series route's ninety are a minute on the foundation -- and gets
    the same values to the bit."""
    m, p, _ = case
    first = ExactOccupancy(m, rel_tol=1e-10)
    ns = [3.0, 1e9]
    want = first.richness_many(ns)
    path = tmp_path / "lines.npz"
    first.save_lines(path)
    cm = _CountingSingles(m)
    later = ExactOccupancy(cm, rel_tol=1e-10)
    swept_at_start = list(cm.orders)
    later.load_lines(path)
    got = later.richness_many(ns)
    assert cm.orders == swept_at_start and not cm.singles
    for a, b in zip(want, got):
        assert a.value == b.value and a.bound == b.bound


def test_a_lines_file_without_power_sums_still_loads(case, tmp_path):
    """Files written before the power sums were kept hold only lines."""
    m, p, eo = case
    eo.richness_many([1e9])
    path = tmp_path / "lines.npz"
    eo.save_lines(path)
    z = dict(np.load(path))
    del z["swept_q"], z["swept_log_m"]
    old = tmp_path / "old.npz"
    np.savez(old, **z)
    fresh = ExactOccupancy(m, rel_tol=1e-10)
    fresh.load_lines(old)
    assert fresh.lines == eo.lines


class _CountingSingles(_CountingModel):
    """Also records each single real order asked of the model."""

    def __init__(self, m):
        super().__init__(m)
        self.singles = []

    def log_mellin(self, q):
        self.singles.append(float(q))
        return self._m.log_mellin(q)


def test_a_line_carries_its_own_real_power_sum(case, tmp_path):
    """Every line starts at t = 0, where it IS M at its real part; the bounds
    read it from there rather than sweeping that real order again -- a full
    sweep per line per process on the foundation."""
    m, p, eo = case
    eo.richness_many([1e9, 1e12])
    path = tmp_path / "lines.npz"
    eo.save_lines(path)
    cm = _CountingSingles(m)
    fresh = ExactOccupancy(cm, rel_tol=1e-10)
    fresh.load_lines(path)
    qs = {ln[0] for ln in fresh.lines}
    for n, v in zip([1e9, 1e12], fresh.richness_many([1e9, 1e12])):
        _held(v, float(np.sum(_seen(p, n))))
    assert not qs & set(cm.singles), (qs, cm.singles)


def test_a_value_no_allowed_q_can_resolve_builds_no_line(case):
    """Where even the best allowed q leaves a line's rounding floor above the
    tolerance, the planner predicts it -- from the kernel's size along the
    line and the planning grid of M(q) -- and builds nothing: three lines
    that could resolve nothing cost the foundation's discovery curve 99 s.
    Every value still comes back with a bound that holds."""
    m, p, eo = case
    fresh = ExactOccupancy(m, rel_tol=1e-24)
    ns = np.geomspace(1e2, 1e6, 5)
    out = fresh.richness_many(ns)
    for n, v in zip(ns, out):
        _held(v, float(np.sum(_seen(p, n))), tol=math.inf)
    assert fresh.lines == [], fresh.lines


def test_the_fused_contour_sums_agree_with_the_matrix_form(case, monkeypatch):
    """Richness and novelty take each line's integral and tail in one C pass
    per parameter, with no nodes-by-parameters matrix; the numpy matrix form
    they replaced must give the same values to rounding, and both must hold
    against the enumerated truth."""
    import BEAM.analytics.exact_occupancy as eo_mod
    m, p, _ = case
    ns = np.geomspace(1e2, 1e9, 9)
    fused = ExactOccupancy(m, rel_tol=1e-10)
    matrix = ExactOccupancy(m, rel_tol=1e-10)
    matrix._fused = False
    built = []
    real_matrix = eo_mod._log_ratio_matrix
    monkeypatch.setattr(eo_mod, "_log_ratio_matrix",
                        lambda w, z: built.append(1) or real_matrix(w, z))
    for name, truth in (("richness_many", lambda n: np.sum(_seen(p, n))),
                        ("novelty_many", lambda n: np.sum(p * _miss(p, n - 1)))):
        built.clear()
        a = getattr(fused, name)(ns)
        assert not built, f"{name} formed a nodes-by-parameters matrix"
        b = getattr(matrix, name)(ns)
        for n, x, y in zip(ns, a, b):
            _held(x, float(truth(n)), tol=math.inf)
            _held(y, float(truth(n)), tol=math.inf)
            assert abs(x.value - y.value) <= 1e-12 * abs(y.value), (name, n, x, y)


def test_the_cohort_private_sums_agree_with_the_taylor_form(case, monkeypatch):
    """private_at_depth takes each line's integral and tail in one C pass per
    depth, from the kernel's exponent in C, with no Taylor series in the
    depth: no polygammas and no matrix per power. The Taylor form it replaced
    must give the same values to rounding, and both must hold against the
    enumerated truth."""
    import BEAM.analytics.exact_occupancy as eo_mod
    m, p, _ = case
    rng = np.random.default_rng(3)
    depths = rng.integers(100, 100_000, size=60).astype(float)
    T = float(depths.sum())
    fused = ExactOccupancy(m, rel_tol=1e-10)
    taylor = ExactOccupancy(m, rel_tol=1e-10)
    taylor._fused = False
    called = []
    real_pg = eo_mod._polygammas
    monkeypatch.setattr(eo_mod, "_polygammas",
                        lambda *a: called.append(1) or real_pg(*a))
    a = fused.private_at_depth(depths, total_depth=T)["private"]
    assert not called, "the fused path took the Taylor series"
    b = taylor.private_at_depth(depths, total_depth=T)["private"]
    assert called, "the comparison never reached the Taylor path"
    for d, x, y in zip(depths, a, b):
        truth = float(np.sum(_seen(p, d) * _miss(p, T - d)))
        _held(x, truth, tol=1e-9)
        _held(y, truth, tol=1e-9)
        assert abs(x.value - y.value) <= 1e-12 * abs(y.value), (d, x, y)


def test_the_cache_lock_uses_windows_locking_where_there_is_no_flock(tmp_path, monkeypatch):
    """On Windows there is no fcntl; the lock file is locked with msvcrt instead."""
    import sys
    import types
    from BEAM.analytics.exact_occupancy import _locked
    calls = []
    fake = types.SimpleNamespace(LK_LOCK=1, LK_UNLCK=0,
                                 locking=lambda fd, mode, n: calls.append((mode, n)))
    monkeypatch.setitem(sys.modules, "fcntl", None)       # import fcntl fails
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    with _locked(tmp_path / "lines.npz"):
        assert calls == [(fake.LK_LOCK, 1)]
    assert calls == [(fake.LK_LOCK, 1), (fake.LK_UNLCK, 1)]
    assert (tmp_path / "lines.npz.lock").exists()


# ── the spread of richness ───────────────────────────────────────

def _variance_direct(p, n):
    """Var[distinct sequences in n draws], from every probability, in long double.

    The pair sum over s != t, (1-p_s-p_t)^n - (1-p_s)^n (1-p_t)^n, expanded in
    powers of p_s p_t: R(2n) - R(n) + sum_k (-1)^k [F_k(n)^2 / C(n,k)
    - C(n,k) F_2k(2n) / C(2n,2k)], with F_k(m) = C(m,k) sum_s p^k (1-p)^(m-k).
    `test_the_reference_variance_is_the_defining_pair_sum` holds it to the
    pair sum itself.
    """
    p = p[p > 0]
    n = int(n)

    def r(m):
        return np.sum(_seen(p, m))

    def f(m, k):
        logs = _log_binom(m, k) + k * np.log(p) + (m - k) * np.log1p(-p)
        return np.sum(np.exp(logs))

    v = r(2 * n) - r(n)
    for k in range(1, min(n, 60) + 1):
        lc = _log_binom(n, k)
        term = f(n, k) ** 2 / np.exp(np.longdouble(lc)) - np.exp(
            np.longdouble(lc - _log_binom(2 * n, 2 * k))) * f(2 * n, 2 * k)
        v += (-1) ** k * term
    return float(v)


def test_the_reference_variance_is_the_defining_pair_sum():
    rng = np.random.default_rng(3)
    for size, n in [(5, 3), (40, 10), (200, 25), (300, 70)]:
        p = rng.dirichlet(np.full(size, 0.3)).astype(np.longdouble)
        miss = _miss(p, n)
        with np.errstate(invalid="ignore"):      # the diagonal, 2 p_s > 1, is zeroed
            pair = _miss(p[:, None] + p[None, :], n) - miss[:, None] * miss[None, :]
        np.fill_diagonal(pair, 0)
        direct = float(np.sum(miss - miss * miss) + np.sum(pair))
        assert _variance_direct(p, n) == pytest.approx(direct, rel=1e-12)


@pytest.mark.parametrize("n", [2, 7, 60, 1000, 10_000, 100_000, 1_000_000])
def test_richness_variance(case, n):
    m, p, occ = case
    _held(occ.richness_variance(n), _variance_direct(p, n), tol=1e-8)


def test_richness_sd_is_the_square_root_with_its_own_bound(case):
    m, p, occ = case
    exact = math.sqrt(_variance_direct(p, 1000))
    _held(occ.richness_sd(1000), exact, tol=1e-8)


def test_one_draw_or_none_has_no_spread(case):
    m, p, occ = case
    for n in (0, 1):
        v = occ.richness_variance(n)
        assert (v.value, v.bound) == (0.0, 0.0)


def test_richness_variance_needs_a_whole_number_of_draws(case):
    m, p, occ = case
    for bad in (2.5, -1, float("nan")):
        with pytest.raises(ValueError):
            occ.richness_variance(bad)
    # 10.0 is 10; cached lines may move the last bit, never past the bounds
    a, b = occ.richness_variance(10.0), occ.richness_variance(10)
    assert abs(a.value - b.value) <= (a.bound + b.bound) * abs(b.value)
