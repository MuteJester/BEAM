"""BeamModel — Ensemble of Dual Graph Experts.

The load-bearing tests here are the ones that check the C implementation
against an INDEPENDENT reference written directly from the specification
(`_reference.py`), rather than against itself. Anything that only checks
C against C would pass just as happily with the forward and backward context
packings swapped, which is the single most likely way to produce a model that
looks plausible and is wrong.
"""

from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest
from _reference import reference_build, reference_scores

from BEAM import BeamModel

AA = "ACDEFGHIKLMNPQRSTVWY"
J_MOTIFS = ("EQYF", "EQFF", "NTIYF", "ETQYF", "YGYTF")


def make_corpus(n=1500, seed=11, lo=11, hi=18):
    """CDR3-shaped strings of EXACTLY the requested length.

    A conserved start, a random middle, and a J-like motif at the end, which
    is the structure the two anchors are supposed to pick up. Sequences are
    built to the target length rather than to a fixed prefix plus motif, so
    that short-length fixtures really are short: getting that wrong silently
    turns a length-3 test into a length-8 one.
    """
    rng = random.Random(seed)
    seqs, abund = [], []
    for _ in range(n):
        length = rng.randint(lo, hi)
        motifs = [m for m in J_MOTIFS if len(m) + 5 <= length]
        j = rng.choice(motifs) if motifs else "F"
        head = "CASS" if length >= len(j) + 5 else "C"
        fill = length - len(head) - len(j)
        mid = "".join(rng.choice(AA) for _ in range(fill))
        seq = head + mid + j
        assert len(seq) == length, (seq, length)
        seqs.append(seq)
        abund.append(rng.randint(1, 40))
    return seqs, abund


@pytest.fixture(scope="module")
def corpus():
    return make_corpus()


@pytest.fixture(scope="module")
def model(corpus):
    seqs, abund = corpus
    return BeamModel(seqs, abundances=abund, order=3)


@pytest.fixture(scope="module")
def small():
    """A deliberately small model for the analytics that cost a full trellis
    sweep per call. Order 2 and short sequences keep the saddlepoint and the
    calibration in seconds rather than minutes, without changing what is being
    checked."""
    seqs, abund = make_corpus(n=600, seed=31, lo=6, hi=9)
    return BeamModel(seqs, abundances=abund, order=2, max_length=10)


# ── Agreement with the independent reference ─────────────────────

@pytest.mark.parametrize("order", [2, 3, 4])
def test_scores_match_reference(corpus, order):
    """The whole pipeline — counting, hierarchical backoff, mixing, the
    partition function — reproduced from the specification independently."""
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=order)
    probe = seqs[:40] + ["CASSLGQGAEQYF", "CASSLGQGAEQYW"]
    probe = [s for s in probe if 11 <= len(s) <= 18]

    ref = reference_build(seqs, abund, order=order)
    want = reference_scores(ref, probe)
    got = np.array([m.log_pseq(s) for s in probe])

    assert np.isfinite(got).all()
    assert np.abs(got - want).max() < 1e-9, (
        f"order {order}: worst deviation {np.abs(got - want).max():.3e}")


def test_partition_matches_reference(corpus):
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=3)
    ref = reference_build(seqs, abund, order=3)
    for length in range(11, 19):
        assert abs(m.log_partition(length) - float(ref.logZ(length))) < 1e-9


def test_path_count_matches_reference(corpus):
    """Support size over the trellis, against the reference DP."""
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2)
    ref = reference_build(seqs, abund, order=2)
    import _reference as R
    want = R.reference_path_counts(ref, range(11, 19))
    for length in range(11, 19):
        assert m.path_count(length) == pytest.approx(want[length], rel=1e-12)


# ── Internal consistency of the DAG ──────────────────────────────

def test_partition_normalises_by_enumeration():
    """The sharpest check available: sum P(s|L) over EVERY sequence of a short
    length must be exactly 1. If the partition function or any context packing
    is wrong, this cannot come out right by accident."""
    seqs, abund = make_corpus(n=400, seed=5, lo=3, hi=4)
    m = BeamModel(seqs, abundances=abund, order=2, max_length=6)
    import itertools
    for length in (3, 4):
        total = 0.0
        logpi = m.log_length_prior(length)
        for t in itertools.product(AA, repeat=length):
            lp = m.log_pseq("".join(t))
            if math.isfinite(lp):
                total += math.exp(lp - logpi)
        assert total == pytest.approx(1.0, abs=1e-9), (
            f"L={length}: sum P(s|L) = {total!r}")


def test_length_prior_normalises(model):
    assert model.length_pmf().sum() == pytest.approx(1.0, abs=1e-12)


def test_hill_number_one_is_exp_entropy(model):
    """`hill_number(1)` is a special case that CALLS the entropy DP, so this
    identity is true by construction and cannot fail. It is kept only as a
    guard against the special case being rewired to something else; the checks
    with teeth are the two below."""
    assert model.hill_number(1.0) == pytest.approx(math.exp(model.entropy()),
                                                   rel=1e-9)


def test_hill_number_one_matches_the_limit_from_either_side(model):
    """D1 must be the alpha -> 1 limit of the GENERAL Mellin formula, which is
    a different code path from the entropy DP the alpha == 1 branch shortcuts
    to. If the entropy DP and the tilted sweep disagreed, this would catch it
    and `test_hill_number_one_is_exp_entropy` would not."""
    h = model.entropy()
    eps = 1e-3
    up = model.log_mellin(1.0 + eps) / (1.0 - (1.0 + eps))
    down = model.log_mellin(1.0 - eps) / (1.0 - (1.0 - eps))
    # ln D_(1+e) = H - e*Var/2 + O(e^2), so the two straddle H and their mean
    # cancels the linear term. Both facts are checked: the bracket catches a
    # sign error, the mean catches a scale error.
    assert min(up, down) < h < max(up, down), (up, h, down)
    # rel 1e-5, not tighter: the O(e^2) term of the expansion is ~1e-6 here.
    assert 0.5 * (up + down) == pytest.approx(h, rel=1e-5)
    assert math.log(model.hill_number(1.0)) == pytest.approx(h, rel=1e-12)


def test_entropy_matches_brute_force():
    """The entropy DP against an exhaustive sum over every short sequence.

    `test_hill_number_one_is_exp_entropy` compares the entropy with itself;
    this compares it with `-sum P ln P` computed one sequence at a time."""
    import itertools
    seqs, abund = make_corpus(n=400, seed=5, lo=3, hi=4)
    m = BeamModel(seqs, abundances=abund, order=2, max_length=6)
    h = 0.0
    for length in (3, 4):
        for t in itertools.product(AA, repeat=length):
            lp = m.log_pseq("".join(t))
            if math.isfinite(lp):
                h -= math.exp(lp) * lp
    assert m.entropy() == pytest.approx(h, rel=1e-9)


def test_hill_numbers_are_ordered(model):
    """Hill numbers are non-increasing in alpha, always."""
    d = [model.hill_number(a) for a in (0.0, 0.5, 1.0, 1.5, 2.0)]
    for a, b in zip(d, d[1:]):
        assert b <= a * (1 + 1e-9), d


def test_support_is_all_of_sequence_space():
    """Documented, not desired, and the reason `path_count` carries a warning.

    Hierarchical backoff leaves no cell at zero, so nothing is dead and the
    support is every peptide of every producible length. That is a property of
    the smoother, not of the grammar, and the number must never be quoted as a
    diversity."""
    seqs, abund = make_corpus(n=300, seed=9, lo=5, hi=6)
    m = BeamModel(seqs, abundances=abund, order=2, max_length=8)
    assert m.path_count(5) == pytest.approx(20.0 ** 5)
    assert m.path_count(6) == pytest.approx(20.0 ** 6)



def test_spectrum_totals(corpus):
    """The counting measure integrates to the support size and the generated
    measure to 1. Both are checked against a different code path."""
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2)

    counting = m.spectrum(measure="counting", bins=512)
    generated = m.spectrum(measure="generated", bins=512,
                           spacing=counting["spacing"])

    assert counting["total"] == pytest.approx(m.path_count(), rel=1e-9)
    assert generated["total"] == pytest.approx(1.0, rel=1e-9)


def test_spectrum_identity(corpus):
    """`sum over bins of multiplicity * mean probability` must be 1.

    This is the check that catches per-bin error, which totals alone cannot
    see: a spectrum can integrate to the right value while being wrong bin by
    bin. The per-bin probability is `generated / counting`, the mass-weighted
    mean, NOT `exp(-surprisal)` at the bin edge, which carries a systematic
    bias of `exp(spacing)` across the bin.
    """
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2)
    c = m.spectrum(measure="counting", bins=512)
    g = m.spectrum(measure="generated", bins=512, spacing=c["spacing"])
    keep = (c["weights"] > 0) & (g["weights"] > 0)
    p = g["weights"][keep] / c["weights"][keep]
    assert float((c["weights"][keep] * p).sum()) == pytest.approx(1.0, rel=1e-9)
    assert p.max() <= 1.0 + 1e-12


def test_spectrum_mean_is_exact_but_the_shape_is_not(corpus):
    """The documented guarantee is precise, and it is not "every bin is right".

    Linear splitting happens at EVERY layer, not once at the end, so a
    sequence's mass diffuses over several bins rather than the two adjacent to
    its exact surprisal. What survives that exactly is the total and the FIRST
    moment; the second is inflated. Both halves are pinned here, because a
    future change that made the spectrum exact bin-by-bin would be an
    improvement, and one that broke the mean would not be caught by
    `test_spectrum_totals`."""
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2)
    lm = m.length_marginals()
    length = int(np.argmax(lm["generated"]))

    s = m.spectrum(measure="generated", length=length, bins=512)
    x, w = s["surprisal"], s["weights"]
    total = w.sum()
    mean_grid = float((x * w).sum() / total)
    var_grid = float((x ** 2 * w).sum() / total) - mean_grid ** 2

    assert mean_grid == pytest.approx(lm["mean_surprisal"][length], rel=1e-12)
    # Each split spreads a point across two adjacent bins, adding at most
    # `spacing**2 / 6` of variance, and there are `length + 1` of them (one per
    # layer plus the sink). The inflation is therefore bounded by a quantity
    # the test can compute rather than by a percentage picked to fit: it must
    # be positive, and it must fit inside that budget.
    inflation = var_grid - lm["variance"][length]
    budget = (length + 1) * s["spacing"] ** 2 / 6.0
    assert 0.0 < inflation <= budget, (inflation, budget)


def test_spectrum_per_length_sums_to_pooled(corpus):
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2)
    pooled = m.spectrum(measure="generated", bins=256)
    acc = np.zeros(256)
    for length in range(1, 28):
        if not math.isfinite(m.log_length_prior(length)):
            continue
        acc += m.spectrum(measure="generated", length=length, bins=256,
                          spacing=pooled["spacing"])["weights"]
    assert np.abs(acc - pooled["weights"]).max() < 1e-12


def test_accumulator_max_bounds_the_grid(corpus):
    """The grid must be sized by the accumulator, not by the surprisal: the DP
    accumulates monotonically and only applies a per-length constant at the
    sink. A grid sized to the surprisal silently drops mass off the top when
    that constant is negative."""
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2)
    hi = m.accumulator_max()
    assert hi > 0
    s = m.spectrum(measure="counting", bins=512)
    occupied = np.flatnonzero(s["weights"] > 0)
    assert occupied.size > 0
    # Linear splitting can push a sliver one bin past the exact value per
    # layer, so the occupied range may run up to max_length+1 bins beyond the
    # bound. What must NOT happen is mass reaching the top, which would mean
    # it fell off the end.
    slack = (int(m.config["max_length"]) + 1) * s["spacing"]
    assert s["surprisal"][occupied[-1]] <= hi + slack
    assert s["weights"][-8:].sum() == 0.0, "mass reached the top of the grid"


# ── Scoring behaviour ────────────────────────────────────────────

def test_batch_matches_scalar(model, corpus):
    seqs, _ = corpus
    batch = model.log_pseq_batch(seqs[:200])
    for i in range(200):
        assert batch[i] == pytest.approx(model.log_pseq(seqs[i]), abs=1e-12)


def test_explain_sums_to_the_score(model, corpus):
    """The per-residue decomposition must reconstruct the score exactly, so it
    can be trusted as an explanation rather than an illustration."""
    seqs, _ = corpus
    cfg = model.config
    wf, wb = cfg["weights"]
    for s in seqs[:20]:
        e = model.explain(s)
        raw = sum(wf * f + wb * b for f, b in zip(e["forward"], e["backward"]))
        want = raw + model.log_length_prior(len(s)) - model.log_partition(len(s))
        assert want == pytest.approx(e["log_pseq"], abs=1e-9)


def test_c_terminus_expert_is_doing_work(model):
    """Swapping the terminal residue for one no J gene produces must cost
    orders of magnitude. A single left-to-right chain cannot see this."""
    good = model.log_pseq("CASSLGQGAEQYF")
    bad = model.log_pseq("CASSLGQGAEQYW")
    assert math.isfinite(good) and math.isfinite(bad)
    assert good - bad > math.log(1000), (good, bad)


def test_rejects_non_amino_acid(model):
    assert model.log_pseq("CASSXQYF") == -math.inf
    assert model.log_pseq("") == -math.inf
    assert model.log_pseq("C" * 40) == -math.inf


def test_unobserved_length_is_impossible(model):
    """A length the model never saw has prior zero, so its sequences have
    probability zero. The prototype floors the prior at 1e-300 instead; that
    differs only outside the training length support, which the foundation
    model covers completely."""
    assert model.log_pseq("CASSF") == -math.inf


# ── Generation ───────────────────────────────────────────────────

def test_sample_is_reproducible(model):
    assert model.sample(20, seed=7) == model.sample(20, seed=7)
    assert model.sample(20, seed=7) != model.sample(20, seed=8)


def test_sample_respects_the_model(model):
    """Draws must be scoreable and follow the length prior."""
    draws = model.sample(400, seed=3)
    assert all(math.isfinite(model.log_pseq(s)) for s in draws)
    pmf = model.length_pmf()
    for s in draws:
        assert pmf[len(s)] > 0


def test_sample_fixed_length(model):
    draws = model.sample(50, length=13, seed=2)
    assert all(len(s) == 13 for s in draws)


def test_sample_frequencies_match_probabilities():
    """Exact trellis sampling, so empirical frequencies must converge to the
    model's own probabilities."""
    seqs, abund = make_corpus(n=300, seed=4, lo=4, hi=4)
    m = BeamModel(seqs, abundances=abund, order=2, max_length=6)
    draws = m.sample(60_000, length=4, seed=5)
    from collections import Counter
    obs = Counter(draws)
    top = [s for s, _ in obs.most_common(10)]
    for s in top:
        want = math.exp(m.log_pseq(s) - m.log_length_prior(4))
        got = obs[s] / len(draws)
        assert got == pytest.approx(want, rel=0.15), (s, got, want)


# ── Counts are additive ──────────────────────────────────────────

def test_subtract_is_additive_inverse(corpus):
    """Holding a cohort out of a foundation model must be a subtraction, not a
    recount of the corpus. That only works if counts are additive."""
    seqs, abund = corpus
    a_seqs, a_ab = seqs[:1000], abund[:1000]
    b_seqs, b_ab = seqs[1000:], abund[1000:]

    full = BeamModel(seqs, abundances=abund, order=2)
    part = BeamModel(b_seqs, abundances=b_ab, order=2)
    only_a = BeamModel(a_seqs, abundances=a_ab, order=2)

    full.subtract(part)
    probe = [s for s in a_seqs[:50]]
    got = np.array([full.log_pseq(s) for s in probe])
    want = np.array([only_a.log_pseq(s) for s in probe])
    assert np.abs(got - want).max() < 1e-9


# ── Persistence ──────────────────────────────────────────────────

def test_round_trip(tmp_path, corpus):
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2)
    path = tmp_path / "model.edge"
    m.save(path)
    back = BeamModel.from_file(path)

    assert back.config == m.config
    assert back.n_observed == m.n_observed
    assert back.total_weight == pytest.approx(m.total_weight)
    for s in seqs[:50]:
        assert back.log_pseq(s) == pytest.approx(m.log_pseq(s), abs=1e-12)


def test_corrupt_file_is_rejected(tmp_path, corpus):
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2)
    path = tmp_path / "model.edge"
    m.save(path)
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(OSError):
        BeamModel.from_file(path)


# ── Configuration guards ─────────────────────────────────────────

def test_coefficients_must_sum_to_one(corpus):
    """The sum is a cross-length calibration knob, not a free parameter."""
    seqs, abund = corpus
    with pytest.raises(ValueError, match="sum to 1"):
        BeamModel(seqs, abundances=abund, coefficients=(0.7, 0.3, 0.15))


def test_rejects_bad_order(corpus):
    seqs, abund = corpus
    with pytest.raises(ValueError):
        BeamModel(seqs, abundances=abund, order=5)


def test_requires_sequences():
    with pytest.raises(ValueError):
        BeamModel([])


def test_skips_unusable_sequences():
    m = BeamModel(["CASSQYF", "CASSXYF", "CASSEQYF"], order=2)
    assert m.n_observed == 2
    assert m.n_skipped == 1


def test_compact_frees_counts(corpus):
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2)
    before = m.memory_bytes
    score = m.log_pseq(seqs[0])
    m.compact()
    assert m.memory_bytes < before
    assert m.log_pseq(seqs[0]) == pytest.approx(score, abs=1e-12)


# ═══════════════════════════════════════════════════════════════
# Mellin transform and diversity
# ═══════════════════════════════════════════════════════════════

def test_log_mellin_at_zero_is_the_support(model):
    """q=0 counts every supported sequence once, so it must reproduce the
    path-count DP, which is a completely different code path."""
    assert model.log_mellin(0.0) == pytest.approx(math.log(model.path_count()),
                                                  rel=1e-9)


def test_log_mellin_at_one_is_zero(model):
    """q=1 is the total probability, which is 1 by construction. This is the
    cheapest check that the tilted sweep and the normalisation agree."""
    assert model.log_mellin(1.0) == pytest.approx(0.0, abs=1e-9)


def test_log_mellin_matches_brute_force():
    """Against an exhaustive sum over every length-3 and length-4 sequence."""
    import itertools
    seqs, abund = make_corpus(n=400, seed=5, lo=3, hi=4)
    m = BeamModel(seqs, abundances=abund, order=2, max_length=6)
    for q in (0.0, 0.5, 1.0, 2.0):
        total = 0.0
        for L in (3, 4):
            for t in itertools.product(AA, repeat=L):
                lp = m.log_pseq("".join(t))
                if math.isfinite(lp):
                    total += math.exp(q * lp)
        assert m.log_mellin(q) == pytest.approx(math.log(total), rel=1e-9), q


def test_hill_number_matches_mellin(model):
    """D_alpha = (sum P^alpha)^(1/(1-alpha)), computed two ways."""
    for a in (0.5, 2.0, 3.0):
        want = math.exp(model.log_mellin(a) / (1.0 - a))
        assert model.hill_number(a) == pytest.approx(want, rel=1e-9), a


def test_diversity_profile_is_ordered(model):
    d = model.diversity_profile()
    assert d["D0"] >= d["D1"] >= d["D2"] > 0
    assert model.effective_diversity() == pytest.approx(d["D1"], rel=1e-12)


def test_hill_curve_shape(model):
    c = model.hill_curve(orders=[0.0, 1.0, 2.0])
    assert list(c["orders"]) == [0.0, 1.0, 2.0]
    assert len(c["hill"]) == 3
    assert np.all(np.diff(c["hill"]) <= 1e-9 * c["hill"][0])


def test_power_sum_matches_log_mellin(model):
    assert model.power_sum(2.0) == pytest.approx(math.exp(model.log_mellin(2.0)),
                                                 rel=1e-12)


# ═══════════════════════════════════════════════════════════════
# Surprisal moments
# ═══════════════════════════════════════════════════════════════

def test_surprisal_mean_is_the_entropy(model):
    """At q=1 the mean surprisal IS the entropy, computed by a different DP."""
    mu = model.surprisal_moments(q=1.0, order=1)
    assert mu[0] == pytest.approx(1.0, abs=1e-12)
    assert mu[1] == pytest.approx(model.entropy(), rel=1e-9)


def test_surprisal_moments_match_brute_force():
    """Exhaustive: every length-3 and length-4 sequence, weighted by its own
    probability. This pins the jet DP against a direct sum."""
    import itertools
    seqs, abund = make_corpus(n=400, seed=5, lo=3, hi=4)
    m = BeamModel(seqs, abundances=abund, order=2, max_length=6)

    for q in (1.0, 2.0):
        wsum = 0.0
        acc = np.zeros(5)
        for L in (3, 4):
            for t in itertools.product(AA, repeat=L):
                lp = m.log_pseq("".join(t))
                if not math.isfinite(lp):
                    continue
                w = math.exp(q * lp)
                u = -lp
                wsum += w
                for k in range(5):
                    acc[k] += w * u ** k
        want = acc / wsum
        got = m.surprisal_moments(q=q, order=4)
        for k in range(5):
            assert got[k] == pytest.approx(want[k], rel=1e-8), (q, k)


def test_surprisal_cumulants_match_moments(model):
    """Cumulants derived from the moments by the standard recursion."""
    mu = model.surprisal_moments(q=1.0, order=4)
    k = model.surprisal_cumulants(q=1.0, order=4)
    assert k[0] == pytest.approx(mu[1], rel=1e-9)                      # mean
    assert k[1] == pytest.approx(mu[2] - mu[1] ** 2, rel=1e-8)         # variance
    third = mu[3] - 3 * mu[1] * mu[2] + 2 * mu[1] ** 3
    assert k[2] == pytest.approx(third, rel=1e-7)


def test_surprisal_variance_matches_sampling(model):
    """The exact variance must agree with the sample variance of draws from
    the model itself, which is an independent route to the same number."""
    draws = model.sample(30000, seed=11)
    u = np.array([-model.log_pseq(s) for s in draws])
    stats = model.surprisal_stats(q=1.0)
    assert u.mean() == pytest.approx(stats["mean"], rel=0.02)
    assert u.var() == pytest.approx(stats["variance"], rel=0.15)


def test_cgf_is_the_shifted_mellin(model):
    for t in (-0.5, 0.0, 0.5):
        assert model.cgf(t) == pytest.approx(model.log_mellin(1.0 + t), rel=1e-12)


def test_cgf_derivative_at_zero_is_minus_entropy(model):
    """d/dt ln E[exp(t ln P)] at t=0 is E[ln P] = -H. A finite difference on
    the Mellin transform must reproduce the entropy DP."""
    h = 1e-4
    slope = (model.cgf(h) - model.cgf(-h)) / (2 * h)
    assert slope == pytest.approx(-model.entropy(), rel=1e-4)


# ═══════════════════════════════════════════════════════════════
# Surprisal range
# ═══════════════════════════════════════════════════════════════

def test_surprisal_range_brackets_every_sequence():
    """Exhaustive: no sequence may fall outside the reported range, and both
    ends must be attained."""
    import itertools
    seqs, abund = make_corpus(n=400, seed=5, lo=3, hi=4)
    m = BeamModel(seqs, abundances=abund, order=2, max_length=6)
    lo, hi = m.surprisal_range()

    seen_lo, seen_hi = math.inf, -math.inf
    for L in (3, 4):
        for t in itertools.product(AA, repeat=L):
            lp = m.log_pseq("".join(t))
            if not math.isfinite(lp):
                continue
            u = -lp
            seen_lo = min(seen_lo, u)
            seen_hi = max(seen_hi, u)
    assert lo == pytest.approx(seen_lo, rel=1e-9)
    assert hi == pytest.approx(seen_hi, rel=1e-9)


def test_dynamic_range_is_positive(model):
    lo, hi = model.surprisal_range()
    assert hi > lo > 0
    assert model.dynamic_range() == pytest.approx((hi - lo) / math.log(10), rel=1e-12)


def test_top_sequence_attains_the_minimum_surprisal(model):
    """The most probable sequence must sit exactly at the low end of the range,
    which links the top-k search and the min-plus sweep."""
    lo, _ = model.surprisal_range()
    (_, best_lp), = model.top_k_sequences(1)
    assert -best_lp == pytest.approx(lo, rel=1e-9)


# ═══════════════════════════════════════════════════════════════
# Per-length marginals
# ═══════════════════════════════════════════════════════════════

def test_length_marginals_decompose_the_totals(model):
    lm = model.length_marginals()
    assert lm["counting"].sum() == pytest.approx(model.path_count(), rel=1e-9)
    assert lm["generated"].sum() == pytest.approx(1.0, rel=1e-12)


def test_length_marginals_generated_is_the_length_prior(model):
    lm = model.length_marginals()
    assert np.abs(lm["generated"] - model.length_pmf()).max() < 1e-12


def test_length_mean_surprisal_reconstructs_the_entropy(model):
    """The entropy is the length-weighted mean of the per-length means. If the
    per-length numbers were wrong this identity would not hold."""
    lm = model.length_marginals()
    h = float((lm["generated"] * lm["mean_surprisal"]).sum())
    assert h == pytest.approx(model.entropy(), rel=1e-9)


def test_length_mean_surprisal_matches_sampling(model):
    """Draw at a fixed length and compare the empirical mean surprisal."""
    lm = model.length_marginals()
    L = int(np.argmax(lm["generated"]))
    draws = model.sample(20000, length=L, seed=5)
    u = np.array([-model.log_pseq(s) for s in draws])
    assert u.mean() == pytest.approx(lm["mean_surprisal"][L], rel=0.02)


# ═══════════════════════════════════════════════════════════════
# Top-k
# ═══════════════════════════════════════════════════════════════

def test_top_k_is_sorted_and_scores_correctly(model):
    top = model.top_k_sequences(25)
    assert len(top) == 25
    lps = [lp for _, lp in top]
    assert lps == sorted(lps, reverse=True)
    for seq, lp in top:
        assert model.log_pseq(seq) == pytest.approx(lp, abs=1e-9)


def test_top_k_is_exact_against_exhaustive_enumeration():
    """The claim is that this is exact, not a beam, so it is checked against a
    full enumeration of a short length. A beam would silently miss sequences
    whose prefix looked unpromising."""
    import itertools
    seqs, abund = make_corpus(n=400, seed=5, lo=3, hi=4)
    m = BeamModel(seqs, abundances=abund, order=2, max_length=6)

    every = []
    for L in (3, 4):
        for t in itertools.product(AA, repeat=L):
            s = "".join(t)
            lp = m.log_pseq(s)
            if math.isfinite(lp):
                every.append((s, lp))
    every.sort(key=lambda kv: -kv[1])

    got = m.top_k_sequences(30)
    assert len(got) == 30
    for (gs, glp), (ws, wlp) in zip(got, every[:30]):
        assert glp == pytest.approx(wlp, abs=1e-9)
        assert gs == ws


def test_top_k_at_a_fixed_length(model):
    top = model.top_k_sequences(10, length=13)
    assert len(top) == 10
    assert all(len(s) == 13 for s, _ in top)


def test_top_k_beats_random_draws(model):
    """The best sequence must be at least as probable as anything sampled."""
    (_, best), = model.top_k_sequences(1)
    draws = model.sample(2000, seed=3)
    assert best >= max(model.log_pseq(s) for s in draws) - 1e-9


def test_top_k_more_than_exists():
    """Asking for more than the model can produce returns what there is.

    With every table cell positive the support is 20**L, so the only way to
    have fewer than 500 sequences is a genuinely short model: length 2 gives
    exactly 400.
    """
    m = BeamModel(["CA", "CF", "SA"], order=2, max_length=3)
    top = m.top_k_sequences(500)
    assert len(top) == 400, "20**2 sequences exist and all should come back"
    assert all(math.isfinite(lp) for _, lp in top)
    assert len({s for s, _ in top}) == 400, "and they must be distinct"


# ═══════════════════════════════════════════════════════════════
# Count addition
# ═══════════════════════════════════════════════════════════════

def test_add_is_the_inverse_of_subtract(corpus):
    seqs, abund = corpus
    a = BeamModel(seqs[:800], abundances=abund[:800], order=2)
    b = BeamModel(seqs[800:], abundances=abund[800:], order=2)
    whole = BeamModel(seqs, abundances=abund, order=2)

    a.add(b)
    probe = seqs[:60]
    assert np.abs(np.array([a.log_pseq(s) for s in probe])
                  - np.array([whole.log_pseq(s) for s in probe])).max() < 1e-9

    a.subtract(b)
    only_a = BeamModel(seqs[:800], abundances=abund[:800], order=2)
    assert np.abs(np.array([a.log_pseq(s) for s in probe])
                  - np.array([only_a.log_pseq(s) for s in probe])).max() < 1e-9


def test_add_accumulates_bookkeeping(corpus):
    seqs, abund = corpus
    a = BeamModel(seqs[:500], abundances=abund[:500], order=2)
    b = BeamModel(seqs[500:900], abundances=abund[500:900], order=2)
    wa, na = a.total_weight, a.n_observed
    a.add(b)
    assert a.n_observed == na + b.n_observed
    assert a.total_weight == pytest.approx(wa + b.total_weight)


# ═══════════════════════════════════════════════════════════════
# Pruning renormalises the length prior
# ═══════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════
# Pruning frontier
# ═══════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════
# Topology
# ═══════════════════════════════════════════════════════════════

def test_topology_counts_live_structure(corpus):
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2)
    t = m.topology(13)
    assert m.is_dag
    assert t["n_nodes"] > 0 and t["n_edges"] > 0
    assert len(t["nodes_by_layer"]) == 14
    assert t["max_layer_nodes"] >= t["mean_layer_nodes"] > 0
    assert t["nodes_by_layer"][0] == 1, "the walk starts in one state"


def test_topology_edges_bound_the_paths(corpus):
    """Every path uses exactly one edge per layer, so the support size is
    bounded above by the product of the per-layer edge counts and below by the
    smallest of them. The previous version of this test asserted only
    `path_count >= 1` and `n_edges >= 13`, which the docstring's claim does not
    follow from and which a badly broken trellis would still satisfy."""
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2)
    length = 13
    t = m.topology(length)
    pc = m.path_count(length)
    layers = t["nodes_by_layer"]

    assert pc >= 1
    assert len(layers) == length + 1
    # A path visits one node per layer, so the support cannot exceed the
    # product of the per-layer live-node counts.
    log_bound = float(np.log(layers[1:].astype(float)).sum())
    assert math.log(pc) <= log_bound + 1e-9, (pc, log_bound)
    # ... and every live node at the last layer terminates at least one path.
    assert pc >= layers[-1]
    # Edges are per (layer, node, residue), so they bound the paths too.
    assert t["n_edges"] >= t["n_nodes"] - 1
    assert 0.0 < t["density"] <= 1.0



# ═══════════════════════════════════════════════════════════════
# Surprisal by depth
# ═══════════════════════════════════════════════════════════════

def test_surprisal_by_depth_reconstructs_the_mean(model):
    """The decomposition must add up EXACTLY to the mean surprisal.

    This is the whole point of reporting a breakdown: if the layer terms, the
    sink term and the per-length normalisation did not sum to the entropy,
    something would be lost or double-counted and the picture would mislead.
    """
    d = model.surprisal_by_depth(q=1.0)
    # rel 1e-9, not the 1e-6 this used to allow: the forward-backward DP is
    # exact to ~1e-13, so a loose tolerance here would hide a real regression.
    assert d["total"] == pytest.approx(model.entropy(), rel=1e-9)
    # `total` is defined in Python as this sum, so on its own the identity
    # below is tautological; it is kept as a guard on the packing. The check
    # with teeth is that the same identity holds under a TILT, where the
    # length weights, the posterior and the constant all change together.
    assert d["by_depth"].sum() + d["length_constant"] == pytest.approx(
        d["total"], rel=1e-12)
    tilted = model.surprisal_by_depth(q=2.0)
    want = float(model.surprisal_moments(q=2.0, order=1)[1])
    assert tilted["total"] == pytest.approx(want, rel=1e-9)
    assert tilted["total"] != pytest.approx(d["total"], rel=1e-3), (
        "q=2 must actually tilt; if it did not, the q argument is ignored")


def test_surprisal_by_depth_finds_the_junction(model):
    """The conserved start costs almost nothing and the middle carries the
    information. If the two anchors were confused this would invert."""
    d = model.surprisal_by_depth(q=1.0)["by_depth"]
    assert d[0] < 0.05, "position 0 is conserved and must be nearly free"
    assert d[5:9].mean() > d[0] * 5, "the junction must dominate"


# ═══════════════════════════════════════════════════════════════
# Length mixture and saddlepoint
# ═══════════════════════════════════════════════════════════════

def test_surprisal_distribution_components(small):
    d = small.surprisal_distribution()
    assert d["weight"].sum() == pytest.approx(1.0, rel=1e-12)
    # The mixture mean must be the entropy.
    assert float((d["weight"] * d["mean"]).sum()) == pytest.approx(
        small.entropy(), rel=1e-9)
    # Law of total variance against the exact global variance.
    var = small.surprisal_stats()["variance"]
    within = float((d["weight"] * d["variance"]).sum())
    between = float((d["weight"] * (d["mean"] - small.entropy()) ** 2).sum())
    assert within + between == pytest.approx(var, rel=1e-6)


def test_surprisal_cdf_is_a_distribution(small):
    lo, hi = small.surprisal_range()
    xs = np.linspace(lo, hi, 40)
    c = small.surprisal_cdf(xs)
    assert np.all(np.diff(c) >= -1e-12), "must be non-decreasing"
    assert 0.0 <= c[0] <= 0.2
    assert c[-1] > 0.95


def test_saddlepoint_reports_non_convergence(small):
    """A point beyond the reachable tilt must be flagged, not answered from a
    tilt that does not solve the saddlepoint equation. Silently returning a
    confident wrong number is the failure mode this guards."""
    lo, hi = small.surprisal_range()
    deep = small.saddlepoint(lo + 1e-6)
    assert deep["converged"] is False, "the flag is the point of this test"
    assert deep["cdf"] == pytest.approx(0.0, abs=1e-9)
    assert deep["pdf"] == pytest.approx(0.0, abs=1e-9)
    far = small.saddlepoint(hi - 1e-6)
    assert far["converged"] is False
    assert far["cdf"] == pytest.approx(1.0, abs=1e-9)
    # and inside the bulk it must converge, and be flagged as having done so
    mu = float(small.surprisal_moments(order=1)[1])
    bulk = small.saddlepoint(mu)
    assert bulk["converged"] is True
    assert 0.0 < bulk["cdf"] < 1.0 and bulk["pdf"] > 0.0


def test_saddlepoint_solves_the_right_equation(small):
    """At the solved t, K'(t) must equal the requested surprisal. This is the
    defining property, and it uses the exact CGF derivatives."""
    mu = float(small.surprisal_moments(order=1)[1])
    for x in (mu - 0.5, mu + 0.8):
        sp = small.saddlepoint(x)
        assert sp["converged"]
        jet = small.cgf_jet(sp["t"])
        assert jet["K1"] == pytest.approx(x, abs=1e-6)


def test_saddlepoint_cdf_is_monotone_and_bracketed(small):
    """Sampled across the BULK of the distribution, not at its extremes: a
    saddlepoint arbitrarily close to the minimum surprisal needs an unbounded
    tilt, and the solve is deliberately clamped rather than allowed to run
    into a degenerate sweep."""
    mu = float(small.surprisal_moments(order=1)[1])
    sd = math.sqrt(small.surprisal_stats()["variance"])
    xs = np.linspace(mu - 1.5 * sd, mu + 1.5 * sd, 7)
    sp = small.saddlepoint(xs)
    assert np.all(np.diff(sp["cdf"]) >= -1e-9)
    assert np.all((sp["cdf"] >= 0) & (sp["cdf"] <= 1))
    assert np.all(sp["pdf"] >= 0)


def test_saddlepoint_is_exactly_zero_outside_the_support(small):
    """Outside the range no sequence exists, so the answer is exact, not an
    extrapolation."""
    lo, hi = small.surprisal_range()
    below = small.saddlepoint(lo - 1.0)
    above = small.saddlepoint(hi + 1.0)
    assert below["cdf"] == 0.0 and below["pdf"] == 0.0
    assert above["cdf"] == 1.0


def test_saddlepoint_agrees_with_sampling(small):
    """The approximation is only in the inversion, so it should track the
    empirical distribution of draws from the model closely."""
    draws = small.sample(20000, seed=17)
    u = np.sort(np.array([-small.log_pseq(s) for s in draws]))
    for frac in (0.25, 0.5, 0.75):
        x = float(np.quantile(u, frac))
        got = small.saddlepoint(x)["cdf"]
        assert got == pytest.approx(frac, abs=0.06), (frac, got)


def test_cgf_jet_matches_the_separate_routes(small):
    """One sweep must give the same K, K' and K'' as three separate calls."""
    for t in (-0.3, 0.0, 0.4):
        jet = small.cgf_jet(t)
        q = 1.0 - t
        assert jet["K"] == pytest.approx(small.log_mellin(q), rel=1e-9)
        k = small.surprisal_cumulants(q=q, order=2)
        assert jet["K1"] == pytest.approx(k[0], rel=1e-9)
        assert jet["K2"] == pytest.approx(k[1], rel=1e-6)


# ═══════════════════════════════════════════════════════════════
# Simulation container
# ═══════════════════════════════════════════════════════════════

def test_simulate_returns_a_simulation_result(model):
    from BEAM._simulation_result import SimulationResult
    sim = model.simulate(50, seed=4)
    assert isinstance(sim, SimulationResult)
    assert len(sim.sequences) == 50
    assert len(sim.log_probs) == 50
    for s, lp, nt in zip(sim.sequences, sim.log_probs, sim.n_tokens):
        assert model.log_pseq(s) == pytest.approx(lp, abs=1e-12)
        assert nt == len(s)


def test_simulate_is_reproducible(model):
    a = model.simulate(20, seed=9)
    b = model.simulate(20, seed=9)
    assert a.sequences == b.sequences


# ═══════════════════════════════════════════════════════════════
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


def test_calibration_flags_an_implausible_sequence(small):
    """A sequence ending in a residue the corpus never terminates on must score
    as an outlier against its own length's null."""
    cal = small.calibrate(bins=1024)
    good = next(s for s in make_corpus(n=20, seed=31, lo=8, hi=8)[0])
    bad = good[:-1] + "W"
    pg = cal.pvalue(len(good), -small.log_pseq(good))
    pb = cal.pvalue(len(bad), -small.log_pseq(bad))
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
# Degenerate corpora
#
# Every one of these builds a model whose tables are almost entirely
# pseudo-count, which is exactly where an off-by-one in the smoothing or the
# length prior stops being visible in the aggregate numbers. The check is
# always the same and always exhaustive: sum P(s|L) over every sequence of
# that length must be 1.
# ═══════════════════════════════════════════════════════════════

def _sum_conditional(m, length):
    """Exhaustive sum of P(s | L) over every length-`L` peptide."""
    import itertools
    logpi = m.log_length_prior(length)
    total = 0.0
    for t in itertools.product(AA, repeat=length):
        lp = m.log_pseq("".join(t))
        if math.isfinite(lp):
            total += math.exp(lp - logpi)
    return total


def test_single_training_sequence_normalises():
    """One sequence, so every count is 1 and every table row is dominated by
    its backoff. The normalisation must still be exact."""
    m = BeamModel(["CASF"], order=2, max_length=6)
    assert m.n_observed == 1 and m.n_skipped == 0
    assert m.length_pmf()[4] == pytest.approx(1.0)
    assert _sum_conditional(m, 4) == pytest.approx(1.0, abs=1e-9)
    assert math.isfinite(m.log_pseq("CASF"))
    assert math.isfinite(m.log_pseq("WWWW")), "backoff leaves nothing dead"


def test_all_identical_sequences_normalise():
    """A degenerate empirical distribution: one sequence repeated. The counts
    concentrate on a single path and everything else lives on the backoff."""
    m = BeamModel(["CASF"] * 50, order=2, max_length=6)
    assert m.n_observed == 50
    assert _sum_conditional(m, 4) == pytest.approx(1.0, abs=1e-9)
    # The observed sequence must still be the most probable one.
    (best, _), = m.top_k_sequences(1)
    assert best == "CASF"


def test_length_one_sequences_normalise():
    """Length 1 is the shortest thing the model can represent, and the one
    where the backward expert's lag never gets a chance to settle during the
    walk: every residue's backward score is paid at the sink instead."""
    m = BeamModel(["C", "A", "S", "C"], order=2, max_length=4)
    assert m.length_pmf()[1] == pytest.approx(1.0)
    assert _sum_conditional(m, 1) == pytest.approx(1.0, abs=1e-12)
    assert m.path_count(1) == pytest.approx(20.0)
    top = m.top_k_sequences(20)
    assert len(top) == 20 and len({s for s, _ in top}) == 20
    assert all(len(s) == 1 for s, _ in top)


def test_one_length_carries_a_single_sequence():
    """A length seen exactly once, alongside a length seen many times. Its
    prior is tiny but non-zero, so it must have a partition function, appear
    in the marginals, and normalise like any other."""
    m = BeamModel(["CASF"] * 60 + ["CASSF"], order=2, max_length=6)
    pmf = m.length_pmf()
    assert pmf[5] == pytest.approx(1.0 / 61.0, rel=1e-12)
    assert math.isfinite(m.log_partition(5))
    assert _sum_conditional(m, 5) == pytest.approx(1.0, abs=1e-9)
    lm = m.length_marginals()
    assert lm["generated"][5] == pytest.approx(pmf[5], rel=1e-12)
    assert lm["counting"][5] == pytest.approx(20.0 ** 5)


def test_max_length_two_is_the_smallest_model():
    """`depth_cap` must be at least 1 and strictly below `max_length`, so 2 is
    the floor. It has to work, not merely be accepted."""
    m = BeamModel(["CA", "CF", "SA"], order=2, max_length=2)
    assert m.config["depth_cap"] == 1
    assert _sum_conditional(m, 2) == pytest.approx(1.0, abs=1e-12)
    assert m.path_count(2) == pytest.approx(400.0)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_degenerate_corpus_matches_the_reference(order):
    """The independent reference, on a corpus small enough that almost every
    table cell is pure backoff. The agreement tests elsewhere all use a corpus
    with hundreds of sequences, where a smoothing bug is diluted."""
    seqs = ["CASF", "CASSF", "CASF", "CAF"]
    m = BeamModel(seqs, order=order, max_length=6)
    ref = reference_build(seqs, order=order, max_len=6)
    probe = ["CASF", "CASSF", "CAF", "WWWW", "YYYYY"]
    got = np.array([m.log_pseq(s) for s in probe])
    want = reference_scores(ref, probe)
    assert np.abs(got - want).max() < 1e-9, (order, got, want)


# ═══════════════════════════════════════════════════════════════
# Argument validation
#
# The C entry points take these as uint8_t and uint32_t and do not populate
# the library's error-message buffer, so an out-of-range value is either
# silently truncated into a different model or surfaces as a bare "invalid
# argument". Both are checked here because both were live.
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("kwargs,field", [
    (dict(order=260), "order"),          # (uint8_t)260 == 4, the default
    (dict(order=0), "order"),
    (dict(max_length=283), "max_length"),   # (uint8_t)283 == 27
    (dict(max_length=1), "max_length"),     # depth_cap has nowhere to go
    (dict(max_length=0), "max_length"),
    (dict(depth_cap=271), "depth_cap"),     # (uint8_t)271 == 15
    (dict(depth_cap=0), "depth_cap"),
])
def test_out_of_range_configuration_is_refused(kwargs, field):
    """Truncation to uint8 must not silently build a different model."""
    base = dict(order=2, max_length=6)
    base.update(kwargs)
    with pytest.raises(ValueError, match=field):
        BeamModel(["CASF"], **base)


def test_unusable_corpus_says_why():
    """Every sequence rejected is the overwhelmingly common build failure, and
    the C layer reports it as "invalid argument"."""
    with pytest.raises(ValueError, match="survived encoding"):
        BeamModel(["XXXX", "ZZZ", ""], order=2, max_length=6)
    with pytest.raises(ValueError, match="survived encoding"):
        BeamModel(["CASF"], abundances=[0.0], order=2, max_length=6)


def test_negative_counts_are_refused(small):
    """`sample(-1)` reached the C layer as 4,294,967,295 draws, which does not
    raise: it allocates tens of gigabytes and never returns."""
    with pytest.raises(ValueError, match="non-negative"):
        small.sample(-1)
    with pytest.raises(ValueError, match="non-negative"):
        small.top_k_sequences(-1)
    assert small.sample(0) == []
    assert small.top_k_sequences(0) == []


def test_out_of_range_length_is_refused(small):
    """A length outside the model reached the C layer as a huge uint32 and
    came back as a silent -inf or a bare "invalid argument"."""
    hi = int(small.config["max_length"])
    for call in (lambda L: small.log_partition(L),
                 lambda L: small.log_length_prior(L),
                 lambda L: small.path_count(L),
                 lambda L: small.topology(L),
                 lambda L: small.sample(1, length=L),
                 lambda L: small.top_k_sequences(1, length=L),
                 lambda L: small.spectrum(length=L, bins=64)):
        with pytest.raises(ValueError, match="length"):
            call(-1)
        with pytest.raises(ValueError, match="length"):
            call(hi + 1)


def test_topology_refuses_a_length_with_no_support(small):
    """Zero means "all lengths" everywhere else, but there is no pooled
    trellis to describe, and an unobserved length has no live structure."""
    with pytest.raises(ValueError, match="at least 1"):
        small.topology(0)
    dead = next(L for L in range(1, int(small.config["max_length"]) + 1)
                if not math.isfinite(small.log_length_prior(L)))
    with pytest.raises(ValueError, match="no sequences of length"):
        small.topology(dead)


def test_spectrum_bin_floor_is_explained(small):
    """The nominal floor is 2, but the transport reserves max_length + 9 bins
    of headroom, so the real floor is higher and the C message did not say so.
    An explicit spacing lifts the restriction, and that must keep working."""
    floor = int(small.config["max_length"]) + 10
    with pytest.raises(ValueError, match=f"at least {floor}"):
        small.spectrum(bins=floor - 1)
    with pytest.raises(ValueError, match="at least 2"):
        small.spectrum(bins=1)
    with pytest.raises(ValueError, match="spacing must be positive"):
        small.spectrum(spacing=-1.0)
    assert small.spectrum(bins=4, spacing=50.0)["weights"].size == 4
    with pytest.raises(ValueError, match=f"at least {floor}"):
        small.calibrate(bins=floor - 1)


def test_moment_order_is_range_checked(small):
    with pytest.raises(ValueError, match="order"):
        small.surprisal_moments(order=17)
    with pytest.raises(ValueError, match="order"):
        small.surprisal_cumulants(order=0)


def test_log_base_is_range_checked(small):
    for base in (1.0, 0.0, -2.0):
        with pytest.raises(ValueError, match="base"):
            small.entropy(base=base)
        with pytest.raises(ValueError, match="base"):
            small.dynamic_range(base=base)


# ═══════════════════════════════════════════════════════════════
# explain
# ═══════════════════════════════════════════════════════════════

def test_explain_reports_the_reason_it_refuses(small):
    """Three different refusals share one C error code, so the wrapper has to
    tell them apart or the caller gets "invalid argument" for all three."""
    with pytest.raises(ValueError, match="outside the twenty amino acids"):
        small.explain("CASSXQYF")
    with pytest.raises(ValueError, match="empty"):
        small.explain("")
    with pytest.raises(ValueError, match="longer than max_length"):
        small.explain("C" * 40)
    with pytest.raises(TypeError):
        small.explain(None)


def test_explain_flags_an_unproducible_length(small):
    """`explain` returns finite per-residue terms for a sequence whose LENGTH
    the model never saw, while `log_pseq` is -inf: the length prior, not any
    residue, is what rules it out. Presenting the decomposition without saying
    so would make it look like a valid explanation of a zero."""
    dead = next(L for L in range(1, int(small.config["max_length"]) + 1)
                if not math.isfinite(small.log_length_prior(L)))
    e = small.explain("C" * dead)
    assert e["length_is_producible"] is False
    assert np.isfinite(e["forward"]).all() and np.isfinite(e["backward"]).all()
    assert e["log_pseq"] == -math.inf

    live = int(np.argmax(small.length_pmf()))
    good = small.sample(1, length=live, seed=1)[0]
    assert small.explain(good)["length_is_producible"] is True


# ═══════════════════════════════════════════════════════════════
# Return-type conventions
# ═══════════════════════════════════════════════════════════════

def test_scalar_arguments_return_scalars(small):
    """One convention across the module: a scalar in, a float out; anything
    with a shape in, an array of that shape out. `surprisal_pdf` and
    `surprisal_cdf` used to return a one-element array for a scalar while
    `saddlepoint` and `pvalue` returned a float."""
    mu = float(small.surprisal_moments(order=1)[1])
    cal = small.calibrate(bins=64)
    live = int(np.argmax(small.length_pmf()))

    assert isinstance(small.surprisal_pdf(mu), float)
    assert isinstance(small.surprisal_cdf(mu), float)
    assert isinstance(cal.pvalue(live, mu), float)
    sp = small.saddlepoint(mu)
    for key in ("x", "pdf", "cdf", "sf", "t"):
        assert isinstance(sp[key], float), key
    assert isinstance(sp["converged"], bool)

    for value in ([mu], np.array([mu]), np.array([mu, mu + 1.0])):
        assert isinstance(small.surprisal_pdf(value), np.ndarray)
        assert isinstance(small.surprisal_cdf(value), np.ndarray)
        assert isinstance(cal.pvalue(live, value), np.ndarray)
        assert isinstance(small.saddlepoint(value)["cdf"], np.ndarray)

    assert small.surprisal_pdf(np.array([mu, mu + 1.0])).shape == (2,)
    assert small.saddlepoint(np.array([mu, mu + 1.0]))["cdf"].shape == (2,)


def test_pvalue_never_exceeds_one(small):
    """The survival function is a normalised cumulative sum, so its first
    entry lands a few ulps above 1. A p-value may not."""
    cal = small.calibrate(bins=64)
    for L in cal.lengths:
        assert cal.pvalue(L, -1e6) <= 1.0
        assert cal.pvalue(L, 1e12) >= 0.0
        assert float(np.max(cal.pvalue(L, np.array([-1e6, 0.0])))) <= 1.0


# ═══════════════════════════════════════════════════════════════
# Set operations: guards and bookkeeping
# ═══════════════════════════════════════════════════════════════

def test_operand_must_be_a_distinct_edge(corpus):
    """`add`/`subtract` alias source and destination in C, so passing the
    model itself doubles or zeroes its own counts. Subtracting a model from
    itself additionally empties the tables, at which point the re-finalize
    fails and there is nothing left to roll back to."""
    seqs, abund = corpus
    m = BeamModel(seqs[:200], abundances=abund[:200], order=2)
    with pytest.raises(ValueError, match="itself"):
        m.add(m)
    with pytest.raises(ValueError, match="itself"):
        m.subtract(m)
    with pytest.raises(TypeError):
        m.add("not a model")
    # and the guard must not have disturbed the model
    assert math.isfinite(m.log_pseq(seqs[0]))


def test_compact_disables_the_count_operations(corpus):
    """`compact` releases the counts, so everything that reads them has to
    say so. `n_nonzero_cells` reported 0, which is indistinguishable from a
    genuinely empty model."""
    seqs, abund = corpus
    m = BeamModel(seqs[:200], abundances=abund[:200], order=2)
    other = BeamModel(seqs[200:300], abundances=abund[200:300], order=2)
    assert m.n_nonzero_cells > 0
    m.compact()
    with pytest.raises(RuntimeError, match="compact"):
        m.n_nonzero_cells
    with pytest.raises(RuntimeError, match="compact"):
        m.add(other)
    with pytest.raises(RuntimeError, match="compact"):
        m.subtract(other)


def test_subtract_restores_the_total_weight(corpus):
    """`total_weight` is the bookkeeping number that survives a subtraction
    correctly, and the one the docstring points callers at."""
    seqs, abund = corpus
    a = BeamModel(seqs[:400], abundances=abund[:400], order=2)
    b = BeamModel(seqs[400:700], abundances=abund[400:700], order=2)
    w0 = a.total_weight
    a.add(b)
    assert a.total_weight == pytest.approx(w0 + b.total_weight)
    a.subtract(b)
    assert a.total_weight == pytest.approx(w0)


def test_subtract_restores_the_sequence_count(corpus):
    """`add` and `subtract` must be exact inverses on the BOOKKEEPING too, not
    just on the counts.

    They were not: `add` adjusted `n_observed` and `n_skipped` while
    `subtract` adjusted only `total_weight`, so every leave-one-cohort-out
    workflow — the thing `subtract` exists for — reported a corpus size that
    was too large. The two functions live in different files, which is how it
    went unnoticed.
    """
    seqs, abund = corpus
    a = BeamModel(seqs[:400], abundances=abund[:400], order=2)
    b = BeamModel(seqs[400:700], abundances=abund[400:700], order=2)
    n0 = a.n_observed
    a.add(b)
    assert a.n_observed == n0 + b.n_observed
    a.subtract(b)
    assert a.n_observed == n0


# ════════════════════════════════════════════════════════════════════════
# Streaming construction
#
# The property that makes a streamed build usable is that it is not an
# approximation of a one-shot build: counts are additive and finalize
# recomputes from scratch, so the two must agree BIT FOR BIT. Anything
# weaker and a convergence curve measured from snapshots is measuring the
# streaming path rather than the model.
# ════════════════════════════════════════════════════════════════════════


def test_builder_is_byte_identical_to_a_one_shot_build(corpus):
    seqs, abund = corpus
    one = BeamModel(seqs, abundances=abund, order=2, max_length=20)
    b = BeamModel.builder(order=2, max_length=20)
    for i in range(0, len(seqs), 137):          # deliberately ragged batches
        b.observe(seqs[i:i + 137], abund[i:i + 137])
    streamed = b.build()

    probe = seqs[:200]
    assert np.array_equal(streamed.log_pseq_batch(probe),
                          one.log_pseq_batch(probe))
    assert streamed.total_weight == one.total_weight
    assert streamed.n_observed == one.n_observed
    assert streamed.n_skipped == one.n_skipped


def test_builder_snapshot_is_independent_of_later_batches(corpus):
    """A convergence curve measures a snapshot and keeps ingesting. If the
    snapshot shared the builder's counts, every earlier checkpoint would
    silently become the final model."""
    seqs, abund = corpus
    b = BeamModel.builder(order=2, max_length=20)
    b.observe(seqs[:300], abund[:300])
    early = b.snapshot()
    early_scores = early.log_pseq_batch(seqs[:50])
    early_weight = early.total_weight

    b.observe(seqs[300:], abund[300:])
    late = b.build()

    assert np.array_equal(early.log_pseq_batch(seqs[:50]), early_scores)
    assert early.total_weight == early_weight
    assert late.total_weight > early_weight
    assert not np.array_equal(late.log_pseq_batch(seqs[:50]), early_scores)

    # And the snapshot really is the model that batch alone would have built.
    reference = BeamModel(seqs[:300], abundances=abund[:300], order=2, max_length=20)
    assert np.array_equal(early_scores, reference.log_pseq_batch(seqs[:50]))


def test_builder_tracks_its_running_totals(corpus):
    seqs, abund = corpus
    b = BeamModel.builder(order=2, max_length=20)
    assert b.n_observed == 0
    b.observe(seqs[:100], abund[:100])
    assert b.n_observed == 100
    assert b.total_weight == pytest.approx(float(sum(abund[:100])))
    b.observe(seqs[100:250], abund[100:250])
    assert b.n_observed == 250


def test_builder_is_spent_after_build(corpus):
    seqs, abund = corpus
    b = BeamModel.builder(order=2, max_length=20)
    b.observe(seqs[:100], abund[:100])
    b.build()
    with pytest.raises(RuntimeError, match="already been build"):
        b.observe(seqs[100:200])
    with pytest.raises(RuntimeError, match="already been build"):
        b.build()


def test_builder_rejects_mismatched_abundances():
    b = BeamModel.builder(order=2, max_length=10)
    with pytest.raises(ValueError, match="abundances has 2 entries"):
        b.observe(["CASSF", "CASSY", "CASSW"], [1.0, 2.0])


def test_from_sequence_file_matches_a_direct_build(tmp_path, corpus):
    seqs, abund = corpus
    path = tmp_path / "corpus.tsv"
    path.write_text("".join(f"{s}\t{a}\n" for s, a in zip(seqs, abund)))

    from_file = BeamModel.from_sequence_file(path, order=2, max_length=20,
                                        chunk_size=97)
    direct = BeamModel(seqs, abundances=abund, order=2, max_length=20)
    assert from_file.total_weight == direct.total_weight
    assert np.array_equal(from_file.log_pseq_batch(seqs[:100]),
                          direct.log_pseq_batch(seqs[:100]))


def test_from_sequence_file_without_weights(tmp_path, corpus):
    """A bare list of sequences, and a two-column file read with the weight
    column switched off, must give the same unweighted model."""
    seqs, _ = corpus
    path = tmp_path / "plain.txt"
    path.write_text("".join(f"{s}\n" for s in seqs))
    from_file = BeamModel.from_sequence_file(path, abundance_column=None,
                                        order=2, max_length=20)
    direct = BeamModel(seqs, order=2, max_length=20)
    assert from_file.total_weight == direct.total_weight
    assert np.array_equal(from_file.log_pseq_batch(seqs[:50]),
                          direct.log_pseq_batch(seqs[:50]))


def test_from_sequence_file_skips_a_header(tmp_path):
    seqs = ["CASSLGF", "CASSYEQF", "CASSWTF"]
    path = tmp_path / "h.tsv"
    path.write_text("cdr3_aa\tcount\n" + "".join(f"{s}\t2\n" for s in seqs))
    m = BeamModel.from_sequence_file(path, has_header=True, order=2, max_length=10)
    assert m.n_observed == 3
    assert m.total_weight == pytest.approx(6.0)


def test_from_dataframe_matches_a_direct_build(corpus):
    seqs, abund = corpus
    table = {"cdr3_aa": seqs, "total_abundance": abund}
    from_df = BeamModel.from_dataframe(table, abundance_column="total_abundance",
                                  order=2, max_length=20)
    direct = BeamModel(seqs, abundances=abund, order=2, max_length=20)
    assert np.array_equal(from_df.log_pseq_batch(seqs[:50]),
                          direct.log_pseq_batch(seqs[:50]))


def test_from_dataframe_names_the_missing_column(corpus):
    seqs, _ = corpus
    with pytest.raises(KeyError, match="no column 'junction_aa'"):
        BeamModel.from_dataframe({"cdr3_aa": seqs}, sequence_column="junction_aa",
                            order=2, max_length=20)


def test_from_parquet_matches_a_direct_build(tmp_path, corpus):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    seqs, abund = corpus
    path = tmp_path / "corpus.parquet"
    pq.write_table(pa.table({"cdr3_aa": seqs,
                             "total_abundance": [float(a) for a in abund]}),
                   path)
    from_pq = BeamModel.from_parquet(path, abundance_column="total_abundance",
                                batch_size=211, order=2, max_length=20)
    direct = BeamModel(seqs, abundances=abund, order=2, max_length=20)
    assert from_pq.total_weight == pytest.approx(direct.total_weight)
    assert np.array_equal(from_pq.log_pseq_batch(seqs[:50]),
                          direct.log_pseq_batch(seqs[:50]))


# ════════════════════════════════════════════════════════════════════════
# Non-destructive set operations
# ════════════════════════════════════════════════════════════════════════


def test_copy_is_independent(corpus):
    """`subtract` is destructive and the counts are the only copy there is, so
    the leave-one-out workflow needs a real clone, not a second reference."""
    seqs, abund = corpus
    base = BeamModel(seqs[:800], abundances=abund[:800], order=2, max_length=20)
    hold = BeamModel(seqs[:200], abundances=abund[:200], order=2, max_length=20)
    before = base.log_pseq_batch(seqs[:50])

    clone = base.copy()
    assert np.array_equal(clone.log_pseq_batch(seqs[:50]), before)
    clone.subtract(hold)
    assert not np.array_equal(clone.log_pseq_batch(seqs[:50]), before)
    assert np.array_equal(base.log_pseq_batch(seqs[:50]), before)


def test_without_a_model_equals_an_explicit_subtract(corpus):
    seqs, abund = corpus
    base = BeamModel(seqs[:800], abundances=abund[:800], order=2, max_length=20)
    hold = BeamModel(seqs[:200], abundances=abund[:200], order=2, max_length=20)

    held_out = base.without(hold)
    manual = base.copy()
    manual.subtract(hold)
    assert np.array_equal(held_out.log_pseq_batch(seqs[:50]),
                          manual.log_pseq_batch(seqs[:50]))
    assert base.total_weight == pytest.approx(float(sum(abund[:800])))


def test_without_sequences_builds_the_matching_operand(corpus):
    seqs, abund = corpus
    base = BeamModel(seqs[:800], abundances=abund[:800], order=2, max_length=20)
    a = base.without(seqs[:200], abundances=abund[:200])
    b = base.without(BeamModel(seqs[:200], abundances=abund[:200], order=2,
                          max_length=20))
    assert np.array_equal(a.log_pseq_batch(seqs[:50]),
                          b.log_pseq_batch(seqs[:50]))
    assert a.total_weight == pytest.approx(
        base.total_weight - float(sum(abund[:200])))


def test_merged_does_not_mutate_either_operand(corpus):
    seqs, abund = corpus
    a = BeamModel(seqs[:400], abundances=abund[:400], order=2, max_length=20)
    b = BeamModel(seqs[400:800], abundances=abund[400:800], order=2, max_length=20)
    wa, wb = a.total_weight, b.total_weight

    merged = a.merged(b)
    assert merged.total_weight == pytest.approx(wa + wb)
    assert a.total_weight == wa
    assert b.total_weight == wb

    inplace = a.copy()
    inplace.add(b)
    assert np.array_equal(merged.log_pseq_batch(seqs[:50]),
                          inplace.log_pseq_batch(seqs[:50]))


def test_without_rejects_abundances_for_a_model_operand(corpus):
    seqs, abund = corpus
    a = BeamModel(seqs[:400], abundances=abund[:400], order=2, max_length=20)
    b = BeamModel(seqs[:100], abundances=abund[:100], order=2, max_length=20)
    with pytest.raises(TypeError, match="already carries its own weights"):
        a.without(b, abundances=[1.0])


# ════════════════════════════════════════════════════════════════════════
# Streaming and file-backed scoring
# ════════════════════════════════════════════════════════════════════════


def test_log_pseq_chunks_reassembles_the_batch(model, corpus):
    seqs, _ = corpus
    probe = seqs[:500]
    chunked = np.concatenate(list(model.log_pseq_chunks(probe, chunk_size=61)))
    assert np.array_equal(chunked, model.log_pseq_batch(probe))


def test_log_pseq_file_matches_the_batch(tmp_path, model, corpus):
    seqs, abund = corpus
    path = tmp_path / "score.tsv"
    path.write_text("".join(f"{s}\t{a}\n" for s, a in zip(seqs, abund)))
    got = model.log_pseq_file(path, chunk_size=97)
    assert np.array_equal(got, model.log_pseq_batch(seqs))


def test_log_pseq_file_fills_a_preallocated_array(tmp_path, model, corpus):
    """The pattern a corpus-scale scan uses: a memmap the size of the corpus,
    filled in place, never a Python list of 900 million floats."""
    seqs, _ = corpus
    path = tmp_path / "plain.txt"
    path.write_text("".join(f"{s}\n" for s in seqs))
    out = np.zeros(len(seqs), dtype=np.float32)
    got = model.log_pseq_file(path, chunk_size=113, out=out)
    assert got is out
    assert np.allclose(out, model.log_pseq_batch(seqs), rtol=1e-6)


def test_log_pseq_file_rejects_a_wrong_sized_out(tmp_path, model, corpus):
    seqs, _ = corpus
    path = tmp_path / "plain.txt"
    path.write_text("".join(f"{s}\n" for s in seqs[:100]))
    with pytest.raises(ValueError, match="not the right length"):
        model.log_pseq_file(path, out=np.zeros(50))
    with pytest.raises(ValueError, match="not the right length"):
        model.log_pseq_file(path, out=np.zeros(500))


def test_log_pseq_parquet_preserves_row_order(tmp_path, model, corpus):
    """Two positionally aligned columns are joined by ROW ORDER and nothing
    else, so a scorer that reordered rows would pair every sequence with some
    other sequence's label."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    seqs, _ = corpus
    path = tmp_path / "s.parquet"
    pq.write_table(pa.table({"cdr3_aa": seqs}), path)
    got = model.log_pseq_parquet(path, batch_size=173)
    assert np.array_equal(got, model.log_pseq_batch(seqs))


def test_pseq_accepts_a_list_and_a_log_flag(model, corpus):
    seqs, _ = corpus
    probe = seqs[:20]
    assert model.pseq(probe, log=True) == pytest.approx(
        model.log_pseq_batch(probe))
    assert model.pseq(probe) == pytest.approx(np.exp(model.log_pseq_batch(probe)))
    assert model.pseq(probe[0]) == pytest.approx(math.exp(
        model.log_pseq(probe[0])))
    assert model.pseq(probe[0], log=True) == pytest.approx(
        model.log_pseq(probe[0]))


# ════════════════════════════════════════════════════════════════════════
# pseq_analysis: spectra, atoms and discovery
# ════════════════════════════════════════════════════════════════════════

#: Grid used by the analysis tests. Small enough to keep a transport in
#: seconds on the `small` fixture, fine enough that the grid error is well
#: below every tolerance asserted against it.
ANALYSIS_BINS = 384


@pytest.fixture(scope="module")
def tiny():
    """A model whose ENTIRE support can be enumerated: 20**3 + 20**4 = 168,000
    sequences. Small enough that every spectrum quantity has an exact
    brute-force counterpart to check against."""
    rng = random.Random(3)
    seqs = ["".join(rng.choice(AA) for _ in range(rng.choice([3, 4])))
            for _ in range(500)]
    abund = [rng.randint(1, 9) for _ in seqs]
    return BeamModel(seqs, abundances=abund, order=2, depth_cap=2, max_length=4)


@pytest.fixture(scope="module")
def tiny_exact(tiny):
    """Every sequence the tiny model can produce, with its exact probability."""
    import itertools
    every = ["".join(t) for L in (3, 4)
             for t in itertools.product(AA, repeat=L)]
    p = np.exp(tiny.log_pseq_batch(every).astype(np.longdouble))
    return every, p


def test_histogram_pair_shares_one_grid_and_hits_its_totals(small):
    a = small.pseq_analysis()
    pair = a.histogram_pair(ANALYSIS_BINS)
    c, g = pair["counting"], pair["generated"]
    assert np.array_equal(c.surprisal, g.surprisal)
    assert c.grid_spacing == g.grid_spacing
    assert c.total_mass == pytest.approx(small.path_count(), rel=1e-9)
    assert g.total_mass == pytest.approx(1.0, rel=1e-12)


def test_histogram_mean_is_the_entropy(small):
    """Linear transport conserves the FIRST moment exactly, so the mean read
    off the grid is not an approximation even though the shape is."""
    a = small.pseq_analysis()
    h = a.histogram(ANALYSIS_BINS, measure="generated")
    assert h.mean == pytest.approx(small.entropy(), rel=1e-9)


def test_histogram_is_cached_not_rebuilt(small):
    a = small.pseq_analysis()
    assert a.histogram(ANALYSIS_BINS) is a.histogram(ANALYSIS_BINS)
    assert a.histogram(ANALYSIS_BINS) is not a.histogram(ANALYSIS_BINS + 1)


def test_histograms_by_length_decompose_the_pooled_spectrum(small):
    a = small.pseq_analysis()
    per_length = a.histograms_by_length(ANALYSIS_BINS, measure="generated")
    marg = a.length_marginals()
    assert set(per_length) == set(marg)
    for L, h in per_length.items():
        assert h.total_mass == pytest.approx(marg[L]["generated"], rel=1e-9)
    total = sum(h.total_mass for h in per_length.values())
    assert total == pytest.approx(1.0, rel=1e-12)
    # And they all sit on the pooled grid, which is what makes them stackable.
    pooled = a.histogram(ANALYSIS_BINS, measure="generated")
    for h in per_length.values():
        assert np.array_equal(h.surprisal, pooled.surprisal)


def test_histograms_by_length_respects_max_length(small):
    a = small.pseq_analysis()
    got = a.histograms_by_length(ANALYSIS_BINS, max_length=7)
    assert got and max(got) <= 7


def test_atom_mass_is_exactly_one_where_exp_surprisal_is_not(small):
    """The atom probability is `generated / counting`, not `exp(-surprisal)`.

    Both spectra are transported on the same grid with the same linear
    splitting, so their ratio is the mass-weighted mean probability of the
    sequences in the bin and `sum(multiplicity * probability)` is the
    generated mass EXACTLY. Reading exp(-surprisal) off the bin's left edge
    instead biases every atom by up to exp(spacing), which this test pins
    down: at a coarse grid the naive route is out by tens of percent, and
    every discovery curve built on it inherits that.
    """
    a = small.pseq_analysis()
    for bins in (128, ANALYSIS_BINS):
        atoms = a.atoms(bins)
        assert atoms.mass == pytest.approx(1.0, rel=1e-12)
        assert atoms.support_size == pytest.approx(small.path_count(), rel=1e-9)

        h = a.histogram(bins, measure="counting")
        keep = h.weights > 0
        naive = float(np.sum(
            h.weights[keep].astype(np.longdouble)
            * np.exp(-h.surprisal[keep].astype(np.longdouble))))
        assert naive > 1.01, (
            "the naive exp(-surprisal) route is supposed to be biased here; "
            "if it is not, this test is no longer proving anything")


def test_discovery_curve_matches_exhaustive_enumeration(tiny, tiny_exact):
    """R(n) and U(n) from the grid spectrum against the same sums taken over
    every one of the 168,000 sequences the model can produce.

    This is the check that the atoms really describe the distribution rather
    than merely being self-consistent: the right-hand side never touches a
    spectrum.
    """
    _, p = tiny_exact
    atoms = tiny.pseq_analysis().atoms(4096)
    assert atoms.support_size == pytest.approx(20 ** 3 + 20 ** 4, rel=1e-9)

    for n, tol in ((1, 1e-9), (10, 1e-5), (1_000, 1e-4),
                   (100_000, 1e-3), (10 ** 8, 1e-3)):
        want_r = float(np.sum(-np.expm1(np.longdouble(n) * np.log1p(-p))))
        want_u = float(np.sum(p * np.exp(np.longdouble(n - 1)
                                         * np.log1p(-p))))
        assert atoms.expected_richness(n) == pytest.approx(want_r, rel=tol)
        assert atoms.novelty(n) == pytest.approx(want_u, rel=tol)


def test_discovery_curve_refines_with_the_grid(tiny, tiny_exact):
    """The grid is the only approximation, so the error must fall as it is
    refined. A bug that biased the atoms would leave a floor instead."""
    _, p = tiny_exact
    n = 100_000
    want = float(np.sum(-np.expm1(np.longdouble(n) * np.log1p(-p))))
    a = tiny.pseq_analysis()
    errors = [abs(a.atoms(b).expected_richness(n) - want) / want
              for b in (512, 2048, 8192)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 1e-5


def test_novelty_is_the_richness_increment(small):
    """U(n) == R(n) - R(n-1) identically. Two different sums over the same
    atoms, so a sign or off-by-one in either shows up here."""
    atoms = small.pseq_analysis().atoms(ANALYSIS_BINS)
    for n in (2, 7, 50, 1000):
        assert atoms.novelty(n) == pytest.approx(
            atoms.expected_richness(n) - atoms.expected_richness(n - 1),
            rel=1e-9)


def test_expected_richness_at_one_draw_is_the_mass(small):
    atoms = small.pseq_analysis().atoms(ANALYSIS_BINS)
    assert atoms.expected_richness(1) == pytest.approx(1.0, rel=1e-9)
    assert atoms.novelty(1) == pytest.approx(1.0, rel=1e-9)
    assert atoms.expected_richness(0) == pytest.approx(0.0, abs=1e-12)


def test_expected_richness_is_monotone_and_bounded(small):
    atoms = small.pseq_analysis().atoms(ANALYSIS_BINS)
    d0 = small.path_count()
    ns = np.logspace(0, 60, 40)
    r = [atoms.expected_richness(n) for n in ns]
    assert all(b >= a for a, b in zip(r, r[1:]))
    assert all(v <= d0 * (1 + 1e-9) for v in r)
    assert atoms.expected_richness(1e300) == pytest.approx(d0, rel=1e-6)


def test_expected_frequency_partitions_the_support(small):
    """Every sequence is drawn 0..n times, so the frequencies sum to D0."""
    atoms = small.pseq_analysis().atoms(ANALYSIS_BINS)
    n = 6
    total = sum(atoms.expected_frequency(n, r) for r in range(n + 1))
    assert total == pytest.approx(small.path_count(), rel=1e-6)
    # f0 is what R(n) says is still unseen.
    assert atoms.expected_frequency(n, 0) == pytest.approx(
        small.path_count() - atoms.expected_richness(n), rel=1e-6)


def test_coverage_matches_exhaustive_enumeration(tiny, tiny_exact):
    _, p = tiny_exact
    atoms = tiny.pseq_analysis().atoms(4096)
    for n in (10, 1_000, 100_000):
        want = float(np.sum(p * -np.expm1(np.longdouble(n) * np.log1p(-p))))
        assert atoms.coverage(n) == pytest.approx(want, rel=1e-3)


def test_discovery_curve_evaluates_many_depths_at_once(small):
    atoms = small.pseq_analysis().atoms(ANALYSIS_BINS)
    curve = atoms.discovery_curve([1, 10, 100])
    assert curve["expected_richness"][0] == pytest.approx(
        atoms.expected_richness(1), rel=1e-12)
    assert curve["novelty"][2] == pytest.approx(atoms.novelty(100), rel=1e-12)
    assert list(curve) == ["n", "expected_richness", "novelty", "coverage"]


def test_atoms_at_one_length_carry_that_length_s_mass(small):
    a = small.pseq_analysis()
    marg = a.length_marginals()
    L = max(marg, key=lambda k: marg[k]["generated"])
    atoms = a.atoms(ANALYSIS_BINS, length=L)
    assert atoms.mass == pytest.approx(marg[L]["generated"], rel=1e-9)
    assert atoms.support_size == pytest.approx(small.path_count(L), rel=1e-9)


# ════════════════════════════════════════════════════════════════════════
# Mellin jets, pooled and per length
# ════════════════════════════════════════════════════════════════════════


def test_derivatives_recover_the_mass_and_the_entropy(model):
    a = model.pseq_analysis()
    d = a.derivatives(1.0, 2)
    assert d[0] == pytest.approx(1.0, rel=1e-12)
    assert -d[1] == pytest.approx(model.entropy(), rel=1e-12)
    # Second derivative gives the surprisal's second raw moment.
    stats = model.surprisal_stats(q=1.0)
    assert d[2] == pytest.approx(stats["variance"] + stats["mean"] ** 2,
                                 rel=1e-10)


def test_derivatives_at_q_zero_is_the_support(model):
    a = model.pseq_analysis()
    assert a.derivatives(0.0, 0)[0] == pytest.approx(model.path_count(),
                                                     rel=1e-9)


def test_mellin_matches_log_mellin(model):
    a = model.pseq_analysis()
    assert math.log(a.mellin(2.0)) == pytest.approx(a.log_mellin(2.0),
                                                    rel=1e-12)
    assert a.cgf(0.0) == pytest.approx(0.0, abs=1e-12)


def test_moments_mean_is_the_entropy(model):
    m = model.pseq_analysis().moments(4)
    assert m["mass"] == pytest.approx(1.0, rel=1e-12)
    assert m["mean"] == pytest.approx(model.entropy(), rel=1e-12)
    assert m["std"] == pytest.approx(math.sqrt(m["variance"]), rel=1e-12)


def test_length_derivatives_q0_is_the_per_length_support(model):
    a = model.pseq_analysis()
    per = a.length_derivatives(0.0, 0)
    for L, jet in per.items():
        assert float(jet[0]) == pytest.approx(model.path_count(L), rel=1e-9)
    assert sum(float(j[0]) for j in per.values()) == pytest.approx(
        model.path_count(), rel=1e-9)


def test_length_derivatives_q1_decomposes_mass_and_entropy(model):
    """Summed over lengths, order 0 must give 1 and order 1 must give minus
    the entropy: the per-length jet has to be a decomposition of the global
    one, not merely per-length numbers that look plausible."""
    per = model.pseq_analysis().length_derivatives(1.0, 2)
    assert sum(float(j[0]) for j in per.values()) == pytest.approx(1.0,
                                                                  rel=1e-12)
    assert -sum(float(j[1]) for j in per.values()) == pytest.approx(
        model.entropy(), rel=1e-9)
    stats = model.surprisal_stats(q=1.0)
    assert sum(float(j[2]) for j in per.values()) == pytest.approx(
        stats["variance"] + stats["mean"] ** 2, rel=1e-9)


@pytest.mark.parametrize("q,order", [(0.0, 0), (0.5, 3), (1.0, 4), (2.0, 2)])
def test_length_derivatives_sum_to_the_pooled_mellin(model, q, order):
    """The per-length jet is now exact at any (q, order), so summing order 0
    over lengths must reproduce M(q) — a different code path."""
    per = model.pseq_analysis().length_derivatives(q, order)
    assert sum(float(j[0]) for j in per.values()) == pytest.approx(
        math.exp(model.log_mellin(q)), rel=1e-9)


def test_length_marginals_are_keyed_by_length(model):
    marg = model.pseq_analysis().length_marginals()
    assert set(marg) == set(model.observed_lengths)
    assert sum(v["generated"] for v in marg.values()) == pytest.approx(1.0,
                                                                      rel=1e-12)
    assert sum(v["counting"] for v in marg.values()) == pytest.approx(
        model.path_count(), rel=1e-9)


def test_edge_thresholding_is_not_applicable(model):
    """BeamModel deliberately has no pruning: hierarchical backoff leaves every
    transition strictly positive, so there is no low-likelihood wiring to
    threshold away. Refusing loudly is the honest answer."""
    with pytest.raises(NotImplementedError, match="no edge pruning"):
        model.pseq_analysis().diversity_under_edge_thresholds([0.0, 0.5])


# ════════════════════════════════════════════════════════════════════════
# SCALE, from the exact null
# ════════════════════════════════════════════════════════════════════════


#: Grid for the SCALE tests. Finer than ANALYSIS_BINS because a quantile is
#: resolved to a bin width while a total is not: at 384 bins the median of the
#: reconstructed null is still a few hundredths of an IQR out. Still ~2s.
CALIBRATION_BINS = 1024


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
# Cross-model summary names
# ════════════════════════════════════════════════════════════════════════


def test_pgen_diagnostics_measures_properness(model):
    d = model.pgen_diagnostics()
    assert d["is_proper"] is True
    assert d["total_mass"] == pytest.approx(1.0, rel=1e-12)
    assert d["length_prior_mass"] == pytest.approx(1.0, rel=1e-12)
    assert d["measure"] == "Pseq"


def test_diversity_profile_carries_entropy_nats(model):
    d = model.diversity_profile()
    assert d["entropy_nats"] == pytest.approx(model.entropy(), rel=1e-12)
    assert d["D1"] == pytest.approx(math.exp(d["entropy_nats"]), rel=1e-9)


def test_length_distribution_is_the_prior(model):
    ld = model.length_distribution
    assert tuple(sorted(ld)) == model.observed_lengths
    assert sum(ld.values()) == pytest.approx(1.0, rel=1e-12)
    assert min(ld) >= 1 and max(ld) <= model.config["max_length"]


def test_a_pre_split_edge_file_is_refused_by_name(tmp_path, small):
    """A model written before the BEAM split must fail with an explanation.

    The format did not change across the split, only the magic did, so a
    pre-BEAM file is byte-for-byte loadable apart from its first four bytes.
    That makes it exactly the case most likely to be mistaken for corruption,
    which is why the loader names it instead of reporting a generic failure.
    """
    good = tmp_path / "m.beam"
    small.save(good)

    legacy = tmp_path / "legacy.beam"
    raw = bytearray(good.read_bytes())
    raw[:4] = (0x45444745).to_bytes(4, "little")     # the old EDGE magic
    legacy.write_bytes(bytes(raw))

    with pytest.raises(OSError, match="written by EDGE"):
        BeamModel.from_file(legacy)

    junk = tmp_path / "junk.beam"
    junk.write_bytes(b"not a model at all, not even close")
    with pytest.raises(OSError, match="not a model file, truncated, or corrupt"):
        BeamModel.from_file(junk)


def test_n_sequences_is_the_ingested_weight(corpus):
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2, max_length=20)
    assert m.n_sequences == pytest.approx(float(sum(abund)))
    assert m.variant == "beam"


def test_calibration_median_converges_with_the_grid(small):
    """The reconstruction is the only approximation in the null, so refining
    the grid must move its median toward the model's own draws. A biased
    quantile inversion would leave a floor instead: before the
    mid-distribution correction this bias was half a bin wide and did not.
    """
    drawn = small.sample(6000, seed=17)
    u = -small.log_pseq_batch(drawn)
    lengths = np.array([len(s) for s in drawn])
    L = int(np.bincount(lengths).argmax())
    truth = float(np.median(u[lengths == L]))
    errors = [abs(small.calibrate(bins=b).median(L) - truth)
              for b in (256, 1024)]
    assert errors[0] > errors[1]
    assert errors[1] < 0.06


# ════════════════════════════════════════════════════════════════════════
# Parallelism
#
# The cap is process-wide, so every test here restores it. The property that
# makes it safe to use is that results do not depend on it: every reduction
# is order-independent or summed in a fixed index order.
# ════════════════════════════════════════════════════════════════════════


@pytest.fixture
def restore_threads():
    from BEAM import threads, set_threads
    before = threads()
    yield
    set_threads(before)


def test_thread_cap_round_trips(model, restore_threads):
    from BEAM import effective_threads, threads, set_threads
    set_threads(2)
    assert threads() == 2
    assert model.threads == 2
    assert 1 <= effective_threads() <= 2
    assert model.effective_threads == effective_threads()

    model.threads = 1
    assert threads() == 1
    assert model.effective_threads == 1

    set_threads(0)
    assert threads() == 0
    assert model.effective_threads >= 1


def test_thread_cap_treats_negative_as_default(model, restore_threads):
    from BEAM import threads
    model.threads = -4
    assert threads() == 0


def test_results_do_not_depend_on_the_thread_count(model, corpus,
                                                   restore_threads):
    """The whole point of a cap being safe to set: pinning it for a benchmark
    must change the timing and not the answer. Both the batch scorer and an
    analytic that reduces over the trellis are checked."""
    seqs, _ = corpus
    probe = seqs[:300]

    model.threads = 1
    one_scores = model.log_pseq_batch(probe)
    one_entropy = model.entropy()
    one_mellin = model.log_mellin(2.0)

    model.threads = 4
    assert np.array_equal(model.log_pseq_batch(probe), one_scores)
    assert model.entropy() == one_entropy
    assert model.log_mellin(2.0) == one_mellin


def test_atoms_feed_the_publicness_model(small):
    """The atom pair is what a sharing prediction consumes, so it has to be
    accepted by `PublicnessModel.expected_counts` as it comes off the
    spectrum — no reshaping, no dtype surgery at the call site."""
    from BEAM import PublicnessModel

    p, c = small.pseq_analysis().atoms(ANALYSIS_BINS).as_arrays()
    depths = np.arange(100, 400, dtype=np.int64)
    pb = PublicnessModel(depths, depth_bins=8)
    out = pb.expected_counts(p, c, levels=np.array([1, 5, len(depths) + 1]))
    counts = np.asarray(out["expected_counts"], dtype=np.float64)
    assert counts.size == 2
    assert np.all(np.isfinite(counts)) and np.all(counts >= 0)


# ════════════════════════════════════════════════════════════════════════
# Fused C entry points
#
# Each of these replaces a loop of separate calls. The load-bearing test for
# every one is that it is BIT-IDENTICAL to the loop it replaces: a fused DP
# that merely agrees to a tolerance is a second implementation to keep in
# step, not a shortcut.
# ════════════════════════════════════════════════════════════════════════


def test_spectrum_pair_is_bit_identical_to_two_calls(small):
    spacing = small._grid_spacing(ANALYSIS_BINS)
    pair = small.spectrum_pair(bins=ANALYSIS_BINS, spacing=spacing)
    c = small.spectrum(measure="counting", bins=ANALYSIS_BINS, spacing=spacing)
    g = small.spectrum(measure="generated", bins=ANALYSIS_BINS, spacing=spacing)
    assert np.array_equal(pair["counting"], c["weights"])
    assert np.array_equal(pair["generated"], g["weights"])
    assert pair["spacing"] == c["spacing"] == g["spacing"]


def test_spectrum_pair_at_one_length_is_bit_identical(small):
    L = max(small.observed_lengths)
    spacing = small._grid_spacing(ANALYSIS_BINS)
    pair = small.spectrum_pair(length=L, bins=ANALYSIS_BINS, spacing=spacing)
    c = small.spectrum(measure="counting", length=L, bins=ANALYSIS_BINS,
                       spacing=spacing)
    assert np.array_equal(pair["counting"], c["weights"])
    assert pair["counting"].sum() == pytest.approx(small.path_count(L),
                                                   rel=1e-9)


def test_spectrum_pair_auto_spacing_matches_the_single_call(small):
    pair = small.spectrum_pair(bins=ANALYSIS_BINS)
    solo = small.spectrum(measure="counting", bins=ANALYSIS_BINS)
    assert pair["spacing"] == solo["spacing"]
    assert np.array_equal(pair["counting"], solo["weights"])


def test_spectra_by_length_rows_are_bit_identical(small):
    spacing = small._grid_spacing(ANALYSIS_BINS)
    got = small.spectra_by_length(measure="generated", bins=ANALYSIS_BINS,
                                  spacing=spacing)
    W = got["weights"]
    assert W.shape[1] == ANALYSIS_BINS
    for L in small.observed_lengths:
        solo = small.spectrum(measure="generated", length=L,
                              bins=ANALYSIS_BINS, spacing=spacing)
        assert np.array_equal(W[L], solo["weights"])
    # Lengths the model does not generate are left at zero, not garbage.
    for L in range(W.shape[0]):
        if L not in small.observed_lengths:
            assert not W[L].any()
    # And the rows sum to the pooled spectrum, which is what one shared grid
    # is for.
    pooled = small.spectrum(measure="generated", bins=ANALYSIS_BINS,
                            spacing=spacing)["weights"]
    assert np.allclose(W.sum(axis=0), pooled, rtol=0, atol=1e-15)


def test_spectra_by_length_does_not_depend_on_the_memory_budget(small):
    """The budget caps CONCURRENCY, never what one length may allocate, so
    squeezing it to a single byte must change the schedule and not the
    answer."""
    from BEAM import memory_budget, set_memory_budget

    spacing = small._grid_spacing(ANALYSIS_BINS)
    before = memory_budget()
    try:
        roomy = small.spectra_by_length(bins=ANALYSIS_BINS, spacing=spacing)
        set_memory_budget(1)
        assert memory_budget() == 1
        cramped = small.spectra_by_length(bins=ANALYSIS_BINS, spacing=spacing)
    finally:
        set_memory_budget(before)
    assert np.array_equal(roomy["weights"], cramped["weights"])
    assert memory_budget() == before


def test_memory_budget_zero_restores_the_default():
    from BEAM import memory_budget, set_memory_budget
    before = memory_budget()
    try:
        set_memory_budget(1 << 20)
        assert memory_budget() == 1 << 20
        set_memory_budget(0)
        assert memory_budget() == 4 * (1 << 30)
    finally:
        set_memory_budget(before)


def test_histograms_by_length_match_the_single_length_route(small):
    """The analysis object now fuses; the fused rows must equal what the
    per-length calls it replaced would have produced."""
    a = small.pseq_analysis()
    fused = a.histograms_by_length(ANALYSIS_BINS, measure="generated")
    for L, h in fused.items():
        solo = small.spectrum(measure="generated", length=L,
                              bins=ANALYSIS_BINS,
                              spacing=small._grid_spacing(ANALYSIS_BINS))
        assert np.array_equal(h.weights, solo["weights"])


def test_histogram_pair_populates_the_single_measure_cache(small):
    """One fused transport has to satisfy a later single-measure request, or
    the fusion costs a sweep instead of saving one."""
    a = small.pseq_analysis()
    pair = a.histogram_pair(ANALYSIS_BINS)
    assert a.histogram(ANALYSIS_BINS, measure="counting") is pair["counting"]
    assert a.histogram(ANALYSIS_BINS, measure="generated") is pair["generated"]


# ════════════════════════════════════════════════════════════════════════
# Per-length tilted moments
# ════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("q", [0.0, 0.5, 1.0, 2.0])
def test_length_moments_mix_to_the_pooled_moments(model, q):
    """`mu[k] == sum_L share[L] * moments[L, k]`.

    The identity the unmixed form exists to make checkable: slot 0 carries
    each length's SHARE of the q-tilted law rather than the constant 1 a raw
    jet would put there, precisely so the mixture can be verified from the
    output alone against `surprisal_moments`, which is a separate DP.
    """
    order = 4
    lm = model.length_moments(q=q, order=order)
    mu = model.surprisal_moments(q=q, order=order)
    share, moments = lm["share"], lm["moments"]

    assert share.sum() == pytest.approx(1.0, rel=1e-12)
    for k in range(order + 1):
        assert float((share * moments[:, k]).sum()) == pytest.approx(
            float(mu[k]), rel=1e-9)


def test_length_moments_share_at_q_one_is_the_length_prior(model):
    lm = model.length_moments(q=1.0, order=1)
    pmf = model.length_pmf()
    assert np.allclose(lm["share"][:len(pmf)], pmf, rtol=1e-9, atol=1e-15)


def test_length_moments_conditional_mean_matches_length_marginals(model):
    """A second, independent C route to E[-ln P | L]."""
    lm = model.length_moments(q=1.0, order=1)
    marg = model.length_marginals()
    for L in model.observed_lengths:
        assert lm["moments"][L, 1] == pytest.approx(
            float(marg["mean_surprisal"][L]), rel=1e-9)


def test_length_moments_ungenerated_lengths_are_zero(model):
    lm = model.length_moments(q=1.0, order=2)
    for L in range(len(lm["share"])):
        if L not in model.observed_lengths:
            assert lm["share"][L] == 0.0


def test_length_derivatives_q0_still_is_the_per_length_support(model):
    """The generalised route must not have broken the q=0 corner the capacity
    tiers read."""
    per = model.pseq_analysis().length_derivatives(0.0, 0)
    for L, jet in per.items():
        assert float(jet[0]) == pytest.approx(model.path_count(L), rel=1e-9)


def test_length_derivatives_q1_still_decomposes_the_entropy(model):
    per = model.pseq_analysis().length_derivatives(1.0, 2)
    assert sum(float(j[0]) for j in per.values()) == pytest.approx(1.0,
                                                                  rel=1e-12)
    assert -sum(float(j[1]) for j in per.values()) == pytest.approx(
        model.entropy(), rel=1e-9)


# ════════════════════════════════════════════════════════════════════════
# Dirichlet-multinomial posterior
# ════════════════════════════════════════════════════════════════════════


def test_posterior_at_kappa_zero_is_a_no_op(corpus):
    seqs, abund = corpus
    donor = BeamModel(seqs[:300], abundances=abund[:300], order=2, max_length=20)
    prior = BeamModel(seqs[300:], abundances=abund[300:], order=2, max_length=20)
    before = donor.log_pseq_batch(seqs[:40])
    donor.posterior(prior, kappa=0.0)
    assert np.array_equal(donor.log_pseq_batch(seqs[:40]), before)


def test_posterior_moves_the_model_toward_the_prior(corpus):
    """The point of the update: with a large kappa the individual's scores sit
    closer to the prior's than they did on the individual's own counts."""
    seqs, abund = corpus
    donor = BeamModel(seqs[:200], abundances=abund[:200], order=2, max_length=20)
    prior = BeamModel(seqs[200:], abundances=abund[200:], order=2, max_length=20)
    probe = seqs[900:1100]

    own = donor.log_pseq_batch(probe)
    ref = prior.log_pseq_batch(probe)
    near = donor.with_prior(prior, kappa=1e6).log_pseq_batch(probe)

    assert np.mean(np.abs(near - ref)) < np.mean(np.abs(own - ref))


def test_posterior_closes_on_the_prior_but_never_reaches_it(corpus):
    """Raising kappa pulls the individual toward the prior, and then stops.

    It stops because `posterior` is not a blend of two models: it adds
    pseudo-counts to ROWS, and three things survive any kappa — rows the
    prior never observed, this model's own hierarchical backoff reapplied to
    the blended counts, and its own Z(L) and pi(L). Anyone reading
    `posterior` as "become the prior at large kappa" would be wrong, so the
    floor is asserted rather than left to be discovered.
    """
    seqs, abund = corpus
    donor = BeamModel(seqs[:200], abundances=abund[:200], order=2, max_length=20)
    prior = BeamModel(seqs[200:], abundances=abund[200:], order=2, max_length=20)
    probe = seqs[900:1100]
    ref = prior.log_pseq_batch(probe)

    def gap(k):
        return float(np.mean(np.abs(
            donor.with_prior(prior, kappa=k).log_pseq_batch(probe) - ref)))

    closing = [gap(k) for k in (0.0, 1.0, 10.0, 100.0)]
    assert all(b < a for a, b in zip(closing, closing[1:])), closing
    assert closing[-1] < 0.1 * closing[0]
    assert gap(1e6) > 0.5, "a huge kappa must not collapse onto the prior"


def test_posterior_leaves_the_sequence_counters_alone(corpus):
    """A pseudo-count is not an observation, so a leave-one-out corpus size
    must not shift when the smoothing strength does."""
    seqs, abund = corpus
    donor = BeamModel(seqs[:300], abundances=abund[:300], order=2, max_length=20)
    prior = BeamModel(seqs[300:], abundances=abund[300:], order=2, max_length=20)
    w, n = donor.total_weight, donor.n_observed
    donor.posterior(prior, kappa=5_000.0)
    assert donor.total_weight == w
    assert donor.n_observed == n


def test_posterior_leaves_the_model_scoreable(corpus):
    """The C clears `finalized` rather than rebuilding, so the wrapper has to
    finalize. If it did not, every score below would be -inf or an error."""
    seqs, abund = corpus
    donor = BeamModel(seqs[:300], abundances=abund[:300], order=2, max_length=20)
    prior = BeamModel(seqs[300:], abundances=abund[300:], order=2, max_length=20)
    donor.posterior(prior, kappa=100.0)
    lp = donor.log_pseq_batch(seqs[:50])
    assert np.all(np.isfinite(lp))
    assert donor.pgen_diagnostics()["is_proper"] is True


def test_with_prior_does_not_mutate_either_operand(corpus):
    seqs, abund = corpus
    donor = BeamModel(seqs[:300], abundances=abund[:300], order=2, max_length=20)
    prior = BeamModel(seqs[300:], abundances=abund[300:], order=2, max_length=20)
    d0 = donor.log_pseq_batch(seqs[:40])
    p0 = prior.log_pseq_batch(seqs[:40])

    out = donor.with_prior(prior, kappa=250.0)
    assert np.array_equal(donor.log_pseq_batch(seqs[:40]), d0)
    assert np.array_equal(prior.log_pseq_batch(seqs[:40]), p0)
    assert not np.array_equal(out.log_pseq_batch(seqs[:40]), d0)

    inplace = donor.copy()
    inplace.posterior(prior, kappa=250.0)
    assert np.array_equal(out.log_pseq_batch(seqs[:40]),
                          inplace.log_pseq_batch(seqs[:40]))


def test_posterior_rejects_bad_operands(corpus):
    seqs, abund = corpus
    donor = BeamModel(seqs[:300], abundances=abund[:300], order=2, max_length=20)
    prior = BeamModel(seqs[300:], abundances=abund[300:], order=2, max_length=20)
    with pytest.raises(ValueError, match="kappa must be non-negative"):
        donor.posterior(prior, kappa=-1.0)
    with pytest.raises(TypeError, match="expects an BeamModel"):
        donor.posterior(object(), kappa=1.0)
    with pytest.raises(ValueError, match="with the model itself"):
        donor.posterior(donor, kappa=1.0)


# ── Strict (closed-world) mode ──────────────────────────────────────────────
#
# The default model can never return zero: the order-0 base level adds one
# pseudo-count to every residue and the probabilities are floored before the
# log, so an unobserved transition arrives as a large finite negative rather
# than -inf. That is what makes D0 equal sum_L 20**L regardless of the data.
# `strict` turns off both, so the model spans only what it has seen.

STRICT_SEQS = ["CASSF", "CASSY", "CTTGW"]


def test_strict_spans_exactly_what_it_saw():
    """D0 is the count of training sequences, not 20**L."""
    m = BeamModel(STRICT_SEQS, max_length=6, strict=True)
    assert m.hill_number(0) == pytest.approx(3.0)
    assert m.hill_number(1) == pytest.approx(3.0)
    # Equally weighted, so each is exactly 1/3 and the entropy is ln 3.
    for s in STRICT_SEQS:
        assert math.exp(m.log_pseq(s)) == pytest.approx(1.0 / 3.0)
    assert m.entropy() == pytest.approx(math.log(3.0))


def test_strict_returns_zero_for_unspellable():
    """A sequence with no supporting path scores exactly zero, not merely small."""
    m = BeamModel(STRICT_SEQS, max_length=6, strict=True)
    # CASSW shares the CASS context but W was never emitted from it. In the
    # open-world model backoff would still price it; here there is no path.
    assert m.log_pseq("CASSW") == -math.inf
    assert m.log_pseq("WWWWW") == -math.inf
    assert math.exp(m.log_pseq("CASSW")) == 0.0


def test_open_world_never_returns_zero_for_an_observed_length():
    """The contrast the flag exists to make: the default floors instead."""
    m = BeamModel(STRICT_SEQS, max_length=6)
    for q in ("CASSW", "WWWWW"):
        assert math.isfinite(m.log_pseq(q))
        assert m.log_pseq(q) < 0.0
    assert m.hill_number(0) == pytest.approx(float(20 ** 5))


def test_strict_is_still_a_proper_distribution():
    """Killing cells must not leak mass; the trellis renormalises over survivors."""
    m = BeamModel(STRICT_SEQS, max_length=6, strict=True)
    d = m.pgen_diagnostics()
    assert d["is_proper"]
    assert d["total_mass"] == pytest.approx(1.0)
    assert d["length_prior_mass"] == pytest.approx(1.0)


def test_strict_trims_the_trellis():
    """`trim`'s general path must run, so topology reports live structure only.

    With no dead cell the trim collapses to pure reachability and every state
    is live at every layer. Under strict the weights decide, and the trellis
    shrinks to the handful of states the corpus actually visited.
    """
    m = BeamModel(STRICT_SEQS, max_length=6, strict=True)
    t = m.topology(5)
    assert t["n_nodes"] < 100          # open-world would be 328,421
    assert m.path_count(5) == pytest.approx(3.0)
    top = m.top_k_sequences(5)
    assert {s for s, _ in top} == set(STRICT_SEQS)


def test_strict_defaults_kappa_to_zero_but_explicit_wins():
    """Backoff off by default under strict, still reachable when asked for."""
    auto = BeamModel(STRICT_SEQS, max_length=6, strict=True)
    assert auto.config["kappa"] == (0.0, 0.0, 0.0)
    assert auto.config["kappa_wide"] == 0.0
    assert auto.config["kappa_len"] == 0.0
    assert auto.config["strict"] is True

    # Strict with backoff on: the floor and add-one are still gone, but a
    # shorter context can carry mass, so more than the training set survives.
    partial = BeamModel(STRICT_SEQS, max_length=6, strict=True,
                   kappa=(20.0, 20.0, 20.0), kappa_wide=20.0, kappa_len=20.0)
    assert partial.config["kappa"] == (20.0, 20.0, 20.0)
    assert partial.hill_number(0) > auto.hill_number(0)

    assert BeamModel(STRICT_SEQS, max_length=6).config["kappa"] == (20.0, 20.0, 20.0)


def test_strict_builder_matches_one_shot():
    """The streamed path must build the same strict model as the one-shot path."""
    b = BeamModel.builder(max_length=6, strict=True)
    b.observe(STRICT_SEQS)
    streamed = b.build()
    one_shot = BeamModel(STRICT_SEQS, max_length=6, strict=True)
    assert streamed.config == one_shot.config
    assert np.array_equal(streamed.log_pseq_batch(STRICT_SEQS),
                          one_shot.log_pseq_batch(STRICT_SEQS))
    assert streamed.hill_number(0) == one_shot.hill_number(0)


def test_strict_abundance_still_weights(corpus):
    """Zero probability is about support; abundance still shapes what survives."""
    m = BeamModel(["CASSF", "CASSY"], abundances=[9.0, 1.0],
             max_length=6, strict=True)
    assert math.exp(m.log_pseq("CASSF")) == pytest.approx(0.9)
    assert math.exp(m.log_pseq("CASSY")) == pytest.approx(0.1)
    assert m.hill_number(0) == pytest.approx(2.0)


def test_strict_survives_save_and_load(tmp_path):
    """`strict` must persist, and this failed silently when it did not.

    The file stores COUNTS, not finalized tables, so load re-finalizes. A
    strict model whose flag was dropped therefore came back with the add-one
    and the probability floor switched on again -- a third model, neither of
    the two on offer: kappa 0 so no backoff, but nothing dead, so D0 returned
    to 20**L and every floored cell contributed about 690 nats. On the 927M
    foundation that moved the surprisal range from ~230 nats to ~17,500 and
    made the spectrum grid useless, with no error anywhere.
    """
    m = BeamModel(STRICT_SEQS, max_length=6, strict=True)
    dst = tmp_path / "strict.model"
    m.save(dst)
    back = BeamModel.from_file(dst)

    assert back.config["strict"] is True
    assert back.config == m.config
    assert back.hill_number(0) == m.hill_number(0) == pytest.approx(3.0)
    assert back.log_pseq("CASSW") == -math.inf
    assert np.array_equal(back.log_pseq_batch(STRICT_SEQS),
                          m.log_pseq_batch(STRICT_SEQS))
    # The surprisal range is what blew up when the flag was lost.
    assert back.accumulator_max() == pytest.approx(m.accumulator_max())


def test_open_world_save_and_load_unchanged(tmp_path):
    """The reserved-word reuse must not disturb existing files."""
    m = BeamModel(STRICT_SEQS, max_length=6)
    dst = tmp_path / "open.model"
    m.save(dst)
    back = BeamModel.from_file(dst)
    assert back.config["strict"] is False
    assert back.hill_number(0) == m.hill_number(0)
    assert np.array_equal(back.log_pseq_batch(STRICT_SEQS),
                          m.log_pseq_batch(STRICT_SEQS))


# ── Sparse strict storage ──────────────────────────────────────────


@pytest.mark.parametrize(("order", "depth_cap", "abundances"), [
    (2, 5, None),
    (3, 5, [2.0, 5.0, 7.0]),
    (4, 1, [3.0, 1.0, 4.0]),
])
def test_sparse_strict_is_bit_identical_to_dense_strict(
        order, depth_cap, abundances):
    """Storage changes allocation only; every public result stays identical."""
    kwargs = dict(order=order, depth_cap=depth_cap, max_length=6, strict=True,
                  abundances=abundances)
    dense = BeamModel(STRICT_SEQS, sparse=False, **kwargs)
    sparse = BeamModel(STRICT_SEQS, sparse=True, **kwargs)

    assert dense.config["sparse"] is False
    assert sparse.config["sparse"] is True

    probe = STRICT_SEQS + ["CASSW", "WWWWW", "CTTGF"]
    assert np.array_equal(dense.log_pseq_batch(probe),
                          sparse.log_pseq_batch(probe))
    assert sparse.log_pseq("CASSW") == -math.inf

    assert dense.n_observed == sparse.n_observed
    assert dense.n_skipped == sparse.n_skipped
    assert dense.total_weight == sparse.total_weight
    assert dense.n_nonzero_cells == sparse.n_nonzero_cells
    assert dense.path_count(5) == sparse.path_count(5)
    assert dense.entropy() == sparse.entropy()
    for alpha in (0.0, 1.0, 2.0):
        assert dense.hill_number(alpha) == sparse.hill_number(alpha)

    dense_topology = dense.topology(5)
    sparse_topology = sparse.topology(5)
    for key in ("length", "n_nodes", "n_edges", "max_layer_nodes",
                "mean_layer_nodes", "density"):
        assert dense_topology[key] == sparse_topology[key]
    assert np.array_equal(dense_topology["nodes_by_layer"],
                          sparse_topology["nodes_by_layer"])
    assert dense.top_k_sequences(10, length=5) == sparse.top_k_sequences(
        10, length=5)
    dense_explain = dense.explain(STRICT_SEQS[0])
    sparse_explain = sparse.explain(STRICT_SEQS[0])
    assert np.array_equal(dense_explain["forward"],
                          sparse_explain["forward"])
    assert np.array_equal(dense_explain["backward"],
                          sparse_explain["backward"])
    assert dense.sample(20, length=5, seed=17) == sparse.sample(
        20, length=5, seed=17)

    for measure in ("counting", "generated"):
        dense_spectrum = dense.spectrum(measure=measure, length=5, bins=32)
        sparse_spectrum = sparse.spectrum(measure=measure, length=5, bins=32)
        assert dense_spectrum["spacing"] == sparse_spectrum["spacing"]
        assert np.array_equal(dense_spectrum["surprisal"],
                              sparse_spectrum["surprisal"])
        assert np.array_equal(dense_spectrum["weights"],
                              sparse_spectrum["weights"])

    assert np.array_equal(dense.surprisal_moments(order=4),
                          sparse.surprisal_moments(order=4))
    assert dense.pgen_diagnostics()["total_mass"] == 1.0
    assert sparse.pgen_diagnostics()["total_mass"] == 1.0

    cfg = dense.config
    expected_cells = 2 * (
        (cfg["depth_cap"] + 1) * 21 ** cfg["order"] * 20
        + cfg["max_length"] * 21 ** 3 * 20
        + (cfg["max_length"] + 1) * cfg["max_length"] * 21 ** 2 * 20
    )
    assert dense.n_cells == sparse.n_cells == expected_cells
    assert dense.memory_bytes >= 20 * sparse.memory_bytes


def test_sparse_selection_and_positive_backoff_fallback():
    """Automatic mode is sparse only in the exact zero-backoff regime."""
    automatic = BeamModel(STRICT_SEQS, order=2, max_length=6, strict=True)
    forced_dense = BeamModel(STRICT_SEQS, order=2, max_length=6, strict=True,
                        sparse=False)
    open_world = BeamModel(STRICT_SEQS, order=2, max_length=6)
    backed_off = BeamModel(
        STRICT_SEQS, order=2, max_length=6, strict=True,
        kappa=(1.0, 1.0, 1.0), kappa_wide=1.0, kappa_len=1.0,
        sparse=None,
    )

    assert automatic.config["sparse"] is True
    assert forced_dense.config["sparse"] is False
    assert open_world.config["sparse"] is False
    assert backed_off.config["sparse"] is False
    assert backed_off.hill_number(0) > automatic.hill_number(0)

    with pytest.raises(ValueError, match="sparse=True"):
        BeamModel(STRICT_SEQS, order=2, max_length=6, strict=True,
             kappa=(1.0, 1.0, 1.0), kappa_wide=1.0, kappa_len=1.0,
             sparse=True)
    with pytest.raises(ValueError, match="sparse=True"):
        BeamModel(STRICT_SEQS, order=2, max_length=6, sparse=True)


def test_sparse_subtract_shrinks_and_copy_is_independent():
    """Rows disappear at zero, while copy and streamed snapshot own storage."""
    prefix = STRICT_SEQS[:2]
    removed = STRICT_SEQS[2:]
    kwargs = dict(order=2, max_length=6, strict=True, sparse=True)

    builder = BeamModel.builder(**kwargs)
    builder.observe(prefix)
    builder_occupancy = builder.n_nonzero_cells
    assert builder_occupancy > 0
    snapshot = builder.snapshot()
    snapshot_scores = snapshot.log_pseq_batch(STRICT_SEQS)
    snapshot_occupancy = snapshot.n_nonzero_cells

    builder.observe(removed)
    assert builder.n_nonzero_cells >= builder_occupancy
    full = builder.build()
    expected_snapshot = BeamModel(prefix, **kwargs)
    assert np.array_equal(snapshot_scores,
                          expected_snapshot.log_pseq_batch(STRICT_SEQS))
    assert snapshot.n_nonzero_cells == snapshot_occupancy
    assert np.array_equal(snapshot.log_pseq_batch(STRICT_SEQS),
                          snapshot_scores)

    original_scores = full.log_pseq_batch(STRICT_SEQS)
    original_occupancy = full.n_nonzero_cells
    held_out = full.copy()
    held_out.subtract(BeamModel(removed, **kwargs))

    assert held_out.n_nonzero_cells < original_occupancy
    assert held_out.n_nonzero_cells == expected_snapshot.n_nonzero_cells
    assert np.array_equal(held_out.log_pseq_batch(STRICT_SEQS),
                          expected_snapshot.log_pseq_batch(STRICT_SEQS))
    assert full.n_nonzero_cells == original_occupancy
    assert np.array_equal(full.log_pseq_batch(STRICT_SEQS), original_scores)

    untouched = full.copy()
    with pytest.raises(ValueError, match="every sequence"):
        untouched.subtract(full.copy())
    assert np.array_equal(untouched.log_pseq_batch(STRICT_SEQS),
                          original_scores)


def test_sparse_add_posterior_without_and_compact_match_dense(tmp_path):
    """Every count mutation supports sparse and mixed-storage operands."""
    kwargs = dict(order=2, max_length=6, strict=True)
    left = STRICT_SEQS[:2]
    right = STRICT_SEQS[2:]
    probes = STRICT_SEQS + ["CASSW", "WWWWW"]

    dense_left = BeamModel(left, sparse=False, **kwargs)
    sparse_left = BeamModel(left, sparse=True, **kwargs)
    dense_right = BeamModel(right, sparse=False, **kwargs)
    sparse_right = BeamModel(right, sparse=True, **kwargs)
    together = BeamModel(STRICT_SEQS, sparse=False, **kwargs)

    dense_plus_sparse = dense_left.copy()
    dense_plus_sparse.add(sparse_right)
    sparse_plus_dense = sparse_left.copy()
    sparse_plus_dense.add(dense_right)
    expected = together.log_pseq_batch(probes)
    assert np.array_equal(dense_plus_sparse.log_pseq_batch(probes), expected)
    assert np.array_equal(sparse_plus_dense.log_pseq_batch(probes), expected)

    dense_post = dense_left.with_prior(sparse_right, kappa=3.0)
    sparse_post = sparse_left.with_prior(dense_right, kappa=3.0)
    assert np.array_equal(dense_post.log_pseq_batch(probes),
                          sparse_post.log_pseq_batch(probes))

    without_rows = sparse_plus_dense.without(right)
    assert without_rows.config["sparse"] is True
    assert np.array_equal(without_rows.log_pseq_batch(probes),
                          sparse_left.log_pseq_batch(probes))

    before = sparse_plus_dense.memory_bytes
    scores = sparse_plus_dense.log_pseq_batch(probes)
    sparse_plus_dense.compact()
    assert sparse_plus_dense.memory_bytes < before
    assert np.array_equal(sparse_plus_dense.log_pseq_batch(probes), scores)
    with pytest.raises(RuntimeError, match="compact"):
        sparse_plus_dense.save(tmp_path / "compacted.edge")


def test_sparse_single_sequence_has_minimal_support():
    model = BeamModel(["CASSF"], order=2, max_length=6, strict=True)
    assert model.config["sparse"] is True
    assert model.n_nonzero_cells == 3 * 2 * len("CASSF")
    assert model.log_pseq("CASSF") == 0.0
    assert model.path_count(5) == 1.0


def test_sparse_results_are_thread_count_independent():
    from BEAM import threads, set_threads

    prior = threads()
    kwargs = dict(order=2, max_length=6, strict=True, sparse=True)
    try:
        set_threads(1)
        serial = BeamModel(STRICT_SEQS * 20, **kwargs)
        set_threads(4)
        parallel = BeamModel(STRICT_SEQS * 20, **kwargs)
    finally:
        set_threads(prior)

    probes = STRICT_SEQS + ["CASSW", "WWWWW"]
    assert np.array_equal(serial.log_pseq_batch(probes),
                          parallel.log_pseq_batch(probes))
    assert serial.entropy() == parallel.entropy()
    assert serial.hill_number(0) == parallel.hill_number(0)


def test_sparse_strict_save_load_and_crc(tmp_path):
    """The sparse mode and exact model survive disk; row payloads stay covered."""
    model = BeamModel(STRICT_SEQS, order=2, max_length=6, strict=True, sparse=True)
    path = tmp_path / "strict-sparse.edge"
    model.save(path)
    restored = BeamModel.from_file(path)

    assert restored.config == model.config
    assert restored.config["sparse"] is True
    assert restored.n_cells == model.n_cells
    assert restored.n_nonzero_cells == model.n_nonzero_cells
    assert restored.hill_number(0) == model.hill_number(0)
    assert np.array_equal(restored.log_pseq_batch(STRICT_SEQS + ["CASSW"]),
                          model.log_pseq_batch(STRICT_SEQS + ["CASSW"]))

    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xA5
    path.write_bytes(data)
    with pytest.raises(OSError):
        BeamModel.from_file(path)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_backward_hoist_is_correct_at_every_order(order, corpus):
    """The backward-lookup hoist is only valid at order 4; prove the rest work.

    `state_weights_sparse` can lift two of its three backward table lookups out
    of the residue loop, but only for some configurations. The settled residue
    sits at ``p = i - K`` with ``K = max(order, 3)``, so its order-J backward
    context is ``s[i-K+1 .. i-K+J]`` and the emitted residue ``s[i]`` is inside
    it exactly when ``J >= K``:

        order 4 (K=4): wide J=4 -> x is the last digit, cw(x) == cw(0) + x
                       mid  J=3 -> invariant, hoistable
                       len  J=2 -> invariant, hoistable
        order 3 (K=3): mid  J=3 -> x IS in the context, NOT hoistable
        order 2 (K=3): wide J=2 -> x is not in the wide context at all,
                                   so cw(x) == cw(0) and "+ x" is wrong

    Hoisting unconditionally drove every partition function to -inf at order 2:
    the model built, reported no error, and could generate nothing. The
    reference values in this file are all order 4, so none of them caught it.
    """
    seqs, abund = corpus
    m = BeamModel(seqs[:600], abundances=abund[:600], order=order, max_length=20,
             strict=True)
    d = m.pgen_diagnostics()
    assert d["is_proper"], f"order {order} is not a proper distribution"
    assert d["total_mass"] == pytest.approx(1.0)
    assert math.isfinite(m.log_partition(m.observed_lengths[0]))
    assert m.hill_number(0) > 0.0
    assert m.hill_number(1) > 0.0
    # every training sequence must be spellable by the model built from it
    scores = m.log_pseq_batch(seqs[:50])
    assert np.isfinite(scores).all(), f"order {order} cannot spell its own input"


# ═══════════════════════════════════════════════════════════════
# Live structure: the frontier's edge masks and the per-length cache
# ═══════════════════════════════════════════════════════════════

def _enumerate_live_trellis(m, length):
    """Nodes and edges of the trellis at one length, counted from OUTSIDE.

    Independent of everything `topology` does. It enumerates every string of
    the requested length, asks the model which ones it can spell at all, and
    then reads the DAG structure straight off the survivors: a live node is a
    (layer, state) pair some spellable string passes through, and a live edge
    is a (layer, state, residue) triple some spellable string uses.

    That is the definition of "live" -- forward-reachable AND able to reach a
    valid ending -- restated as a property of complete paths, which is the one
    form that cannot be derived from the sweep it is checking. Only usable for
    lengths small enough to enumerate: 20**4 is 160,000 strings.
    """
    import itertools
    every = ["".join(t) for t in itertools.product(AA, repeat=length)]
    lp = m.log_pseq_batch(every)
    nodes, edges = set(), set()
    K = 3                                  # state_order for order 2 and 3
    for seq, v in zip(every, lp):
        if not math.isfinite(v):
            continue
        for i in range(length + 1):
            state = tuple(seq[i - K + j] if 0 <= i - K + j < i else "."
                          for j in range(K))
            nodes.add((i, state))
            if i < length:
                edges.add((i, state, seq[i]))
    by_layer = [0] * (length + 1)
    for i, _ in nodes:
        by_layer[i] += 1
    return len(nodes), len(edges), by_layer


@pytest.mark.parametrize("strict", [True, False])
def test_topology_matches_an_enumerated_trellis(strict):
    """`topology` counts what a brute-force enumeration of the support counts.

    This is the test that would catch the frontier's edge masks being wrong.
    `path_count` and `topology` no longer read a single edge WEIGHT: they read
    twenty bits per state that the frontier distilled from the weights while
    it was discovering reachability. A wrong bit adds or drops an edge, and
    nothing else in this file would notice, because every other check compares
    the sweeps against each other.
    """
    rng = random.Random(19)
    seqs = ["".join(rng.choice(AA) for _ in range(4)) for _ in range(120)]
    m = BeamModel(seqs, order=2, max_length=5, strict=strict)
    want_nodes, want_edges, want_layers = _enumerate_live_trellis(m, 4)
    t = m.topology(4)
    assert t["n_nodes"] == want_nodes
    assert t["n_edges"] == want_edges
    assert [int(v) for v in t["nodes_by_layer"]] == want_layers


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("strict", [True, False])
def test_live_structure_is_identical_on_every_repeat_call(order, strict):
    """Cached and uncached answers are the same double, not a close one.

    The per-length live structure is cached on the model, so the second call
    to any of these runs different code from the first. `==` rather than
    `approx` on purpose: a cache that shifts the last bit is a bug, not a
    tolerance.
    """
    seqs, abund = make_corpus(n=300, seed=5, lo=6, hi=9)
    m = BeamModel(seqs, abundances=abund, order=order, max_length=10, strict=strict)
    for length in m.observed_lengths:
        first = m.path_count(length)
        assert m.path_count(length) == first
        a = m.topology(length)
        b = m.topology(length)
        assert (a["n_nodes"], a["n_edges"]) == (b["n_nodes"], b["n_edges"])
        assert list(a["nodes_by_layer"]) == list(b["nodes_by_layer"])
    assert m.hill_number(0) == m.hill_number(0)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_live_structure_is_rebuilt_after_every_mutation(order):
    """A mutated model answers as a model built that way from the start.

    The live structure is a projection of the finalized tables, so anything
    that changes the counts has to invalidate it. Comparing against a FRESH
    build rather than merely against "something different" is what makes this
    a staleness test: a cache that was not dropped keeps returning the old
    model's structure, which is still a perfectly self-consistent answer.
    """
    seqs, abund = make_corpus(n=200, seed=7, lo=6, hi=9)
    more, more_ab = make_corpus(n=120, seed=8, lo=6, hi=9)
    kw = dict(order=order, max_length=10, strict=True)
    length = 8

    base = BeamModel(seqs, abundances=abund, **kw)
    other = BeamModel(more, abundances=more_ab, **kw)
    both = BeamModel(seqs + more, abundances=abund + more_ab, **kw)

    warm = base.path_count(length)                     # fills the cache
    assert base.topology(length)["n_nodes"] > 0

    merged = base.merged(other)                        # copy + add
    assert merged.path_count(length) == both.path_count(length)
    assert (merged.topology(length)["n_nodes"]
            == both.topology(length)["n_nodes"])

    back = merged.copy()
    back.subtract(other)
    assert back.path_count(length) == warm
    assert back.topology(length)["n_nodes"] == base.topology(length)["n_nodes"]

    # posterior mixes in a prior; the point is only that the answer moves off
    # the cached one rather than being served from it.
    post = base.with_prior(other, kappa=50.0)
    assert post.path_count(length) > 0
    assert base.path_count(length) == warm, "the operand must be untouched"


def test_live_cache_does_not_survive_a_save_and_load(tmp_path):
    """A loaded model builds its own structure, and gets the same answer.

    The file stores counts, so loading re-finalizes. Nothing about the live
    structure is serialised and nothing should be: it is a projection of
    tables that the reader rebuilds.
    """
    seqs, abund = make_corpus(n=200, seed=13, lo=6, hi=9)
    m = BeamModel(seqs, abundances=abund, order=2, max_length=10, strict=True)
    want = {L: m.path_count(L) for L in m.observed_lengths}
    topo = {L: m.topology(L)["n_edges"] for L in m.observed_lengths}
    path = tmp_path / "live.edge"
    m.save(path)
    back = BeamModel.from_file(path)
    for L, v in want.items():
        assert back.path_count(L) == v
        assert back.topology(L)["n_edges"] == topo[L]


@pytest.mark.parametrize("threads", [1, 4, 8])
def test_live_structure_does_not_depend_on_the_thread_count(threads):
    """Thread count changes which thread builds which length, never the answer.

    The cache is filled from inside the parallel per-length loop, so this is
    also the test that two threads racing to publish the same length cannot
    produce two different structures.
    """
    import BEAM
    seqs, abund = make_corpus(n=400, seed=23, lo=8, hi=12)
    saved = BEAM.threads()
    try:
        BEAM.set_threads(1)
        m = BeamModel(seqs, abundances=abund, order=2, max_length=14, strict=True)
        want_pc = {L: m.path_count(L) for L in m.observed_lengths}
        want_topo = {L: m.topology(L)["n_edges"] for L in m.observed_lengths}
        want_d0 = m.hill_number(0)

        BEAM.set_threads(threads)
        m2 = BeamModel(seqs, abundances=abund, order=2, max_length=14, strict=True)
        assert m2.hill_number(0) == want_d0
        for L, v in want_pc.items():
            assert m2.path_count(L) == v
            assert m2.topology(L)["n_edges"] == want_topo[L]
    finally:
        BEAM.set_threads(saved)
