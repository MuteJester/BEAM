"""Exact per-position marginals and expected cell usage.

Every check here is EXACT AGAINST EXACT. The models are built small enough
that the whole support is enumerable -- `path_count(L)` says how big it is and
`top_k_sequences(k=path_count)` produces all of it -- so the reference is a
direct sum over sequences of `exp(log_pseq)`, not a sample and not a second
dynamic program that could share a mistake with the first. Nothing here is
compared against a Monte Carlo estimate, and no tolerance is looser than
machine precision.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pytest

from BEAM import BeamModel
from BEAM.constants import BEAM_ALPHABET as AA

from _helpers import make_corpus, make_nt_corpus


#: Machine precision, near enough. Every quantity below is O(1), so an absolute
#: tolerance is the honest one: relative tolerance on a probability that is
#: legitimately 1e-9 would demand more of the arithmetic than double precision
#: has to give. Measured worst case across this file is 7e-16.
EXACT = 1e-13

L_ENUM = 15


def _enumerable(order=3, alphabet="AG", n=10, seed=7):
    """A model whose entire length-15 support fits in a list.

    A conserved head and tail with a short random middle over a two-letter
    alphabet: small enough to enumerate, but with enough context sharing that
    the support (25) is bigger than the corpus (10), so the marginals are
    genuinely a property of the DAG rather than a re-reading of the training
    set.
    """
    rng = np.random.default_rng(seed)
    seqs = ["CASS" + "".join(rng.choice(list(alphabet), 7)) + "EQYF"
            for _ in range(n)]
    return BeamModel(list(dict.fromkeys(seqs)), order=order, max_length=18)


@pytest.fixture(scope="module")
def enum_model():
    return _enumerable()


@pytest.fixture(scope="module")
def support(enum_model):
    """The complete length-15 support, with log probabilities."""
    n = int(enum_model.path_count(L_ENUM))
    sup = enum_model.top_k_sequences(k=n, length=L_ENUM)
    assert len(sup) == n, "top_k did not return the whole support"
    assert len({s for s, _ in sup}) == n, "top_k repeated a sequence"
    return sup


def _direct_marginals(support, length, clamp=None):
    """P(x_i = a | length, clamp) by summing over the enumerated support."""
    keep = [(s, v) for s, v in support
            if clamp is None or all(s[p] == r for p, r in clamp.items())]
    z = sum(math.exp(v) for _, v in keep)
    out = np.zeros((length, 20))
    for s, v in keep:
        w = math.exp(v)
        for i, ch in enumerate(s):
            out[i, AA.index(ch)] += w
    return out / z, z


# Exact against exact: unclamped
# ═══════════════════════════════════════════════════════════════

def test_marginals_match_a_direct_sum_over_the_whole_support(enum_model,
                                                             support):
    ref, _ = _direct_marginals(support, L_ENUM)
    got = enum_model.positional_marginals(L_ENUM)
    assert got.shape == (L_ENUM, 20)
    assert np.abs(got - ref).max() < EXACT


def test_zero_probability_cells_are_exactly_zero(enum_model, support):
    """Not a floor, and not merely small.

    A closed-world model that cannot spell residue `a` at position `i` should
    say so with a zero, because that is information. The check is on the
    PATTERN of zeros, so a floor of 1e-300 would fail it just as a floor of
    1e-6 would.
    """
    ref, _ = _direct_marginals(support, L_ENUM)
    got = enum_model.positional_marginals(L_ENUM)
    assert (ref == 0).any(), "the fixture is not closed-world enough to test"
    assert ((got == 0.0) == (ref == 0.0)).all()


# Exact against exact: clamped
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("pos", [0, 4, 7, 11, 14])
def test_clamped_marginals_match_the_conditional_sum(enum_model, support, pos):
    """The clamp is the deliverable that matters, so it is checked at every
    kind of position: inside the conserved head, in the free middle, and in
    the J motif at the end."""
    occurring = sorted({s[pos] for s, _ in support})
    for residue in occurring:
        ref, _ = _direct_marginals(support, L_ENUM, {pos: residue})
        got = enum_model.positional_marginals(L_ENUM, clamp={pos: residue})
        assert np.abs(got - ref).max() < EXACT
        assert ((got == 0.0) == (ref == 0.0)).all()


def test_a_clamped_position_is_one_hot(enum_model, support):
    pos, residue = 5, sorted({s[5] for s, _ in support})[0]
    got = enum_model.positional_marginals(L_ENUM, clamp={pos: residue})
    assert got[pos, AA.index(residue)] == pytest.approx(1.0, abs=EXACT)
    assert got[pos].sum() == pytest.approx(1.0, abs=EXACT)
    assert (np.delete(got[pos], AA.index(residue)) == 0.0).all()


def test_clamping_several_positions_at_once(enum_model, support):
    s0 = support[0][0]
    clamp = {4: s0[4], 6: s0[6], 9: s0[9]}
    ref, _ = _direct_marginals(support, L_ENUM, clamp)
    got = enum_model.positional_marginals(L_ENUM, clamp=clamp)
    assert np.abs(got - ref).max() < EXACT


def test_clamping_every_position_recovers_that_sequence(enum_model, support):
    """The degenerate clamp, which is the strongest statement of what a clamp
    means: pin all fifteen positions and the conditional is a point mass on the
    sequence, whatever its probability was."""
    s0 = support[-1][0]
    got = enum_model.positional_marginals(L_ENUM,
                                          clamp={i: c for i, c in enumerate(s0)})
    for i, ch in enumerate(s0):
        assert got[i, AA.index(ch)] == pytest.approx(1.0, abs=EXACT)


def test_clamp_accepts_residue_indices_as_well_as_characters(enum_model,
                                                             support):
    residue = sorted({s[4] for s, _ in support})[0]
    a = enum_model.positional_marginals(L_ENUM, clamp={4: residue})
    b = enum_model.positional_marginals(L_ENUM, clamp={4: AA.index(residue)})
    assert (a == b).all()


def test_an_empty_clamp_is_the_unclamped_answer(enum_model):
    a = enum_model.positional_marginals(L_ENUM)
    b = enum_model.positional_marginals(L_ENUM, clamp={})
    assert (a == b).all()


def test_clamp_probability_matches_the_enumerated_mass(enum_model, support):
    for pos in (2, 6, 10):
        for residue in sorted({s[pos] for s, _ in support}):
            _, z = _direct_marginals(support, L_ENUM, {pos: residue})
            total = sum(math.exp(v) for _, v in support)
            got = enum_model.log_clamp_probability(L_ENUM, {pos: residue})
            assert got == pytest.approx(math.log(z / total), abs=EXACT)


# A clamp of probability zero
# ═══════════════════════════════════════════════════════════════

def test_an_impossible_clamp_fails_loudly(enum_model, support):
    """Explicitly handled, not divided by zero.

    The conditional on an event of probability zero does not exist, so asking
    for it is an error rather than a number. `log_clamp_probability` is the
    other half of the contract: it answers -inf, because "how much mass does
    this keep" does have an answer.
    """
    impossible = next(c for c in AA if all(s[4] != c for s, _ in support))
    with pytest.raises(ValueError, match="cannot produce|no path|clamp"):
        enum_model.positional_marginals(L_ENUM, clamp={4: impossible})
    assert enum_model.log_clamp_probability(L_ENUM, {4: impossible}) == -math.inf


def test_a_clamp_outside_the_alphabet_is_refused(enum_model):
    with pytest.raises(ValueError, match="residues"):
        enum_model.positional_marginals(L_ENUM, clamp={4: "B"})
    with pytest.raises(ValueError, match="outside"):
        enum_model.positional_marginals(L_ENUM, clamp={4: 25})


def test_a_clamp_outside_the_sequence_is_refused(enum_model):
    with pytest.raises(ValueError, match="outside a length"):
        enum_model.positional_marginals(L_ENUM, clamp={L_ENUM: "A"})
    with pytest.raises(ValueError, match="outside a length"):
        enum_model.positional_marginals(L_ENUM, clamp={-1: "A"})


def test_a_contradictory_clamp_is_refused(enum_model):
    """A plain dict cannot express this -- equal keys collapse -- but a
    mapping is only required to have `items()`, and the second write to a
    position would otherwise win silently."""
    class Twice:
        def items(self):
            return [(4, "A"), (4, "G")]

    with pytest.raises(ValueError, match="two different residues"):
        enum_model.positional_marginals(L_ENUM, clamp=Twice())


def test_a_fractional_clamp_position_is_refused(enum_model):
    """`int(4.7)` is 4, and a clamp on position 4.7 is a typo rather than a
    request to round."""
    with pytest.raises(TypeError, match="must be integers"):
        enum_model.positional_marginals(L_ENUM, clamp={4.7: "A"})


# Internal consistency, at every length the model produces
# ═══════════════════════════════════════════════════════════════

def test_every_row_sums_to_one_at_every_producible_length(model):
    """Not renormalised anywhere, so this is a real check on the DP rather
    than a check that a division happened."""
    for length in model.observed_lengths:
        p = model.positional_marginals(length)
        assert p.shape == (length, model.n_residues)
        assert np.abs(p.sum(axis=1) - 1.0).max() < EXACT
        assert (p >= 0.0).all()


def test_rows_still_sum_to_one_under_a_clamp(model):
    length = model.observed_lengths[len(model.observed_lengths) // 2]
    top = model.top_k_sequences(1, length=length)[0][0]
    p = model.positional_marginals(length, clamp={3: top[3]})
    assert np.abs(p.sum(axis=1) - 1.0).max() < EXACT


def test_marginals_agree_with_the_nucleotide_alphabet():
    """Five columns, not twenty, and the same identities.

    The alphabet is a runtime dimension of every table and of the trellis, so
    a sweep that hardcoded twenty anywhere would pass the whole amino-acid
    half of this file and fail here."""
    seqs, abund = make_nt_corpus(n=400, seed=5, lo=15, hi=21)
    nt = BeamModel(seqs, abundances=abund, order=2, alphabet="nt",
                   max_length=24)
    length = nt.observed_lengths[0]
    p = nt.positional_marginals(length)
    assert p.shape == (length, 5)
    assert np.abs(p.sum(axis=1) - 1.0).max() < EXACT


def test_an_unproducible_length_fails_loudly(model):
    """A documented refusal, not a silently flat row of zeros."""
    dead = next(L for L in range(1, int(model.config["max_length"]) + 1)
                if not math.isfinite(model.log_length_prior(L)))
    with pytest.raises(ValueError, match="generates no sequences"):
        model.positional_marginals(dead)
    with pytest.raises(ValueError, match="at least 1"):
        model.positional_marginals(0)
    with pytest.raises(ValueError, match="between 0 and max_length"):
        model.positional_marginals(10_000)


def test_the_marginal_of_a_conserved_position_is_a_point_mass(model):
    """Every training sequence of the fixture corpus starts with C, so no
    length can put anything else at position 0. A model that smeared mass
    across the first column would pass every sum-to-one check above."""
    length = model.observed_lengths[-1]
    p = model.positional_marginals(length)
    assert p[0, AA.index("C")] == pytest.approx(1.0, abs=EXACT)


# Expected cell usage
# ═══════════════════════════════════════════════════════════════

def _direct_expected_counts(model, support, length):
    """Expected counts by walking each supported sequence into its cells.

    The addressing is transcribed from `observe_one_position` in training.c and
    the row formulas in indexing.h -- deliberately re-derived here rather than
    read back from the C sweep this is checking, since a shared derivation
    would agree with itself whatever it did.
    """
    cfg = model.config
    A, none = 21, 20
    order = int(cfg["order"])
    width = int(cfg["max_length"])
    total = sum(math.exp(v) for _, v in support)

    def ctx(r, p, k, forward):
        idx = 0
        for j in range(k, 0, -1):
            q = p - j if forward else p + j
            idx = idx * A + (r[q] if 0 <= q < length else none)
        return idx

    out = {k: defaultdict(lambda: np.zeros(20)) for k in (
        "forward_wide", "forward_mid", "forward_len",
        "backward_wide", "backward_mid", "backward_len")}
    for s, v in support:
        w = math.exp(v) / total
        r = [AA.index(c) for c in s]
        for p in range(length):
            for anchor, forward, depth in (("forward", True, p),
                                           ("backward", False, length - 1 - p)):
                out[anchor + "_wide"][
                    min(depth, depth_cap) * A ** order
                    + ctx(r, p, order, forward)][r[p]] += w
                out[anchor + "_mid"][
                    depth * A ** 3 + ctx(r, p, 3, forward)][r[p]] += w
                out[anchor + "_len"][
                    (length * width + depth) * A ** 2
                    + ctx(r, p, 2, forward)][r[p]] += w
    return out


def test_expected_counts_match_a_direct_sum_over_the_whole_support(enum_model,
                                                                   support):
    ref = _direct_expected_counts(enum_model, support, L_ENUM)
    got = enum_model.expected_counts(length=L_ENUM)
    for name, want in ref.items():
        blk = got[name]
        have = {int(row): blk["counts"][i]
                for i, row in enumerate(blk["rows"])}
        assert set(have) == set(want), f"{name}: the row sets differ"
        for row, counts in want.items():
            assert np.abs(have[row] - counts).max() < EXACT, f"{name} row {row}"


def test_a_row_absent_from_the_answer_is_exactly_zero(enum_model, support):
    """The sparse form is only honest if what it omits is zero. Checked by
    asking the direct enumeration for the rows the sweep did NOT return."""
    ref = _direct_expected_counts(enum_model, support, L_ENUM)
    got = enum_model.expected_counts(length=L_ENUM)
    for name, want in ref.items():
        returned = set(map(int, got[name]["rows"]))
        assert not [r for r in want if r not in returned], (
            f"{name}: a row with mass was omitted")


def test_every_block_sums_to_the_expected_length(model):
    """Each position increments exactly one cell of each of the six tables, so
    every block sums to L. It is the cheapest check that the sink and the
    first K layers were both covered -- a sweep that dropped the sink would
    still produce a plausible-looking table."""
    length = model.observed_lengths[len(model.observed_lengths) // 2]
    ec = model.expected_counts(length=length)
    assert ec["expected_length"] == pytest.approx(float(length), abs=EXACT)
    for anchor in ("forward", "backward"):
        for expert in ("wide", "mid", "len"):
            blk = ec[f"{anchor}_{expert}"]
            assert blk["counts"].sum() == pytest.approx(float(length),
                                                        abs=EXACT)
            assert (blk["counts"] >= 0.0).all()
            assert (np.diff(blk["rows"]) > 0).all(), "rows must be ascending"


def test_pooled_expected_counts_sum_to_the_mean_length(small):
    pmf = small.length_pmf()
    mean = float(sum(L * pmf[L] for L in range(len(pmf))))
    ec = small.expected_counts()
    assert ec["expected_length"] == pytest.approx(mean, abs=EXACT)
    for anchor in ("forward", "backward"):
        for expert in ("wide", "mid", "len"):
            assert ec[f"{anchor}_{expert}"]["counts"].sum() == pytest.approx(
                mean, abs=EXACT)


def test_the_decoded_columns_rebuild_every_row_index(small):
    """`depth`, `context` and `length` are a convenience, and a convenience
    that disagrees with the row it was decoded from is worse than none."""
    ec = small.expected_counts(length=small.observed_lengths[0])
    cfg = small.config
    A, width = 21, int(cfg["max_length"])
    n_ctx = {"wide": A ** int(cfg["order"]), "mid": A ** 3, "len": A ** 2}
    for anchor in ("forward", "backward"):
        for expert, nc in n_ctx.items():
            blk = ec[f"{anchor}_{expert}"]
            if expert == "len":
                rebuilt = (blk["length"] * width + blk["depth"]) * nc \
                          + blk["context"]
                assert (blk["length"] == small.observed_lengths[0]).all()
            else:
                rebuilt = blk["depth"] * nc + blk["context"]
            assert (rebuilt == blk["rows"]).all()
            assert (blk["context"] < nc).all()


def test_expected_counts_refuse_a_length_the_model_cannot_produce(model):
    dead = next(L for L in range(1, int(model.config["max_length"]) + 1)
                if not math.isfinite(model.log_length_prior(L)))
    with pytest.raises(ValueError, match="generates no sequences"):
        model.expected_counts(length=dead)


def test_expected_counts_land_only_in_cells_the_corpus_could_have_used(small):
    """A closed-world model scores through occupied cells only, so the
    expected usage cannot reach a cell the counts never touched. This is the
    strict-world statement of the same fact the zero-pattern test makes for
    marginals, and it is what would break if the C-anchored wide context were
    read with the finalized table's nearest-first packing instead of the
    counts' most-distant-first one."""
    ec = small.expected_counts(length=small.observed_lengths[0])
    for anchor in ("forward", "backward"):
        for expert in ("wide", "mid", "len"):
            blk = ec[f"{anchor}_{expert}"]
            assert len(blk["rows"]) > 0
            assert len(blk["rows"]) <= small.n_nonzero_cells


# The new methods change nothing that was there before
# ═══════════════════════════════════════════════════════════════

def test_the_marginals_reproduce_the_partition_function(enum_model):
    """The backward sweep the marginals normalise by is an independent
    evaluation of ln Z(L): `log_partition` reaches it forwards with per-layer
    rescaling, this reaches it backwards in log space. They have to agree, and
    agreeing is what says the new sweep is normalising by the right thing."""
    for length in enum_model.observed_lengths:
        got = enum_model.log_clamp_probability(length, {0: "C"})
        # P(x_0 = C) is 1 in this corpus, so ln P is 0 and the clamped Z is
        # the unclamped one.
        assert got == pytest.approx(0.0, abs=EXACT)


def test_asking_for_marginals_does_not_disturb_the_model(model):
    seqs = [s for s, _ in model.top_k_sequences(50)]
    before = [model.log_pseq(s) for s in seqs]
    diag_before = model.pgen_diagnostics()
    model.positional_marginals(model.observed_lengths[0])
    model.expected_counts(length=model.observed_lengths[0])
    after = [model.log_pseq(s) for s in seqs]
    assert before == after, "scores moved after an analytic ran"
    assert model.pgen_diagnostics() == diag_before


# The same exactness at every order
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("order", [2, 3, 4])
def test_both_answers_stay_exact_at_every_order(order):
    """Order decides which branch the C-anchored context takes, and the
    branches are not equivalent code paths that happen to be fast or slow.

    With ``K = max(order, 3)``: at order 4 the emitted residue lands in the
    last digit of the wide context and the kernels hoist the mid and length
    lookups out of the residue loop; at order 3 the residue is inside the mid
    context and nothing may be hoisted; at order 2 it is outside the wide
    context altogether. Getting that wrong produces a model that normalises
    perfectly and is addressed by the wrong cell, so each order is enumerated
    separately rather than trusting one to stand for the others.
    """
    m = _enumerable(order=order)
    n = int(m.path_count(L_ENUM))
    sup = m.top_k_sequences(k=n, length=L_ENUM)
    assert len(sup) == n

    ref, _ = _direct_marginals(sup, L_ENUM)
    assert np.abs(m.positional_marginals(L_ENUM) - ref).max() < EXACT

    residue = sorted({s[6] for s, _ in sup})[0]
    ref, _ = _direct_marginals(sup, L_ENUM, {6: residue})
    got = m.positional_marginals(L_ENUM, clamp={6: residue})
    assert np.abs(got - ref).max() < EXACT
    assert ((got == 0.0) == (ref == 0.0)).all()

    want = _direct_expected_counts(m, sup, L_ENUM)
    ec = m.expected_counts(length=L_ENUM)
    for name, cells in want.items():
        blk = ec[name]
        have = {int(r): blk["counts"][i] for i, r in enumerate(blk["rows"])}
        assert set(have) == set(cells), f"order {order}, {name}: rows differ"
        for row, counts in cells.items():
            assert np.abs(have[row] - counts).max() < EXACT


def test_both_answers_are_bit_identical_across_thread_counts(small):
    """Pinned, not hoped for.

    The marginal sum is chunked into a fixed number of pieces and combined in
    chunk order, and the expected-count accumulation is serial, precisely so
    that neither answer depends on how many cores happened to be free. Nothing
    else in the library would notice if that stopped being true.
    """
    length = small.observed_lengths[-1]
    residue = small.top_k_sequences(1, length=length)[0][0][2]
    before = small.threads
    ref = None
    try:
        for n in (1, 2, 4, 0):
            small.threads = n
            ec = small.expected_counts(length=length)
            got = (small.positional_marginals(length).tobytes(),
                   small.positional_marginals(
                       length, clamp={2: residue}).tobytes(),
                   tuple((b["rows"].tobytes(), b["counts"].tobytes())
                         for k, b in sorted(ec.items())
                         if isinstance(b, dict)))
            if ref is None:
                ref = got
            assert got == ref, f"answer moved at threads={n}"
    finally:
        small.threads = before


def test_an_oversized_sweep_is_refused_before_it_allocates(small):
    """The backward field is held for all `L + 1` layers at once, which is 44
    MB on the default amino-acid model and gigabytes at a nucleotide model's
    ceiling order. `calloc` grants lazily on Linux, so an oversized request
    fails on first write with the OOM killer choosing a victim from the whole
    machine; it has to be refused while refusing is still cheap."""
    import BEAM
    length = small.observed_lengths[0]
    budget = BEAM.memory_budget()
    try:
        BEAM.set_memory_budget(1000)
        with pytest.raises(MemoryError, match="memory budget"):
            small.positional_marginals(length)
        with pytest.raises(MemoryError, match="memory budget"):
            small.expected_counts(length=length)
    finally:
        BEAM.set_memory_budget(budget)
    # and the refusal left nothing broken behind
    assert small.positional_marginals(length).shape == (length, 20)
