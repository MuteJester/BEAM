"""What this package refuses, and that each refusal names its own reason.

Two groups. Configurations that cannot form a mixture at all are refused at
construction, because the object is ill-defined from the moment it is
assembled and finding that out three minutes into a Gram matrix helps nobody.
Questions with no exact answer are refused at the call, because the alternative
is a number that reads like every other number here.

Each case pins the CLASS rather than the message text, with the message checked
only for the fragment a caller would act on. A test that matches a whole
message pins the prose and breaks when it improves.
"""
from __future__ import annotations

import math

import pytest

import BEAM
from BEAM import BeamModel, Mixture
from BEAM.errors import (BeamError, BeamTypeError, BeamValueError,
                         BudgetExceeded, ModelMismatch, NoLivePaths)
from BEAM.mixture import (IncompatibleMembers, MixtureError, MixtureTooLarge,
                          UnsupportedOrder)
from _tiny import tiny_model


# ── Members that cannot share a trellis ─────────────────────────

@pytest.mark.parametrize("differs", [
    {"order": 3},
    {"max_length": 8},
    {"alphabet": "nt"},
])
def test_incompatible_members_are_refused_at_construction(tiny_a, differs):
    """One differing field is enough: the state index stops being shared."""
    other = tiny_model(9, letters="ACGT" if "alphabet" in differs else "ACDG",
                       **differs)
    with pytest.raises(IncompatibleMembers) as exc:
        Mixture([tiny_a, other])
    assert "shape mismatch" in str(exc.value)


def test_an_incompatible_mixture_is_catchable_the_broad_ways_too(tiny_a):
    """`IncompatibleMembers` keeps `ModelMismatch` and `BeamError` as bases.

    Existing code that catches the library-wide class must keep working, which
    is the whole reason the hierarchy is layered rather than flat.
    """
    other = tiny_model(9, order=3)
    for cls in (IncompatibleMembers, ModelMismatch, MixtureError, BeamError):
        with pytest.raises(cls):
            Mixture([tiny_a, other])


def test_a_non_model_member_is_refused_by_position(tiny_a):
    with pytest.raises(BeamTypeError) as exc:
        Mixture([tiny_a, "not a model"])
    assert "member 1" in str(exc.value)


def test_an_empty_mixture_is_refused():
    with pytest.raises(BeamValueError):
        Mixture([])


# ── Weights: refused, never normalised ──────────────────────────

def test_weights_that_do_not_sum_to_one_are_refused(tiny_a, tiny_b):
    """Documented behaviour is to refuse, so it refuses.

    Raw counts pasted in as weights and a member dropped from one list but not
    the other both land here, and normalising would give each of them a
    plausible answer.
    """
    with pytest.raises(BeamValueError) as exc:
        Mixture([tiny_a, tiny_b], [3.0, 7.0])
    assert "sum to 1" in str(exc.value)


def test_negative_weights_are_refused(tiny_a, tiny_b):
    with pytest.raises(BeamValueError) as exc:
        Mixture([tiny_a, tiny_b], [1.5, -0.5])
    assert "non-negative" in str(exc.value)


def test_the_wrong_number_of_weights_is_refused(tiny_a, tiny_b):
    with pytest.raises(BeamValueError) as exc:
        Mixture([tiny_a, tiny_b], [0.2, 0.3, 0.5])
    assert "one entry per member" in str(exc.value)


def test_non_finite_weights_are_refused(tiny_a, tiny_b):
    with pytest.raises(BeamValueError):
        Mixture([tiny_a, tiny_b], [float("nan"), 1.0])
    with pytest.raises(BeamValueError):
        Mixture([tiny_a, tiny_b], [float("inf"), 1.0])


def test_a_zero_weight_member_is_allowed_and_contributes_nothing(tiny_a,
                                                                 tiny_b):
    """Excluding a member by weight is meaningful, so it is not refused."""
    mix = Mixture([tiny_a, tiny_b], [1.0, 0.0])
    assert mix.hill_number(2.0) == pytest.approx(tiny_a.hill_number(2.0),
                                                 rel=1e-12)


def test_weights_are_a_copy(tiny_a, tiny_b):
    mix = Mixture([tiny_a, tiny_b])
    w = mix.weights
    w[0] = 99.0
    assert mix.weights[0] == pytest.approx(0.5)


# ── Orders with no exact answer ─────────────────────────────────

def test_q_equals_one_raises_and_names_the_two_honest_paths(tiny_a, tiny_b):
    """The refusal has to be actionable, or it just moves the problem.

    A bare float from `hill_number(1.0)` would be read once, pasted into a
    table, and separated from the caveat that lives only in prose. Refusing
    costs the caller one line; the message has to say which line.
    """
    mix = Mixture([tiny_a, tiny_b])
    with pytest.raises(UnsupportedOrder) as exc:
        mix.hill_number(1.0)
    msg = str(exc.value)
    assert "entropy_bounds()" in msg
    assert "entropy_estimate(" in msg
    assert "does not factor" in msg


@pytest.mark.parametrize("q", [1.0, 1.5, 0.5, -1.0, -2.0, 2.5])
def test_every_inexact_order_raises(tiny_a, tiny_b, q):
    with pytest.raises(UnsupportedOrder):
        Mixture([tiny_a, tiny_b]).hill_number(q)


def test_a_list_containing_q_equals_one_raises_rather_than_dropping_it(
        tiny_a, tiny_b):
    """A partial answer to a list request is worse than no answer.

    The caller would get a shorter array and no reason, and the entry that
    vanished is exactly the one that needed explaining. Validated before
    anything is computed, so the failure is immediate rather than after the
    other orders have been paid for.
    """
    mix = Mixture([tiny_a, tiny_b])
    with pytest.raises(UnsupportedOrder):
        mix.hill_numbers([0.0, 1.0, 2.0])
    with pytest.raises(UnsupportedOrder):
        mix.hill_numbers([2.0, 3.0, 1.0])
    # and the exact-only list still works
    assert mix.hill_numbers([0.0, 2.0, 3.0]).shape == (3)


def test_there_is_no_flag_that_softens_the_refusal(tiny_a, tiny_b):
    """`hill_number` takes an order and nothing else, deliberately."""
    import inspect
    sig = inspect.signature(Mixture.hill_number)
    assert list(sig.parameters) == ["self", "q"]


def test_sampling_at_an_exactly_computable_order_is_refused(tiny_a, tiny_b):
    """Ignoring `samples` there would be a silent no-op."""
    mix = Mixture([tiny_a, tiny_b])
    with pytest.raises(BeamValueError) as exc:
        mix.beta_diversity(2.0, samples=100)
    assert "only meaningful at q=1" in str(exc.value)


def test_log_power_sum_refuses_q_zero(tiny_a, tiny_b):
    with pytest.raises(UnsupportedOrder):
        Mixture([tiny_a, tiny_b]).log_power_sum(0.0)


# ── -inf is an answer and must survive ──────────────────────────

def test_a_sequence_no_member_spells_scores_minus_infinity(far_a, far_b):
    """Never floored, never smoothed. Closed world all the way to the caller."""
    mix = Mixture([far_a, far_b])
    assert mix.log_pseq("WWWWW") == -math.inf
    assert mix.pseq("WWWWW") == 0.0
    assert mix.log_pseq_batch(["WWWWW"])[0] == -math.inf


def test_a_sequence_only_one_member_spells_keeps_its_share(far_a, far_b):
    """Not -inf, and exactly the weighted probability under the one member."""
    from _tiny import support
    s = next(iter(support(far_a)))
    assert far_b.log_pseq(s) == -math.inf
    mix = Mixture([far_a, far_b], [0.25, 0.75])
    assert mix.log_pseq(s) == pytest.approx(
        math.log(0.25) + far_a.log_pseq(s), rel=1e-13)


def test_a_batch_of_all_unspellable_sequences_is_all_minus_infinity(far_a,
                                                                    far_b):
    """The all-dead reduction must not become NaN.

    Shifting by the maximum gives `-inf - -inf` when every term is `-inf`, and
    NaN is the one answer that is worse than either a number or a refusal.
    """
    mix = Mixture([far_a, far_b])
    out = mix.log_pseq_batch(["WWWWW", "WWWWWW", "YYYYY"])
    assert all(v == -math.inf for v in out)


# ── Lengths and indices ─────────────────────────────────────────

def test_a_length_no_member_produces_has_no_within_length_weights(tiny_a,
                                                                  tiny_b):
    mix = Mixture([tiny_a, tiny_b])
    missing = next(L for L in range(1, mix.max_length + 1)
                   if L not in mix.observed_lengths)
    with pytest.raises(NoLivePaths) as exc:
        mix.member_weights(missing)
    assert "no member" in str(exc.value)


@pytest.mark.parametrize("i", [-1, 2, 99])
def test_a_member_index_out_of_range_is_refused(tiny_a, tiny_b, i):
    with pytest.raises(BeamValueError):
        Mixture([tiny_a, tiny_b]).inner_product(0, i)


def test_a_sample_count_below_two_has_no_standard_error(tiny_a, tiny_b):
    with pytest.raises(BeamValueError) as exc:
        Mixture([tiny_a, tiny_b]).entropy_estimate(1)
    assert "standard error" in str(exc.value)


# ── The budget, and a defect it once hid ────────────────────────

def test_a_cap_that_is_not_a_power_of_two_still_counts_correctly(tiny_a,
                                                                 tiny_b,
                                                                 sup_a, sup_b):
    """Regression: the probe mask is `slots - 1`, so `slots` must be a power
    of two. It was computed as `2 * cap`, which is one only when the caller's
    ceiling happens to be -- every derived ceiling was, so nothing failed until
    a cap was passed by hand.
    """
    want = float(len(set(sup_a) | set(sup_b)))
    for cap in (1000, 4097, 12345):
        got = Mixture([tiny_a, tiny_b]).support_size(cap=cap)
        assert got.value == want, f"cap={cap}"


def test_a_ceiling_of_one_is_honoured_rather_than_quietly_raised(tiny_a,
                                                                 tiny_b):
    """An explicit cap is an instruction, not a hint."""
    with pytest.raises(MixtureTooLarge):
        Mixture([tiny_a, tiny_b]).support_size(cap=1)


def test_the_budget_refusal_is_catchable_the_broad_way_too(tiny_a, tiny_b):
    for cls in (MixtureTooLarge, BudgetExceeded, MemoryError, MixtureError):
        with pytest.raises(cls):
            Mixture([tiny_a, tiny_b]).support_size(cap=1)


def test_the_classes_are_exported_from_the_package_root():
    """A caller cannot catch what it cannot import."""
    for name in ("Mixture", "MixtureError", "IncompatibleMembers",
                 "UnsupportedOrder", "MixtureTooLarge", "EntropyBounds",
                 "EntropyEstimate", "SupportSize", "BetaDiversity"):
        assert hasattr(BEAM, name), name
        assert name in BEAM.__all__, name
