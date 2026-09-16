"""Depth-matched Hill profiles: the draws, the fits, and what the result says.

The three gates the implementation was built against, then the contract.

1. The draws are exactly those of the plain loop -- one numpy generator,
   repertoire-major, replicate-minor, `rng.multinomial(D, a / a.sum())`.
2. The profiles are a serial refit of those draws with `hill_profile`, bit
   for bit, at any worker count.
3. They match the same refit with `hill_numbers` to machine precision.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from BEAM import (BeamModel, DepthMatchedProfiles, depth_matched_hill_profiles,
                  set_threads, threads)
from BEAM.diversity import DEFAULT_ORDERS
from BEAM.diversity import depth_matched as dm
from BEAM.errors import BeamTypeError, BeamValueError

from _helpers import make_corpus, make_nt_corpus

CFG = dict(order=2, max_length=12)
B = 3


def bits(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64)).view(np.uint64)


def repertoire(seed, n, maker=make_corpus, **kw):
    """Distinct sequences with integer template counts."""
    s, a = maker(n=n, seed=seed, **kw)
    merged = {}
    for x, y in zip(s, a):
        merged[x] = merged.get(x, 0) + y
    return list(merged), np.array(list(merged.values()), dtype=np.float64)


@pytest.fixture(scope="module")
def reps():
    return [repertoire(1, 500, lo=6, hi=11), repertoire(2, 350, lo=6, hi=11),
            repertoire(3, 420, lo=6, hi=11)]


def reference_draws(abundances, n_bootstraps, seed):
    """The algorithm as specified, written out as the plain loop."""
    depth = int(min(float(a.sum()) for a in abundances))
    rng = np.random.default_rng(seed)
    out = []
    for a in abundances:
        p = a / a.sum()
        for _ in range(n_bootstraps):
            draws = rng.multinomial(depth, p)
            hit = np.flatnonzero(draws)
            out.append((hit, draws[hit].astype(np.float64)))
    return depth, out


def serial(reps, draws, orders, how, weighting="abundance"):
    rows = []
    for k, (hit, counts) in enumerate(draws):
        seqs = reps[k // B][0]
        members = [seqs[j] for j in hit]
        model = BeamModel(members,
                          abundances=counts if weighting == "abundance" else None,
                          **CFG)
        rows.append(getattr(model, how)(orders))
    return np.array(rows).reshape(len(reps), B, len(orders))


# ── The three gates ─────────────────────────────────────────────

def test_gate1_the_draws_are_the_reference_draws(reps):
    abus = [a for _, a in reps]
    depth, seed_used, got = dm._draws(abus, B, 20260916)
    want_depth, want = reference_draws(abus, B, 20260916)
    assert depth == want_depth and seed_used == 20260916
    assert len(got) == len(want)
    for (i, b, hit, counts), (w_hit, w_counts) in zip(got, want):
        assert np.array_equal(hit, w_hit)
        assert (bits(counts.astype(np.float64)) == bits(w_counts)).all()
        assert counts.sum() == depth


@pytest.mark.parametrize("weighting", ["abundance", "presence"])
def test_gate2_profiles_are_a_serial_refit_bit_for_bit(reps, weighting):
    _, draws = reference_draws([a for _, a in reps], B, 7)
    want = serial(reps, draws, DEFAULT_ORDERS, "hill_profile", weighting)
    for n_workers in (1, 3, 4):
        r = depth_matched_hill_profiles(
            [s for s, _ in reps], [a for _, a in reps], n_bootstraps=B, seed=7,
            weighting=weighting, n_workers=n_workers, **CFG)
        assert (bits(r.profiles) == bits(want)).all(), n_workers
        assert np.array_equal(r.n_distinct_drawn,
                              np.array([h.size for h, _ in draws]).reshape(3, B))


def test_gate3_profiles_match_hill_numbers_to_machine_precision(reps):
    _, draws = reference_draws([a for _, a in reps], B, 7)
    want = serial(reps, draws, DEFAULT_ORDERS, "hill_numbers")
    r = depth_matched_hill_profiles([s for s, _ in reps], [a for _, a in reps],
                                    n_bootstraps=B, seed=7, **CFG)
    assert np.max(np.abs(r.profiles - want) / np.abs(want)) <= 1e-12


# ── The contract ────────────────────────────────────────────────

def test_the_depth_is_templates_not_distinct_sequences():
    """The repertoire with fewer distinct sequences has more templates, so a
    depth read off the wrong total would come out as the other one's."""
    a = (["CASSAF", "CASSGF", "CASSHF"], np.array([50.0, 40.0, 30.0]))
    b = ([f"CASS{x}{y}F" for x in "ACDEFGH" for y in "KLM"], np.ones(21))
    r = depth_matched_hill_profiles([a[0], b[0]], [a[1], b[1]], n_bootstraps=2,
                                    seed=1, orders=[0.0, 1.0, 2.0], **CFG)
    assert r.depth == 21
    assert list(r.n_templates) == [120, 21] and list(r.n_distinct) == [3, 21]


def test_one_repertoire_is_a_resample_at_its_own_depth(reps):
    s, a = reps[1]
    r = depth_matched_hill_profiles([s], [a], n_bootstraps=2, seed=3,
                                    orders=[0.0, 2.0], **CFG)
    assert r.depth == int(a.sum()) and r.profiles.shape == (1, 2, 2)


def test_no_abundances_means_one_template_per_sequence(reps):
    seqs = [s for s, _ in reps]
    r = depth_matched_hill_profiles(seqs, n_bootstraps=1, seed=3,
                                    orders=[2.0], **CFG)
    assert r.depth == min(len(s) for s in seqs)


def test_an_unseeded_run_records_the_seed_that_reproduces_it(reps):
    seqs, abus = [s for s, _ in reps[:2]], [a for _, a in reps[:2]]
    r = depth_matched_hill_profiles(seqs, abus, n_bootstraps=2, orders=[2.0],
                                    **CFG)
    again = depth_matched_hill_profiles(seqs, abus, n_bootstraps=2,
                                        orders=[2.0], seed=r.seed, **CFG)
    assert isinstance(r.seed, int)
    assert (bits(r.profiles) == bits(again.profiles)).all()


def test_the_thread_cap_is_restored_even_when_a_fit_fails(reps, monkeypatch):
    seqs, abus = [s for s, _ in reps[:2]], [a for _, a in reps[:2]]
    set_threads(5)
    try:
        depth_matched_hill_profiles(seqs, abus, n_bootstraps=2, orders=[2.0],
                                    seed=1, n_workers=2, **CFG)
        assert threads() == 5

        class Exploding(BeamModel):
            def hill_profile(self, orders):
                raise RuntimeError("boom")

        monkeypatch.setattr(dm, "BeamModel", Exploding)
        with pytest.raises(RuntimeError, match="boom"):
            depth_matched_hill_profiles(seqs, abus, n_bootstraps=2,
                                        orders=[2.0], seed=1, n_workers=2,
                                        **CFG)
        assert threads() == 5
    finally:
        set_threads(0)


def test_workers_times_threads_stays_within_the_budget():
    set_threads(8)
    try:
        assert dm._plan_threads(None, 40) == (4, 2)
        assert dm._plan_threads(3, 40) == (3, 2)
        assert dm._plan_threads(64, 40) == (8, 1)    # capped, not oversubscribed
        assert dm._plan_threads(4, 2) == (2, 4)      # never more than the tasks
    finally:
        set_threads(0)


def test_nucleotide_repertoires_work(reps):
    nt = [repertoire(5, 200, maker=make_nt_corpus, lo=9, hi=15),
          repertoire(6, 150, maker=make_nt_corpus, lo=9, hi=15)]
    r = depth_matched_hill_profiles([s for s, _ in nt], [a for _, a in nt],
                                    n_bootstraps=2, seed=2, orders=[0.5, 2.0],
                                    alphabet="nt", order=3, max_length=15)
    assert np.all(np.isfinite(r.profiles)) and r.config["alphabet"] == "nt"


# ── What the result is ──────────────────────────────────────────

def test_the_result_is_frozen_and_is_not_a_number(reps):
    r = depth_matched_hill_profiles([s for s, _ in reps[:2]],
                                    [a for _, a in reps[:2]], n_bootstraps=2,
                                    seed=1, orders=[0.0, 1.0, 2.0], **CFG)
    assert isinstance(r, DepthMatchedProfiles)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.depth = 1
    for name in ("orders", "profiles", "n_distinct_drawn", "n_templates",
                 "n_distinct"):
        assert not getattr(r, name).flags.writeable, name
    with pytest.raises(TypeError):
        r.config["order"] = 3
    for obj in (r, r.resampling_band()):
        assert not hasattr(obj, "__float__")
        with pytest.raises(TypeError):
            float(obj)


def test_the_band_is_named_and_described_as_what_it_is(reps):
    r = depth_matched_hill_profiles([s for s, _ in reps], [a for _, a in reps],
                                    n_bootstraps=4, seed=1, **CFG)
    band = r.resampling_band(lo=10, hi=90)
    assert band.lower.shape == band.upper.shape == (3, DEFAULT_ORDERS.size)
    assert np.all(band.lower <= band.median) and np.all(band.median <= band.upper)
    assert "not a confidence interval" in repr(band)
    assert "confidence interval" in type(band).__doc__
    assert not hasattr(r, "confidence_interval")
    with pytest.raises(ValueError):
        r.resampling_band(lo=90, hi=10)


# ── What is refused, and why ────────────────────────────────────

@pytest.mark.parametrize("abundances, match", [
    ([np.array([0.5, 0.25, 0.25]), np.ones(3)], "non-integer"),
    ([np.array([2.0, 0.0, 1.0]), np.ones(3)], "not positive"),
    ([np.array([2.0, np.nan, 1.0]), np.ones(3)], "non-finite"),
    ([np.ones(2), np.ones(3)], "shape"),
    ([np.ones(3)], "2 repertoires"),
])
def test_bad_abundances_are_refused(abundances, match):
    seqs = [["CASSAF", "CASSGF", "CASSHF"], ["CASSKF", "CASSLF", "CASSMF"]]
    with pytest.raises(BeamValueError, match=match):
        depth_matched_hill_profiles(seqs, abundances, n_bootstraps=1, **CFG)


@pytest.mark.parametrize("seqs, match", [
    (["CASSAF", "CASSAF"], "more than once"),
    (["CASSAF", "CASS" + "A" * 20 + "F"], "longer than max_length"),
    (["CASSAF", "CASSXF"], "does not accept"),
    (["CASSAF", ""], "empty"),
])
def test_sequences_a_fit_would_skip_are_refused(seqs, match):
    with pytest.raises(BeamValueError, match=match):
        depth_matched_hill_profiles([seqs, ["CASSKF"]], n_bootstraps=1, **CFG)


def test_bad_arguments_are_refused():
    seqs = [["CASSAF", "CASSGF"], ["CASSKF", "CASSLF"]]
    with pytest.raises(BeamTypeError):
        depth_matched_hill_profiles("CASSAF")
    with pytest.raises(BeamTypeError):
        depth_matched_hill_profiles(["CASSAF", "CASSGF"])
    with pytest.raises(BeamValueError):
        depth_matched_hill_profiles([])
    with pytest.raises(BeamValueError):
        depth_matched_hill_profiles(seqs, n_bootstraps=0)
    with pytest.raises(BeamTypeError):
        depth_matched_hill_profiles(seqs, n_bootstraps=True)
    with pytest.raises(BeamValueError):
        depth_matched_hill_profiles(seqs, weighting="unweighted")
    with pytest.raises(BeamValueError):
        depth_matched_hill_profiles(seqs, orders=[])
    with pytest.raises(BeamValueError):
        depth_matched_hill_profiles(seqs, orders=[1.0, float("nan")])
    with pytest.raises(BeamValueError):
        depth_matched_hill_profiles(seqs, seed=-1)
    with pytest.raises(BeamTypeError):
        depth_matched_hill_profiles(seqs, abundances=[[1.0, 1.0], [1.0, 1.0]],
                                    orders=[1.0], sequences=["CASSF"])
    with pytest.raises(BeamTypeError):
        depth_matched_hill_profiles(seqs, orders=[1.0], odrer=3)
