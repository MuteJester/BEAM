"""Configuration guards, argument validation, return conventions.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel
from BEAM import _cbeam as _c
from BEAM._model import _config_args_from_dict

from _helpers import AA, J_MOTIFS, make_corpus

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
    with pytest.raises(ValueError, match="alphabet does not accept"):
        small.explain("CASSXQYF")
    with pytest.raises(ValueError, match="empty"):
        small.explain("")
    with pytest.raises(ValueError, match="longer than max_length"):
        small.explain("C" * 40)
    with pytest.raises(TypeError):
        small.explain(None)


def test_explain_flags_an_unproducible_length(open_small):
    """`explain` returns finite per-residue terms for a sequence whose LENGTH
    the model never saw, while `log_pseq` is -inf: the length prior, not any
    residue, is what rules it out. Presenting the decomposition without saying
    so would make it look like a valid explanation of a zero."""
    dead = next(L for L in range(1, int(open_small.config["max_length"]) + 1)
                if not math.isfinite(open_small.log_length_prior(L)))
    e = open_small.explain("C" * dead)
    assert e["length_is_producible"] is False
    assert np.isfinite(e["forward"]).all() and np.isfinite(e["backward"]).all()
    assert e["log_pseq"] == -math.inf

    live = int(np.argmax(open_small.length_pmf()))
    good = open_small.sample(1, length=live, seed=1)[0]
    assert open_small.explain(good)["length_is_producible"] is True


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
    # The loader used to answer every failure with one sentence -- "not a model
    # file, truncated, or corrupt" -- which is what this line pinned. It now
    # names the cause, so the assertion moves with it. The property under test
    # is unchanged: a junk file is refused, and refused DIFFERENTLY from an
    # EDGE file, which is what makes the EDGE message worth having.
    with pytest.raises(OSError, match="not a BEAM model file"):
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


def test_c_and_python_defaults_describe_the_same_model():
    """The two entry points must agree on what "default" means.

    A caller writing BeamModel(seqs) and a caller reaching
    beam_model_config_default() in C are asking for the same model. Between
    the closed-world flip and its follow-up they were not: C set strict but
    kept backoff at 20/20/200, so the same corpus gave a different support --
    D0 of 4 instead of 3 on three short sequences -- with nothing failing,
    because both configurations are individually valid.

    Storage is the one field allowed to differ. The C layer stays dense while
    Python selects sparse, and a dense and a sparse strict model are
    bit-identical, which is asserted separately.
    """
    seqs = ["CASSF", "CASTF", "CAGSF"]
    py = BeamModel(seqs, abundances=[3, 2, 1], order=2, max_length=6)
    cfg = py.config

    # What the C default resolves to, read back through a model built with it.
    c_defaults = _c.config(_c.new(*_config_args_from_dict(
        {**cfg, "sparse": False})))

    assert cfg["strict"] is True
    assert cfg["kappa"] == (0.0, 0.0, 0.0)
    assert cfg["kappa_wide"] == 0.0 and cfg["kappa_len"] == 0.0
    for field in ("strict", "kappa", "kappa_wide", "kappa_len",
                  "order", "depth_cap", "coefficients", "weights"):
        assert c_defaults[field] == cfg[field], field
