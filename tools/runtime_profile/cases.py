"""
Every piece of BEAM functionality the profile measures.

A CASE is (name, group, fn, kind). `fn` takes a context object and returns the
value to record. Two things are recorded for every case: how long it took, and
WHAT IT RETURNED -- the second is what turns this from a benchmark into a
regression guard. An optimisation that is faster and wrong shows up as a
changed digest, not as a faster number.

`kind` says how the return value is captured:
  scalar   a float/int/bool/str, compared exactly
  array    a numpy array, stored as .npy and compared elementwise
  struct   a dict/list of the above, flattened
  opaque   timed but not compared (things with no stable value)
"""
import numpy as np

CASES = []


def case(name, group, kind="struct", tiers=("tiny", "small", "foundation"),
         needs=(), reps_by_tier=None):
    """`reps_by_tier` caps repetitions per tier.

    The spectrum family runs for tens of minutes on the foundation model --
    METHOD.md puts one spectrum build at ~20 min at order 4 -- so repeating it
    three times would cost hours for a variance estimate nobody reads. Those
    cases take one rep there and the table records `reps` alongside the mean.
    """
    def deco(fn):
        CASES.append(dict(name=name, group=group, kind=kind, fn=fn,
                          tiers=set(tiers), needs=set(needs),
                          reps_by_tier=reps_by_tier or {}))
        return fn
    return deco


HEAVY = {"foundation": 1}


# ── Construction, persistence ───────────────────────────────────
# Timed by the harness itself (they produce the context), registered here so
# they appear in the table in the right order.

# ── Introspection: cheap properties ─────────────────────────────
@case("n_states", "introspect", "scalar")
def _(c): return c.m.n_states


@case("n_cells", "introspect", "scalar")
def _(c): return c.m.n_cells


@case("n_nonzero_cells", "introspect", "scalar")
def _(c): return c.m.n_nonzero_cells


@case("n_observed", "introspect", "scalar")
def _(c): return c.m.n_observed


@case("n_skipped", "introspect", "scalar")
def _(c): return c.m.n_skipped


@case("n_sequences", "introspect", "scalar")
def _(c): return c.m.n_sequences


@case("total_weight", "introspect", "scalar")
def _(c): return c.m.total_weight


@case("memory_bytes", "introspect", "scalar")
def _(c): return c.m.memory_bytes


@case("is_dag", "introspect", "scalar")
def _(c): return bool(c.m.is_dag)


@case("variant", "introspect", "scalar")
def _(c): return str(c.m.variant)


@case("observed_lengths", "introspect", "array")
def _(c): return np.asarray(c.m.observed_lengths, dtype=np.int64)


@case("length_distribution", "introspect", "struct")
def _(c):
    d = c.m.length_distribution
    return d if isinstance(d, dict) else np.asarray(d, dtype=np.float64)


@case("config", "introspect", "struct")
def _(c):
    cfg = c.m.config
    return cfg if isinstance(cfg, dict) else dict(vars(cfg))


# ── Scoring ─────────────────────────────────────────────────────
@case("log_pseq_single", "scoring", "scalar")
def _(c): return c.m.log_pseq(c.probe[0])


@case("log_pseq_batch_1k", "scoring", "array")
def _(c): return np.asarray(c.m.log_pseq_batch(c.probe[:1000]))


@case("log_pseq_batch_100k", "scoring", "array")
def _(c): return np.asarray(c.m.log_pseq_batch(c.probe_big))


@case("pseq", "scoring", "scalar")
def _(c): return float(c.m.pseq(c.probe[0]))


@case("explain", "scoring", "struct")
def _(c): return c.m.explain(c.probe[0])


@case("log_length_prior", "scoring", "array")
def _(c):
    return np.asarray([c.m.log_length_prior(L) for L in c.m.observed_lengths])


@case("log_partition", "scoring", "array")
def _(c):
    return np.asarray([c.m.log_partition(L) for L in c.m.observed_lengths])


# ── Diversity / Hill ────────────────────────────────────────────
@case("path_count_total", "diversity", "scalar")
def _(c): return c.m.path_count()


@case("path_count_by_length", "diversity", "array")
def _(c):
    return np.asarray([c.m.path_count(L) for L in c.m.observed_lengths])


@case("entropy", "diversity", "scalar")
def _(c): return c.m.entropy()


@case("hill_number_q1", "diversity", "scalar")
def _(c): return c.m.hill_number(1.0)


@case("hill_numbers", "diversity", "array")
def _(c): return np.asarray(c.m.hill_numbers([0.0, 0.5, 1.0, 2.0, 3.0]))


@case("hill_curve", "diversity", "struct", reps_by_tier=HEAVY)
def _(c): return c.m.hill_curve()


@case("effective_diversity", "diversity", "scalar")
def _(c): return c.m.effective_diversity()


@case("diversity_profile", "diversity", "struct")
def _(c): return c.m.diversity_profile()


@case("log_mellin", "diversity", "array")
def _(c): return np.asarray([c.m.log_mellin(q) for q in (0.0, 0.5, 1.0, 2.0)])


@case("power_sum", "diversity", "array")
def _(c): return np.asarray([c.m.power_sum(q) for q in (0.5, 1.0, 2.0)])


# ── Surprisal / moments ─────────────────────────────────────────
@case("surprisal_moments", "surprisal", "array")
def _(c): return np.asarray(c.m.surprisal_moments(q=1.0, order=4))


@case("surprisal_cumulants", "surprisal", "array")
def _(c): return np.asarray(c.m.surprisal_cumulants(q=1.0, order=4))


@case("surprisal_range", "surprisal", "array")
def _(c): return np.asarray(c.m.surprisal_range())


@case("surprisal_stats", "surprisal", "struct")
def _(c): return c.m.surprisal_stats(q=1.0)


@case("surprisal_by_depth", "surprisal", "struct")
def _(c): return c.m.surprisal_by_depth(q=1.0)


@case("surprisal_distribution", "surprisal", "struct")
def _(c): return c.m.surprisal_distribution()


@case("surprisal_pdf", "surprisal", "array")
def _(c): return np.asarray(c.m.surprisal_pdf(c.grid))


@case("surprisal_cdf", "surprisal", "array")
def _(c): return np.asarray(c.m.surprisal_cdf(c.grid))


@case("accumulator_max", "surprisal", "scalar")
def _(c): return c.m.accumulator_max()


@case("dynamic_range", "surprisal", "scalar")
def _(c): return c.m.dynamic_range()


@case("cgf", "surprisal", "array")
def _(c): return np.asarray([c.m.cgf(t) for t in (-0.5, 0.0, 0.5)])


@case("cgf_jet", "surprisal", "struct")
def _(c): return c.m.cgf_jet(0.5)


@case("saddlepoint", "surprisal", "struct")
def _(c): return c.m.saddlepoint(c.m.surprisal_stats(q=1.0)["mean"])


# ── Spectra (the heavy ones) ────────────────────────────────────
@case("spectrum", "spectrum", "struct", reps_by_tier=HEAVY)
def _(c): return c.m.spectrum(bins=512)


@case("spectrum_counting", "spectrum", "struct", reps_by_tier=HEAVY)
def _(c): return c.m.spectrum(measure="counting", bins=512)


@case("spectrum_pair", "spectrum", "struct", reps_by_tier=HEAVY)
def _(c): return c.m.spectrum_pair(bins=512)


@case("spectra_by_length", "spectrum", "struct", reps_by_tier=HEAVY)
def _(c): return c.m.spectra_by_length(bins=512)


# ── Per-length structure ────────────────────────────────────────
@case("length_marginals", "structure", "struct")
def _(c): return c.m.length_marginals()


@case("length_moments", "structure", "struct")
def _(c): return c.m.length_moments(q=1.0, order=4)


@case("length_pmf", "structure", "array")
def _(c): return np.asarray(c.m.length_pmf())


@case("topology", "structure", "struct")
def _(c): return c.m.topology(c.mid_length)


# ── Generation / search ─────────────────────────────────────────
@case("top_k_20", "generate", "struct")
def _(c): return c.m.top_k_sequences(20)


@case("top_k_100", "generate", "struct")
def _(c): return c.m.top_k_sequences(100)


@case("sample_10k", "generate", "struct")
def _(c):
    s = c.m.sample(10_000, seed=20260827)
    return {"n": len(s), "first20": s[:20], "sorted_digest": sorted(s)[:5]}


@case("simulate_10k", "generate", "opaque")
def _(c): return c.m.simulate(10_000, seed=20260827)


# ── Calibration / analysis objects ──────────────────────────────
@case("calibrate", "analysis", "opaque", reps_by_tier=HEAVY)
def _(c): return c.m.calibrate(bins=512)


@case("pseq_analysis", "analysis", "opaque", reps_by_tier=HEAVY)
def _(c): return c.m.pseq_analysis()


@case("pgen_diagnostics", "analysis", "struct")
def _(c): return c.m.pgen_diagnostics()


# ── Model algebra ───────────────────────────────────────────────
@case("copy", "algebra", "opaque")
def _(c): return c.m.copy()


@case("add", "algebra", "scalar")
def _(c):
    x = c.m.copy()
    x.add(c.other)
    return x.total_weight


@case("subtract", "algebra", "scalar")
def _(c):
    x = c.m.copy()
    x.subtract(c.other)          # `other` is a 10% cohort, not a full copy
    return x.total_weight


@case("merged", "algebra", "scalar")
def _(c): return c.m.merged(c.other).total_weight


@case("posterior", "algebra", "scalar")
def _(c):
    x = c.m.copy()
    x.posterior(c.other, 100.0)
    return x.n_nonzero_cells


@case("with_prior", "algebra", "scalar")
def _(c): return c.m.with_prior(c.other, 100.0).n_nonzero_cells


@case("without", "algebra", "scalar")
def _(c): return c.m.without(c.holdout_seqs).total_weight


@case("compact", "algebra", "scalar")
def _(c):
    x = c.m.copy()
    x.compact()
    return x.memory_bytes


# ── Publicness / occupancy ──────────────────────────────────────
# Downstream of the spectrum, so these guard the composition rather than a
# single call: if a spectrum optimisation shifts a bin, `expected_counts`
# is where it becomes visible as a changed prediction.

@case("publicness_build", "publicness", "opaque")
def _(c): return c.publicness()


@case("publicness_pmf", "publicness", "array")
def _(c): return np.asarray(c.publicness().pmf(1e-9))


@case("publicness_moments", "publicness", "struct")
def _(c):
    mean, var = c.publicness().moments(np.array([1e-12, 1e-9, 1e-6]))
    return {"mean": np.asarray(mean), "var": np.asarray(var)}


@case("publicness_expected_counts", "publicness", "struct", reps_by_tier=HEAVY)
def _(c):
    # The counting spectrum is (surprisal, multiplicity); the occupancy model
    # wants (probability, multiplicity). This composition is the whole point
    # of the case: a spectrum change lands here as a changed prediction.
    sp = c.m.spectrum(measure="counting", bins=512)
    return c.publicness().expected_counts(np.exp(-sp["surprisal"]),
                                          sp["weights"])


@case("publicness_depth_spread", "publicness", "struct")
def _(c):
    pm = c.publicness()
    return {"depth_spread": float(pm.depth_spread),
            "n_repertoires": int(pm.n_repertoires),
            "n_depth_groups": int(pm.n_depth_groups),
            "k_fft": int(pm.k_fft)}


@case("publicness_bin_mass", "publicness", "array")
def _(c):
    return np.asarray(c.publicness().bin_mass(
        1e-9, np.array([0.0, 1.0, 2.0, 5.0, 10.0, 50.0])))
