"""Hill profiles of several repertoires at one common sequencing depth.

Put each sequence in a bag as many times as its abundance, draw ``D`` from the
bag with replacement, fit a model to what was drawn, profile it, and repeat.
``D`` is the smallest template total among the repertoires in the call, so the
deeper ones are thinned to the shallowest rather than the shallowest padded.

Three rules decide the shape of the code below, and each one is a result the
caller would otherwise lose silently.

EVERY REPLICATE IS A FRESH MODEL of exactly the sequences drawn, with exactly
the drawn counts. Not one model per repertoire re-weighted per replicate, and
no count table, support or trellis shared between replicates. In the closed
world a model's support -- which cells exist -- is part of the answer and
differs between replicates, and ``D0`` is that support.

THE DRAWS ARE NUMPY'S, from one generator, repertoire-major and
replicate-minor: ``rng.multinomial(D, a / a.sum())`` for each replicate of
each repertoire in turn. Given the same seed they are exactly the draws a
plain loop would make, which is what lets a result be checked against one.

ALL DRAWS HAPPEN FIRST, and only then are the replicates fitted, possibly in
parallel. The generator is consumed in one fixed order before any thread
starts, so no result can depend on the thread count or on scheduling.
"""
from __future__ import annotations

import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Sequence

import numpy as np

from ..errors import BeamTypeError, BeamValueError
from ..model import alphabets
from ..model.facade import BeamModel, _orders_vector
from ..runtime import effective_threads, set_threads, threads
from .results import DepthMatchedProfiles

__all__ = ["DEFAULT_ORDERS", "depth_matched_hill_profiles"]

#: Orders 0 to 3 in steps of 0.1: 31 points, rounded so 0.3 is 0.3.
DEFAULT_ORDERS = np.round(np.arange(0.0, 3.0001, 0.1), 4)
DEFAULT_ORDERS.setflags(write=False)

_WEIGHTINGS = ("abundance", "presence")


# ── Validation ──────────────────────────────────────────────────

def _repertoires(repertoires: Any) -> list[list[str]]:
    if isinstance(repertoires, (str, bytes)) or not isinstance(
            repertoires, Iterable):
        raise BeamTypeError(
            "repertoires must be a list of sequence lists, one per repertoire")
    reps = []
    for i, r in enumerate(repertoires):
        if isinstance(r, (str, bytes)):
            raise BeamTypeError(
                f"repertoire {i} is a single string; each repertoire must be a "
                f"list of sequences (pass [seqs], not seqs, for one)")
        seqs = list(r)
        if not seqs:
            raise BeamValueError(f"repertoire {i} is empty")
        reps.append(seqs)
    if not reps:
        raise BeamValueError("no repertoires were given")
    return reps


def _abundances(abundances: Any, reps: list[list[str]]) -> list[np.ndarray]:
    """Per-repertoire template counts, as float64 holding exact integers.

    INTEGERS, because ``D`` counts templates: it is the number of draws from a
    bag holding each sequence ``a`` times, and a bag cannot hold 2.5 of one.
    Refusing a fractional abundance is also what stops the commonest mistake
    here: frequencies that sum to 1 would make ``D`` equal 1, and every
    replicate a model of one sequence.
    """
    if abundances is None:
        return [np.ones(len(s), dtype=np.float64) for s in reps]
    if isinstance(abundances, (str, bytes)) or not isinstance(
            abundances, Iterable):
        raise BeamTypeError(
            "abundances must be a list of arrays, one per repertoire, or None")
    abus = list(abundances)
    if len(abus) != len(reps):
        raise BeamValueError(
            f"abundances has {len(abus)} entries but there are {len(reps)} "
            f"repertoires")
    out = []
    for i, (a, s) in enumerate(zip(abus, reps)):
        try:
            a = np.array(a if isinstance(a, np.ndarray) else list(a),
                         dtype=np.float64)
        except (TypeError, ValueError):
            raise BeamTypeError(
                f"abundances[{i}] must be a sequence of numbers") from None
        if a.ndim != 1 or a.size != len(s):
            raise BeamValueError(
                f"abundances[{i}] has shape {a.shape} but repertoire {i} has "
                f"{len(s)} sequences")
        if not np.all(np.isfinite(a)):
            raise BeamValueError(f"abundances[{i}] holds a non-finite value")
        if not np.all(a > 0):
            raise BeamValueError(
                f"abundances[{i}] holds a value that is not positive; drop "
                f"those sequences instead, a zero share is never drawn")
        if not np.all(a == np.floor(a)):
            raise BeamValueError(
                f"abundances[{i}] holds a non-integer; abundances here are "
                f"template counts, because the common depth is a number of "
                f"templates drawn. Frequencies would make it 1")
        if a.sum() >= 2.0 ** 53:
            raise BeamValueError(
                f"abundances[{i}] totals {a.sum():.3g} templates, past the "
                f"range a float64 counts exactly")
        out.append(a)
    return out


def _check_sequences(reps: list[list[str]], config: dict[str, Any]) -> None:
    """Refuse, up front, any sequence a fit would silently skip.

    A model skips a sequence longer than ``max_length`` or outside its
    alphabet. Here that would be worse than a lost sequence: every template of
    it that a replicate draws vanishes from that replicate, so the replicate is
    no longer at depth ``D`` and nothing says so.

    Duplicates are refused too. The draw is over the entries given, so a
    sequence listed twice gets two shares of the bag; the fit would merge them
    back and not notice, but ``n_distinct_drawn`` would count it twice.
    Nucleotide input is case-folded by the model, so ``acgt`` and ``ACGT`` are
    duplicates there.
    """
    alphabet = config.get("alphabet", alphabets.DEFAULT_ALPHABET)
    info = alphabets.info(alphabet)
    max_length = config.get("max_length")
    if max_length is None:
        max_length = int(info["default_max_length"])
    syms = info["symbols"]
    fold = alphabet == "nt"
    allowed = frozenset(syms + syms.lower()) if fold else frozenset(syms)
    for i, seqs in enumerate(reps):
        for s in seqs:
            if not isinstance(s, str):
                raise BeamTypeError(
                    f"repertoire {i} holds a {type(s).__name__}, not a string")
            if not s or len(s) > max_length or not allowed.issuperset(s):
                why = ("is empty" if not s else
                       f"is longer than max_length={max_length}"
                       if len(s) > max_length else
                       f"has characters the {alphabet!r} alphabet does not "
                       f"accept: {alphabets.offenders(alphabet, s)}")
                raise BeamValueError(
                    f"repertoire {i}: {s!r} {why}. A fit would skip it, and "
                    f"every template of it a resample drew would be lost from "
                    f"that resample, which would then not be at the common "
                    f"depth. Filter the repertoire first")
        keys = [s.upper() for s in seqs] if fold else seqs
        if len(set(keys)) != len(keys):
            raise BeamValueError(
                f"repertoire {i} lists a sequence more than once; merge the "
                f"duplicates and sum their abundances")


def _check_config(config: dict[str, Any]) -> None:
    params = inspect.signature(BeamModel.__init__).parameters
    allowed = set(params) - {"self", "sequences", "abundances"}
    for k in config:
        if k in ("sequences", "abundances"):
            raise BeamTypeError(
                f"{k!r} is set per resample by this function and cannot be "
                f"passed through")
        if k not in allowed:
            raise BeamTypeError(
                f"unexpected model configuration {k!r}; BeamModel takes "
                f"{sorted(allowed)}")


def _count(name: str, value: Any, lo: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise BeamTypeError(f"{name} must be an integer, got {value!r}")
    if value < lo:
        raise BeamValueError(f"{name} must be at least {lo}, got {value}")
    return int(value)


# ── The draw ────────────────────────────────────────────────────

def _draws(abundances: list[np.ndarray], n_bootstraps: int,
           seed: int | None) -> tuple[int, int, list[tuple[int, int, np.ndarray, np.ndarray]]]:
    """Every replicate's draw, in the one order the result depends on.

    Returns ``(depth, seed_used, draws)`` with ``draws`` a list of
    ``(repertoire, replicate, hit, counts)``: ``hit`` the indices of the
    sequences drawn at least once, ascending, and ``counts`` their drawn
    template counts. Exactly

        rng = numpy.random.default_rng(seed)
        for each repertoire, in order:
            p = a / a.sum()
            for each replicate:
                d = rng.multinomial(depth, p)
                hit = numpy.flatnonzero(d); counts = d[hit]

    ``default_rng(SeedSequence(s))`` is ``default_rng(s)`` -- an integer seed
    is wrapped in exactly that SeedSequence -- so going through one changes no
    draw and buys a seed that can be recorded when the caller passed ``None``.
    """
    ss = np.random.SeedSequence(seed)
    rng = np.random.default_rng(ss)
    depth = int(min(float(a.sum()) for a in abundances))
    out = []
    for i, a in enumerate(abundances):
        p = a / a.sum()
        for b in range(n_bootstraps):
            d = rng.multinomial(depth, p)
            hit = np.flatnonzero(d)
            out.append((i, b, hit, d[hit]))
    return depth, int(ss.entropy), out


# ── Threads ─────────────────────────────────────────────────────

def _plan_threads(n_workers: int | None, n_tasks: int) -> tuple[int, int]:
    """``(workers, threads_per_fit)`` with their product within the budget.

    Each fit is itself parallel: BEAM's OpenMP regions default to one thread
    per core. Running replicates side by side on top of that multiplies the
    two, and a run of three unpinned processes on 48 cores carried about 144
    threads and went about twice as slow per item. So the thread budget -- the
    cap already set with ``BEAM.set_threads``, else what OpenMP would start --
    is SPLIT between replicates and the threads inside each.

    The default is two threads per replicate -- ``budget // 2`` workers -- and
    at most eight replicates at a time. On 16 threads the donor-scale
    benchmark (two repertoires of 180,000 sequences, 20 resamples each) took
    43.3 s with one worker of 16 threads, 25.1 s with four of four, and 23.4 s
    with eight of two and with sixteen of one; per-fit parallelism runs out
    long before sixteen threads, because a fit's lengths are few and unequal.
    Eight of two is taken over sixteen of one because it holds half as many
    models at once. The cap of eight bounds that memory on a larger machine,
    where the best split was not measured.
    """
    cap = threads()
    budget = cap if cap > 0 else effective_threads()
    budget = max(1, int(budget))
    if n_workers is None:
        workers = min(8, max(1, budget // 2))
    else:
        workers = _count("n_workers", n_workers, 1)
    workers = max(1, min(workers, budget, n_tasks))
    return workers, max(1, budget // workers)


# ── The entry point ─────────────────────────────────────────────

def depth_matched_hill_profiles(
    repertoires: Sequence[Sequence[str]],
    abundances: Sequence[Sequence[float]] | None = None,
    *,
    n_bootstraps: int = 20,
    orders: Iterable[float] | None = None,
    seed: int | None = None,
    weighting: str = "abundance",
    n_workers: int | None = None,
    **config: Any,
) -> DepthMatchedProfiles:
    """Hill profiles of ``k`` repertoires, resampled to one template depth.

    ``D`` is the smallest template total -- ``sum(abundances)``, never the
    distinct count -- among the repertoires passed. For each repertoire in the
    order given, and each of ``n_bootstraps`` replicates, ``D`` templates are
    drawn with replacement in proportion to abundance, collapsed to
    per-sequence counts, and a fresh :class:`BeamModel` is fitted to them and
    profiled with :meth:`BeamModel.hill_profile`.

    **Profiles are comparable only within one call.** ``D`` belongs to the
    set of repertoires passed together: adding or removing one can change it,
    and with it every other repertoire's profiles.

    Args:
        repertoires: ``k`` lists of sequences, each without duplicates and
            each sequence one the model accepts -- a sequence a fit would skip
            would silently take its drawn templates out of the depth, so it is
            refused here instead. ``k = 1`` is allowed: ``D`` is then that
            repertoire's own total, and the result is a plain resample of it.
        abundances: ``k`` arrays of positive integer template counts, or
            ``None`` for one template per sequence.
        n_bootstraps: Resamples per repertoire.
        orders: Hill orders; default :data:`DEFAULT_ORDERS`, 0 to 3 in steps of
            0.1.
        seed: Seeds the one numpy generator every draw comes from. ``None``
            draws fresh entropy and records it in the result's ``seed``.
        weighting: ``"abundance"`` fits each resample with its drawn counts;
            ``"presence"`` fits it with one per distinct sequence drawn. The
            draw is abundance-proportional either way.
        n_workers: Resamples fitted at once. Default: half the thread budget,
            at most 8, so each fit gets two threads. The budget is split
            between them for the duration of the call -- see the note below.
        **config: Passed to every :class:`BeamModel`.

    Returns:
        A :class:`DepthMatchedProfiles`. Its ``profiles`` are identical,
        bit for bit, whatever ``n_workers`` is.

    .. note::
       For the duration of the call the process-wide thread cap
       (:func:`BEAM.set_threads`) is set to ``budget // n_workers`` so that
       the replicates running side by side do not oversubscribe the machine,
       and restored afterwards, also on error. Other BEAM work in the process
       sees that cap meanwhile, and two of these calls must not overlap in one
       process. Results never depend on the cap.
    """
    reps = _repertoires(repertoires)
    abus = _abundances(abundances, reps)
    if orders is None:
        orders = DEFAULT_ORDERS
    q = _orders_vector(orders)
    if q.size == 0:
        raise BeamValueError("orders is empty")
    n_bootstraps = _count("n_bootstraps", n_bootstraps, 1)
    if weighting not in _WEIGHTINGS:
        raise BeamValueError(
            f"weighting must be one of {_WEIGHTINGS}, got {weighting!r}")
    if seed is not None:
        seed = _count("seed", seed, 0)
    _check_config(config)
    _check_sequences(reps, config)

    depth, seed_used, draws = _draws(abus, n_bootstraps, seed)
    k = len(reps)
    profiles = np.empty((k, n_bootstraps, q.size), dtype=np.float64)
    distinct = np.empty((k, n_bootstraps), dtype=np.int64)

    def fit(item):
        i, b, hit, counts = item
        seqs = reps[i]
        members = [seqs[j] for j in hit]
        weights = counts.astype(np.float64) if weighting == "abundance" else None
        model = BeamModel(members, abundances=weights, **config)
        if model.n_skipped:
            # Unreachable after `_check_sequences`; kept because the cost of
            # being wrong about that is a replicate silently short of D.
            raise BeamValueError(
                f"a resample of repertoire {i} lost {model.n_skipped} "
                f"sequences to the fit, so it is not at depth {depth}")
        return i, b, model.hill_profile(q), hit.size

    workers, per_fit = _plan_threads(n_workers, len(draws))
    previous = threads()
    set_threads(per_fit)
    try:
        if workers == 1:
            results = map(fit, draws)
            for i, b, prof, nd in results:
                profiles[i, b] = prof
                distinct[i, b] = nd
        else:
            ex = ThreadPoolExecutor(max_workers=workers,
                                    thread_name_prefix="beam-depth-matched")
            try:
                futures = [ex.submit(fit, d) for d in draws]
                for f in futures:
                    i, b, prof, nd = f.result()
                    profiles[i, b] = prof
                    distinct[i, b] = nd
            finally:
                ex.shutdown(wait=True, cancel_futures=True)
    finally:
        set_threads(previous)

    return DepthMatchedProfiles(
        orders=q, depth=depth, weighting=weighting,
        n_bootstraps=n_bootstraps, seed=seed_used, profiles=profiles,
        n_distinct_drawn=distinct,
        n_templates=np.array([int(a.sum()) for a in abus], dtype=np.int64),
        n_distinct=np.array([len(s) for s in reps], dtype=np.int64),
        config=config)
