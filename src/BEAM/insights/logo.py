"""Sequence logos from a finalized BEAM model."""
from __future__ import annotations

import math

import numpy as np

from .. import _cbeam as _c
from ..constants import BEAM_ALPHABET, BEAM_NT_ALPHABET
from .results import Logo, LogoSet

#: The surprisal a band edge takes when the caller means "and everything
#: rarer". Resolved against the grid by the C layer, which is the only place
#: that knows where the grid ends.
_TOP = math.inf


def _edges_from(pseq_above, pseq_below, surprisal_above, surprisal_below,
                edges, length):
    """Turn the caller's condition into ascending surprisal edges.

    Surprisal is the native axis, so `surprisal_*` and `edges` need no
    conversion; `pseq_*` does, and the reversal lives here, at the boundary
    the caller thinks in. Inside the library everything is nats ascending, and
    a probability threshold entering there would invert an axis in the middle
    of code with no other reason to think about probabilities.

    Note which way each pair runs, because "above" means opposite things on
    the two axes: `pseq_above` and `surprisal_below` select the SAME
    sequences, the common ones.
    """
    given = {k: v for k, v in (("pseq_above", pseq_above),
                               ("pseq_below", pseq_below),
                               ("surprisal_above", surprisal_above),
                               ("surprisal_below", surprisal_below),
                               ("edges", edges)) if v is not None}
    if len(given) > 1:
        raise ValueError(
            f"give at most one of pseq_above, pseq_below, surprisal_above, "
            f"surprisal_below or edges; they are ways of saying which "
            f"sequences the logo is over, and you gave "
            f"{', '.join(sorted(given))}")
    if not given:
        return None
    if edges is not None:
        e = np.ascontiguousarray(np.asarray(edges, dtype=np.float64).ravel())
        if e.size < 2:
            raise ValueError(
                f"{e.size} band edge(s) define no band; two ascending edges "
                f"define one")
        return e

    if surprisal_above is not None or surprisal_below is not None:
        cut = float(surprisal_above if surprisal_above is not None
                    else surprisal_below)
        if not math.isfinite(cut) or cut < 0.0:
            raise ValueError(
                f"a surprisal threshold is a finite number of nats and is "
                f"never negative, got {cut!r}. A negative one is almost "
                f"always ln(p) where -ln(p) was meant")
        # Rarer is HIGHER surprisal, so "above" is the upper band. This is the
        # opposite way round from the probability keywords, which is why they
        # are named for their own axis rather than sharing one word.
        if surprisal_above is not None:
            return np.array([cut, _TOP], dtype=np.float64)
        return np.array([-_TOP, cut], dtype=np.float64)

    t = float(pseq_above if pseq_above is not None else pseq_below)
    if not (0.0 < t < 1.0):
        raise ValueError(
            f"a Pseq threshold is a probability strictly between 0 and 1, "
            f"got {t!r}")
    cut = -math.log(t)
    # Above in probability is BELOW in surprisal, and vice versa.
    if pseq_above is not None:
        return np.array([-_TOP, cut], dtype=np.float64)
    return np.array([cut, _TOP], dtype=np.float64)


def _residues(alphabet_code: int) -> str:
    return BEAM_NT_ALPHABET if alphabet_code else BEAM_ALPHABET


def _build(raw, length, residues, max_bits, max_nats):
    (prob, hb, hn, eb, en, ib, inn, sup, cons, consp, pm,
     lo, hi, log_mass, mass, fraction, res_err) = raw
    R = len(residues)
    f = np.float64
    return Logo(
        probability=np.frombuffer(prob, dtype=f).reshape(length, R).copy(),
        height_bits=np.frombuffer(hb, dtype=f).reshape(length, R).copy(),
        height_nats=np.frombuffer(hn, dtype=f).reshape(length, R).copy(),
        entropy_bits=np.frombuffer(eb, dtype=f).copy(),
        entropy_nats=np.frombuffer(en, dtype=f).copy(),
        information_bits=np.frombuffer(ib, dtype=f).copy(),
        information_nats=np.frombuffer(inn, dtype=f).copy(),
        n_support=np.frombuffer(sup, dtype=np.uint32).copy(),
        consensus=np.frombuffer(cons, dtype=np.uint32).copy(),
        consensus_probability=np.frombuffer(consp, dtype=f).copy(),
        position_mass=(None if pm is None
                       else np.frombuffer(pm, dtype=f).copy()),
        residues=residues,
        lo_nats=lo, hi_nats=hi, log_mass=log_mass, mass=mass,
        fraction=fraction,
        resolution_error=(None if math.isnan(res_err) else res_err),
        max_entropy_bits=max_bits, max_entropy_nats=max_nats)


def _args(g, length, edges, counting, bins, spacing, reference_residues,
          max_resolution_error):
    return (g, int(length), None if edges is None else edges.tobytes(),
            bool(counting), int(bins), float(spacing),
            int(reference_residues), float(max_resolution_error))


def logos(model, length, *, pseq_above=None, pseq_below=None,
          surprisal_above=None, surprisal_below=None, edges=None,
          counting=False, bins=0, spacing=0.0, reference_residues=0,
          max_resolution_error=0.0) -> LogoSet:
    """Sequence logos for one length, optionally split by Pseq.

    Args:
        length: the sequence length. Logos are per length -- position 5 of a
            length-12 sequence and position 5 of a length-18 one are
            different distances from the C terminus, so there is no pooled
            form.
        pseq_above: keep only sequences with ``Pseq > t``. One logo.
        pseq_below: keep only sequences with ``Pseq <= t``. One logo.
        surprisal_above: keep only sequences with ``-ln Pseq >= s``, in nats
            -- the RARE ones. Note the reversal: ``surprisal_above`` and
            ``pseq_below`` select the same sequences, and so do
            ``surprisal_below`` and ``pseq_above``. Surprisal is the native
            axis, so these need no conversion.
        surprisal_below: keep only sequences with ``-ln Pseq < s``, in nats.
        edges: surprisal edges in nats, ascending, for full control;
            ``n`` edges give ``n - 1`` logos. ``inf`` is allowed at either
            end and is resolved against the grid.
        counting: weight every producible sequence 1 rather than by its own
            probability. On an OPEN-WORLD model this gives a flat logo at
            every position, because the add-one makes every sequence
            producible and the counting marginal uniform.
        bins, spacing: the surprisal grid, used only when a band is asked
            for. Both default to a choice made from the model.
        reference_residues: the alphabet size information is measured
            against. Defaults to the model's own. Pass 4 for the conventional
            2-bit DNA axis -- BEAM has five nucleotide residues, because N is
            emitted rather than a wildcard.
        max_resolution_error: refuse rather than return a logo whose grid
            error exceeds this.

    Returns:
        :class:`~BEAM.insights.results.LogoSet`.

    The unconditional generated logo is EXACT: no grid is involved at all.
    Any Pseq condition needs the surprisal grid, and then
    ``logo.resolution_error`` says how far it was from resolving the band.

    Raises:
        ValueError: for a length the model does not generate, edges that do
            not ascend, a band outside the surprisal this length can attain,
            or a grid coarser than asked for.
        RuntimeError: for a band holding no mass. That is refused rather than
            returned, because a logo of nothing comes back as maximum
            information content at every position.

        >>> s = model.insights.logos(15, surprisal_above=20.7)   # nats
        >>> s = model.insights.logos(15, pseq_below=1e-9)          # the same
        >>> s[0].consensus_sequence
        'CASSLGGGGGYEQYF'
        >>> s[0].information_bits.round(2)[:4]
        array([4.32, 3.91, 2.87, 2.34])
    """
    e = _edges_from(pseq_above, pseq_below, surprisal_above, surprisal_below,
                    edges, length)
    a = _args(model._g, length, e, counting, bins, spacing,
              reference_residues, max_resolution_error)
    (raw, n_res, alpha, ref, max_bits, max_nats, on_grid, whole, spc,
     exact, cnt) = _c.insights_logos(*a)
    residues = _residues(alpha)[:n_res]
    return LogoSet(
        logos=[_build(r, int(length), residues, max_bits, max_nats)
               for r in raw],
        length=int(length), residues=residues, reference_residues=ref,
        max_entropy_bits=max_bits, max_entropy_nats=max_nats,
        on_grid_total=on_grid, length_total=whole,
        spacing=(None if math.isnan(spc) else spc),
        exact=bool(exact), counting=bool(cnt))


def logo(model, length, **kw) -> Logo:
    """One logo, for a condition that defines exactly one band.

    The common case. Asking for several bands through this raises rather than
    silently returning the first, because "the logo" of four bands is not a
    question with an answer.
    """
    s = logos(model, length, **kw)
    if len(s) != 1:
        raise ValueError(
            f"this condition defines {len(s)} bands, not one; use logos() "
            f"and pick, or narrow the condition")
    return s[0]


def cost(model, length, *, pseq_above=None, pseq_below=None,
         surprisal_above=None, surprisal_below=None, edges=None,
         counting=False, bins=0, spacing=0.0, reference_residues=0,
         max_resolution_error=0.0) -> dict:
    """What :func:`logos` will cost, without running it.

    ``bytes`` is the figure that will be checked against the memory budget,
    from the same expression -- so a caller can compare against
    :func:`BEAM.memory_budget` and warn, rather than catch a refusal. It
    slightly UNDERSTATES a banded sweep: the backward grid carries an origin
    offset that is only known once the sink weights are built.
    """
    e = _edges_from(pseq_above, pseq_below, surprisal_above, surprisal_below,
                    edges, length)
    a = _args(model._g, length, e, counting, bins, spacing,
              reference_residues, max_resolution_error)
    states, edge_count, nbytes = _c.insights_logo_cost(*a)
    return {"states": states, "edges": edge_count, "bytes": nbytes}
