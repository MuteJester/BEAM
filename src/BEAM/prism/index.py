"""PRISM: the one-edit repair factor and its per-position posterior.

A prism takes a beam and splits it into components. PRISM takes one BEAM score
and splits it into per-position evidence, from a single pass: computing the
sequence-wide factor means visiting every edit history exactly once, so the
localisation is free rather than a second machine.

WHAT IT IS. Two hypotheses about an observed sequence ``s``:

``H0``
    ``s`` was drawn from the model.
``H1``
    a clean ``x`` was drawn from the model, one of substitution / insertion /
    deletion was chosen with probability 1/3, a legal site and residue were
    chosen uniformly, and the single edit produced ``s``.

``B = Q/P`` is the likelihood ratio, the PRISM factor. Large ``B`` means a
one-edit repair explains the sequence better than the model does.

WHY THAT PHRASING AND NOT "NOISE". A sequencing error gives a large ``B``. So
does a genuinely rare biological sequence that happens to sit one edit from a
common one. The model cannot tell them apart, and the name should not pretend
otherwise.

THE PROPERTY WORTH THE WHOLE DESIGN. Because ``B`` is a likelihood ratio,
``E[B] <= 1`` under the model, so it is an e-value and Markov gives
``P(B >= t) <= 1/t`` directly. A threshold is a false-positive BOUND with no
calibration table, no simulated null and nothing fitted. The index this
replaces needed four separate simulated nulls to say less.

The bound is under the model's own law. On real corpus sequences it holds
comfortably rather than tightly: measured, ``B >= 100`` flags 0.27% of clean
corpus CDR3s against the 1% the bound allows. Conservative, and valid.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..runtime import _c


class Prism:
    """One sequence's repair factor and the posterior over where the edit was.

    Attributes:
        sequence: the sequence scored.
        log_b: ``log B``. ``+inf`` when the model cannot spell the sequence but
            can spell a repair, which is the strongest one-edit evidence there
            is. ``-inf`` when no single edit reaches the model's support.
            ``nan`` when neither the sequence nor any repair is in it, which
            means "not explainable by one edit" rather than "clean".
        log_q, log_p0: the two halves of the ratio.
        substitution: per-residue log evidence that this residue was
            substituted, length ``L``.
        insertion: per-residue log evidence that this residue is a spurious
            insertion, length ``L``.
        deletion: per-GAP log evidence that a residue is missing there, length
            ``L + 1``. Deletions live between residues, not on them, which is
            why this array is one longer.

    The three arrays are unnormalised; :meth:`posterior` divides by ``Q``.
    """

    __slots__ = ("sequence", "log_b", "log_q", "log_p0",
                 "substitution", "insertion", "deletion", "n_residues")

    def __init__(self, *, sequence, log_b, log_q, log_p0,
                 substitution, insertion, deletion, n_residues):
        #: Residues the scoring model emits. The channel denominators below
        #: are counts of possible edits, so they are alphabet-sized: there are
        #: `n_residues - 1` substitutions at a site and `n_residues`
        #: insertions before it.
        self.n_residues = n_residues
        self.sequence = sequence
        self.log_b = log_b
        self.log_q = log_q
        self.log_p0 = log_p0
        self.substitution = substitution
        self.insertion = insertion
        self.deletion = deletion

    @property
    def spellable(self) -> bool:
        """Whether the model gives the sequence non-zero probability."""
        return math.isfinite(self.log_p0)

    def posterior(self) -> dict[str, np.ndarray]:
        """``P(edit history | s, H1)``, summing to 1 across all three arrays.

        This is a posterior from the SAME alternative that produced the
        detection score, not a separately calibrated attribution. That is the
        point: the thing that says "flagged" and the thing that says "here" are
        one object, so they cannot disagree.
        """
        n = len(self.sequence)
        if not math.isfinite(self.log_q):
            return {"substitution": np.full(n, np.nan),
                    "insertion": np.full(n, np.nan),
                    "deletion": np.full(n + 1, np.nan)}
        # THE PER-SITE ARRAYS CARRY NO CHANNEL DENOMINATOR. C applies
        # (R-1)L, RL and L+1 to the aggregated channel totals only, so a posterior formed
        # from the raw arrays does not sum to 1 -- it came out at 215 the first
        # time this was checked. Each site's contribution is its own log
        # evidence, less its channel's operation count, less the 1/3 type
        # prior; dividing by Q then normalises exactly.
        subs = float(self.n_residues - 1)     # every residue but this one
        ins = float(self.n_residues)           # every residue, before this one
        denom = {"substitution": math.log(subs * n) if n else math.inf,
                 "insertion": math.log(ins * n) if n else math.inf,
                 "deletion": math.log(n + 1.0)}
        base = self.log_q + math.log(3.0)
        return {k: np.exp(np.asarray(v, dtype=np.float64) - denom[k] - base)
                for k, v in (("substitution", self.substitution),
                             ("insertion", self.insertion),
                             ("deletion", self.deletion))}

    def site(self) -> tuple[str, int, float]:
        """The most probable edit history: ``(kind, index, posterior)``.

        ``index`` is a residue index for a substitution or insertion and a GAP
        index for a deletion.
        """
        p = self.posterior()
        best = ("substitution", 0, -np.inf)
        for kind, arr in p.items():
            if arr.size and np.any(np.isfinite(arr)):
                i = int(np.nanargmax(arr))
                if arr[i] > best[2]:
                    best = (kind, i, float(arr[i]))
        return best

    def __repr__(self) -> str:
        k, i, q = self.site()
        return (f"Prism({self.sequence!r}, log_b={self.log_b:.3f}, "
                f"site={k}@{i} p={q:.3f})")


class PrismScorer:
    """A model prepared for scoring many sequences.

    WHAT IS ACTUALLY REUSED, and what is not.

    Each sequence costs ``40L + 20`` scorings of one-edit neighbours, and those
    neighbours are specific to it, so there is no large table to precompute
    once and share. What the prepared form does buy:

    * **The candidates never reach Python.** They are edited in place in the
      encoded buffer inside C. Building them as Python strings, which is what
      a straightforward implementation does, was the dominant cost.
    * **The batch runs under OpenMP with the GIL released**, scheduled
      dynamically because cost grows with length and CDR3s vary two-fold.
    * **The model is validated and the capsule resolved once**, not per call.

    Measured against the same computation driven from Python: 12,058 seq/s
    against 2,527, about 5x. That is the honest figure; the constant is a
    string-building constant, not an algorithmic one, and no amount of caching
    changes the ``40L + 20`` scorings the definition requires.

    A memo across sequences was considered and left out. Repertoire members do
    share neighbours, but the hit rate depends entirely on how related the
    input is, so it would make throughput unpredictable and the cache would
    outweigh the model on large inputs.
    """

    __slots__ = ("_model", "_g")

    def __init__(self, model):
        # Imported here rather than at module scope: BeamModel constructs
        # analytics objects, so a top-level import would be a cycle.
        from ..model.facade import BeamModel
        if not isinstance(model, BeamModel):
            raise TypeError(f"model must be a BeamModel, got "
                            f"{type(model).__name__}")
        self._model = model
        self._g = model._g

    @property
    def model(self):
        return self._model

    def factor(self, sequences, omega: int = 1):
        """``log B`` for one sequence or many.

        Returns a ``float`` for a single string and an ``np.ndarray``
        otherwise. A sequence the model's alphabet cannot encode scores
        ``nan`` rather than raising, because one bad row in a repertoire file
        must not lose the other million.

        ``omega`` is how many edits the alternative allows.

        **It is not an accuracy dial.** ``Q_k = T^k P`` and ``T`` is an
        averaging operator, so raising omega SPREADS mass rather than
        accumulating it: for a singly-corrupted sequence ``Q_1`` concentrates
        on the clean parent while ``Q_2`` dilutes that across the parent's own
        neighbours, and ``B`` typically falls. Measured on the 927M foundation,
        mean ``log B`` went -1.105 at omega=1 to -2.000 at omega=2.

        Raise it when you are looking for sequences with TWO edits, which is
        the alternative it actually names. Each level multiplies the work by
        about ``40L + 20``: measured, 1,949 seq/s at omega=1 against 6.6 at
        omega=2, so this is a tool for a flagged subset rather than a
        repertoire.

        The guarantee is unaffected at any omega, since ``T`` maps a
        probability law to a probability law.
        """
        single = isinstance(sequences, str)
        seqs = [sequences] if single else list(sequences)
        out = np.asarray(_c.prism_factor(self._g, seqs, int(omega)),
                         dtype=np.float64)
        return float(out[0]) if single else out

    def explain(self, sequence: str, omega: int = 1) -> Prism:
        """:class:`Prism` for one sequence: the factor and the posterior."""
        if not isinstance(sequence, str):
            raise TypeError("explain takes a single sequence; use factor() "
                            "for a batch")
        d = _c.prism_explain(self._g, sequence, int(omega))
        return Prism(
            sequence=sequence,
            n_residues=self._model.n_residues,
            log_b=float(d["log_b"]), log_q=float(d["log_q"]),
            log_p0=float(d["log_p0"]),
            substitution=np.asarray(d["sub"], dtype=np.float64),
            insertion=np.asarray(d["ins"], dtype=np.float64),
            deletion=np.asarray(d["del"], dtype=np.float64))

    def flag(self, sequences, threshold: float = 1.0, omega: int = 1):
        """Boolean flags at ``B >= threshold``.

        **The default is 1, and it is the only non-arbitrary point in the
        method.** ``B = 1`` is where the likelihood ratio crosses one: a
        one-edit repair explains the sequence exactly as well as the model
        does. Above it, a repair is the better explanation. That needs no
        convention, no calibration and no clean set, and it is what the number
        means rather than a cutoff chosen to hit a rate.

        Any ``threshold > 1`` also carries a Markov bound, since ``E[B] <= 1``
        makes ``B`` an e-value: ``P(B >= t) <= 1/t``. So ``threshold=100``
        guarantees at most 1% false positives on the model's own draws. That
        guarantee is real but LOOSE -- Markov assumes nothing about the
        distribution's shape -- and on measured data it delivers roughly a
        fifth of what it allows, which is detection paid for nothing. Measured
        on a 250,000-sequence benchmark:

        ===========  ============  ==========  ======
        threshold    Markov bound  actual FPR  TPR
        ===========  ============  ==========  ======
        ``B >= 1``   100%          21.75%      42.79%
        ``B >= 10``  10%           2.22%       18.41%
        ``B >= 100`` 1%            0.39%       7.13%
        ===========  ============  ==========  ======

        If a specific false-positive rate is required and a clean set exists,
        an empirical threshold beats the bound by a wide margin: 1% is reached
        at ``B >= 17``, with twice the detection of ``B >= 100``.

        A NOTE ON WHAT ``B >= 1`` FLAGS. It flags about a fifth of genuine
        CDR3s, and that is not a defect. Public sequences are dense, so for
        many real sequences a one-edit neighbour genuinely IS more probable.
        The statistic is answering the question honestly; whether a fifth is
        an acceptable rate is a decision about the application.
        """
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        b = self.factor(sequences, omega=omega)
        t = math.log(threshold)
        return (b >= t) if not isinstance(b, float) else bool(b >= t)

    def __repr__(self) -> str:
        return f"PrismScorer({self._model!r})"
