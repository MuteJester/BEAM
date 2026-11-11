"""An independent reference implementation of BeamModel, for the tests to check
the C against.

This is deliberately written straight from the specification in
`include/beam/model.h`, in the most obvious way, with no shared code with
`src/native/`. It is slow and that is
fine: it exists so that a test failure means the two disagree, rather than
meaning the C disagrees with itself.

The things it is here to catch, in order of how likely they are to go wrong
and how quietly they would do so:

  1. The FORWARD context is packed from the residues BEFORE the target and the
     BACKWARD context from those AFTER it. Swapping the two produces a model
     that trains, scores, normalises, and is wrong.
  2. Depth is EXACT and measured from the anchoring terminus: `i` forward and
     `L-1-i` backward. An off-by-one there addresses a well-formed row of the
     wrong depth, which is silent.
  3. Each row is normalised over the residues BEFORE logs are taken, so every
     observed row sums to 1 and every unobserved one is all-zero. That is the
     closed world, and it is what makes `Z(L)` the Bhattacharyya coefficient
     between the two chains rather than an arbitrary constant.
  4. The two directions combine log-linearly with weights that sum to 1.

The C repacks its finalized C-anchored table nearest-first for cache reasons.
That is storage, not semantics, and this file keeps one packing throughout: as
long as the counting and the lookups agree with each other, the probabilities
are the same.
"""

from __future__ import annotations

import math

import numpy as np


class Alphabet:
    """The symbols a reference model is built over.

    Written from the specification like everything else here: the radix is the
    residues plus one sentinel that pads contexts at the termini and is never
    emitted, and nucleotide input folds case while amino-acid input does not.
    """

    def __init__(self, name, symbols, fold_case=False):
        self.name = name
        self.symbols = symbols
        self.residues = len(symbols)
        self.radix = len(symbols) + 1
        self.none = len(symbols)
        self.code = {c: i for i, c in enumerate(symbols)}
        if fold_case:
            self.code.update({c.lower(): i for i, c in enumerate(symbols)})


AA_ALPHABET = Alphabet("aa", "ACDEFGHIKLMNPQRSTVWY")
NT_ALPHABET = Alphabet("nt", "ACGTN", fold_case=True)
ALPHABETS = {"aa": AA_ALPHABET, "nt": NT_ALPHABET}

# The amino-acid dimensions under their historical names, for callers that
# import them. Nothing below reads these: every function takes its alphabet.
RESIDUES = AA_ALPHABET.residues
ALPHABET = AA_ALPHABET.radix
NONE = AA_ALPHABET.none
AA = AA_ALPHABET.symbols
CODE = AA_ALPHABET.code


class ReferenceBeamModel:
    """One log-probability table per direction, plus the per-length constants.

    `lp[d][depth][ctx][x]` is `log q_d(x | depth, ctx)`.
    """

    def __init__(self, order, max_len, weights, lp, log_pi,
                 alpha=AA_ALPHABET):
        self.alpha = alpha
        self.order = order
        # The state holds exactly the residues one context needs.
        self.state_order = order
        self.max_len = max_len
        self.weights = weights
        self.lp = lp
        self.log_pi = log_pi
        self._logz = {}

    # ── contexts ─────────────────────────────────────────────────

    @staticmethod
    def ctx_forward(s, pos, k, radix):
        """The `k` residues before `pos`, most-distant-first."""
        idx = 0
        for j in range(k, 0, -1):
            q = pos - j
            idx = idx * radix + (s[q] if q >= 0 else radix - 1)
        return idx

    @staticmethod
    def ctx_backward(s, pos, k, radix, length):
        """The `k` residues after `pos`, most-distant-first."""
        idx = 0
        for j in range(k, 0, -1):
            q = pos + j
            idx = idx * radix + (s[q] if q < length else radix - 1)
        return idx

    # ── scoring ──────────────────────────────────────────────────

    def raw(self, s):
        """Sum of both directions over every residue, unnormalised."""
        A = self.alpha.radix
        n = len(s)
        total = 0.0
        for p in range(n):
            x = s[p]
            f = self.lp[0][p][self.ctx_forward(s, p, self.order, A)][x]
            b = self.lp[1][n - 1 - p][
                self.ctx_backward(s, p, self.order, A, n)][x]
            if f == -math.inf or b == -math.inf:
                return -math.inf
            total += self.weights[0] * f + self.weights[1] * b
        return total

    def logZ(self, length):
        if length not in self._logz:
            self._logz[length] = self._logz_trellis(length)
        return self._logz[length]

    def _layer(self, i, length):
        """Edge weights of one trellis layer, as an (n_states, R) array.

        Written with array operations rather than loops, but structurally
        independent of the C: the state digits are re-derived here from the
        packing rule rather than assumed.
        """
        A, R = self.alpha.radix, self.alpha.residues
        K = self.state_order
        S = A ** K
        states = np.arange(S)
        slots = [(states // A ** (K - 1 - t)) % A for t in range(K)]

        # Forward: the context IS the state, since the state holds exactly
        # `order` residues and a context is a suffix of it.
        a = self.lp[0][i][states % A ** self.order]       # (S, R)
        w = np.where(np.isneginf(a), -np.inf, self.weights[0] * a)

        if i < K:
            return w

        # Backward: settles the residue emitted K steps ago.
        scored = states // A ** (K - 1)
        r = length - 1 - (i - K)
        xs = np.arange(R)[None, :]

        idx = np.zeros((S, R), dtype=np.int64)
        for j in range(self.order, 0, -1):
            cj = xs if j == K else slots[j][:, None]
            idx = idx * A + cj
        # Clamp the residue index so the gather is in bounds for states whose
        # oldest slot is a sentinel; those rows are masked out below.
        sc = np.minimum(scored, R - 1)[:, None]
        b = self.lp[1][r][idx, sc]
        w = np.where(np.isneginf(b), -np.inf, w + self.weights[1] * b)
        return np.where(scored[:, None] < R, w, -np.inf)

    def _tail(self, length):
        """Sink weights, as an (n_states,) array."""
        A, R, NON = self.alpha.radix, self.alpha.residues, self.alpha.none
        K = self.state_order
        S = A ** K
        states = np.arange(S)
        slots = [(states // A ** (K - 1 - t)) % A for t in range(K)]

        total = np.zeros(S)
        for r in range(min(K, length)):
            pos = K - 1 - r
            scored = slots[pos]
            idx = np.zeros(S, dtype=np.int64)
            for j in range(self.order, 0, -1):
                cj = slots[pos + j] if j <= r else np.full(S, NON)
                idx = idx * A + cj
            sc = np.minimum(scored, R - 1)
            b = self.lp[1][r][idx, sc]
            mix = np.where(np.isneginf(b), -np.inf, self.weights[1] * b)
            total = np.where(scored < R, total + mix, -np.inf)
        return total

    def _logz_trellis(self, length):
        """Forward sweep in log space over the layered DAG."""
        A, R = self.alpha.radix, self.alpha.residues
        K = self.state_order
        S = A ** K
        shift = A ** (K - 1)
        cur = np.full(S, -np.inf)
        cur[S - 1] = 0.0                       # all-sentinel start

        for i in range(length):
            w = self._layer(i, length)
            m = (cur[:, None] + w).reshape(A, shift, R)
            nxt = np.full((shift, A), -np.inf)
            nxt[:, :R] = np.logaddexp.reduce(m, axis=0)
            cur = nxt.ravel()

        return float(np.logaddexp.reduce(cur + self._tail(length)))

    def log_pseq(self, seq):
        code = self.alpha.code
        if any(c not in code for c in seq):
            return -math.inf
        s = [code[c] for c in seq]
        n = len(s)
        if n == 0 or n > self.max_len or not math.isfinite(self.log_pi[n]):
            return -math.inf
        raw = self.raw(s)
        if raw == -math.inf:
            return -math.inf
        return raw + self.log_pi[n] - self.logZ(n)


# ── construction ─────────────────────────────────────────────────

def reference_build(seqs, abundances=None, *, order=4, max_len=27,
                    weights=(0.5, 0.5), alpha=None, alpha_length=None,
                    alphabet="aa"):
    """Count, smooth by the hierarchical prior, take logs. The whole model.

    Every row is the posterior mean of a Dirichlet centred on its parent,

        P_k(x) = (n_k(x) + alpha[k] * P_{k-1}(x)) / (n_k + alpha[k]),

    level k keeping the k residues nearest the scored one, level 0 the depth
    alone, and the uniform 1/R below that. Computed here the plain way, in
    dense arrays: a level's counts are the top counts summed over the
    residues it drops, and a parent is broadcast to its children by tiling.
    It shares no code with the library's sparse, depth-by-depth build.
    """
    if alpha is None:
        alpha = 10.0
    if isinstance(alpha, (int, float)):
        alpha = (float(alpha),) * (order + 1)
    alpha = tuple(float(a) for a in alpha)
    alpha_length = 10.0 if alpha_length is None else float(alpha_length)
    alph = ALPHABETS[alphabet] if isinstance(alphabet, str) else alphabet
    A, R = alph.radix, alph.residues
    n_ctx = A ** order
    if abundances is None:
        abundances = [1.0] * len(seqs)

    cnt = [np.zeros((max_len, n_ctx, R)) for _ in range(2)]
    cnt_length = np.zeros(max_len + 1)

    for seq, w in zip(seqs, abundances):
        if any(c not in alph.code for c in seq):
            continue
        s = [alph.code[c] for c in seq]
        n = len(s)
        if n == 0 or n > max_len:
            continue
        cnt_length[n] += w
        for p in range(n):
            x = s[p]
            cnt[0][p][ReferenceBeamModel.ctx_forward(s, p, order, A)][x] += w
            cnt[1][n - 1 - p][
                ReferenceBeamModel.ctx_backward(s, p, order, A, n)][x] += w

    lp = []
    for d in range(2):
        # Contexts are packed most-distant-first, so a level-k context is the
        # top context's LOW k digits: summing out the high digits is a reshape.
        prev = np.full((max_len, 1, R), 1.0 / R)
        for k in range(order + 1):
            n_k = cnt[d].reshape(max_len, A ** (order - k), A ** k, R).sum(axis=1)
            parent = np.tile(prev, (1, A, 1)) if k else prev
            tot = n_k.sum(axis=2, keepdims=True)
            prev = (n_k + alpha[k] * parent) / (tot + alpha[k])
        lp.append(np.log(prev))

    tot = cnt_length.sum()
    log_pi = np.full(max_len + 1, -np.inf)
    log_pi[1:] = np.log((cnt_length[1:] + alpha_length / max_len)
                        / (tot + alpha_length))

    m = ReferenceBeamModel(order, max_len, weights, lp, log_pi, alph)

    # Finalize renormalises the length prior over the lengths whose trellis is
    # non-empty, so a length the tables cannot spell does not keep its share.
    live = [L for L in range(1, max_len + 1)
            if math.isfinite(log_pi[L]) and math.isfinite(m.logZ(L))]
    if live:
        shift = math.log(sum(math.exp(log_pi[L]) for L in live))
        for L in range(max_len + 1):
            log_pi[L] = (log_pi[L] - shift if L in live else -math.inf)
        m._logz = {L: m._logz[L] for L in live}
    return m


def reference_scores(model, seqs):
    return [model.log_pseq(s) for s in seqs]


def reference_path_counts(model, lengths):
    """Live sequences of each length, by enumeration over the trellis."""
    out = {}
    for L in lengths:
        A, R = model.alpha.radix, model.alpha.residues
        K = model.state_order
        S = A ** K
        shift = A ** (K - 1)
        cur = np.zeros(S)
        cur[S - 1] = 1.0
        for i in range(L):
            w = model._layer(i, L)
            live = (~np.isneginf(w)).astype(float)
            m = (cur[:, None] * live).reshape(A, shift, R)
            nxt = np.zeros((shift, A))
            nxt[:, :R] = m.sum(axis=0)
            cur = nxt.ravel()
        tail = (~np.isneginf(model._tail(L))).astype(float)
        out[L] = float((cur * tail).sum())
    return out
