"""An independent reference implementation of BeamModel, for the tests to check
the C against.

This is deliberately written straight from the specification in
`include/beam/model.h` and `.private/edge/METHOD.md` (working tree, not
shipped), in the most
obvious way, with no shared code with `src/native/`. It is slow and that
is fine: it exists so that a test failure means the two disagree, rather than
meaning the C disagrees with itself.

The things it is here to catch, in order of how likely they are to go wrong
and how quietly they would do so:

  1. The FORWARD context is packed from the residues BEFORE the target and the
     BACKWARD context from those AFTER it, both most-distant-first. Swapping
     the two produces a model that trains, scores, normalises, and is wrong.
  2. The hierarchical backoff drops the MOST DISTANT context residue at each
     level, so an order-k estimate falls back on the order-(k-1) one it
     refines.
  3. The wide expert's backoff base is the order-(k-1) table at the matching
     DEPTH BUCKET, with depths at or beyond the cap sharing one bucket built
     from depth-aggregated counts.
  4. The three experts combine log-linearly, and the two directions combine
     log-linearly on top of that.
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
    """Log-probability tables plus the per-length constants."""

    def __init__(self, order, depth_cap, max_len, coefs, weights,
                 lp_wide, lp_mid, lp_len, log_pi, alpha=AA_ALPHABET):
        self.alpha = alpha
        self.order = order
        # The mid expert is order 3, so the DAG state can never be narrower.
        # Conflating the two silently reads a neighbouring cell for order < 3.
        self.state_order = max(order, 3)
        self.depth_cap = depth_cap
        self.max_len = max_len
        self.coefs = coefs
        self.weights = weights
        self.lp_wide = lp_wide          # [dir][depth_bucket][ctx_k][x]
        self.lp_mid = lp_mid            # [dir][depth][ctx_3][x]
        self.lp_len = lp_len            # [dir][L][depth][ctx_2][x]
        self.log_pi = log_pi
        self._logz = {}

    # ── context packing ──────────────────────────────────────────

    @staticmethod
    def ctx_forward(s, pos, k, radix):
        """c_j = s[pos - j], packed most-distant-first."""
        none = radix - 1
        idx = 0
        for j in range(k, 0, -1):
            q = pos - j
            idx = idx * radix + (s[q] if q >= 0 else none)
        return idx

    @staticmethod
    def ctx_backward(s, pos, k, radix):
        """c_j = s[pos + j], packed most-distant-first."""
        none = radix - 1
        idx = 0
        n = len(s)
        for j in range(k, 0, -1):
            q = pos + j
            idx = idx * radix + (s[q] if q < n else none)
        return idx

    # ── one residue, one direction ───────────────────────────────

    def mix(self, d, length, depth, cw, cm, cl, x):
        bucket = min(depth, self.depth_cap)
        a = self.lp_wide[d][bucket][cw][x]
        b = self.lp_mid[d][depth][cm][x]
        c = self.lp_len[d][length][depth][cl][x]
        if -math.inf in (a, b, c):
            return -math.inf
        return self.coefs[0] * a + self.coefs[1] * b + self.coefs[2] * c

    def raw(self, s):
        """Sum of both directions over every residue, unnormalised."""
        A = self.alpha.radix
        n = len(s)
        total = 0.0
        for p in range(n):
            x = s[p]
            f = self.mix(0, n, p,
                         self.ctx_forward(s, p, self.order, A),
                         self.ctx_forward(s, p, 3, A),
                         self.ctx_forward(s, p, 2, A), x)
            b = self.mix(1, n, n - 1 - p,
                         self.ctx_backward(s, p, self.order, A),
                         self.ctx_backward(s, p, 3, A),
                         self.ctx_backward(s, p, 2, A), x)
            if f == -math.inf or b == -math.inf:
                return -math.inf
            total += self.weights[0] * f + self.weights[1] * b
        return total

    # ── partition function, by brute-force enumeration where it is
    #    affordable and by the trellis otherwise ──────────────────

    def logZ(self, length):
        if length in self._logz:
            return self._logz[length]
        z = self._logz_trellis(length)
        self._logz[length] = z
        return z

    def _layer(self, i, length):
        """Edge weights of one trellis layer, as an (n_states, 20) array.

        Written with array operations rather than loops, but structurally
        independent of the C: the state digits are re-derived here from the
        packing rule rather than assumed.
        """
        A, R = self.alpha.radix, self.alpha.residues
        K = self.state_order
        S = A ** K
        states = np.arange(S)
        slots = [(states // A ** (K - 1 - t)) % A for t in range(K)]

        def gather(d, depth, cw, cm, cl, x_axis_last):
            bucket = min(depth, self.depth_cap)
            a = self.lp_wide[d][bucket][cw]
            b = self.lp_mid[d][depth][cm]
            c = self.lp_len[d][length][depth][cl]
            return a, b, c

        # ── forward: every context is a suffix of the state ──
        a, b, c = gather(0, i,
                         states % A ** self.order,
                         states % A ** 3,
                         states % A ** 2, True)
        mix = (self.coefs[0] * a + self.coefs[1] * b + self.coefs[2] * c)
        mix = np.where(np.isneginf(a) | np.isneginf(b) | np.isneginf(c),
                       -np.inf, mix)
        w = self.weights[0] * mix                       # (S, R)

        if i < K:
            return w

        # ── backward: settles the residue emitted K steps ago ──
        scored = states // A ** (K - 1)                 # (S,)
        r = length - 1 - (i - K)
        xs = np.arange(R)[None, :]                      # (1, R)

        def pack(order_k):
            idx = np.zeros((S, R), dtype=np.int64)
            for j in range(order_k, 0, -1):
                cj = xs if j == K else slots[j][:, None]
                idx = idx * A + cj
            return idx

        bw, bm, bl = pack(self.order), pack(3), pack(2)
        bucket = min(r, self.depth_cap)
        # Clamp the residue index so the gather is in bounds for states whose
        # oldest slot is a sentinel; those rows are masked out below.
        sc = np.minimum(scored, R - 1)[:, None]
        a = self.lp_wide[1][bucket][bw, sc]
        b = self.lp_mid[1][r][bm, sc]
        c = self.lp_len[1][length][r][bl, sc]
        bmix = (self.coefs[0] * a + self.coefs[1] * b + self.coefs[2] * c)
        bmix = np.where(np.isneginf(a) | np.isneginf(b) | np.isneginf(c),
                        -np.inf, bmix)
        w = np.where(scored[:, None] < R, w + self.weights[1] * bmix,
                     -np.inf)
        return w

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

            def pack(order_k):
                idx = np.zeros(S, dtype=np.int64)
                for j in range(order_k, 0, -1):
                    cj = slots[pos + j] if j <= r else np.full(S, NON)
                    idx = idx * A + cj
                return idx

            bucket = min(r, self.depth_cap)
            sc = np.minimum(scored, R - 1)
            a = self.lp_wide[1][bucket][pack(self.order), sc]
            b = self.lp_mid[1][r][pack(3), sc]
            c = self.lp_len[1][length][r][pack(2), sc]
            mix = (self.coefs[0] * a + self.coefs[1] * b + self.coefs[2] * c)
            mix = np.where(np.isneginf(a) | np.isneginf(b) | np.isneginf(c),
                           -np.inf, mix)
            total = np.where(scored < R, total + self.weights[1] * mix,
                             -np.inf)
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

def _hierarchy(counts, n_depth, order, kappa, alpha):
    """Backoff from order 0 up to `order`, dropping the most distant context
    residue at each level. Returns the probability table at every order."""
    A, R = alpha.radix, alpha.residues
    levels = [counts]
    for _ in range(order):
        c = levels[-1]
        levels.append(c.reshape(n_depth, A, -1, R).sum(1))
    levels = levels[::-1]                       # levels[j] keeps j residues

    p = levels[0] + 1.0                         # Laplace alpha = 1
    p = p / p.sum(-1, keepdims=True)
    out = [p]
    for j in range(1, order + 1):
        c = levels[j].reshape(n_depth, -1, R)
        parent = out[-1].reshape(n_depth, -1, R)
        n_ctx = c.shape[1]
        rep = n_ctx // parent.shape[1]
        q = c + kappa[j - 1] * np.tile(parent, (1, rep, 1))
        tot = q.sum(-1, keepdims=True)
        q = np.divide(q, tot, out=np.zeros_like(q), where=tot > 0)
        out.append(q)
    return out


def reference_build(seqs, abundances=None, *, order=4, depth_cap=15,
                    max_len=27, coefs=(0.60, 0.25, 0.15),
                    kappa=(20.0, 20.0, 20.0), kappa_wide=20.0,
                    kappa_len=200.0, weights=(0.5, 0.5), alphabet="aa"):
    """Count, smooth, and assemble, straight from the specification.

    `depth_cap` is clamped to `max_len - 1` exactly as `beam_model_new`
    clamps it. Without that, a reference built at a short `max_len` keeps the
    default cap of 15 while the C model silently reduces it, and the two
    disagree for a reason that has nothing to do with the thing under test.
    """
    if abundances is None:
        abundances = [1.0] * len(seqs)
    alpha = ALPHABETS[alphabet] if isinstance(alphabet, str) else alphabet
    A, R, CODE_ = alpha.radix, alpha.residues, alpha.code
    depth_cap = min(depth_cap, max_len - 1)
    n_bucket = depth_cap + 1
    n_ctx_w = A ** order

    cw = [np.zeros((n_bucket, n_ctx_w, R)) for _ in range(2)]
    cm = [np.zeros((max_len, A ** 3, R)) for _ in range(2)]
    cl = [np.zeros((max_len + 1, max_len, A ** 2, R))
          for _ in range(2)]
    clen = np.zeros(max_len + 1)

    for seq, w in zip(seqs, abundances):
        if not seq or len(seq) > max_len or any(c not in CODE_ for c in seq):
            continue
        s = [CODE_[c] for c in seq]
        n = len(s)
        clen[n] += w
        for p in range(n):
            x = s[p]
            for d, depth, ctx in (
                (0, p, ReferenceBeamModel.ctx_forward),
                (1, n - 1 - p, ReferenceBeamModel.ctx_backward),
            ):
                cw[d][min(depth, depth_cap), ctx(s, p, order, A), x] += w
                cm[d][depth, ctx(s, p, 3, A), x] += w
                cl[d][n, depth, ctx(s, p, 2, A), x] += w

    log_pi = np.where(clen > 0, np.log(np.where(clen > 0, clen / clen.sum(), 1)),
                      -np.inf)

    lp_wide, lp_mid, lp_len = [], [], []
    for d in range(2):
        mid = _hierarchy(cm[d], max_len, 3, kappa, alpha)
        agg = _hierarchy(cm[d][depth_cap:].sum(0)[None], 1, 3, kappa, alpha)

        # length-conditioned expert: order 2, backing off to the order-2
        # marginal of the mid counts at the same depth
        base2 = mid[2]
        arr = np.zeros_like(cl[d])
        for L in range(max_len + 1):
            q = cl[d][L] + kappa_len * base2
            tot = q.sum(-1, keepdims=True)
            arr[L] = np.divide(q, tot, out=np.zeros_like(q), where=tot > 0)

        # wide expert: order K, backing off to order K-1 at the matching bucket
        k1 = order - 1
        n_ctx1 = A ** k1
        wide = np.zeros_like(cw[d])
        for bucket in range(n_bucket):
            base = mid[k1][bucket] if bucket < depth_cap else agg[k1][0]
            q = cw[d][bucket] + kappa_wide * np.tile(base,
                                                     (n_ctx_w // n_ctx1, 1))
            tot = q.sum(-1, keepdims=True)
            wide[bucket] = np.divide(q, tot, out=np.zeros_like(q), where=tot > 0)

        def to_log(p):
            return np.log(np.maximum(p, 1e-300))

        lp_wide.append(to_log(wide))
        lp_mid.append(to_log(mid[3]))
        lp_len.append(to_log(arr))

    return ReferenceBeamModel(order, depth_cap, max_len, coefs, weights,
                         lp_wide, lp_mid, lp_len, log_pi, alpha)


def reference_scores(model, seqs):
    return np.array([model.log_pseq(s) for s in seqs])


def reference_path_counts(model, lengths):
    """Exact support size per length over the backward-trimmed trellis."""
    A, R = model.alpha.radix, model.alpha.residues
    K = model.state_order
    S = A ** K
    shift = A ** (K - 1)
    out = {}
    for length in lengths:
        layers = [model._layer(i, length) for i in range(length)]
        tail = model._tail(length)

        alive = [None] * (length + 1)
        alive[length] = np.isfinite(tail)
        for i in range(length - 1, -1, -1):
            nb = alive[i + 1].reshape(shift, A)[:, :R]
            ok = np.isfinite(layers[i]) & np.tile(nb, (A, 1))
            alive[i] = ok.any(1)

        cur = np.zeros(S)
        cur[S - 1] = 1.0 if alive[0][S - 1] else 0.0
        for i in range(length):
            nb = alive[i + 1].reshape(shift, A)[:, :R]
            ok = np.isfinite(layers[i]) & np.tile(nb, (A, 1))
            contrib = np.where(ok, cur[:, None], 0.0)
            nxt = np.zeros((shift, A))
            nxt[:, :R] = contrib.reshape(A, shift, R).sum(0)
            cur = nxt.ravel()
        out[length] = float(cur[np.isfinite(tail)].sum())
    return out
