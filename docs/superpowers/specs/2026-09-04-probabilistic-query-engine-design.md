# Probabilistic query engine

**Status:** approved design, implementation staged
**Branch:** `probabilistic-queries`
**Date:** 2026-09-04

## Goal

A C foundation for answering exact probabilistic questions about a finalized
BEAM model, over which Python is a binding layer and nothing more.

The questions are of the form "what is the probability, expectation, variance,
entropy, or most likely value of *this*, given *that*" -- where *that* may be
any regular language over the alphabet, not merely a fixed residue at a fixed
position. `P(x_j | x_i = a)` must fall out as the simplest cell of a much
larger space, not be the interface the space is built around.

Every answer is exact. Nothing in this subsystem samples, estimates, or
approximates. A question that cannot be answered exactly is refused with a
reason, not answered approximately without one.

## Non-goals

- Sampling or Monte Carlo estimation of anything. The library's identity is
  exactness; a query API that quietly estimates would cost more than it gives.
- Expectations of arbitrary user functions of a whole sequence. Only
  functionals that decompose over the trellis are computable in closed form.
  Everything else is refused before any work is done.
- Context-free or unbounded-counting constraints. The constraint language is
  regular, deliberately: regular is exactly what a finite product with the
  trellis can absorb.
- Changing any existing numeric result. `positional_marginals`,
  `log_clamp_probability`, `expected_counts`, PRISM and every scoring path
  must be bit-for-bit unchanged.
- Inferring an evolutionary or recombination process. The cross-length indel
  algebra compares what the model assigns to two strings; it does not model an
  alignment, and the documentation must say so plainly.

## The organizing idea: two axes

Every exact query is an independent choice on two axes. Confusing them is what
makes query APIs sprawl.

**Axis 1 -- the event.** What is conditioned on, or whose probability is
asked. A `BeamConstraint`.

**Axis 2 -- the quantity.** What the sum-product accumulates as it walks the
same graph. A semiring.

|  | log-prob | expectation | second-order | max-plus |
|---|---|---|---|---|
| **layer masks** | `P(x_i=a)`, `P(clamp)` | `E[count(C) \| clamp]` | `Var`, `H(X \| clamp)` | MAP given clamp |
| **automaton** | `P(contains CASSQ)` | `E[length \| >=2 cysteines]` | `H(X \| motif)` | MAP under a regex |

The grid is the deliverable. Any new question is a new cell or a new
functional, never a new engine.

## Constraints

A constraint denotes a subset of `R^L` for each length `L`. It has two parts,
and the pair is a deliberate cost ladder.

### Layer masks -- the free tier

`mask[i]` is a bitset over the `R` residues: the set of values position `i`
may take. The denoted language is `{s : s_i in mask[i] for all i}`.

Masks delete edges. They do not add states, so a mask-only constraint costs
exactly what an unconstrained sweep costs. Conjunction is a bitwise AND, so
composing clamps is free. Masks express: a clamped residue, a residue set
("position 5 is aromatic"), a prefix, a suffix, a negation at a position.

This tier is not a convenience wrapper over the general one. It is the common
case, and it must stay free.

### The automaton -- the general tier

`(Q, delta, q0, F)` over residue indices. The denoted language is
`{s : delta*(q0, s) in F}`. Anything referring to more than one position at
once lives here: a motif anywhere, a bounded count, a union of two mask
constraints, any regular expression.

**The transition function is layer-indexed**, `delta[i] : Q x R -> Q`, with a
`uniform` flag selecting a single shared table when every layer agrees. Motifs
and counters are uniform; a disjunction of two position-specific masks is not.
Layer-indexing costs `L * Q * R` `uint32` -- 9,600 entries at `L=30, Q=16,
R=20`, which is nothing -- and it removes an entire class of "cannot be
expressed" from the design.

A constraint's language is the **intersection** of its masks and its
automaton. `|Q| = 1` means mask-only, and the engine takes the free path.

### Composition

| operation | result |
|---|---|
| `and(mask, mask)` | mask, bitwise AND -- free |
| `and(A, B)` with automata | product automaton, `\|Q_A\| * \|Q_B\|` |
| `or(A, B)` | requires automata; masks are lifted first |
| `not(A)` | complement of `F`; masks lift to a layer-indexed automaton |

Builders: `position`, `position_set`, `prefix`, `suffix`, `motif_anywhere`,
`count_range`, `and`, `or`, `not`.

## The product graph

`BeamProduct` is a constraint resolved against one length.

States are pairs `(s, q)`. The live set per layer is built by intersecting the
existing `BeamLive` frontier -- which already stores per-layer reached states
and edge bits, and is already cached -- with the automaton states that are
both forward-reachable from `q0` and backward-co-reachable to `F`.

**Co-reachability is a correctness requirement, not an optimization.** Without
it a constraint that no path satisfies yields `Z = 0` and every conditional
divides by zero. With it, the empty live set is detected at construction and
reported as `BEAM_ERR_NO_LIVE_PATHS` naming the layer where the language
died.

### Gathering by target requires the inverse transition

The existing forward sweep accumulates **by target** -- each target state
gathers from its `A` sources -- specifically so that no two threads write the
same accumulator, and the product engine must keep that property.

The trellis half inverts by construction: `next(s, x) = (s % state_shift) * A
+ x`, so a target names its own last residue and the sources differ only in
the digit that falls off the end. The automaton half does not invert: several
`q` may reach one `q'` on the same residue.

So `automaton.c` precomputes the inverse relation `delta_inv[i][x][q'] -> {q}`
when the product is built. `|Q|` is small and this is cheap; the alternative,
scattering from sources, would either race or need per-thread accumulators
over the whole product state space.

Zero-mass conditions are routine, not exceptional: in a 4,000-sequence
amino-acid model at order 3 and length 14, 38 of 280 `(position, residue)`
pairs had zero marginal. Aggregate queries such as a mutual-information matrix
must treat those as contributing zero. Only a query that asks *directly* about
an impossible event raises.

## The semiring engine

One driver, one graph traversal, four algebras.

| semiring | carrier | gives |
|---|---|---|
| `logprob` | `double` (log) | `Z(L, C)`, edge posteriors, marginals |
| `expect` | `(log_p, e)` | `E[f]` for any additive functional `f` |
| `expect2` | `(log_p, e_f, e_g, m2)` | `Var[f]`, `Cov[f, g]` -- univariate variance is the case `g = f` |
| `maxplus` | `double` + backpointer | constrained MAP |

### Dispatch must be compile-time

The semiring ops cannot be function pointers in the inner loop. This repo has
already paid for that lesson once: `edge_weights.c` is compiled three times
over because GCC lowers OpenMP regions before constant propagation, and
indirect dispatch inside the kernel cost 31-51%.

`engine_kernels.h` follows the same house pattern -- no include guard,
included once per semiring with the ops as macros, exactly as
`edge_weights_kernels.h` is included once per alphabet.

### Entropy is a functional, not a semiring

`H(X | L, C) = log Z - E[log w]`, and `log w(s) = sum_i log w_i(s_i)` is
additive. So entropy needs no algebra of its own -- it is the `expect`
semiring with `f = LOG_WEIGHT`. Conditional entropy, and therefore mutual
information, follow from the same one pass.

### Additive functionals

`f(s) = sum_i g(i, x_i)`, represented as either an `L x R` table of `double`
or the distinguished kind `LOG_WEIGHT`. This is the whole exactly-computable
vocabulary, and it covers every statistic on the approved list:

| statistic | representation |
|---|---|
| `1{x_i = a}` | table, one non-zero cell |
| `count(a)` | table, column `a` all ones |
| surprisal `-ln P` | `LOG_WEIGHT`, negated, plus `log Z` |
| any weighted positional score | arbitrary table |

`E[length | C]` is not a functional -- length is fixed within a sweep. It is
computed by the aggregation layer below.

State-dependent functionals `g(i, s, x)` are a documented extension point.
`expected_counts` already aggregates per edge, so the precedent exists; it is
not in the initial stages.

### Numerical form: carry conditional means

The textbook expectation semiring carries `(p, p*f)`, which goes negative and
cannot live in log space.

This engine carries `(log p, e)` where `e` is the mean over paths *through
that node*:

- `(x) tensor` -- `(lp1 + lp2, e1 + e2)`. Valid at the forward-backward join
  by conditional independence given the state; valid on an edge because the
  edge contributes a deterministic increment.
- `(+) mixture` -- `lp = logaddexp(lp1, lp2)`, `sigma = exp(lp1 - lp)`,
  `e = sigma*e1 + (1-sigma)*e2`.
- second order merges by the centred parallel-axis form,
  `m2 = sigma*m2_1 + (1-sigma)*m2_2 + sigma*(1-sigma)*(e1-e2)^2`, which is
  stable where the uncentred form cancels catastrophically.

No signed-log machinery, no negative carriers, and the `-inf` (zero-weight)
case falls out of `sigma = 0`.

## Pairwise joints: the optimization is a constraint

All-pairs joints look like they need a hand-written kernel that propagates `R`
tagged forward messages. They do not.

An automaton with `R+1` states -- "before position `i`", then one state per
value observed at `i` -- records `x_i` in the product state. Running edge
posteriors on that product yields `P(x_j = b, x_i = a)` for **every** `j` and
every `a`, from one sweep at cost `x(R+1)`.

So all pairwise joints cost `L` product sweeps, sharing one backward pass,
with no code outside `automaton.c`. The measured alternative -- `L*R` separate
clamped sweeps -- was 0.54 s at order 3 and about 19 s at order 4 for a
length-14 model.

This is the design's own argument for itself: the optimization that would have
been a special-purpose kernel is instead an eight-line automaton builder.

## Aggregation across lengths

The engine's unit is one length. Queries whose condition is length-agnostic
sum over the live length range, weighted by the model's length prior:

    P(L | C) = P(L) * P(C | L) / sum_L' P(L') * P(C | L')

`E[length | C]` and `P(seq | C)` are then thin layers over per-length results.
`log_pseq` already carries the length prior, so cross-length comparison is
coherent without new machinery.

## Cross-length algebra: insertions and deletions

Given an observed `s` of length `L` and a radius `k`:

- `N+_k(s)` -- distinct strings of length `L+k` that reduce to `s` by deleting
  `k` positions.
- `N-_k(s)` -- distinct strings of length `L-k` obtained by deleting `k`.

Reported as the log-odds `log P(s') - log P(s)` against the unmodified
sequence, and as a posterior over the neighbourhood once normalized. The
length prior cancels in the posterior and does not in the odds, so both are
exposed.

**Degeneracy.** Inserting `S` at gap 2, 3 or 4 of `CASSF` yields the same
string. Candidates are deduplicated by string, and the `(gap, residue)` grid
attributes each distinct string to its **leftmost** gap, with aliased cells
zeroed so the grid sums to one.

**This is not an inferred edit.** The model has no alignment and no indel
process. The quantity is a ratio of what the model assigns to two strings that
happen to differ by one residue. PRISM answers the different, sequence-
conditioned question -- "given this observed sequence, where was it edited" --
and the documentation must point at it by name so the two are never confused.

**Cost.** `log_pseq` measures 0.66 us, so `k=1` is 320 scorings (0.2 ms),
`k=2` is 54,400 (0.04 s), `k=3` is 5.4M (3.6 s). `k=1` gets the incremental
optimization -- for a fixed gap only `O(k)` context terms vary with the
inserted residue, so prefix and suffix scores are reused. `k>=2` enumerates
directly under the cost guard; optimizing it before anyone has asked would be
premature.

## Cost model and refusal

Every query is planned before it runs. `plan.c` estimates live product states
per layer, edge relaxations, and peak bytes -- `(L+1) * live * sizeof(carrier)`
for the backward field.

A query over budget is **refused with a number**, in the house voice, naming
the knob. This is not defensive decoration: a 206 GB allocation has taken this
machine down twice, and a query API where `mutual_information_matrix()` at
order 6 has no speed bump would be a defect.

`beam_query_estimate` exposes the same numbers without running anything, so
the Python layer can warn before a 19-second call rather than after it.

### Error codes are reused, not invented

The existing vocabulary already covers this subsystem, and adding enum values
to a public error type churns an ABI the verification gate records.

| condition | code | precedent |
|---|---|---|
| unsatisfiable constraint | `BEAM_ERR_NO_LIVE_PATHS` | `positional_marginals` raises it for exactly this |
| bad position, residue or length | `BEAM_ERR_PARAM_OUT_OF_RANGE` | every existing validator |
| over the memory budget | `BEAM_ERR_ALLOC` + message | the construction guard in `lifecycle.c` |
| a query with no exact answer | `BEAM_ERR_INVALID_ARG` + message | argument validation generally |

A new code is added only when a stage proves it needs one. Stage 1 needs
none. The Python layer's exception hierarchy is built over these codes plus
the accompanying message, which `set_beam_error` already preserves for
`BEAM_ERR_ALLOC`.

## File layout

```
include/beam/query.h              the whole public surface

src/native/query/
    constraint.c      masks, composition, validation
    automaton.c       DFA builders: literal, set, motif, count, and/or/not
    product.c         trellis x constraint: liveness, co-reachability, sizing
    engine.c          the sum-product driver
    engine_kernels.h  kernel body, included once per semiring (no guard)
    semiring.h        carrier types and op macros
    posterior.c       edge posteriors and marginals under a constraint
    expectation.c     additive functionals, E / Var / Cov
    information.c     entropy, conditional entropy, mutual information
    indel.c           cross-length neighbourhood algebra
    plan.c            cost estimation and refusal

src/native/bindings/python/query_bindings.c
src/BEAM/query/                   ergonomics only; no mathematics
tests/native/test_query_*.c
tests/python/query/
```

`src/BEAM/query/` holds argument validation, residue-character to index
translation, numpy packaging and error translation. If a probability is
computed there, the layering has failed.

## Staging

Each stage ships its C and its native tests, and must leave the numeric half
of the verification gate green before the next begins.

**Python bindings are deferred to one stage at the end**, rather than shipped
per stage. Two reasons, and the second is the stronger one. The goal is a C
foundation that Python later exposes, so a binding written beside each stage
would be designing the Python surface six times from six partial views of it.
And `RERECORDED.md` is explicit that baselines are re-recorded once at the end
of a branch, not per commit -- "re-recording them per commit would have fused
two unrelated changes into one diff that nobody could audit" -- and every
binding touches three of the eight baselines. Deferring them keeps each C
stage a diff whose numeric consequences are readable.

| stage | contents | anchor |
|---|---|---|
| 1 | `constraint` (masks only), `product`, `logprob` engine, `posterior`, `plan` | reproduces `positional_marginals` and `log_clamp_probability` bit-for-bit |
| 2a | `automaton.c`: motif, count, mask lifting, and/or/not | the language enumerated exhaustively against naive matchers |
| 2b | `product.c` and the product engine | motif and count probabilities checked against enumeration |
| 3 | `expect` semiring, functionals, `expectation.c` | `E` and `Var` checked against enumeration; entropy against `-sum p log p` |
| 4 | `information.c`: MI, conditional entropy, matrices | joints via the tagging automaton, checked against clamped sweeps |
| 5 | `indel.c` | neighbourhood scores checked against direct `log_pseq` |
| 6 | `maxplus`, constrained MAP | checked against `top_k_sequences` under a clamp |
| 7 | Python bindings and `src/BEAM/query/` | the Python surface checked against the C one, baselines re-recorded once |

### Stage 2 splits, and 2a ships an explicit refusal

The automaton is pure data -- it needs no model and no sweep -- so its language
can be gated on its own, by enumerating a small alphabet exhaustively and
comparing against a naive matcher. That is a decisive test and it does not
have to wait for the product engine, so `automaton.c` lands first.

Between 2a and 2b the mask engine can be handed a constraint it cannot honour.
It refuses, by name, rather than reading the masks and ignoring the automaton:
an engine that answered `P(contains CASSQ)` as though the motif were not there
would return a confidently wrong number, which is worse than no number. The
refusal is tested at all three entry points, and 2b deletes it.

## Testing

The standing rule on this project applies: tests that could fail. Specifically

- **Enumeration.** For small models -- short lengths, low order, a reduced
  alphabet -- enumerate the support and compare every quantity against the
  brute-force sum. This is the primary weapon and it is decisive.
- **Bit-for-bit reproduction.** Stage 1's anchor is free: the mask-only,
  `|Q|=1`, log-prob path *is* the existing code, and must agree to the last
  bit, not to a tolerance.
- **Algebraic identities that a wrong engine fails.** Marginals sum to one;
  `MI(i,j) = H(i) + H(j) - H(i,j)`; `MI(i,i) = H(i)`; MI is symmetric and
  non-negative; `Var >= 0`; conditioning on a certain event is the identity;
  `sum_a P(x_i=a) P(x_j=b | x_i=a) = P(x_j=b)`.
- **Automaton correctness against a naive matcher.** Whether a string is in
  the language, checked by a direct scan for every string in a small support.
- **Impossible and degenerate constraints.** Empty language, contradictory
  masks, a motif longer than the length, `k` larger than the sequence.
- **Mutation testing.** Every regression test must be shown to fail against a
  deliberately broken engine before it is trusted -- and the rebuild must be
  verified to have happened, since a `make test && build_ext` chain has
  silently short-circuited on this project before and nearly recorded a false
  negative.
- **Thread independence.** Results identical across `OMP_NUM_THREADS`.
- **Bounded execution.** Everything that builds a model runs under
  `verification/bounded`.

## Risks

| risk | mitigation |
|---|---|
| Product state explosion at high order | `plan.c` refuses with a number before allocating |
| A new engine drifts from the existing one | stage 1's bit-for-bit anchor; gate re-recorded per stage |
| Semiring dispatch costs performance | compile-time instantiation, measured single-threaded, interleaved, min-of-N |
| Indel results read as inferred edits | named in the docstring, PRISM pointed at explicitly |
| Scope sprawl across six stages | each stage is independently green and revertible |
