# Probability mixtures

`Mixture` is the distribution

```
pbar(s) = sum_i w_i P_i(s),     w_i >= 0,   sum_i w_i = 1
```

— pick a donor by `w`, then pick a sequence from that donor. It is what beta diversity between individuals is defined on.

```python
from BEAM import BeamModel, Mixture

mix = Mixture([donor_a, donor_b, donor_c])       # uniform weights
mix.hill_number(2.0)                             # exact
mix.beta_diversity(2.0).beta                     # exact, 1 <= beta <= M
mix.entropy_bounds()                             # exact bracket
mix.entropy_estimate(10000, seed=1)              # sampled, with a standard error
mix.support_size()                               # exact union of the supports
```

## `merged()` is not this

`BeamModel.merged` and `add` sum edge **counts** and refinalize. That gives the model you would get by **training on the pooled corpus**, which is a different distribution with a strictly larger support: a pooled path may splice an edge learned from one donor onto an edge learned from another and so spell a sequence no single donor spells.

The gap is not a rounding detail. Pooling 64 donors at a fixed 30,000-sequence budget raises `D0` by **24.3%** while the number of transitions rises 0.57% — almost all of the excess is recombination, not new evidence. On the toy fixtures in `tests/python/mixture/` the same effect gives a union of 211 against a pooled support of 233.

`D0(union) <= D0(merged)` always. Nothing in this package is implemented in terms of the count algebra.

## What is exact

The three Hill orders are not equally tractable, and the API does not pretend they are.

| order | status | how |
|---|---|---|
| `q = 0` | **exact** | a forward sweep carrying the set of members still consistent with the path |
| `q = 1` | **no closed form** | exact bounds, plus an unbiased estimate with a standard error |
| integer `q >= 2` | **exact** | `M(M+1)/2` product sweeps over the shared trellis |
| everything else | refused | `1.5`, `-1`, `0.5` — none of them factor |

`sum_s pbar(s)^q` for integer `q` expands into a sum over q-tuples of members,

```
sum_s pbar^q = sum over (f1..fq) of  w_f1 ... w_fq * <P_f1, .., P_fq>
```

and each inner product is one forward sweep whose edge weight is the **sum** of the members' edge weights. That is the whole engine.

`q = 1` has no such expansion, because `log(sum_i w_i P_i(s))` does not factor over edges. So:

```python
mix.hill_number(1.0)                # raises UnsupportedOrder
mix.hill_numbers([0.0, 1.0, 2.0])   # raises too — no partial answers
```

This is deliberate. A bare float gets read once, pasted into a table, and separated from the caveat that lives only in prose. Refusing costs the caller one line and removes the possibility. `entropy_bounds()` and `entropy_estimate()` are the two honest paths, and both return objects that carry their own exactness.

There is no flag that softens it.

## `q = 1`: bounds and an estimate

```
sum_i w_i H_i  <=  H(pbar)  <=  sum_i w_i H_i + H(w)
```

Both sides come from the members' own exact entropies and the weights, so they cost nothing, and **both are attained** — identical members sit on the lower bound, members that share nothing sit on the upper one. Restated as diversity that is `1 <= D_beta,1 <= exp(H(w))`, which for uniform weights is `1 <= D_beta,1 <= M`.

`log pbar(s)` *is* exactly computable for any single `s`, so sampling from the mixture and averaging `-log pbar(s)` is unbiased and the only error is Monte Carlo:

```python
e = mix.entropy_estimate(20000, seed=1)
e.value, e.stderr, e.exact          # 5.119, 0.0047, False
e.bounds.brackets(e.value)          # True
```

## Beta diversity, and the trap in it

```python
r = mix.beta_diversity(2.0)
r.gamma, r.alpha, r.beta, r.exact
```

`D_alpha` is the **power mean of order `1 - q`** of the members' own Hill numbers — arithmetic at `q = 0`, geometric at `q = 1`, harmonic at `q = 2`. The naive arithmetic mean at every order is wrong everywhere except `q = 0`, and it fails quietly: it inflates alpha and so under-reports beta. On a measured two-donor case with no shared sequence, where beta must be exactly 2, the arithmetic version returns **1.325**.

`alpha_diversity(q)` is shipped so that nobody has to assemble this themselves.

`1 <= D_beta <= M` is a property of **the mixture**. It says nothing about a pooled-trained model, whose diversity can exceed the sum of the members' because it spells sequences none of them do.

At `q = 1` the gamma has no closed form, so `beta` comes back `None` with the exact bracket filled in; pass `samples=` for an estimate alongside.

```python
r = mix.beta_diversity(1.0, samples=20000, seed=1)
r.beta, r.exact          # None, False
r.lower, r.upper         # 1.0, 2.0
r.estimate, r.stderr     # 1.566, 0.0074
```

## Per-length structure

Members carry different length priors, so the mixture's prior is `pibar(L) = sum_i w_i pi_i(L)` and the effective weights **within** a length are `w_i pi_i(L) / pibar(L)`, not `w_i`.

```python
mix.length_distribution      # {L: pibar(L)}
mix.member_weights(14)       # the renormalised within-length weights
```

`log_pseq` needs no special handling here — each `P_i(s)` already carries its own `pi_i(L)`, so the re-weighting falls out of the sum. Anything conditioned on a length does need it, which is why `member_weights` is exposed rather than kept private: it is the single easiest thing to get wrong, because using `w_i` unchanged still produces a normalised distribution and plausible numbers.

## Weights are refused, not normalised

```python
Mixture([a, b], [3.0, 7.0])     # BeamValueError: weights must sum to 1
```

Raw counts pasted in as weights, and a member dropped from one list but not the other, both land here. Normalising would give each of them a plausible answer. Divide by the sum yourself and the intent is on the page.

Negative, non-finite, and wrong-length weight vectors are refused too. A weight of exactly zero is allowed: excluding a member is meaningful.

## Cost, and how to spend it

Every `q = 2` quantity is `M(M+1)/2` inner products. The honest unit is
**core-seconds per pair**, because the wall clock depends entirely on which
axis you parallelise:

| models | core-seconds per pair |
|---|---|
| donor-sized (30k sequences, order 4, `max_length` 27) | **0.11–0.15** |
| the 927M-sequence foundation model (same config) | ~3.3 |

**Ask for the whole matrix, not its entries.** A single inner product spreads
over the model's ~25 lengths and stops paying almost immediately — measured,
110 ms of core time per pair at one thread, 107 ms at four, and **399 ms at
sixteen**: 3.6× the work for 4.4× the wall clock. `log_gram_matrix()` instead
hands one pair to each thread:

```
16 donors, order 4, 136 pairs
  threads=1    19.87s   146.1 ms/pair   speedup 1.00x   efficiency 100%
  threads=4     4.64s    34.1 ms/pair   speedup 4.28x   efficiency 107%
  threads=16    2.65s    19.5 ms/pair   speedup 7.50x   efficiency  47%
```

Better than the length axis at every thread count, but **not linear**: the
sweep is memory-bandwidth bound, so one host saturates around 7–8× whichever
axis you use. Across separate machines it does scale linearly — each has its
own memory system — so a cohort-scale Gram matrix is a cluster job sized in
**node-days**, not a core count on one box.

Worked, for a 38,538-repertoire cohort: 742,607,991 pairs at 0.146 core-seconds
is **3.4 core-years**, which is ~170 days on one 16-core node and ~5 days on 32
of them. The single-core figure is real, but it is not the wall clock.

There is no shortcut. `sum_i w_i P_i(s)` does not factor over edges, and
carrying an `M × M` accumulator per trellis state would fuse the pairs into one
sweep with exactly the same operation count and a state vector nobody can
afford.

Two things are **not** `O(M²)` and are often billed as if they were:

- `alpha_diversity(q)` is the power mean of each member's own Hill number —
  `O(M)`, no pairs at all.
- `entropy_bounds()` is `O(M)` — per-member entropies and `H(w)`.

## What limits `M`

**Only `q = 0`.** The union sweep carries one bit per member in a `uint64_t`,
so 64 members is structural there. Nothing else has that bound: the inner
product never distinguishes the members it multiplies, so `q >= 2`,
`alpha_diversity` and `entropy_bounds` work at any size.

That was not always true. The ceiling used to sit on `Mixture` itself, so a
256-member cohort could not be constructed at all and every quantity went with
it — including the three that have nothing to do with the bitmask.

```python
Mixture(models_256).hill_number(2.0)            # fine
Mixture(models_256).alpha_diversity(2.0)        # fine
Mixture(models_256).support_size()              # BeamValueError, naming the sweep
Mixture(models_256).support_size(exact=False)   # bounds, free, any M
```

Within 64 members, what binds the union is the number of distinct
`(state, member-set)` pairs alive per layer — the quantity that could in
principle reach `2^M`. Measured on donors of 6,000 sequences at order 2,
`max_length` 12, varying how much the members disagree:

| members drawn from | peak at `M = 8` | at `M = 32` | at `M = 64` | growth per doubling |
|---|---|---|---|---|
| 18 of 20 residues (near-identical) | 8,860 | 34,957 | 69,628 | ×1.99 — linear |
| 12 of 20 (moderate) | 7,024 | 28,549 | 58,733 | ×2.00 → ×2.06 |
| 7 of 20 (strongly disjoint) | 3,105 | 25,716 | 104,037 | ×2.20 → ×4.05 |

So the `2^M` blow-up is real but slow, and `M = 64` peaks near 100,000 pairs —
about 5 MB against a default budget of 4.29 GB. `support_size()` reports the
peak it reached in `peak_states`.

**What runs out before any of this** is holding the models: an order-4,
`max_length` 27 donor model is ~218 MB, so 32 of them is 7 GB.

## Refusals

| exception | means | also a |
|---|---|---|
| `MixtureError` | base of every refusal here | `BeamError` |
| `IncompatibleMembers` | the members cannot share a trellis | `ModelMismatch`, `RuntimeError` |
| `UnsupportedOrder` | that Hill order has no exact answer | `BeamValueError`, `ValueError` |
| `MixtureTooLarge` | exactly answerable, too large for the budget | `BudgetExceeded`, `MemoryError` |

Members must agree on alphabet, `order`, `max_length` and `depth_cap` — the same four fields `add` checks, for the same reason: a mixture reads all its members at **one state index**, which is sound only because a state index is a function of the configuration and never of the data. Disagreement is caught at construction, not at the first analytic.

A model may appear more than once. Unlike `add` and `subtract`, nothing here mutates a member, so aliasing is harmless and a mixture of a model with itself is a useful degenerate case.

Compacted models are accepted: the sweeps read the finalized tables, not the raw counts.

## For contributors

The C side is `include/beam/mixture.h` and `src/native/mixture/`:

- `inner_product.c` — the product sweep. A near-clone of `tilted_log_z` in `math/diversity.c` with the tilt replaced by a sum of members' weights. Two traps are documented there: the `alive` mask must be the **intersection** of the reached sets, because `beam_layer_weights_live` leaves dead rows *untouched* rather than filling them with `-inf`; and the weights are summed into one accumulator as each is built, so the footprint is independent of `k`.
- `support.c` — the bitmask union sweep, and why it is not inclusion–exclusion over the `2^M` intersections (those terms alternate in sign and each is astronomically large, so double precision loses the answer to cancellation).
- `shape.c` — the compatibility gate.

The identity `<P, P> = 1 / D2` puts this kernel and the pre-existing Mellin path on the same number; it is asserted in `tests/native/test_mixture.c` and, at full scale on the foundation model, in `tests/python/mixture/test_kernel.py`.
