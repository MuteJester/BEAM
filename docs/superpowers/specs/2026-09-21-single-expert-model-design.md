# Single-expert BEAM: the probability model

Status: PROPOSED. Definition agreed. Implementation and test plans pending.

Branch: `feature/single-expert-model`.

---

## 1. What this changes

Today each direction's per-position term is a weighted geometric mean of three
expert tables of different context order, one of which is additionally
conditioned on sequence length, and the depth index is clamped at
`depth_cap = 15`.

This replaces that with a single table per direction at exact depth.

The bidirectional structure, the closed world, the length prior, the trellis and
the partition function are unchanged.

**The model is to have exactly three free parameters: `k`, `w_f`, `w_b`.**
Everything else is either derived or fixed by the data.

## 2. The model

Let `s = s_0 ... s_{L-1}` over an alphabet `A` of `R = 20` residues, with a
sentinel symbol for padding giving radix 21. Let `k` be the context order,
default 4.

### The per-position term

```
   psi_i(s)  =  w_f * log q_f( s_i | i,      s_{i-k} ... s_{i-1} )
             +  w_b * log q_b( s_i | L-1-i,  s_{i+1} ... s_{i+k} )

   w_f = w_b = 0.5,      w_f + w_b = 1
```

Contexts are sentinel-padded where they run off either end. Depths are EXACT:
`i` from the N-terminus and `L-1-i` from the C-terminus. There is no cap.

### The two tables

```
   q_f( x | d, c )        q_b( x | d, c )
```

One row per `(depth, k-mer context)` per direction, each row a distribution over
the `R` residues:

```
   sum_x q( x | d, c )  =  1        for every observed row
```

An unobserved row, or an unobserved residue within an observed row, is zero.
This is the closed world and it is the only source of `-inf`.

### The sequence probability

```
   log P(s)  =  sum_{i=0}^{L-1} psi_i(s)  +  log pi(L)  -  log Z(L)

   Z(L)      =  sum over t in A^L of  exp( sum_i psi_i(t) )
```

with `pi(L)` the abundance-weighted observed length distribution.

## 3. Properties that now hold

**Length is determined exactly at every position.** With exact depths,
`d_f(i) + d_b(i) = i + (L-1-i) = L-1` for every `i` and every `L`, with no
exceptions and no degradation at long lengths. The model is fully length-aware
without any length-conditioned table, which is what the order-2 expert was
nominally for.

**And it still pools across lengths.** Each row is estimated from every training
sequence that carried that context at that distance from that terminus,
whatever its length. Length awareness lives in the COMBINATION of the two
depths; length pooling lives in the ESTIMATES. Under the previous design those
two were in tension, because the only explicitly length-aware table was also the
one whose counts were split by length.

**Each direction alone is a proper distribution.** Every row is a normalised
conditional and the forward contexts are a function of the prefix, so
`prod_i q_f(...)` telescopes to 1 over `A^L` by the chain rule, and likewise
backward over the suffix. The model is exactly a weighted geometric mean of two
proper autoregressive chains, one read from each terminus.

**`Z(L)` acquires a closed form.** With `w_f = w_b = 0.5`,

```
   Z(L)  =  sum over t in A^L of  sqrt( P_f(t | L) * P_b(t | L) )
```

the **Bhattacharyya coefficient** between the two chains. By Cauchy-Schwarz
`Z(L) <= 1`, with equality if and only if the two chains agree. So
`-log Z(L) >= 0` is the Bhattacharyya distance and measures exactly how much the
N-anchored and C-anchored readings disagree at that length. Under the mixture it
was the same kind of object over six chains, and the extra four contributed a
sub-normalisation of about 0.06 nats per position that had no interpretation.

**The support is the intersection of two supports, not six.** Position `i` is
live if and only if both lookups succeed.

**The weights must still sum to 1.** The constraint was never about the three
experts. The total weight per position must be 1 or `log W` scales with `L`,
which leaves within-length ranking untouched and breaks cross-length comparison.
It now falls entirely on `w_f + w_b`.

## 4. What is removed

- the order-3 mid table and the order-2 length-conditioned len table, both
  directions
- the mixing coefficients `coef_wide`, `coef_mid`, `coef_len` and their
  sum-to-one validation
- the three-way presence test and log-linear combination in `mix_values` and
  `mix_sparse`, and the duplicated presence tests in the DP edge kernels
- `depth_cap` and the `beam_depth_part` clamp
- the backoff strengths addressing the mid ladder and the len table

## 5. What is kept

Context order `k` (default 4), `max_length`, the closed world (strict, sparse,
zero backoff), the length prior, the partition function, the trellis whose state
is the last `k` residues, and every dynamic program built on it. The state stays
`21^k` per layer. Layer `i` already knows `i`, so reading row `(i, c)` instead
of `(min(i, depth_cap), c)` costs nothing structurally.

## 6. Table geometry

Rows per direction: `max_length * A^k`. At the defaults, `27 * 21^4 = 5,250,987`
rows per direction, `210,039,480` cells across both directions against
`147,805,560` for the current three-expert stack.

Values are stored only for observed rows. The row directory is one `uint32` per
possible row while mutable, and seals to the rank-select form at 1.25 bits per
row once finalized, about 1.6 MB for the whole directory.

## 7. Variants deliberately not taken

**Conditioning the table on length**, indexed by
`(direction, length, depth, context)`. Rejected on measurement: occupied rows
fall from 1.99 to 1.51 observed continuations out of 20, dead positions rise
from 33.4% to 42.9%, positions no model in a 20-donor panel can score rise from
13.9% to 24.2%, row misses more than double while cell misses stay flat, and it
would need about 605,000 distinct sequences to reach the miss rate the current
table reaches at 98,000 against a largest available repertoire of 269,000, with
the learning-curve exponent smaller so the gap widens with more data.

The underlying reason is that length conditioning removes the pooling that
produces generalisation. 73% of the positions a model can spell were never
present as a 9-mer in its own training strings, and that comes from sharing
observations across sequences of different lengths.

It is also unnecessary here: section 3 shows exact depths already determine
length at every position.

**Keeping the three-expert mixture.** Its recorded justification is an average
precision measurement for publicness prediction, 0.8467 at AP>=25 against 0.8255
for the best single expert. Measured separately, the order-3 expert contributes
+0.006 nats per position against the order-2 expert's +0.058, while holding 0.25
of the coefficient and a third of the lookups in the hottest kernel in the
library. The mixture also costs coverage, because the support is an intersection
of six supports rather than two: dead positions 33.4% with the mixture against
31.7% for the wide table alone.

## 8. Predicted consequences, to be confirmed

| quantity | current | predicted | basis |
|---|---:|---:|---|
| lookups per position | 6 | 2 | by construction |
| sub-normalisation per position | 0.0606 nats | 0 | Hoelder, and it vanishes |
| per-position calibration | reference | about -0.06 nats | measured shrinkage loss |
| dead positions from dropping the mixture | 33.4% | 31.7% | measured, wide-only gate |
| dead positions from uncapping depth | 31.7% | worse, by an unmeasured amount | see below |

Uncapping depth has NOT been measured. The recorded cap sweep covers 4, 8 and
15, and reach falls monotonically as the cap rises: 49.0% at cap 4, 40.5% at
cap 8, 23.3% at cap 15. Uncapping continues in that direction. The effect should
be small for this cohort, because depths at or beyond 15 occur only in sequences
longer than 15 and only at their last few positions, and the modal CDR3 is 15
residues. But it is a prediction, not a measurement, and it must be measured.

That same sweep also recorded that at `depth_cap = 15` the model assigns more
mass to a stranger's held-out sequences than to its own donor's, a ratio of
0.960, which cap 4 flips to 1.254. That measurement is one donor, one foreign
control and a single split, and it was taken on coverage and held-out mass
rather than average precision. It is the strongest available evidence that the
depth axis is mis-specified, and it points toward LESS depth resolution, not
more. Removing the cap moves the other way. **This is the one place where the
agreed definition and the available evidence disagree, and the disagreement
should be resolved by measurement on this branch rather than by argument.**

## 9. This is a different model, not a refactor

No quantity will be bit-identical and none should be expected to be. The
validation question is whether the conclusions survive and whether any headline
metric gets worse, not whether numbers match.

`.private/edge/METHOD.md` section 7f records that average precision and held-out
likelihood are ANTAGONISTIC in this model past order 3, that order-4-alone has
worse held-out NLL (29.102) than plain order 2 (28.678), and that suppressing
the mid-sequence order-4 factor improves held-out NLL while destroying average
precision. It ends: "Any future proposal must declare which of the two it is
optimising."

**This proposal optimises interpretability and likelihood tractability, not
average precision.** That is the declaration.

## 10. Acceptance criterion

1. **Figures.** The five pipelines and their measurable survival criteria are
   in the test plan, section 6. They run on ig05, and the pre-change numbers
   must be recorded there BEFORE deploying, because afterwards the old build is
   gone.

2. **Publicness average precision.** The benchmark this proposal is nominally
   judged against, 0.8467 against 0.8255, **cannot be re-run**: the evaluation
   pool, the labels and the scoring harness are all lost. A working substitute
   is committed in this repository at
   `.private/manuscript/figures/supplementary/figure_S7_publicness_pr/`, which
   draws the same pool as METHOD.md section 4b and records `beam-foundation` at
   0.7716 / 0.8089 / 0.8383 / **0.8587**, reproducing the recorded figures to
   about 0.002.

   **Proposed criterion: AP>=25 of at least 0.836 on that pool**, which is the
   committed 0.8587 less the +0.0212 the three-expert stack was recorded as
   buying over the best single expert.

   **This is a floor, not a target.** Section 9 declares this proposal
   optimises likelihood, and METHOD.md section 7f records that the two are
   antagonistic. So a held-out NLL improvement is the success condition, and
   the AP number states how much publicness performance the project is willing
   to spend.

3. **Variant or replacement.** Unresolved. See the implementation plan, Q6.

4. **Three numbers in section 8 have no source in this repository**: the
   depth-cap sweep, the own-versus-foreign ratio and the wide-only dead rate.
   The first two come from a session memory dated 2026-09-02 recording one
   donor, one foreign control and a single split. Both are cheaply regenerable
   on the current build, since `depth_cap = max_length - 1` and
   `coefficients = (1, 0, 0)` are legal configurations today. **Regenerate them
   before relying on section 8 either way.**
