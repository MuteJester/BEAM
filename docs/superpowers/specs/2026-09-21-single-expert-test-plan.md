# Single-expert BEAM: test and validation plan

Companion to `2026-09-21-single-expert-model-design.md`.

---

## 0. The sequencing that makes this safe

Seven measurements were taken against the CURRENT build before writing this.
Five of the six new invariants below already pass on the three-expert model.
**Land those five first, before any code changes.** They then act as the
change's own control group: they stay green through the rewrite, and any
failure is unambiguously the rewrite's fault.

The two that currently FAIL are the sharp, falsifiable statements of the new
model. Land them as expected failures with the measured current values
recorded, and flipping them to real assertions is the definition of done.

| | currently | must become |
|---|---|---|
| **Invariant 1**, row sums | forward 0.5548 to 1.0, backward 0.7360 to 1.0 | exactly 0 or exactly 1 |
| **Invariant 5**, a position's term vs total length | moves by 1.9e-2 at p=0 | exactly 0 |

## 1. A structural gift

**The nucleotide alphabet already runs at exact depth.**
`src/native/core/alphabet_nt.c:87` sets `default_depth_cap = 255`, clamped to
`max_length - 1`, with the comment "Resolve every depth. The cap is a
concession to the amino-acid wide table's size at radix 21, and radix 6 does
not need it."

The uncapped-depth code path is shipped, exercised by 39 cases in
`tests/native/test_alphabet.c`, and correct. **Uncapping is de-risked. Only its
coverage cost on amino acids is unmeasured.**

## 2. Two measurements available on the current build, today, with no new code

Both are legal configurations of the shipping library:

1. `coefficients=(1, 0, 0)` isolates the VALUE contribution of the mid and len
   experts. It does not remove the veto, so it isolates values, not support.
2. `depth_cap = max_length - 1` is the uncapping ablation.

**This regenerates the depth-cap sweep the design leans on, without writing a
line of new C.** It is the cheapest available de-risking and should happen
before implementation, not after.

## 3. The six invariants

### 1. Row normalisation
Every row of `q_f` and `q_b` sums to exactly 1 or exactly 0. Nothing between.
Test through `explain` in Python (substituting a residue at position `p`
changes neither context, so it sweeps exactly one row), and exhaustively in C
over every row of a small model. Tolerance `abs=1e-12`, about 20 ulp.

**Currently false, and no existing test can see it**, because `Z(L)` divides
the deficit out exactly.

### 2. Each direction alone is a proper chain, and the leak is exact
```
   M_f(L) + Leak_f(L) = 1      exactly
```
where the leak is the mass on prefixes that walk into an unobserved row.
Measurable with no new C: set `weights=(1.0, 0.0)` and
`exp(log_partition(L))` is the forward chain's mass over the live set.
Verified against enumeration to 1.2e-15.

Report `1 - exp(log_partition(L))` per length as the branch's coverage
measurement. This is the honest replacement for the missing depth-cap sweep.

### 3. `Z(L)` is the Bhattacharyya coefficient
Define, from three models over the same corpus at weights `(1,0)`, `(0,1)` and
`(0.5,0.5)`:
```
   D(L) = log Z_f + log Z_b - 2 log Z_half  >=  0        Cauchy-Schwarz
   log Z_half  <=  0
```
`D(L)` is the disagreement and `1 - Z_f` is the leak. `Z <= 1` alone cannot
separate the two causes; these two numbers do.

**Test A, the model where the chains provably agree.** Corpus is all `2^6 = 64`
strings over `{A, C}`. Every observed row is exactly uniform over the two
letters, so `Z(6) = 1` exactly. Measured on the current build:
`log Z = 4.4408920985006262e-16`, every sequence at exactly `-6 ln 2`.

Note it is marginally POSITIVE, so the assertion must be `abs=1e-12` around
zero, never `<= 0.0`.

**Test C, the cheap continuous monitor.** `2 log Z_half <= log Z_f + log Z_b`
and `log Z_half <= 1e-12` at every observed length, on the production model,
no enumeration. It also exercises zero anchor weights in both directions,
dense and sparse, so it doubles as the `0 * -inf` regression test in Python,
where today that exists only in C.

### 4. `log P` sums to a proper distribution
`pgen_diagnostics` (`src/BEAM/model/facade.py:472`) remains valid **unchanged**
and should be kept. But it must not be presented as validating the new model:
`Z(L)` is DEFINED as the sum of `exp(raw)` over `A^L`, so `sum_s P(s) = 1`
holds for any per-position weight function whatsoever. It reports
`is_proper: True` today while every row is sub-normalised.

One addition worth making: decompose the joint check into
`sum_{s in A^L} P(s) == exp(log_length_prior(L))` per length, so an error in
`Z(L)` at one length cannot be cancelled by an error in `pi` at another.

### 5. The depth pair determines length
`d_f(i) + d_b(i) = L - 1` at every position. The observable consequence, and
the test with teeth: **a position's term must not move with the total sequence
length.** Measured spread today, which must go to zero:

| | spread |
|---|---:|
| forward, p=0 | 1.936e-02 |
| forward, p=2 | 7.481e-04 |
| backward, r=0 | 6.087e-02 |
| backward, r=4 | 7.481e-04 |

Tolerance `abs=1e-15`: the two terms are literally the same table cell.

### 6. The DP and the direct scorer agree
This exists today as `tests/native/test_model.c:1192`, whose own docstring says
"If the two ever diverge the analytics describe a different model from the one
being scored, and nothing else in this file would notice." Its coverage is one
order, one length, one sequence, dense only, both sides finite.

Widen on five axes: orders 2/3/4 (order 4 is the only one where the dense
backward hoist fires); dense AND sparse (the `0 * -inf` NaN shipped because the
regression test built dense while the Python layer selects sparse); lengths
`L < k`, `L == k`, `L == k+1`, `L == max_length`; a DEAD sequence, asserting
both sides are `-inf`; and a Python mirror at production scale through
`log_clamp_probability(L, full) + log_length_prior(L) == log_pseq(s)`, verified
to agree at 8.9e-16.

## 4. The headline metric is not reproducible, and here is what replaces it

The average precision benchmark the design is nominally judged against
(0.8467 for the stack against 0.8255 for the best single expert) **cannot be
re-run.** Four independent blockers, all verified:

1. The evaluation pool `ordered_sequences_S27` and `design_probe/mask.npy` are
   bare filenames with no recorded directory, and are not on this filesystem.
2. The AP harness imports `load_eval` and `aprow` from an ephemeral scratchpad
   path that no longer exists. Both functions are lost.
3. The reference score vector is in the same dead path.
4. The counting path hardcodes two directories that do not exist.

**A working substitute is committed in this repo.**
`.private/manuscript/figures/supplementary/figure_S7_publicness_pr/` draws the
same pool as METHOD.md section 4b (3,000,000 uniform over the 73,245,676
distinct sequences of 768 held-out Emerson donors, seed 20260823) and its
result is committed: `beam-foundation` at **0.7716 / 0.8089 / 0.8383 /
0.8587**, reproducing the recorded EDGE figures to about 0.002. Competitor
baselines for OLGA, SoNIA and LZGraph are committed alongside on identical
sequences and labels.

**Recommended acceptance criterion for design section 10 item 2:** the new
model must reach **AP>=25 of at least 0.836** on that pool, which is the
committed 0.8587 less the +0.0212 the three-expert stack was recorded as
buying. It is defensible because it is exactly the quantity the mixture's
recorded justification claims, expressed on a pool that can actually be re-run.

**And it should be framed as a floor, not a target.** The design declares it
optimises likelihood tractability, not average precision, and METHOD.md section
7f records that the two are antagonistic past order 3. So a held-out NLL
improvement is the success condition and the AP number states how much
publicness performance the project is willing to spend.

## 5. Differential validation: support-matched or nothing

Recorded in `.private/position_likelihood_matrix/THREE_EXPERTS.md`: 7.20% of
positions alive under the shipped experts are dead under a narrower gate, and
the shipped mixture scores those at -1.839 nats while the narrower one scores
them minus infinity. Removing them from a mean improves it by roughly 0.05 to
0.10 nats per position, **larger than the entire measured contribution of the
mid expert (+0.006)**. A comparison on each model's own live set is not merely
biased, it is biased by more than the effect being measured.

So: fit both on identical data, score identical held-out sequences, compare on
the **intersection** of live positions, and report four abstention numbers
beside every score: `dead_old`, `dead_new`, `old_alive_new_dead` (the trap),
`new_alive_old_dead`.

Decompose the delta into its three causes, since a single number cannot
attribute a regression: dropping mid and len, uncapping depth, and removing the
veto. The first two are measurable on the current build (section 2).

## 6. The figure regression suite

Five pipelines, with measurable survival criteria rather than eyeballs:

| figure | survives if |
|---|---|
| R(N) | `R(n) -> D0` still asserted; `R(n)` at both anchors within 0.5 decades; curve monotone |
| R(N) held-out | the median over-prediction ratio (currently 1.49x over 707 repertoires) does not move further from 1, and **no repertoire flips to under-prediction** |
| publicness distribution | total predicted/observed in [0.9, 1.15] (currently 1.057) and median absolute log10 ratio at most 0.10 (currently 0.060) |
| public/private ratio | the templates-convention over-prediction keeps its sign and rough magnitude (currently 1.61x); the distinct-convention ratio stays in [0.9, 1.1]. Its three no-tolerance identities must hold EXACTLY, those are correctness not survival |
| Hill spectrum | the scripts' own cross-checks hold at their own tolerances (D1 against the Mellin derivative at 3.5e-15, the 16,384-bin spectrum reproducing D1 and D2 at 1.1e-04). `D0` may move by orders of magnitude and that is expected; **`D1` must stay within 0.3 decades** |

Several of these scripts already validate themselves. Those self-checks are
correctness gates and must pass unchanged; the thresholds above are survival
criteria layered on top.

**The suite cannot run on this machine.** `BEAM_MODELS` is unset and
`beam_full_strict.beam` is on ig05. **Record the pre-change numbers there
BEFORE deploying**, because after deployment the old build is gone.

## 7. Re-recording the baselines

Nine of the ten `verification/` checks will move, because the model is
different. The procedure follows the precedent in `RERECORDED.md`:
`make clean` first (an incremental `ar rcs` once nearly baked a symbol from a
deleted source into the baseline), rebuild the extension (a stale `.so` once
reported a 15.01x memory ratio for a 1.36x change), run under `.venv/bin/python`
(a bare interpreter collects nothing from the parquet tests and six cases look
deleted), and **re-record once at the end of the branch in a commit that
touches nothing else.**

Preconditions, all of which must hold, because re-recording to make a failure
go away is the hazard:

- every "bug signal" test passing **with no tolerance loosened**; a loosened
  tolerance is a silent re-record
- `exact.py` green with its worst margin still well under 100% (currently 5.7%)
- all six invariants passing
- `tests/reference/beam_model.py` **ported, not deleted**. A re-record with the
  independent reference deleted records the C against nothing.
- `numeric.json` differences explained by category. All of `o2/*`, `o3/*` and
  `o4/*` must move; if any block is unchanged, a code path was missed.

## 8. What cannot be validated with what exists

- The headline AP number cannot be re-run. Use the S7 substitute.
- **Three numbers the design leans on have no source in this repository**: the
  depth-cap sweep (49.0 / 40.5 / 23.3%), the 0.960 / 1.254 own-versus-foreign
  ratio, and the 31.7% wide-only dead rate. The first two come from a session
  memory dated 2026-09-02 recording one donor, one foreign control and a single
  split; the third from an ad-hoc measurement. Section 2 regenerates the first
  two cheaply. **The design's section 8 calls the 0.960 ratio "the strongest
  available evidence that the depth axis is mis-specified", and it points AWAY
  from uncapping, so it must be re-measured or the design must stop leaning on
  it.**
- The `_gen` alphabet kernel specialisation is unreachable and will remain
  unverified. Known, documented, accepted.
