# Single-expert BEAM: implementation plan

Companion to `2026-09-21-single-expert-model-design.md`.

---

## 1. The geometry, derived once

```
   row = d * n_ctx + c,    n_ctx = A^k,    d in [0, max_length),  c in [0, A^k)
   ix  = row * R + x
   rows_per_dir = max_length * A^k
```

The surviving formula inherits the **mid** expert's shape, not the wide one:
the depth axis is exactly `max_length` because forward depth is at most `L-1`
and backward depth is at most `L-1`. `beam_row_mid` already assumes this;
`beam_row_wide` did not, because it routed through `beam_depth_part`.

Check: `27 * 21^4 = 5,250,987` rows per direction, `210,039,480` cells across
both, against `147,805,560` today. Matches the design's section 6.

**Six consequences that are not renames.**

1. Wide row count goes up **1.69x**, not down.
2. The sparse partition count goes from `n_depth_wide` to `max_length`.
3. `layer_max`'s directory slice `base = beam_depth_part(m,i) * nw` becomes
   `base = i * n_ctx`, and its bound test now varies all the way to
   `max_length - 1`.
4. The training pass's parallel task count goes from `depth_cap + 1` to
   `max_length`, and its merged-final-bucket branch disappears.
5. `finalize`'s depth-aggregated backoff base exists ONLY because the wide
   table has a merged cap bucket. With exact depths it is dead code regardless
   of whether backoff survives at all.
6. `layer_max`'s `stride = S / nw` becomes identically 1 once
   `state_order == order`, so its inner repeat loop is dead.

**Row-index overflow is unchanged.** The worst case today is already
`max_length * A^k` with `depth_cap` clamped to `max_length - 1`. The
`UINT32_MAX` guard keeps exactly its current margin. Say so in the commit; a
reviewer will assume regression.

**The memory-budget precheck does get worse.** Dense amino-acid defaults go
from 2.36 GB to 3.36 GB against a 4 GiB budget, so the margin falls from 1.82x
to 1.28x, and a dense model with `max_length >= 35` is newly refused. Sparse is
unaffected in practice. At least one existing test builds dense at
`max_length = 60` and will start failing with a budget error: 3.38 GB today,
about 7.47 GB uncapped.

## 2. Six design questions the model design does not settle

**Q1. Does `state_order` become `order`?** The design says the state stays
`21^k`, which implies dropping `max(order, 3)`. At `k = 4` nothing moves. At
`k = 2` the trellis narrows from `21^3` to `21^2`, which changes `n_states`,
`path_count`, `topology` and `hill_number(0)` for every order-2 model in the
suite, and the backward settle lag drops from 3 to 2.

**Q2. Does any backoff survive?** The design says exactly three free
parameters, but its removal list names the mid-ladder and len strengths and
**not `kappa_wide`**. There is a hard number attached: rebuilding a backoff
ladder over the single table costs about **882 MB transient per direction**
against roughly 42 MB today, because level `j = k` alone is 840 MB at the
defaults. A 21x regression on every non-strict build and load. That argues for
deleting all five and letting `strict` be the only smoothing switch.

**Q3. Is `w_f + w_b == 1` a hard refusal?** Today only non-negativity is
validated, and `(1.0, 0.0)` is a working configuration used to score one
direction alone, with explicit kernel guards added because it produced NaNs.
**It also conflicts with an existing test** that deliberately sets both weights
to 100.0. Either the rule is softened or that test changes.

**Q4. Does the surviving table keep the backward nearest-first repack?** Its
recorded justification, one contiguous block per layer sweep instead of 3.9
million scattered lookups over half a gigabyte, is unchanged and in fact
stronger now the table is 1.69x larger. Keeping it preserves the
counts-versus-finalized packing asymmetry, which the source calls "the single
most likely way to produce a model that looks plausible and is wrong".

**Q5. Header layout on the format bump.** Keep 144 bytes with the dead words
zeroed, preserving the static assertion as a tripwire and freeing 40 bytes of
genuine spare, or compact and reassert.

**Q6. Variant or replacement?** A variant means both row formulas, both mixing
paths and both file formats coexisting inside the hottest kernel in the
library, which the source records as already costing 3x object code for one
specialisation axis. A replacement makes every published model unreadable. The
plan assumes replacement; choosing a variant invalidates large parts of it.

## 3. The on-disk format: old files cannot be migrated

The model file stores six per-expert blocks, direction-major, expert-minor.
The loader accepts formats 1 through 5.

**Migration is impossible in principle.** The wide counts are stored at
**clamped** depth, so for every depth at or beyond 15 the information about
which exact depth an observation came from was destroyed at training time. The
mid table has exact depths but only order 3; the len table has exact depths but
only order 2. **No old file contains an order-`k` exact-depth table anywhere.**

**And failing to bump the format is worse than bumping it.** A format-5 file
written by the old build has the same magic, the same 144-byte header and the
same format number as one written by the new build. Nothing rejects it, the
payload extents are simply wrong, and it surfaces as a **checksum failure**, a
corrupt-file diagnosis for a perfectly intact file.

So: bump to format 6, keep the older constants only so the loader can name what
it refuses, and refuse by name with a message saying the depth clamp destroyed
the information and the model must be retrained. There is precedent and wording
already in the file for refusing the pre-BEAM format.

Five on-disk compatibility fixtures are compared by hex float and are
deliberately never regenerated, on the stated grounds that a regenerated
fixture "stops being evidence about the past and starts being a recording of
the present". They become refusal tests; the fixtures stay as evidence.

## 4. Four places a mistake would be silent

**The scorer and the DP computing different models.** The presence test is
duplicated in eight places, and the edge kernel is textually included three
times, so six of those exist in triplicate in the object code. Patch fewer than
all of them and `Z(L)` normalises a different model than the one scoring. No
bit-level gate catches it, because every such gate compares two builds of the
same model, and `pgen_diagnostics` cannot see it because `Z(L)` normalises
whatever the DP computed.
*Caught by:* invariant 6 widened on all five axes, plus the production-scale
Python route `log_clamp_probability(L, full) + log_length_prior(L) ==
log_pseq(s)`.

**The two backward context packings becoming interchangeable.** Today
most-distant-first (counts) and nearest-first (finalized) apply to different
tables, so a mix-up is a type error in practice. After the merge they apply to
the same table and the compiler cannot tell them apart.
*Caught by:* the existing state-versus-sequence agreement test, **plus a new
assertion that the two forms are NOT equal** for a context with distinct
digits. The suite currently checks agreement where it should agree but not
disagreement where it must disagree.

**`expected_counts` keeping the old row arithmetic.** It is the second
independent copy of the row formula. Get it wrong and the blocks are still
well-formed, still ascending, and **still sum to the expected length**, which
is the identity the suite checks, because that is a property of the weights and
not of the addresses.
*Caught by:* training on a single known sequence and asserting that every
returned row index is one that sequence's observation would have incremented. A
set-containment check on addresses, not a sum check on values. That assertion
does not currently exist.

**The sparse forward short-circuit probing a different row than the scorer
reads.** Three sites must use the exact-depth row. If one keeps a clamp, the
disagreement appears only at depths 15 and beyond, that is, only in sequences
longer than 15 and only at their last few positions. The modal CDR3 would score
correctly and the tail wrongly, and aggregate metrics would barely move.
*Caught by:* extending the layer-max tests past `max_length` rather than the
short fixtures they use now, and a byte comparison of the live and full layer
weights at depths in `[15, max_length)` on a model built from sequences of
length 20 and up.

## 5. Ordered sequence

The build **cannot** stay green throughout. Deleting the table-kind enum breaks
roughly thirty files at once, and that is deliberate: the enum is the
compiler's index of every site.

1. **Green.** Add the `w_f + w_b == 1` validation with tests, alone. It is the
   only net-new rule and is independent of everything else.
2. **Green.** Land the five invariants that already pass, plus the two that
   fail as expected failures with their measured values recorded.
3. **Green.** Run the two ablations available on the current build
   (`coefficients=(1,0,0)` and `depth_cap = max_length - 1`).
4. **Green.** Port the independent reference implementation **from the design,
   not from the new C code**. Deriving it from the implementation makes the
   agreement test a tautology. It shrinks from 412 lines to about 120.
5. **RED begins.** Geometry core: the model state, indexing, the internal
   contract header.
6. **RED.** Table access and mixing. Deleting the table-kind enum here is what
   produces the full site list.
7. **RED.** Config and lifecycle. Resolve Q2 here; it decides how much of
   smoothing and finalize survives.
8. **RED.** Producers and consumers in dependency order, ending with the
   bindings and the Python facade. The wire format between the binding and the
   Python config is positional and must change in the same commit.
9. **Green.** Build, then the four checks in section 4, before touching any
   test that measures a figure.
10. **Green.** Tests: deletes, then geometry rewrites, then the budget tests
    last, since their new thresholds can only be measured once 9 passes.
11. **Green.** Re-measure the two sparse-storage constants, whose tuning
    assumed the current table geometry. Land each measurement with its
    constant, or neither.
12. **Then** the design's section 8 open measurement: the uncapped-depth
    effect, which the design says is unmeasured and which the available
    evidence points against.

## 6. Scale of the mechanical work

36 sites in the DP edge kernels, each compiled three times. Roughly 23 C tests
and 19 Python tests break correctly; about 40 further call sites are pure
keyword churn. **That will look like a catastrophic red suite and is nothing of
the kind.** The tests that matter are the ones classified as bug signals in the
test plan, and they are green today.
