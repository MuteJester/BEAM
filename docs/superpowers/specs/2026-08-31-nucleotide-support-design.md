# Nucleotide alphabet support

**Status:** approved design, implementation staged
**Branch:** `nucleotide-support`
**Date:** 2026-08-31

## Goal

Let a BEAM model be built over nucleotide sequences using exactly the model
that is used for amino acids today: the same two anchored experts, the same
log-linear mixing, the same hierarchical backoff, the same exact dynamic
programs. Only the alphabet changes.

The amino-acid path must be unchanged. Not "equivalent" and not "within
tolerance" -- unchanged, verified bit for bit.

## Non-goals

- Codon or reading-frame awareness. A nucleotide model is a model over a
  five-symbol alphabet and knows nothing about translation.
- Translating between the two alphabets, or building one model from the other.
- Retuning the mixing coefficients or backoff strengths for nucleotides. Those
  are measured numbers; changing them needs a measurement, and this change
  provides none.
- Raising the amino-acid context order. The order ceiling rises only for
  nucleotides.

## The alphabets

| | symbols | residues `R` | radix `A` | sentinel | default order | max order | default `max_length` | default `depth_cap` |
|---|---|---|---|---|---|---|---|---|
| `aa` | `ACDEFGHIKLMNPQRSTVWY` | 20 | 21 | 20 | 4 | 4 | 27 | 15 |
| `nt` | `ACGTN` | 5 | 6 | 5 | 6 | 9 | 81 | resolve every depth |

`N` is an emitted residue, not an error and not a wildcard. A sequence
containing `N` is scoreable; the model learns a distribution over `N`;
generation may emit it; path counts include `N`-bearing sequences.

Nucleotide input is case-folded on ingest, so `acgtn` and `ACGTN` are the same
sequence. This is pure input normalisation and costs nothing, because it lives
in the `[256]` decode table. The amino-acid decode table is untouched and
still rejects lowercase.

Any byte outside an alphabet's symbol set is rejected exactly as a non-amino
acid is today: the sequence is skipped at build time and scores `-inf`.

### Why order 6 is the nucleotide default

An amino acid is a compressed codon -- 21 values where three nucleotides need
`6^3 = 216`. Matching the amino-acid model's context *span* would need order
12, which is `6^12` states and not buildable. Measured against the real
allocation in `beam_compute_log_z`:

| nt config | states `6^k` | log Z buffers per call | finalize work vs AA order 4 |
|---|---|---|---|
| order 6 | 46,656 | 3.5 MB | 0.53x |
| order 7 | 279,936 | 21 MB | 3.2x |
| order 9 | 10,077,696 | 766 MB | 114x |
| *AA order 4 (today)* | *194,481* | *38 MB* | *1x* |

Order 6 is two codons and is cheaper than today's amino-acid model, so every
dynamic program stays interactive. Order 9 is three codons, remains reachable
for anyone who asks for it, and is documented as expensive.

## Architecture

### Three layers

1. **`BeamAlphabet`**, a public enum in the config: `BEAM_ALPHABET_AA = 0`,
   `BEAM_ALPHABET_NT = 1`. `AA` is zero deliberately, so
   `beam_model_config_default()`'s opening `memset` and every existing
   zero-initialised config keep meaning "amino acid" without a code change.

2. **`BeamAlphabetInfo`**, an internal descriptor holding everything
   alphabet-shaped in one place: `residues`, `radix`, `none`, the `[256]`
   decode table, the `radix`-wide encode table, the symbol string, the
   display name, and the per-alphabet defaults and ceilings from the table
   above. Looked up by `beam_alphabet_info(BeamAlphabet)`.

3. **Derived dimensions on `BeamModel_`**: `n_residues`, `n_alphabet`,
   `none_idx`, and a pointer to the descriptor. These sit beside the eight
   dimensions the struct already carries as runtime `uint32_t`
   (`n_states`, `n_ctx_wide`, ...), so this is the pattern the struct already
   uses rather than a new one.

### What the refactor actually touches

About twenty dynamic-program kernels already open with

```c
const uint32_t A = BEAM_ALPHABET;
const uint32_t R = BEAM_RESIDUES;
```

Each becomes a two-line edit reading `m->n_alphabet` and `m->n_residues`. The
table-addressing helpers `beam_ix_wide/mid/len` already take
`const BeamModel *m`, so they need no signature change.

The context-packing helpers in `indexing.h` take no model. Each gains one
`uint32_t radix` parameter, with the sentinel derived as `radix - 1`. Every
call site already has `A` in scope, so each call site is a one-token edit.

Three genuinely fixed-size arrays are sized by a new `BEAM_MAX_RESIDUES`:

- `src/native/dp/edge_weights.c:292` -- `double buf[BEAM_RESIDUES]`
- `src/native/io/model_file.c:187` -- `double values[BEAM_RESIDUES]`
- `src/native/core/alphabet.c:47` -- `beam_idx_to_char[BEAM_ALPHABET]`

`BEAM_MAX_LENGTH` rises from 27 to 255, the `uint8_t` limit, for both
alphabets, making `max_length` a genuinely free model parameter. Per-alphabet
*defaults* are unchanged for amino acids (27), so no existing behaviour moves.
Stack buffers sized by `BEAM_MAX_LENGTH` grow from 27 to 255 entries, which is
2 KB for a `double` array and therefore not a concern.

### The order-ceiling overflow guard

Raising `BEAM_MAX_ORDER` to 9 globally would be a bug. `beam_ipow_u32` is a
plain `uint32_t` multiply loop that wraps silently, and an amino-acid model at
order 9 asks for `21^9 = 794,280,046,581` states. So:

- the order ceiling is **per-alphabet**, carried in the descriptor (4 for
  amino acids, 9 for nucleotides);
- `beam_config_valid` gains an explicit check that computes `A^state_order` in
  `uint64_t` and rejects overflow with a typed error rather than allocating a
  wrapped, nonsense-sized table;
- `BEAM_MAX_ORDER` survives only as the array-sizing bound for `slot[]` and
  `level_counts[]`, and rises to 9 for that purpose alone.

### File format (done)

`BEAM_FORMAT` goes 2 -> 3. The header had no reserved space left -- both spare
words were spent on `strict` and `storage_sparse` -- so the format-3 header
**appends** two words, `alphabet` and a new `reserved`, taking it from 136 to
144 bytes.

Appending rather than grouping `alphabet` with the other `uint32`s where it
belongs is deliberate, and it buys the whole compatibility story: an older
header is then exactly a PREFIX of a newer one. The reader takes the 136
common bytes, looks at `format`, and asks for the tail only when it is there.
Grouping the field would have meant two layouts to parse and two chances to
get an offset wrong. Reading `sizeof h` unconditionally would swallow the
first eight bytes of a format-2 payload and report a corrupt file for a
perfectly good one.

- Format 3 reads the field; formats 1 and 2 leave it zero, which is
  `BEAM_ALPHABET_AA`, which is what they are.
- An alphabet the build does not know is refused rather than guessed at.
- A format-3 file read by an older build fails its CRC and is refused, so the
  incompatibility is safe in both directions.
- `reserved` is written zero and ignored. Both earlier spare words were spent,
  which is why this needed a format bump at all; leaving one costs eight bytes
  a file and saves the next bump.

The field is load-bearing rather than cosmetic: alphabet size fully determines
the payload layout, so without it a nucleotide model reads back either as CRC
corruption or, worse, as an amino-acid model of the same shape.
`test_a_nucleotide_file_relabelled_as_amino_acid_is_rejected` repairs the CRC
after relabelling on purpose, so what the loader is left to catch is the
mislabelling itself and not a checksum.

`BEAM_SPARSE_RECORD_BYTES` became runtime (`4 + R * 8`) in the kernel bite;
the `block[]` staging buffer is sized by `BEAM_MAX_RESIDUES`.

**The one baseline that moved.** `saved-model bytes` was re-recorded, and the
change was proved rather than assumed: the same fixture saved by the commits
either side differs by exactly eight bytes, the format word 2 -> 3, and the
trailing CRC -- with the payload byte-identical. `numeric` and `thread
independence` stayed green across it. See `verification/baseline/RERECORDED.md`.

### Python surface (done)

`alphabet="aa"` is the default everywhere; `"nt"` is opt-in.

```python
BeamModel(seqs)                                  # aa, unchanged
BeamModel(seqs, alphabet="nt")                   # order 6, max_length 81
BeamModel(seqs, alphabet="nt", order=9)          # allowed, expensive
BeamModel.builder(alphabet="nt")
BeamModel.from_parquet(p, alphabet="nt")         # column defaults to "cdr3"
m.alphabet                                       # -> "aa" | "nt"
m.n_residues                                     # -> 20 | 5
```

`order`, `depth_cap` and `max_length` became `None`-defaulted and are resolved
from the alphabet, so `BeamModel(seqs)` still produces exactly order 4,
depth_cap 15, max_length 27.

**Those defaults are read from the C descriptor, not restated.** A new binding
`_cbeam.alphabet_info(name)` serves the descriptor verbatim, and
`BEAM.model.alphabets` is a thin reader over it. This is the one design point
worth insisting on: `config.c` records a real bug from the last time each
layer kept its own copy of a default -- they disagreed, silently, and both
produced a model. A second axis has the same failure mode, so there is nothing
to keep in step. `test_python_and_c_resolve_the_same_defaults` checks it
directly for both alphabets.

Also here:

- `config` now reports `alphabet`, so `_config_args_from_dict` round-trips it
  and `merged`, `without`, `posterior` and the copy path all preserve it.
- `_check_scoreable` and `_build_failure_reason` name the model's own
  alphabet and list its residues.
- The `from_*` sequence-column default follows the alphabet: `cdr3_aa` for
  amino acids, `cdr3` for nucleotides, which are the AIRR fields for a
  junction's translation and the junction itself. Reading the wrong one skips
  every row -- not subtle, but silent until the corpus comes back empty.
- PRISM's channel denominators became `(R-1)L` and `RL` in **both**
  `prism.c` and `prism/index.py`. They were 19 and 20 in each, and fixing only
  Python made the nucleotide posterior sum to 4.75 instead of 1, which is how
  the second copy was found.

There is deliberately **no** alphabet auto-detection. `ACGT` is a valid
amino-acid sequence, so any heuristic silently mislabels short or
low-complexity input.

**End to end, with nothing but the alphabet given:**

```
BeamModel(3000 nt CDR3s, alphabet="nt")
  -> order 6, max_length 81, 46,656 states, 64 MB
     log_pseq == log_pseq(lower case), support 3,657, entropy 7.57 nats
     generation reproduces the V and J motifs
```

## File layout

New:

- `src/native/core/alphabet_aa.c` -- amino-acid tables and descriptor
- `src/native/core/alphabet_nt.c` -- nucleotide tables and descriptor
- `src/BEAM/model/alphabets.py` -- the Python-side registry
- `tests/native/test_alphabet.c` -- alphabet unit tests
- `tests/python/model/test_nucleotide.py` -- nucleotide model tests

Repurposed:

- `src/native/core/alphabet.c` -- keeps `beam_encode`, becomes the registry
  and dispatch; the amino-acid tables move out to `alphabet_aa.c`

Each new `.c` file needs one line in `build_support/native_sources.txt`, in
the alphabetical block for its directory. New private headers go in
`src/native/internal/beam/` and are added to the `internal.h` umbrella. C test
files are globbed by the Makefile and need no registration.

## Staging

Each phase ends with the verification gate and a green light before the next
begins.

**Phase 0 -- alphabet becomes runtime, amino acid still the only alphabet.**
No new behaviour, no new API, no format change. The gate must show `numeric`,
`saved-model bytes` and `thread independence` unchanged. This is the safety
proof: if all 78 numeric quantities still match as hex floats, the refactor is
provably inert, and no nucleotide bug found later can be an amino-acid
regression.

**Phase 1 -- the alphabet registry and the nucleotide tables.** Descriptors,
the config field, per-alphabet defaults and ceilings, the overflow guard. C
only.

**Phase 2 -- file format v3.** Header field, load compatibility for formats 1
and 2, runtime sparse record size.

**Phase 3 -- the Python surface.** `alphabet=` on the constructors, the
facade, alphabet-aware validation messages, PRISM, docs.

**Phase 4 -- tests and baselines.** Nucleotide corpus fixtures,
`tests/native/test_alphabet.c`, parametrising the independent NumPy reference
in `tests/reference/beam_model.py` (which carries its own hardcoded copy of
the alphabet), and re-recording the two baselines that legitimately move.

## Carried requirements

Found while implementing, and recorded because each is a landmine a passing
test suite would not find. All five are now discharged.

1. ~~**`BEAM_MAX_ORDER` must rise to 9.**~~ Done: it sizes
   `slot[BEAM_MAX_ORDER + 1]` and `level_counts[BEAM_MAX_ORDER + 1]`, and is
   now the largest of the per-alphabet ceilings rather than the validation
   limit.
2. ~~**`indexing.h` clamps `order` to `BEAM_MAX_ORDER`.**~~ Correct now that
   the bound is 9 and no alphabet's ceiling exceeds it, so the clamp cannot
   fire for a legal config. Had it stayed at 4, a nucleotide model at order 9
   would have had its backward context silently truncated -- no crash, no
   error, a quietly wrong model.
3. ~~**The cross-alphabet shape check is untested.**~~ Done:
   `test_model_algebra_refuses_across_alphabets` builds two models agreeing on
   order, length and depth and differing only in alphabet, and checks that
   `add`, `subtract` and `posterior` all refuse.
4. ~~**`beam_model_new` refuses every alphabet but amino acid.**~~ Lifted.
   The test that guarded it had a stated lifetime and was removed with it.
5. ~~**Error messages still say "the 20 amino acids".**~~ They now name the
   model's own alphabet and list its residues.

### New, from enabling nucleotides

6. ~~**A nucleotide model cannot be saved.**~~ Done: format 3 appends the
   alphabet to the header, nucleotide models round-trip bit-identically, and
   the refusal and the test that pinned it are both gone.

## Nucleotide models, measured

The default configuration -- order 6, `max_length` 81, every depth resolved --
builds and scores:

| | states | cells | build | peak RSS |
|---|---|---|---|---|
| sparse (what Python selects) | 46,656 | 40.4 M | 0.03 s | 67 MB |
| dense (the C default) | 46,656 | 40.4 M | 0.14 s | 321 MB |

Dense at the nucleotide defaults is 0.65 GB of tables against 2.36 GB for the
amino-acid defaults, so the alphabet that needs three times the length still
costs a third as much.

The two strongest checks are enumerative rather than self-referential. The
partition function is summed over all 125 length-3 nucleotide sequences and
must reach 1, which exercises the context packing, the backward lag, the sink
weights and the normalisation at radix 6 in one assertion. And `path_count` is
compared against the count of sequences that actually score finitely, which
crosses the trellis, the liveness frontier and the edge weights against
enumeration. Corrupting the context packing to base `radix - 1` fails both,
along with 45 of the amino-acid model tests.

## The measured cost of a runtime alphabet

Recorded after the kernels were migrated, because it contradicts what the
design assumed.

The design chose a runtime alphabet over compiling the core twice, on the
stated basis that the amino-acid path faced a "small perf risk in a few inner
loops". Measured, the risk is not small. Single-threaded, interleaved A/B,
min of nine runs each, on a quiesced machine:

| operation | before | after | ratio |
|---|---|---|---|
| `path_count` | 64.2 ms | 96.7 ms | **1.51x** |
| `spectrum` | 120.0 ms | 167.4 ms | **1.40x** |
| `hill_number(2)` | 77.8 ms | 102.5 ms | **1.32x** |
| `entropy` | 84.2 ms | 110.0 ms | **1.31x** |
| `topology` | 0.17 ms | 0.18 ms | 1.06x |

**The cause is confirmed, not guessed.** Temporarily pinning `A` and `R` back
to compile-time constants in the fifteen kernel files restores every number to
baseline (0.98-1.02x). Nothing structural changed; the compiler simply lost
the constants it was folding. `for (x = 0; x < R; ++x)` with `R = 20` unrolls
and vectorises and with a runtime `R` does neither, and `t % A` / `t / A` with
`A = 21` is a multiply-shift and with a runtime `A` is two hardware divides.

A first attempt to fix only the division -- a `beam_split_state` helper that
branched on the two known radices -- was written, measured, and removed: it
made no measurable difference, because the division is only part of what the
constants were buying.

**A first measurement of this was worthless and nearly misled the work.** It
ran multi-threaded on a machine at load average 9.67 and reported a 20%
regression; re-running the identical build three times gave a spread of 6% to
750% on the same operations. Any benchmark of this codebase must pin threads
to one and interleave the two builds, or it is measuring the machine.

### Where the cost actually is

Narrowed by pinning the constants back one file at a time, rather than
reasoned about:

| pinned | cold path_count | warm path_count |
|---|---|---|
| nothing (as committed) | 97.6 ms | 44.7 ms |
| `edge_weights.c` + `liveness.c` | 63.3 ms | 28.8 ms |
| `liveness.c` alone | 98.9 ms | 44.5 ms |
| **`edge_weights.c` alone** | **64.8 ms** | **27.7 ms** |
| *pre-migration reference* | *64.9 ms* | *28.5 ms* |

**The whole regression is in `edge_weights.c`.** Nothing else in the fifteen
migrated files costs anything measurable, which is what one would expect: that
file holds `state_weights_dense`, `state_weights_sparse` and the layer sweeps,
and every dynamic program reaches its inner loop through them. The other
files' `A` and `R` sit in loops that run once per layer, not once per edge.

### Why the obvious fix does not work

Three attempts, none of which moved the number:

1. specialising the `t % A` / `t / A` split alone, across the analytics;
2. dispatching `state_weights_dense` / `state_weights_sparse` at their call
   sites;
3. dispatching whole exported functions -- `beam_layer_weights_live`,
   `beam_tail_weights`, `beam_layer_max` -- into `BEAM_INLINE` bodies taking
   the dimensions as parameters.

The cause is OpenMP. Rebuilt with `-fopenmp` removed, the same dispatch
recovers the entire regression (29.3 ms against a pinned ceiling of 28.6 ms).
With it, nothing. GCC lowers a parallel region into an outlined function early,
and the region's variables reach that function through a data struct, so a
radix that became a literal at the inlining site arrives inside the loop as a
runtime load. Moving the dispatch *inside* the region did not help either,
which suggests the outlining also blocks the `always_inline` that was supposed
to carry the literal in.

### What was done, and the four things that did not work

`edge_weights.c` is now compiled once per alphabet. Its function bodies moved
verbatim to `edge_weights_kernels.h`, which has no include guard and is
included three times -- with the amino-acid dimensions, the nucleotide ones,
and the model's own -- each inclusion defining a suffixed set of file-static
functions. The five public entry points became dispatchers on the model's
radix. Nothing about the code inside changed, including every OpenMP pragma.

| operation | before | after | ratio |
|---|---|---|---|
| `path_count` | 64.8 ms | 63.9 ms | 0.99x |
| `spectrum` | 120.3 ms | 120.5 ms | 1.00x |
| `entropy` | 83.3 ms | 85.1 ms | 1.02x |
| `hill_number(2)` | 75.4 ms | 77.3 ms | 1.03x |

Four attempts to recover the speed *without* duplicating the text were
measured, and none of them moved the number:

1. specialising the `t % A` / `t / A` state split across the analytics;
2. dispatching the two leaf kernels at their call sites;
3. dispatching the exported functions into `always_inline` bodies taking the
   dimensions as parameters;
4. extracting each parallel loop's body into a specialised slice called from
   inside the region.

Rebuilt without `-fopenmp`, attempt 3 recovers everything -- 29.3 ms against a
pinned ceiling of 28.6. That is the tell: GCC lowers a parallel region into an
outlined function early and passes its variables through a data struct, so a
radix that is a literal at the call site arrives inside the loop as a runtime
load. Only text that already contains the literal before that lowering keeps
it, which is what including the file three times achieves and what no amount
of dispatching does.

Attempt 4 taught something separate and worth keeping: `beam_tail_weights`
alone *did* recover under it, because its loop body is one self-contained call
per state. That was enough to bring `path_count` back to 1.01x while entropy
and the spectrum stayed 15% down -- which is how the remaining cost was
localised to the two layer sweeps.

The cost is object size: this file goes from 9.1 KB of text to 26.6 KB, and
`libbeam.a` grows 11%. The pre-existing `-Wmaybe-uninitialized` on `slot` in
`indexing.h` now appears 15 times rather than 5, being inlined into three
copies; it is the same false positive, not a new one.

The other fourteen migrated files keep the runtime alphabet, where it costs
nothing measurable.

## The independent reference

`tests/reference/beam_model.py` is a from-scratch NumPy reimplementation
written against the specification, sharing no code with `src/native/`. It is
the only thing in the suite that can catch a dimension which is consistently
wrong in both the C and its own tests. It carried its own hardcoded copy of
the amino-acid alphabet, so nucleotide models had none of that.

It is now parameterised by an `Alphabet`, and
`test_reference_agreement.py` checks scores at orders 2, 3 and 4, the
partition function, and path counts over the radix-6 trellis as well as the
radix-21 one. The five amino-acid agreement tests passing unchanged through
the parameterisation is what says the refactor was faithful: the reference is
independent, so if it had drifted, the amino-acid comparison would have
broken.

Two mutations were run to check the tests bite, and the pair is more
informative than either alone:

- **Swapping two nucleotide residue indices** (G and T) is caught by the
  alphabet table tests and NOT by the reference. That is correct rather than a
  gap: a consistent relabelling of indices is an isomorphism, so no score
  moves. What it breaks is decoding and file interchange, which is where it is
  caught.
- **Narrowing the mid expert's backward context from order 3 to 2**, in the
  trellis only, leaves a perfectly self-consistent model that still normalises
  -- `test_the_partition_normalises_over_nucleotides` passes. Eight of the ten
  agreement tests fail, including all three nucleotide score tests. That is
  the class of bug the reference exists for, and nothing else in the suite
  sees it.

The nucleotide corpus generator deliberately does not build sequences on codon
boundaries. The model knows nothing about reading frame, and a corpus that
respected frame could hide a packing error that happens to respect it too.

## What the independent review found

Two reviewers were run over the finished branch, one static and one
adversarial, and both found the same defect independently.

**`BeamModel(seqs, alphabet="nt", strict=False)` segfaulted** — the documented
open world at the alphabet's own default order. `finalize.c` indexed the wide
expert's backoff base at `order - 1` into `double *mid_p[4]`, holding the mid
hierarchy's orders 0 to 3. Amino acids cap at order 4, so that index was always
in range and always a level that had been built; a nucleotide model at order 6
read two slots past the end of a stack array.

Enlarging the array would only have moved the failure: `beam_build_hierarchy`
builds the mid levels from `cnt_mid`, an order-3 count table, so `mid_p[5]`
would have been NULL. There is no order-4 context anywhere in the model to
build it from, and `kappa` has three entries — the smoothing hierarchy is
*specified* only to order 3.

The fix is therefore not a bigger array. The wide expert takes one backoff
step, and its base is now the order below **capped at 3**: the deepest level
the hierarchy provides. For every amino-acid order this changes nothing, since
the ceiling there is 4. A nucleotide model above order 4 now backs off one
bigger step rather than indexing a level that was never built. The arrays are
also sized by `BEAM_MAX_ORDER + 1`, matching what `smoothing.c` and
`indexing.h` already did — that sweep missed this file.

**Why nothing caught it.** Every nucleotide model in either suite was order 3,
or used the closed-world default where all five backoff strengths are zero and
the branch is skipped entirely. The C tests asserted that `beam_config_valid`
*accepts* order 5 and 9 without ever building one. The gap was in the tests,
not in the gate, which is why every baseline was green over a configuration
that crashed the process.

Three lesser findings were also raised, and all three are closed:

- The three sparse row counts are cast to `uint32_t` without a guard. The
  widest reaches 2.57e9 against a 4.29e9 limit at the largest configuration
  today's ceilings allow -- inside it, but by less than a factor of two.
  `beam_model_new` now checks all three, and
  `test_the_widest_table_still_fits_a_uint32_row_index` pins the headroom that
  makes the check unreachable, so raising a ceiling breaks a test rather than
  quietly relying on the margin.
- `model.h` documented the backoff strengths as defaulting to 20 / 20 / 20, 20
  and 200 while `config.c` set all five to zero. Stale since the closed world
  became the default, and pre-existing rather than introduced here. The comment
  now says what the code does and where the open-world values come from.
- The `_gen` copy of the edge-weight kernels is unreachable, since the dispatch
  picks `_aa` or `_nt` for every alphabet that exists, so nothing checks it
  agrees with them bit for bit. Left in place -- it is what keeps a third
  alphabet correct -- but the header now says out loud that it is unverified
  until that day, rather than letting a reader assume the triple compilation is
  covered three times over.

Everything else the reviewers attacked held. The bit-for-bit claim was verified
beyond the gate: 1304 of 1335 quantities compared as hex floats across 31
configurations were identical, and every one of the 31 differences was the
`config` dict gaining an `alphabet` key — zero numeric differences anywhere.
The partition function was enumerated over 270 nucleotide configurations the
suite does not cover, worst error 5.7e-14. Real format-2 files written by the
pre-branch build load bit-identically. The performance claim was reproduced at
0.98-1.03x.

## The second review round

The fix above was reviewed again, by two more agents pointed at it rather than
at the branch. Both confirmed it — one deriving the index arithmetic from the
packing invariants rather than from the test, the other proving it positively
with a C probe: for every wide row whose count row is empty, the finalized row
must equal the order-3 base row, and it does at every order 2 to 9, worst
difference 4.4e-16. That probe also discriminates, since a deliberately wrong
reduction differs from order 4 up. And keeping the enlarged arrays while
restoring the unclamped index still segfaults, which confirms the clamp rather
than the array size is what was load-bearing.

Five things attributable to those two commits were found and closed:

1. **The tripwire claim was wrong on one of its two knobs.** Raising
   `BEAM_MAX_ORDER` from 9 to 20 left the suite green, because the headroom
   test reads the descriptor's `max_order`, not the global bound. The real
   missing invariant was that the two cannot diverge: an alphabet allowing an
   order past `BEAM_MAX_ORDER` would have `indexing.h` silently CLAMP rather
   than fail. That assertion now exists, and raising a descriptor's ceiling
   past the bound fails two tests.
2. **The guard's message could contradict its own condition** — it tested the
   maximum of three row counts and reported the maximum of two.
3. **The guard's message was unreachable from Python.** The bindings replaced
   every `beam_model_new` failure with "invalid BeamModel configuration",
   discarding the reason. They now pass it through, and the facade no longer
   blames the corpus for a failure the corpus did not cause: an out-of-memory
   build used to surface as "no training sequence survived encoding".
4. **The coverage was still thin where it mattered.** The C test built orders
   2 to 6 and the Python one 4 to 7, while the alphabet allows 9 — the same
   shape of gap that hid the original defect, one layer down. Both now go to
   the ceiling.
5. **The normalisation test never reached the aggregated backoff branch.** Its
   `depth_cap` was at or above the length being enumerated, so every position
   took the per-depth base and `agg_p` was untouched. Writing the fix for that
   surfaced a bug in the test rather than the code: enumerating a length the
   corpus never spelled sums to zero, correctly, because that length has no
   prior. It now observes two lengths, caps depth at 1, and enumerates both.

## Verification contract

The gate at `verification/check.py` compares every quantity as a hex float, so
it fails on a one-bit change that a tolerance-based assertion would pass.

**Recorded starting state on this branch (2026-08-31), before any change:**

| check | state | why |
|---|---|---|
| `_cbeam callables` | FAIL | uncommitted PRISM work adds `prism_explain`, `prism_factor` |
| `C public symbols` | FAIL | uncommitted PRISM work adds `beam_prism_factor*` |
| `Python public API` | FAIL | uncommitted PRISM work adds `Prism`, `Attribution`, `ScaleIndex`, ... |
| `pytest inventory` | FAIL | uncommitted PRISM work adds `tests/python/analytics/test_prism.py` |
| `C test inventory` | **OK** | |
| `numeric` (78 keys) | **OK** | |
| `saved-model bytes` | **OK** | |
| `thread independence` | **OK** | |

Also recorded: 310 collected pytest cases, 108 C public symbols, 66 passing
native test cases.

The contract for every phase of this work is therefore:

1. `numeric`, `saved-model bytes` and `thread independence` stay **OK**,
   except at Phase 2 where the format bump moves `saved-model bytes` by
   design and it is re-recorded deliberately.
2. The four surface checks gain only what the phase intends and nothing else,
   compared against the frozen "before" lists rather than against the stale
   committed baseline.
3. All 310 existing pytest cases keep passing, and every new test is added
   rather than substituted.

The pre-existing PRISM-driven baseline staleness is **not** fixed as part of
this work. It is a separate change on separate evidence, and folding it in
would destroy the ability to attribute a gate movement to this work.
