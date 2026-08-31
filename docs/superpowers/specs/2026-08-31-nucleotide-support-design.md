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

### File format

`BEAM_FORMAT` goes 2 -> 3. The header has no reserved space left -- both spare
words were spent on `strict` and `storage_sparse` -- so `alphabet` is appended
as a `uint32_t`, taking the header from 136 to 140 bytes.

- Format 3 reads the field.
- Formats 1 and 2 load as amino acid, which is what they are.
- Saving always writes format 3.

This field is load-bearing rather than cosmetic: alphabet size fully
determines the payload layout, so without it, opening a nucleotide model as an
amino-acid one surfaces as CRC corruption instead of a typed error. The
project's own format notes argue that a file which loads under the wrong
identity is worse than one that refuses with a reason.

`BEAM_SPARSE_RECORD_BYTES` becomes runtime (`4 + R * 8`); the `block[]` stack
buffer is sized by `BEAM_MAX_RESIDUES`.

### Python surface

`alphabet="aa"` is the default everywhere; `"nt"` is opt-in.

```python
BeamModel(seqs)                                  # aa, unchanged
BeamModel(seqs, alphabet="nt")                   # order 6, max_length 81
BeamModel(seqs, alphabet="nt", order=9)          # allowed, expensive
BeamModel.builder(alphabet="nt")
BeamModel.from_parquet(p, alphabet="nt")         # column defaults to "cdr3"
m.alphabet                                       # -> "aa" | "nt"
```

- `order`, `max_length` and `depth_cap` resolve their defaults from the
  alphabet when not given, the same way `kappa` already resolves from
  `strict`. The C layer and the Python layer must resolve them identically;
  `config.c` records a past bug where those two disagreed silently, and a
  second axis has exactly the same failure mode.
- The config wire tuple gains `alphabet` as a trailing int, so existing
  positional parsing is undisturbed.
- `_check_scoreable` and `_build_failure_reason` become alphabet-aware, and
  their messages name the actual alphabet instead of "the twenty amino acids".
- `from_*` column defaults: `"cdr3_aa"` for amino acids as today, `"cdr3"`
  (the AIRR nucleotide field) for nucleotides.
- PRISM's `19 * L` and `20 * L` literals become `(R - 1) * L` and `R * L` in
  both `src/native/model/prism.c` and `src/BEAM/prism/index.py`.

There is deliberately **no** alphabet auto-detection. `ACGT` is a valid
amino-acid sequence, so any heuristic silently mislabels short or
low-complexity input.

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

Found while implementing, and required of the bite that makes the kernels read
the alphabet. Recorded here because each is a landmine that a passing test
suite would not find.

1. **`BEAM_MAX_ORDER` must rise to 9.** It sizes `slot[BEAM_MAX_ORDER + 1]` in
   `indexing.h` and `edge_weights.c`, and `level_counts[BEAM_MAX_ORDER + 1]`
   in `smoothing.c`. It is left at 4 while nucleotide models are refused,
   because raising it early buys nothing and costs the byte-identical-object
   evidence.
2. **`indexing.h:78` clamps `order` to `BEAM_MAX_ORDER`.** A nucleotide model
   at order 9 would have its backward context silently truncated to order 4 --
   no crash, no error, just a quietly wrong model. The clamp must read the
   alphabet's ceiling or go.
3. **The cross-alphabet shape check is untested.** `add`, `subtract` and
   `posterior` now include the alphabet, because two models can agree on
   order, length and depth and still have different row widths. Testing it
   needs two models of different alphabets, so it must be tested as soon as
   nucleotide models build.
4. **`beam_model_new` refuses every alphabet but amino acid.** Removing that
   refusal is the last step of the kernel bite, and is the moment nucleotide
   support actually exists. Until then it is what keeps a radix-6 table from
   being indexed at radix 21.
5. **Error messages still say "the 20 amino acids"** in `encoding.c`,
   `scoring.c` and `prism.c`. They should name the model's own alphabet.

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
