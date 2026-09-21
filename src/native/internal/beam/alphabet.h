/**
 * @file alphabet.h
 * @brief Alphabet descriptors, the registry, and sequence encoding.
 *
 * Internal to src/native/. Not a public header and not installed.
 * Included through beam/internal.h, which is the umbrella every
 * translation unit still uses; include this directly only once the
 * owning file has been narrowed to it.
 */
#ifndef BEAM_INTERNAL_ALPHABET_H
#define BEAM_INTERNAL_ALPHABET_H

#include "beam/model.h"

/* ── Alphabet descriptors ────────────────────────────────────── */

/**
 * Everything alphabet-shaped, in one place.
 *
 * The model itself carries only the derived dimensions it needs in inner
 * loops (`n_residues`, `n_alphabet`); this is the table those are read from at
 * construction, plus the defaults and ceilings that depend on the alphabet
 * rather than on the caller.
 */
typedef struct {
    BeamAlphabet id;
    const char  *name;        /**< "aa" | "nt", as the Python layer spells it */

    uint8_t residues;         /**< R: emitted symbols                        */
    uint8_t radix;            /**< A: R + 1, the context-packing base        */
    uint8_t none;             /**< the sentinel's index, == residues         */

    /**
     * Context order of the wide expert: the default when the caller gives
     * none, and the largest this alphabet accepts.
     *
     * The ceiling is per-alphabet because it is really a bound on `A^order`.
     * `beam_ipow_u32` is a `uint32_t` multiply loop that wraps silently, and
     * 21^9 is 794,280,046,581 -- an amino-acid model at the nucleotide
     * ceiling would allocate a wrapped, nonsense-sized table rather than
     * fail. Nucleotides can afford order 9 because 6^9 is only 10,077,696.
     */
    uint8_t default_order;
    uint8_t max_order;

    /** Default longest sequence accepted. Nucleotides need three times the
     *  amino-acid length to say the same thing. */
    uint8_t default_max_length;

    /**
     * Default depth cap of the wide expert.
     *
     * 255 means "resolve every depth": `beam_model_new` already clamps a cap
     * at or beyond `max_length` down to `max_length - 1`, which resolves each
     * depth exactly. The cap exists to keep the amino-acid wide table
     * affordable at radix 21, and that cost does not arise at radix 6.
     */

    /**
     * Byte → residue index, `BEAM_RESIDUE_INVALID` for anything else.
     *
     * An invalid byte is marked as such rather than mapping to 0 and
     * colliding with the first residue, so this table decides validity on its
     * own and needs no separate predicate beside it.
     */
    const uint8_t *char_to_idx;

    /**
     * Residue index → character, `radix` entries plus a NUL. The last entry
     * is the sentinel's display character, '.'; the first `residues` are the
     * alphabet's symbols in index order.
     */
    const char *idx_to_char;
} BeamAlphabetInfo;

/** The descriptor for `id`, or NULL if `id` names no alphabet. */
const BeamAlphabetInfo *beam_alphabet_info(BeamAlphabet id);

/** The descriptor named `name` ("aa", "nt"), or NULL. NULL `name` is NULL. */
const BeamAlphabetInfo *beam_alphabet_by_name(const char *name);

extern const BeamAlphabetInfo beam_alphabet_aa;
extern const BeamAlphabetInfo beam_alphabet_nt;

extern const uint8_t beam_aa_char_to_idx[256];
extern const uint8_t beam_nt_char_to_idx[256];

/* The invariants every kernel assumes. Held here rather than tested, because
 * a violation must not survive to run time: the sentinel sharing an index
 * with a residue would corrupt every context, silently. */
_Static_assert(BEAM_AA_RADIX == BEAM_AA_RESIDUES + 1u,
               "the amino-acid radix is its residues plus one sentinel");
_Static_assert(BEAM_AA_NONE == BEAM_AA_RESIDUES,
               "the amino-acid sentinel sits just past the last residue");
_Static_assert(BEAM_NT_RADIX == BEAM_NT_RESIDUES + 1u,
               "the nucleotide radix is its residues plus one sentinel");
_Static_assert(BEAM_NT_NONE == BEAM_NT_RESIDUES,
               "the nucleotide sentinel sits just past the last residue");
_Static_assert(BEAM_MAX_RESIDUES >= BEAM_AA_RESIDUES
               && BEAM_MAX_RESIDUES >= BEAM_NT_RESIDUES,
               "BEAM_MAX_RESIDUES must cover every alphabet it sizes for");
_Static_assert(BEAM_MAX_RADIX >= BEAM_AA_RADIX
               && BEAM_MAX_RADIX >= BEAM_NT_RADIX,
               "BEAM_MAX_RADIX must cover every alphabet it sizes for");
_Static_assert(BEAM_RESIDUE_INVALID > BEAM_MAX_RADIX,
               "the invalid marker must not collide with a residue index");

/* ── Encoding ────────────────────────────────────────────────── */

/**
 * Encode a NUL-terminated sequence into residue indices, using the model's
 * own alphabet.
 *
 * Returns false if the sequence is empty, longer than the model's
 * `max_length`, or contains a byte that is not a residue of its alphabet.
 */
bool beam_encode(const BeamModel *m, const char *seq,
                 uint8_t *out, uint32_t *out_len);


#endif /* BEAM_INTERNAL_ALPHABET_H */
