/**
 * @file alphabet_aa.c
 * @brief The amino-acid alphabet: 20 residues and one sentinel.
 *
 * Twenty residues in the conventional single-letter order, plus one sentinel
 * at index 20 that pads contexts at the termini and is never emitted.
 *
 * It used to carry a second encoding as well -- `beam_aa_to_bit`, the
 * FlashBack walk dictionary's bitmask, whose own section header opened by
 * admitting it was "unrelated to the model's own alphabet". The two disagreed
 * deliberately and in opposite directions: `'X'` and lower case are refused
 * here and folded to bit 31 there, and `'@'` and `'$'` are residues there and
 * not here. Neither had a caller, and the only thing that referenced either
 * was `beam_aa_bitmask` in common.h, which had no caller either and mapped
 * every non-standard byte to the same bit, so an all-lower-case sequence and
 * an all-`X` sequence produced identical masks. Both are gone. A maintainer
 * editing "the amino-acid table" now has one table in front of them.
 */
#include "beam/common.h"
#include "beam/model.h"
#include "beam/alphabet.h"

/* ── Descriptor tables ───────────────────────────────────────── */

const uint8_t beam_aa_char_to_idx[256] = {
    /* Spelled out rather than filled at first use, so it is genuinely const
     * and thread-safe with no initialisation order to get wrong.
     *
     * Every unlisted byte is 0 by C's aggregate rules, and 0 is 'A', so the
     * table cannot be left to say that: each entry is set explicitly and
     * BEAM_RESIDUE_INVALID marks a non-residue, which is what lets one lookup
     * decide both the index and the validity.
     *
     * Lower case is deliberately absent. An amino-acid sequence is upper case
     * by convention, and folding it here would be a silent widening of what
     * the model accepts. */
    [0 ... 'A' - 1] = BEAM_RESIDUE_INVALID,
    ['A'] = 0,
    ['B'] = BEAM_RESIDUE_INVALID,
    ['C'] = 1,  ['D'] = 2,  ['E'] = 3,  ['F'] = 4,  ['G'] = 5,
    ['H'] = 6,  ['I'] = 7,
    ['J'] = BEAM_RESIDUE_INVALID,
    ['K'] = 8,  ['L'] = 9,  ['M'] = 10, ['N'] = 11,
    ['O'] = BEAM_RESIDUE_INVALID,
    ['P'] = 12, ['Q'] = 13, ['R'] = 14, ['S'] = 15, ['T'] = 16,
    ['U'] = BEAM_RESIDUE_INVALID,
    ['V'] = 17, ['W'] = 18,
    ['X'] = BEAM_RESIDUE_INVALID,
    ['Y'] = 19,
    ['Z' ... 255] = BEAM_RESIDUE_INVALID,
    /* The gaps are named rather than left to a default that later entries
     * override: B, J, O, U, X and Z are the letters that are not amino acids,
     * and spelling them out is what lets -Woverride-init stay on. X in
     * particular is the one a caller is most likely to send, and it is
     * rejected -- the model does not guess at an unknown residue. */
};

static const char aa_idx_to_char[BEAM_AA_RADIX + 1] =
    "ACDEFGHIKLMNPQRSTVWY.";

const BeamAlphabetInfo beam_alphabet_aa = {
    .id                 = BEAM_ALPHABET_AA,
    .name               = "aa",
    .residues           = BEAM_AA_RESIDUES,
    .radix              = BEAM_AA_RADIX,
    .none               = BEAM_AA_NONE,
    /* Order 4 is the measured default and also the ceiling: 21^5 is 4.1
     * million contexts per depth, which the wide table cannot carry. */
    .default_order      = 4,
    .max_order          = 4,
    .default_max_length = 27,
    .char_to_idx        = beam_aa_char_to_idx,
    .idx_to_char        = aa_idx_to_char,
};
