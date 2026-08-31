/**
 * @file alphabet_aa.c
 * @brief The amino-acid alphabet: 20 residues, plus the walk-dictionary
 *        bit mapping that is unrelated to it.
 *
 * Twenty residues in the conventional single-letter order, plus one sentinel
 * at index 20 that pads contexts at the termini and is never emitted.
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
    .default_depth_cap  = 15,
    .char_to_idx        = beam_aa_char_to_idx,
    .idx_to_char        = aa_idx_to_char,
};


/* ── Walk-dictionary bit mapping ─────────────────────────────── */

/*
 * Unrelated to the model's own alphabet: this is the FlashBack walk
 * dictionary's bitmask encoding, which has its own sentinels and its own
 * treatment of non-standard residues.
 */

uint8_t beam_aa_to_bit(char c) {
    switch (c) {
        case 'A': return  0; case 'C': return  1; case 'D': return  2;
        case 'E': return  3; case 'F': return  4; case 'G': return  5;
        case 'H': return  6; case 'I': return  7; case 'K': return  8;
        case 'L': return  9; case 'M': return 10; case 'N': return 11;
        case 'P': return 12; case 'Q': return 13; case 'R': return 14;
        case 'S': return 15; case 'T': return 16; case 'V': return 17;
        case 'W': return 18; case 'Y': return 19;
        case '@': return 20; /* start sentinel */
        case '$': return 21; /* end sentinel */
        default:  return 31; /* non-standard → high bit, ignored in bitmask */
    }
}
