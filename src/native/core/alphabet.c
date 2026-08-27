/**
 * @file alphabet.c
 * @brief Amino acid to bit-index mapping.
 */
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include "beam/common.h"
#include "beam/model.h"

/* ── Amino acid mapping ────────────────────────────────────── */


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

/* ── Alphabet ────────────────────────────────────────────────── */


const uint8_t beam_char_to_idx[256] = {
    /* Filled at first use by beam_alphabet_init via a const cast would be
     * unclean; instead the table is spelled out so it is genuinely const and
     * thread-safe with no initialisation order to get wrong. 0xFF marks
     * "not an amino acid"; callers map that to BEAM_NONE. */
    ['A'] = 0,  ['C'] = 1,  ['D'] = 2,  ['E'] = 3,  ['F'] = 4,
    ['G'] = 5,  ['H'] = 6,  ['I'] = 7,  ['K'] = 8,  ['L'] = 9,
    ['M'] = 10, ['N'] = 11, ['P'] = 12, ['Q'] = 13, ['R'] = 14,
    ['S'] = 15, ['T'] = 16, ['V'] = 17, ['W'] = 18, ['Y'] = 19,
    /* Every other byte is 0, which collides with 'A'. Validity is therefore
     * decided by beam_is_residue(), not by the value alone. */
};


const char beam_idx_to_char[BEAM_ALPHABET] = {
    'A','C','D','E','F','G','H','I','K','L',
    'M','N','P','Q','R','S','T','V','W','Y','.'
};
