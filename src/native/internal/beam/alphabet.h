/**
 * @file alphabet.h
 * @brief Alphabet tables and encoding.
 *
 * Internal to src/native/. Not a public header and not installed.
 * Included through beam/internal.h, which is the umbrella every
 * translation unit still uses; include this directly only once the
 * owning file has been narrowed to it.
 */
#ifndef BEAM_INTERNAL_ALPHABET_H
#define BEAM_INTERNAL_ALPHABET_H

#include "beam/model.h"

/* ── Alphabet ────────────────────────────────────────────────── */

/** ASCII → residue index, or BEAM_NONE for anything not an amino acid. */
extern const uint8_t beam_char_to_idx[256];

/** Residue index → ASCII. Index BEAM_NONE maps to '.'. */
extern const char beam_idx_to_char[BEAM_ALPHABET];

/**
 * Encode a NUL-terminated sequence into residue indices.
 * Returns false if the sequence is empty, longer than `max_len`, or contains
 * a character outside the 20 amino acids.
 */
bool beam_encode(const BeamModel *m, const char *seq,
                     uint8_t *out, uint32_t *out_len);

/**
 * True for the twenty amino acids BEAM encodes.
 *
 * Kept here as `static inline` rather than exported: it sits in the
 * per-character loop of every encode, and it is what decides validity,
 * because the char-to-index table maps every unknown byte to 0, which
 * collides with 'A'.
 */
static inline bool beam_is_residue(unsigned char c) {
    return c == 'A' || c == 'C' || c == 'D' || c == 'E' || c == 'F'
        || c == 'G' || c == 'H' || c == 'I' || c == 'K' || c == 'L'
        || c == 'M' || c == 'N' || c == 'P' || c == 'Q' || c == 'R'
        || c == 'S' || c == 'T' || c == 'V' || c == 'W' || c == 'Y';
}

#endif /* BEAM_INTERNAL_ALPHABET_H */
