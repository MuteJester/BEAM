/**
 * @file alphabet_nt.c
 * @brief The nucleotide alphabet: A C G T N, plus one sentinel.
 *
 * `N` is an emitted residue rather than an error or a wildcard. A sequence
 * containing it is scoreable, the model learns a distribution over it,
 * generation may emit it, and path counts include N-bearing sequences. The
 * alternative -- rejecting it, as the amino-acid alphabet rejects 'X' -- would
 * silently drop every read that carries an ambiguous base.
 *
 * Its cost is the radix: five residues make the packing base 6 rather than 5,
 * and a wide context of order k costs 6^k. At the default order 6 that is
 * 46,656 contexts, which is a quarter of what the amino-acid model already
 * carries.
 */
#include "beam/common.h"
#include "beam/model.h"
#include "beam/alphabet.h"

/* ── Descriptor tables ───────────────────────────────────────── */

const uint8_t beam_nt_char_to_idx[256] = {
    /* Both cases map to the same index. Nucleotide data arrives lower case
     * often enough -- FASTA, several AIRR exports -- that rejecting it would
     * turn a whole corpus into skipped sequences with nothing to explain why,
     * and case carries no meaning to fold away. This is input normalisation
     * only: 'acgt' and 'ACGT' are one sequence, not two.
     *
     * The amino-acid table deliberately does not do this. */
    [0   ... 'A' - 1] = BEAM_RESIDUE_INVALID,
    ['A'] = 0,
    ['B'] = BEAM_RESIDUE_INVALID,
    ['C'] = 1,
    ['D' ... 'F'] = BEAM_RESIDUE_INVALID,
    ['G'] = 2,
    ['H' ... 'M'] = BEAM_RESIDUE_INVALID,
    ['N'] = 4,
    ['O' ... 'S'] = BEAM_RESIDUE_INVALID,
    ['T'] = 3,
    ['U' ... 'a' - 1] = BEAM_RESIDUE_INVALID,
    ['a'] = 0,
    ['b'] = BEAM_RESIDUE_INVALID,
    ['c'] = 1,
    ['d' ... 'f'] = BEAM_RESIDUE_INVALID,
    ['g'] = 2,
    ['h' ... 'm'] = BEAM_RESIDUE_INVALID,
    ['n'] = 4,
    ['o' ... 's'] = BEAM_RESIDUE_INVALID,
    ['t'] = 3,
    ['u' ... 255] = BEAM_RESIDUE_INVALID,
    /* Every range is spelled out rather than defaulted-then-overridden, so
     * -Woverride-init stays on and a stray duplicate would be caught.
     *
     * U falls in a rejected range on purpose. Reading it as T would be the
     * model deciding, on the caller's behalf, that their input is RNA. The
     * remaining IUPAC codes -- R, Y, S, W, K, M, B, D, H, V -- are rejected
     * for the same reason: N says "this base is unknown", which the model can
     * represent, while R says "this base is A or G", which it cannot. */
};

static const char nt_idx_to_char[BEAM_NT_RADIX + 1] = "ACGTN.";

const BeamAlphabetInfo beam_alphabet_nt = {
    .id                 = BEAM_ALPHABET_NT,
    .name               = "nt",
    .residues           = BEAM_NT_RESIDUES,
    .radix              = BEAM_NT_RADIX,
    .none               = BEAM_NT_NONE,
    /*
     * Order 6 is two codons, and it is the default because it is the largest
     * context that leaves every dynamic program cheaper than the amino-acid
     * model already is: 6^6 = 46,656 states against 21^4 = 194,481.
     *
     * Matching the amino-acid model's context SPAN would need order 12, since
     * an amino acid is a compressed codon -- 21 values where three
     * nucleotides need 216. That is 6^12 states and not buildable. Order 9,
     * three codons, is reachable for a caller who asks: it is a 10,077,696
     * state trellis, roughly 114 times the finalize work and 766 MB of
     * partition-function buffers per thread.
     */
    .default_order      = 6,
    .max_order          = 9,
    /* Three bases to the amino acid, against the amino-acid default of 27. */
    .default_max_length = 81,
    /* Resolve every depth. The cap is a concession to the amino-acid wide
     * table's size at radix 21, and radix 6 does not need it. */
    .char_to_idx        = beam_nt_char_to_idx,
    .idx_to_char        = nt_idx_to_char,
};
