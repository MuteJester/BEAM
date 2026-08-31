/**
 * @file test_alphabet.c
 * @brief Tests for the alphabet descriptors and registry.
 *
 * The expected mappings below are written out by hand rather than derived
 * from the descriptors. Deriving them would make every test circular: a
 * transposed pair in the table under test would appear in the expectation
 * too, and the suite would agree with the bug.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "beam/common.h"
#include "beam/model.h"
#include "beam/alphabet.h"

static int pass_count = 0, fail_count = 0;

#include "test_utils.h"

/* ── Independent expectations ────────────────────────────────── */

static const char AA_SYMS[] = "ACDEFGHIKLMNPQRSTVWY";
static const char NT_UPPER[] = "ACGTN";
static const char NT_LOWER[] = "acgtn";

/** The index `c` must map to, or BEAM_RESIDUE_INVALID. */
static unsigned expect_aa(int c) {
    for (unsigned i = 0; i < 20u; ++i)
        if (AA_SYMS[i] == c) return i;
    return BEAM_RESIDUE_INVALID;
}

static unsigned expect_nt(int c) {
    for (unsigned i = 0; i < 5u; ++i)
        if (NT_UPPER[i] == c || NT_LOWER[i] == c) return i;
    return BEAM_RESIDUE_INVALID;
}

typedef unsigned (*ExpectFn)(int);

/* ── Descriptor integrity ────────────────────────────────────── */

static void test_descriptors_are_internally_consistent(void) {
    for (unsigned k = 0; k < BEAM_ALPHABET_COUNT; ++k) {
        const BeamAlphabetInfo *a = beam_alphabet_info((BeamAlphabet)k);
        ASSERT_MSG(a != NULL, "every alphabet id must resolve");
        ASSERT_MSG((unsigned)a->id == k, "descriptor must report its own id");
        ASSERT_MSG(a->name && a->name[0], "every alphabet needs a name");
        ASSERT_MSG(a->char_to_idx && a->idx_to_char, "tables must be present");

        ASSERT_MSG(a->radix == a->residues + 1u,
                   "radix is the residues plus exactly one sentinel");
        ASSERT_MSG(a->none == a->residues,
                   "the sentinel sits just past the last residue");
        ASSERT_MSG(a->residues <= BEAM_MAX_RESIDUES,
                   "BEAM_MAX_RESIDUES must cover the arrays it sizes");
        ASSERT_MSG(a->radix <= BEAM_MAX_RADIX,
                   "BEAM_MAX_RADIX must cover the arrays it sizes");

        ASSERT_MSG(strlen(a->idx_to_char) == a->radix,
                   "idx_to_char holds exactly radix characters");
        ASSERT_MSG(a->idx_to_char[a->none] == '.',
                   "the sentinel displays as '.'");
    }
    PASS();
}

static void test_defaults_sit_inside_their_own_ceilings(void) {
    for (unsigned k = 0; k < BEAM_ALPHABET_COUNT; ++k) {
        const BeamAlphabetInfo *a = beam_alphabet_info((BeamAlphabet)k);

        ASSERT_MSG(a->default_order >= 2 && a->default_order <= a->max_order,
                   "the default order must be a legal order");
        /* The state is max(order, 3), so an alphabet whose ceiling is below 3
         * would build a state too narrow to feed the order-3 mid expert. */
        ASSERT_MSG(a->max_order >= 3, "the state can never be narrower than 3");

        ASSERT_MSG(a->default_max_length >= 2,
                   "depth_cap must be at least 1 and below max_length");
        ASSERT_MSG(a->default_max_length <= BEAM_MAX_LENGTH_CEILING,
                   "max_length is a uint8_t and cannot exceed its ceiling");

        ASSERT_MSG(a->default_depth_cap >= 1, "a depth cap of 0 is not legal");
    }
    PASS();
}

/*
 * The property that makes a per-alphabet order ceiling necessary rather than
 * tidy. beam_ipow_u32 is a uint32_t multiply loop with no overflow check, so
 * a ceiling that lets A^order exceed UINT32_MAX does not fail -- it wraps,
 * and the model allocates a table sized by the wrapped value.
 */
static void test_the_order_ceiling_keeps_context_counts_in_range(void) {
    for (unsigned k = 0; k < BEAM_ALPHABET_COUNT; ++k) {
        const BeamAlphabetInfo *a = beam_alphabet_info((BeamAlphabet)k);

        /* The state order is max(order, 3), and it is the widest thing
         * exponentiated, so it is what must stay in range. */
        unsigned state_order = a->max_order > 3u ? a->max_order : 3u;

        uint64_t wide = 1, wrapped = 1;
        for (unsigned i = 0; i < state_order; ++i) {
            wide *= a->radix;
            wrapped = (uint32_t)(wrapped * a->radix);
        }
        ASSERT_MSG(wide <= UINT32_MAX,
                   "radix^max_order must fit the uint32 it is stored in");
        ASSERT_MSG(wide == wrapped,
                   "a uint32 multiply loop must reach the same value");
    }

    /*
     * And the reason it cannot be one global ceiling: the nucleotide ceiling
     * applied to the amino-acid radix overflows by six orders of magnitude.
     * If this ever stops being true, a shared BEAM_MAX_ORDER became safe and
     * the per-alphabet field could go.
     */
    const BeamAlphabetInfo *aa = beam_alphabet_info(BEAM_ALPHABET_AA);
    const BeamAlphabetInfo *nt = beam_alphabet_info(BEAM_ALPHABET_NT);
    uint64_t cross = 1;
    for (unsigned i = 0; i < nt->max_order; ++i) cross *= aa->radix;
    ASSERT_MSG(cross > UINT32_MAX,
               "the nucleotide ceiling must be unsafe at the amino-acid "
               "radix, which is why the ceiling is per-alphabet");
    PASS();
}

/* ── Character mapping ───────────────────────────────────────── */

/* Every one of the 256 byte values, not a sampled few: an accidental extra
 * designated initialiser is invisible to any test that only checks the
 * characters it expects to be valid. */
static void check_all_bytes(const BeamAlphabetInfo *a, ExpectFn expect,
                            int *ok, const char **why) {
    for (int c = 0; c < 256; ++c) {
        unsigned want = expect(c);
        unsigned got  = a->char_to_idx[c];
        if (got != want) {
            *ok = 0;
            *why = "a byte maps to the wrong residue index";
            return;
        }
        if (want != BEAM_RESIDUE_INVALID && want >= a->residues) {
            *ok = 0;
            *why = "a residue index is outside the alphabet";
            return;
        }
    }
}

static void test_every_byte_maps_exactly_as_specified(void) {
    int ok = 1;
    const char *why = "";

    check_all_bytes(beam_alphabet_info(BEAM_ALPHABET_AA), expect_aa, &ok, &why);
    ASSERT_MSG(ok, why);
    check_all_bytes(beam_alphabet_info(BEAM_ALPHABET_NT), expect_nt, &ok, &why);
    ASSERT_MSG(ok, why);
    PASS();
}

/*
 * No byte may encode to the sentinel's index. The sentinel pads contexts at
 * the termini, so a character sharing its index would make "no residue here"
 * and "this residue" the same context digit, silently corrupting every table
 * lookup that spans an end.
 */
static void test_no_character_encodes_to_the_sentinel(void) {
    for (unsigned k = 0; k < BEAM_ALPHABET_COUNT; ++k) {
        const BeamAlphabetInfo *a = beam_alphabet_info((BeamAlphabet)k);
        for (int c = 0; c < 256; ++c)
            ASSERT_MSG(a->char_to_idx[c] != a->none,
                       "a character encodes to the sentinel index");
    }
    PASS();
}

/* The valid indices must be exactly 0..residues-1, each reachable. A gap
 * would leave a table row that no sequence can ever address. */
static void test_residue_indices_are_dense(void) {
    for (unsigned k = 0; k < BEAM_ALPHABET_COUNT; ++k) {
        const BeamAlphabetInfo *a = beam_alphabet_info((BeamAlphabet)k);
        int seen[BEAM_MAX_RESIDUES] = {0};

        for (int c = 0; c < 256; ++c) {
            unsigned ix = a->char_to_idx[c];
            if (ix != BEAM_RESIDUE_INVALID) seen[ix] = 1;
        }
        for (unsigned i = 0; i < a->residues; ++i)
            ASSERT_MSG(seen[i], "a residue index is unreachable from any byte");
    }
    PASS();
}

static void test_index_and_character_round_trip(void) {
    for (unsigned k = 0; k < BEAM_ALPHABET_COUNT; ++k) {
        const BeamAlphabetInfo *a = beam_alphabet_info((BeamAlphabet)k);
        for (unsigned i = 0; i < a->residues; ++i) {
            unsigned char ch = (unsigned char)a->idx_to_char[i];
            ASSERT_MSG(a->char_to_idx[ch] == i,
                       "index -> character -> index must be the identity");
        }
    }
    PASS();
}

/* ── Per-alphabet behaviour ──────────────────────────────────── */

static void test_nucleotides_fold_case_and_amino_acids_do_not(void) {
    const BeamAlphabetInfo *nt = beam_alphabet_info(BEAM_ALPHABET_NT);
    for (unsigned i = 0; i < 5u; ++i)
        ASSERT_MSG(nt->char_to_idx[(unsigned char)NT_LOWER[i]]
                   == nt->char_to_idx[(unsigned char)NT_UPPER[i]],
                   "a lower-case base must encode as its upper-case self");

    /* Folding the amino acids too would silently widen what the model
     * accepts, so the asymmetry is deliberate and checked. */
    const BeamAlphabetInfo *aa = beam_alphabet_info(BEAM_ALPHABET_AA);
    for (int c = 'a'; c <= 'z'; ++c)
        ASSERT_MSG(aa->char_to_idx[c] == BEAM_RESIDUE_INVALID,
                   "the amino-acid alphabet must stay upper case only");
    PASS();
}

static void test_ambiguity_beyond_N_is_still_rejected(void) {
    const BeamAlphabetInfo *nt = beam_alphabet_info(BEAM_ALPHABET_NT);
    /* The other IUPAC codes, and U. Accepting N is a decision about missing
     * data; accepting R or Y would be the model guessing which base. */
    const char *rejected = "RYSWKMBDHVUryswkmbdhvu";
    for (const char *p = rejected; *p; ++p)
        ASSERT_MSG(nt->char_to_idx[(unsigned char)*p] == BEAM_RESIDUE_INVALID,
                   "only ACGTN may encode; other IUPAC codes must not");
    PASS();
}

/*
 * A C G T and N are all amino acids too -- Ala, Cys, Gly, Thr, Asn -- and they
 * encode to entirely different indices in the two alphabets. This is why
 * there is no alphabet auto-detection: "ACGT" is a perfectly good peptide,
 * and a heuristic would silently build the wrong model from it.
 */
static void test_the_alphabets_overlap_with_different_indices(void) {
    const BeamAlphabetInfo *aa = beam_alphabet_info(BEAM_ALPHABET_AA);
    const BeamAlphabetInfo *nt = beam_alphabet_info(BEAM_ALPHABET_NT);

    int differ = 0;
    for (unsigned i = 0; i < 5u; ++i) {
        unsigned char ch = (unsigned char)NT_UPPER[i];
        ASSERT_MSG(aa->char_to_idx[ch] != BEAM_RESIDUE_INVALID,
                   "every base is also a valid amino acid");
        if (aa->char_to_idx[ch] != nt->char_to_idx[ch]) differ = 1;
    }
    ASSERT_MSG(differ,
               "the same byte must mean different residues in the two "
               "alphabets, which is what makes detection unsafe");
    PASS();
}

/* ── Equivalence with the table this replaces ────────────────── */

/*
 * The load-bearing test of this change. The kernels still read the legacy
 * beam_char_to_idx, which reports 0 for an unknown byte and needs
 * beam_is_residue() alongside it to distinguish that from 'A'. The descriptor
 * table folds both into one lookup. Nothing may score differently when the
 * kernels switch over, so the two must agree on all 256 bytes.
 */
static void test_the_descriptor_table_matches_the_legacy_pair(void) {
    const BeamAlphabetInfo *aa = beam_alphabet_info(BEAM_ALPHABET_AA);

    for (int c = 0; c < 256; ++c) {
        unsigned ix = aa->char_to_idx[c];
        if (beam_is_residue((unsigned char)c)) {
            ASSERT_MSG(ix != BEAM_RESIDUE_INVALID,
                       "a legacy residue must stay valid");
            ASSERT_MSG(ix == beam_char_to_idx[c],
                       "a legacy residue must keep its index");
        } else {
            ASSERT_MSG(ix == BEAM_RESIDUE_INVALID,
                       "a byte the legacy pair rejects must stay rejected");
        }
    }
    PASS();
}

static void test_the_descriptor_decode_matches_the_legacy_decode(void) {
    const BeamAlphabetInfo *aa = beam_alphabet_info(BEAM_ALPHABET_AA);
    for (unsigned i = 0; i < BEAM_ALPHABET; ++i)
        ASSERT_MSG(aa->idx_to_char[i] == beam_idx_to_char[i],
                   "the decode table must be unchanged, sentinel included");
    PASS();
}

/* ── Registry ────────────────────────────────────────────────── */

static void test_the_registry_resolves_ids_and_names(void) {
    ASSERT_MSG(beam_alphabet_info(BEAM_ALPHABET_AA) == &beam_alphabet_aa,
               "the amino-acid id must resolve to its descriptor");
    ASSERT_MSG(beam_alphabet_info(BEAM_ALPHABET_NT) == &beam_alphabet_nt,
               "the nucleotide id must resolve to its descriptor");

    ASSERT_MSG(beam_alphabet_by_name("aa") == &beam_alphabet_aa,
               "\"aa\" must resolve");
    ASSERT_MSG(beam_alphabet_by_name("nt") == &beam_alphabet_nt,
               "\"nt\" must resolve");

    for (unsigned k = 0; k < BEAM_ALPHABET_COUNT; ++k) {
        const BeamAlphabetInfo *a = beam_alphabet_info((BeamAlphabet)k);
        ASSERT_MSG(beam_alphabet_by_name(a->name) == a,
                   "every descriptor must be findable by its own name");
    }
    PASS();
}

static void test_the_registry_refuses_what_it_does_not_know(void) {
    ASSERT_MSG(beam_alphabet_info((BeamAlphabet)BEAM_ALPHABET_COUNT) == NULL,
               "an id past the last alphabet must not resolve");
    ASSERT_MSG(beam_alphabet_info((BeamAlphabet)9999) == NULL,
               "a wild id must not resolve");

    ASSERT_MSG(beam_alphabet_by_name(NULL) == NULL, "NULL is not a name");
    ASSERT_MSG(beam_alphabet_by_name("") == NULL, "empty is not a name");
    ASSERT_MSG(beam_alphabet_by_name("AA") == NULL, "lookup is case sensitive");
    ASSERT_MSG(beam_alphabet_by_name("aa ") == NULL, "no trimming");
    ASSERT_MSG(beam_alphabet_by_name("a") == NULL, "no prefix matching");
    ASSERT_MSG(beam_alphabet_by_name("protein") == NULL, "no aliases");
    PASS();
}

/* Names cross into Python and into saved-model diagnostics, so they must be
 * distinct and stable. */
static void test_alphabet_names_are_distinct(void) {
    for (unsigned i = 0; i < BEAM_ALPHABET_COUNT; ++i)
        for (unsigned j = i + 1; j < BEAM_ALPHABET_COUNT; ++j)
            ASSERT_MSG(strcmp(beam_alphabet_info((BeamAlphabet)i)->name,
                              beam_alphabet_info((BeamAlphabet)j)->name) != 0,
                       "two alphabets share a name");
    PASS();
}

/* ── Runner ──────────────────────────────────────────────────── */

int main(void) {
    printf("\nBEAM Unit Tests - Alphabets\n");
    printf("==============================================================\n\n");

    printf("[descriptor]\n");
    RUN_TEST(test_descriptors_are_internally_consistent);
    RUN_TEST(test_defaults_sit_inside_their_own_ceilings);
    RUN_TEST(test_the_order_ceiling_keeps_context_counts_in_range);

    printf("\n[mapping]\n");
    RUN_TEST(test_every_byte_maps_exactly_as_specified);
    RUN_TEST(test_no_character_encodes_to_the_sentinel);
    RUN_TEST(test_residue_indices_are_dense);
    RUN_TEST(test_index_and_character_round_trip);

    printf("\n[per-alphabet]\n");
    RUN_TEST(test_nucleotides_fold_case_and_amino_acids_do_not);
    RUN_TEST(test_ambiguity_beyond_N_is_still_rejected);
    RUN_TEST(test_the_alphabets_overlap_with_different_indices);

    printf("\n[equivalence with the legacy tables]\n");
    RUN_TEST(test_the_descriptor_table_matches_the_legacy_pair);
    RUN_TEST(test_the_descriptor_decode_matches_the_legacy_decode);

    printf("\n[registry]\n");
    RUN_TEST(test_the_registry_resolves_ids_and_names);
    RUN_TEST(test_the_registry_refuses_what_it_does_not_know);
    RUN_TEST(test_alphabet_names_are_distinct);

    printf("\n==============================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
