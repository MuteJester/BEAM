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
#include <math.h>

#include "beam/common.h"
#include "beam/model.h"
#include "beam/crc32c.h"
#include "beam/alphabet.h"
#include "beam/model_build.h"   /* beam_config_valid */

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
        /* BEAM_MAX_ORDER sizes every order-indexed array -- `slot[]` in the
         * context packers, the prior's strengths `alpha[]`, its lower-level
         * tables and radix powers, and finalize's per-level `cnt[]`, `tot[]`
         * and `prob[]`. It is only a safe bound while no alphabet lets a
         * caller ask for more, and `indexing.h` would silently CLAMP rather
         * than fail if one did. */
        ASSERT_MSG(a->max_order <= BEAM_MAX_ORDER,
                   "an alphabet cannot allow an order past the bound that "
                   "sizes the arrays indexed by it");

        ASSERT_MSG(a->default_max_length >= 2,
                   "a model of one residue is not a model");
        ASSERT_MSG(a->default_max_length <= BEAM_MAX_LENGTH_CEILING,
                   "max_length is a uint8_t and cannot exceed its ceiling");

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

/*
 * A sparse table addresses its rows with a uint32, and the row counts are
 * cast to one at construction. `beam_model_new` guards that cast, but the
 * guard cannot fire at today's ceilings -- the widest count reaches 2.57e9
 * against a limit of 4.29e9. This pins the headroom rather than the guard, so
 * that raising BEAM_MAX_ORDER or BEAM_MAX_LENGTH_CEILING far enough breaks a
 * test and makes someone decide, instead of quietly relying on a factor of
 * less than two.
 */
static void test_the_widest_table_still_fits_a_uint32_row_index(void) {
    const uint32_t max_buckets = BEAM_MAX_LENGTH_CEILING;   /* depth_cap + 1 */
    for (unsigned k = 0; k < BEAM_ALPHABET_COUNT; ++k) {
        const BeamAlphabetInfo *a = beam_alphabet_info((BeamAlphabet)k);

        uint64_t n_ctx = 1;
        for (unsigned i = 0; i < a->max_order; ++i) n_ctx *= a->radix;
        const uint64_t rows_wide = (uint64_t)max_buckets * n_ctx;

        const uint64_t rows_len = (uint64_t)(BEAM_MAX_LENGTH_CEILING + 1u)
                                * BEAM_MAX_LENGTH_CEILING
                                * (uint64_t)a->radix * a->radix;

        ASSERT_MSG(rows_wide <= UINT32_MAX,
                   "the wide table's row count must fit a uint32 at every "
                   "configuration the ceilings allow");
        ASSERT_MSG(rows_len <= UINT32_MAX,
                   "and so must the length-conditioned table's");
    }
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

/* ── Configuration ───────────────────────────────────────────── */

/*
 * Adding a field to BeamModelConfig must not have moved a single default.
 * The expected values are written out rather than read from the descriptor
 * or from BEAM_DEFAULT_ALPHA, so that editing either cannot quietly redefine
 * what "unchanged" means.
 */
static void test_the_amino_acid_defaults_are_unchanged(void) {
    BeamModelConfig c;
    beam_model_config_default(&c);

    ASSERT_MSG(c.alphabet == BEAM_ALPHABET_AA, "the default is amino acid");
    ASSERT_MSG(c.order == 4, "order");
    ASSERT_MSG(c.max_length == 27, "max_length");
    ASSERT_MSG(c.weight_forward == 0.5 && c.weight_backward == 0.5,
               "anchor weights");
    /* Every slot, not only 0..order: a caller who raises the order after
     * defaulting must not find an unset strength at the new top level. */
    for (unsigned k = 0; k <= BEAM_MAX_ORDER; ++k)
        ASSERT_MSG(c.alpha[k] == 10.0, "every prior level starts at 10");
    ASSERT_MSG(c.alpha_auto && c.alpha_length_auto,
               "and by default the counts estimate the strengths");
    ASSERT_MSG(c.alpha_length == 10.0, "the length prior starts at 10");
    /* Sparse, because a dense table under the prior materialises every row
     * there is, which is gigabytes at order 4. */
    ASSERT_MSG(c.sparse, "sparse storage is the default");
    PASS();
}

static void test_per_alphabet_defaults_come_from_the_descriptor(void) {
    BeamModelConfig aa, nt;
    beam_model_config_default_for(BEAM_ALPHABET_AA, &aa);
    beam_model_config_default_for(BEAM_ALPHABET_NT, &nt);

    ASSERT_MSG(aa.order == 4 && aa.max_length == 27,
               "the amino-acid defaults");
    ASSERT_MSG(nt.order == 6 && nt.max_length == 81,
               "two codons of context over three bases per residue");

    /* What the alphabet must NOT decide. These were measured on amino acids
     * and carrying them across is the deliberate choice; re-tuning them needs
     * its own measurement. */
    ASSERT_MSG(nt.weight_forward == aa.weight_forward
               && nt.weight_backward == aa.weight_backward,
               "anchor weights are alphabet-independent");
    /* The prior is a starting point that `fit_alpha` replaces for a real
     * corpus, so it is the same for both alphabets -- and written out as a
     * literal for each, since comparing the two to each other would pass if
     * both moved together. */
    for (unsigned k = 0; k <= BEAM_MAX_ORDER; ++k)
        ASSERT_MSG(aa.alpha[k] == 10.0 && nt.alpha[k] == 10.0,
                   "every prior level starts at 10 in both alphabets");
    ASSERT_MSG(aa.alpha_length == 10.0 && nt.alpha_length == 10.0,
               "the length prior starts at 10 in both alphabets");
    ASSERT_MSG(aa.sparse && nt.sparse, "both alphabets default to sparse");
    PASS();
}

/* The reason BEAM_ALPHABET_AA is zero. A caller who zeroed a config has
 * always been asking for amino acids, and must keep getting them. */
static void test_a_zeroed_config_still_means_amino_acid(void) {
    BeamModelConfig c;
    memset(&c, 0, sizeof c);
    ASSERT_MSG(c.alphabet == BEAM_ALPHABET_AA,
               "a zeroed config must mean amino acid");
    PASS();
}

static void test_defaults_refuse_an_unknown_alphabet(void) {
    BeamModelConfig c;
    beam_model_config_default(&c);
    BeamModelConfig before = c;
    beam_model_config_default_for((BeamAlphabet)9999, &c);
    ASSERT_MSG(memcmp(&c, &before, sizeof c) == 0,
               "an unknown alphabet must leave the config untouched");
    PASS();
}

/* ── Validation ──────────────────────────────────────────────── */

static BeamModelConfig valid_for(BeamAlphabet a) {
    BeamModelConfig c;
    beam_model_config_default_for(a, &c);
    /* The nucleotide default asks to resolve every depth, which only
     * beam_model_new clamps; beam_config_valid sees the raw value. */
    return c;
}

static void test_the_order_bound_is_the_alphabet_s_own(void) {
    BeamModelConfig aa = valid_for(BEAM_ALPHABET_AA);
    ASSERT_MSG(beam_config_valid(&aa), "the amino-acid default must be valid");

    aa.order = 5;
    ASSERT_MSG(!beam_config_valid(&aa),
               "amino acids stop at order 4: 21^5 per depth is unaffordable");

    /* The same order the amino acids refuse is ordinary for nucleotides, and
     * this is the whole point of a per-alphabet ceiling. */
    BeamModelConfig nt = valid_for(BEAM_ALPHABET_NT);
    nt.order = 5;
    ASSERT_MSG(beam_config_valid(&nt), "order 5 is fine at radix 6");
    nt.order = 9;
    ASSERT_MSG(beam_config_valid(&nt), "9 is the nucleotide ceiling");
    nt.order = 10;
    ASSERT_MSG(!beam_config_valid(&nt), "and 10 is past it");
    PASS();
}

static void test_validation_refuses_an_unknown_alphabet(void) {
    BeamModelConfig c = valid_for(BEAM_ALPHABET_AA);
    c.alphabet = (BeamAlphabet)7;
    ASSERT_MSG(!beam_config_valid(&c),
               "a config whose alphabet cannot be resolved is not valid");
    PASS();
}

static void test_max_length_is_free_up_to_the_ceiling(void) {
    BeamModelConfig c = valid_for(BEAM_ALPHABET_AA);
    /* Longer than the old fixed limit of 27, which is the point: nucleotide
     * sequences are three times the length of the amino acids they encode. */
    c.max_length = 100;
    ASSERT_MSG(beam_config_valid(&c), "max_length is a free parameter now");

    c.max_length = 0;
    ASSERT_MSG(!beam_config_valid(&c), "a zero-length model is not a model");
    PASS();
}

/**
 * Every way a config can be rejected says which rule rejected it.
 *
 * The refusal used to be assembled by the caller from a fixed list -- order,
 * max_length, depth_cap, the coefficient sum, the anchor weights -- each
 * printed with its value and its range. A rule not on that list could not be
 * named, and at the time two were not. Tripping one of them produced five
 * fields, every one of them inside its stated range, and no reason. It cost
 * two build cycles in one afternoon, twice, because a message that lists only
 * valid things reads as though nothing is wrong.
 *
 * So this pins the property rather than the wording: for each way to make a
 * config invalid, the reason must contain a token that belongs to THAT rule
 * and could not have come from any other. A message naming everything would
 * pass a test that only checked the message was non-empty. The prior's
 * strengths are held to the index: a refusal of alpha[2] that named alpha[0]
 * would send the caller to the wrong level.
 */
static void test_every_config_refusal_names_its_own_rule(void) {
    struct { const char *what; const char *token; BeamModelConfig c; } cases[12];
    uint32_t n = 0;

    cases[n].what = "order past the alphabet ceiling"; cases[n].token = "order=";
    cases[n].c = valid_for(BEAM_ALPHABET_AA); cases[n].c.order = 9; ++n;

    cases[n].what = "a zero max_length"; cases[n].token = "max_length=";
    cases[n].c = valid_for(BEAM_ALPHABET_AA); cases[n].c.max_length = 0; ++n;

    cases[n].what = "a negative anchor weight"; cases[n].token = "anchor weights";
    cases[n].c = valid_for(BEAM_ALPHABET_AA);
    cases[n].c.weight_backward = -0.5; ++n;

    /* A zero strength would let an unobserved residue reach probability zero
     * at that level, bringing the closed world back through one row; a
     * negative or non-finite one is not a Dirichlet at all. The cases run
     * from the bottom level to the order itself. Each FIXES its strengths:
     * the default estimates them from the counts, and a value about to be
     * overwritten by the estimate is not the caller's to get wrong. */
    cases[n].what = "a zero strength at level 0"; cases[n].token = "alpha[0]=";
    cases[n].c = valid_for(BEAM_ALPHABET_AA); cases[n].c.alpha_auto = false;
    cases[n].c.alpha[0] = 0.0; ++n;

    cases[n].what = "an infinite strength at level 1"; cases[n].token = "alpha[1]=";
    cases[n].c = valid_for(BEAM_ALPHABET_AA); cases[n].c.alpha_auto = false;
    cases[n].c.alpha[1] = INFINITY; ++n;

    cases[n].what = "a NaN strength at level 2"; cases[n].token = "alpha[2]=";
    cases[n].c = valid_for(BEAM_ALPHABET_AA); cases[n].c.alpha_auto = false;
    cases[n].c.alpha[2] = NAN; ++n;

    cases[n].what = "a negative strength at the order"; cases[n].token = "alpha[4]=";
    cases[n].c = valid_for(BEAM_ALPHABET_AA); cases[n].c.alpha_auto = false;
    cases[n].c.alpha[4] = -1.0; ++n;

    /* The nucleotide default order reaches levels the amino acids never
     * have. */
    cases[n].what = "a zero strength at nucleotide level 6";
    cases[n].token = "alpha[6]=";
    cases[n].c = valid_for(BEAM_ALPHABET_NT); cases[n].c.alpha_auto = false;
    cases[n].c.alpha[6] = 0.0; ++n;

    cases[n].what = "a zero length strength"; cases[n].token = "alpha_length=";
    cases[n].c = valid_for(BEAM_ALPHABET_AA); cases[n].c.alpha_length_auto = false;
    cases[n].c.alpha_length = 0.0; ++n;

    cases[n].what = "an infinite length strength"; cases[n].token = "alpha_length=";
    cases[n].c = valid_for(BEAM_ALPHABET_AA); cases[n].c.alpha_length_auto = false;
    cases[n].c.alpha_length = INFINITY; ++n;

    cases[n].what = "a NaN length strength"; cases[n].token = "alpha_length=";
    cases[n].c = valid_for(BEAM_ALPHABET_AA); cases[n].c.alpha_length_auto = false;
    cases[n].c.alpha_length = NAN; ++n;

    static char msg[512];
    char why[BEAM_CONFIG_REASON_MAX];
    for (uint32_t i = 0; i < n; ++i) {
        const char *r = beam_config_invalid_reason(&cases[i].c, why, sizeof why);
        if (!r) {
            snprintf(msg, sizeof msg, "%s was accepted", cases[i].what);
            ASSERT_MSG(false, msg);
        }
        if (!strstr(r, cases[i].token)) {
            snprintf(msg, sizeof msg,
                     "%s was refused without naming its rule: expected \"%s\" "
                     "in \"%s\"", cases[i].what, cases[i].token, r);
            ASSERT_MSG(false, msg);
        }
    }
    /* And a valid config gets no sentence at all, so the reason is a reason
     * and not something every config carries. */
    BeamModelConfig ok = valid_for(BEAM_ALPHABET_AA);
    ASSERT_MSG(beam_config_invalid_reason(&ok, why, sizeof why) == NULL,
               "a valid config produced a rejection reason");
    PASS();
}

/*
 * Strengths past the order are documented as ignored, and must be: a caller
 * who lowers the order of a config built for a higher one leaves stale
 * values in the upper slots, and refusing the config for a level the model
 * does not have would be refusing it for nothing.
 */
static void test_prior_strengths_past_the_order_are_ignored(void) {
    BeamModelConfig c = valid_for(BEAM_ALPHABET_AA);
    c.alpha_auto = false;
    c.order = 2;
    c.alpha[3] = -1.0;
    c.alpha[BEAM_MAX_ORDER] = NAN;
    ASSERT_MSG(beam_config_valid(&c),
               "an unusable strength above the order must not be refused");

    c.alpha[2] = -1.0;
    ASSERT_MSG(!beam_config_valid(&c),
               "and the same value AT the order must be");

    /* Left to the data, the same slot holds nothing the caller chose: the
     * estimate overwrites it at finalize, so it is not refused. */
    c.alpha_auto = true;
    ASSERT_MSG(beam_config_valid(&c),
               "a strength the counts will estimate is not validated");
    PASS();
}

/* ── Construction ────────────────────────────────────────────── */

static void test_a_model_reports_the_alphabet_it_was_built_with(void) {
    BeamModelConfig c = valid_for(BEAM_ALPHABET_AA);
    BeamModel *m = beam_model_new(&c);
    ASSERT_MSG(m != NULL, "the amino-acid model must build");

    ASSERT_MSG(beam_model_alphabet(m) == BEAM_ALPHABET_AA, "reports aa");
    ASSERT_MSG(beam_model_n_residues(m) == BEAM_AA_RESIDUES, "20 residues");
    ASSERT_MSG(beam_model_radix(m) == BEAM_AA_RADIX, "radix 21");

    /* The dimensions must be exactly what the preprocessor produced before
     * they were derived from the descriptor. */
    ASSERT_MSG(beam_model_n_states(m) == 21u * 21u * 21u * 21u,
               "21^4 states at order 4");
    beam_model_destroy(m);
    PASS();
}

static void test_construction_refuses_an_unknown_alphabet(void) {
    BeamModelConfig c = valid_for(BEAM_ALPHABET_AA);
    c.alphabet = (BeamAlphabet)42;
    BeamModel *m = beam_model_new(&c);
    ASSERT_MSG(m == NULL, "an unresolvable alphabet must not build");
    PASS();
}

/*
 * NOTE: add, subtract and posterior now include the alphabet in their shape
 * check, because two models can agree on order, length and depth and still
 * have tables of different widths -- which would walk the operand off the end
 * of its allocation. That check has no test here on purpose: exercising it
 * needs two models of different alphabets, and nucleotide models cannot be
 * built until the kernels read the alphabet. It is recorded as a required
 * test of the bite that makes them buildable, rather than covered by a test
 * that asserts something easier and looks like it covered it.
 */

/* A model built at a length the old fixed ceiling forbade must build, score,
 * and agree with a shorter model on a sequence they both accept. */
static void test_a_model_longer_than_the_old_ceiling_builds_and_scores(void) {
    const char *seqs[] = {"CASSLGQAYEQYF", "CASSLGQAYEQFF", "CASRDRAYNTIYF"};
    const double ab[] = {3.0, 1.0, 2.0};

    BeamModelConfig lng = valid_for(BEAM_ALPHABET_AA);
    lng.max_length = 60;
    /*
     * Order 2, and that is not incidental. Finalize sweeps the partition
     * function over every state of every layer for every length, so its cost
     * grows with the state space and faster than linearly with max_length:
     * at the default order 4 this one model takes tens of seconds, at order
     * 2 a fraction of one. The claim here is about the LENGTH ceiling, which
     * the order does not touch.
     */
    lng.order = 2;
    BeamModel *m = beam_model_new(&lng);
    ASSERT_MSG(m != NULL, "a max_length of 60 must build now");
    ASSERT_MSG(beam_model_config(m)->max_length == 60, "and keep its length");

    ASSERT_MSG(beam_model_observe(m, seqs, ab, 3) == BEAM_OK, "observe");
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalize");

    const double lp = beam_model_log_pseq(m, "CASSLGQAYEQYF");
    ASSERT_MSG(lp < 0.0 && isfinite(lp),
               "an observed sequence must score finitely at length 60");

    /*
     * The length boundary, which is also what exercises the scratch buffers
     * at capacity. Every scoring path encodes into a stack array, and while
     * those were sized by the old fixed 27 and max_length could reach 255, a
     * sequence this long wrote off the end of one. A build that still had an
     * undersized buffer would smash its stack here rather than fail a
     * comparison.
     *
     * Under the hierarchical prior `log_pseq` is -INFINITY only for what the
     * model cannot read, so it now tells the boundary apart by itself: sixty
     * A's were never observed and must still score, and sixty-one must not.
     * `explain` is kept alongside, because it reports the length it read and
     * refuses with an error rather than a value.
     */
    char at_limit[61], past_limit[62];
    memset(at_limit, 'A', 60);    at_limit[60] = '\0';
    memset(past_limit, 'A', 61);  past_limit[61] = '\0';

    const double lp_at = beam_model_log_pseq(m, at_limit);
    ASSERT_MSG(isfinite(lp_at) && lp_at < lp,
               "an unseen sequence of exactly max_length must score finitely, "
               "and below an observed one");
    ASSERT_MSG(beam_model_log_pseq(m, past_limit) == -INFINITY,
               "one residue past max_length cannot be read");

    double fwd[BEAM_MAX_LENGTH_CEILING], bwd[BEAM_MAX_LENGTH_CEILING];
    uint32_t got_len = 0;
    ASSERT_MSG(beam_model_explain(m, at_limit, fwd, bwd, &got_len) == BEAM_OK,
               "a sequence of exactly max_length must be read");
    ASSERT_MSG(got_len == 60, "and read at its full length");
    ASSERT_MSG(beam_model_explain(m, past_limit, fwd, bwd, &got_len) != BEAM_OK,
               "and one residue longer must be refused");

    beam_model_destroy(m);
    PASS();
}

/* ── Nucleotide models ───────────────────────────────────────── */

/* Small on purpose. A nucleotide model at its own defaults is order 6 over 81
 * positions, and finalize sweeps the partition function over every state of
 * every layer for each of the 81 lengths -- correct, and not something a unit
 * test should wait for. */
static BeamModel *nt_model(uint32_t order, uint32_t max_length) {
    BeamModelConfig c;
    beam_model_config_default_for(BEAM_ALPHABET_NT, &c);
    c.order = (uint8_t)order;
    c.max_length = (uint8_t)max_length;
    return beam_model_new(&c);
}

static void test_a_nucleotide_model_builds_and_scores(void) {
    BeamModel *m = nt_model(3, 12);
    ASSERT_MSG(m != NULL, "a nucleotide model must build now");
    ASSERT_MSG(beam_model_alphabet(m) == BEAM_ALPHABET_NT, "reports nt");
    ASSERT_MSG(beam_model_n_residues(m) == 5, "five residues");
    ASSERT_MSG(beam_model_radix(m) == 6, "radix six");
    ASSERT_MSG(beam_model_n_states(m) == 6u * 6u * 6u, "6^3 states at order 3");

    const char *seqs[] = { "TGTGCCAGCAGT", "TGTGCCAGCAGC", "TGTGCCAGTTCA" };
    const double ab[] = { 4.0, 2.0, 1.0 };
    ASSERT_MSG(beam_model_observe(m, seqs, ab, 3) == BEAM_OK, "observe");
    ASSERT_MSG(beam_model_n_observed(m) == 3, "all three accepted");
    ASSERT_MSG(beam_model_n_skipped(m) == 0, "none skipped");
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalize");

    const double lp = beam_model_log_pseq(m, "TGTGCCAGCAGT");
    ASSERT_MSG(isfinite(lp) && lp < 0.0,
               "an observed nucleotide sequence must score finitely");
    beam_model_destroy(m);
    PASS();
}

/*
 * Every nucleotide string of length 1..max_length, all 3,905 of them, in
 * order, through `visit`. Written with a counter in base 5 over the letters
 * rather than through the alphabet descriptor, so a mis-ordered table cannot
 * make the enumeration skip the string it mis-orders.
 */
typedef void (*VisitFn)(const char *seq, uint32_t len, void *ctx);

static void for_every_nucleotide_string(uint32_t max_length, VisitFn visit,
                                        void *ctx) {
    char buf[16];
    uint32_t digit[16];
    for (uint32_t len = 1; len <= max_length; ++len) {
        for (uint32_t i = 0; i < len; ++i) digit[i] = 0;
        for (;;) {
            for (uint32_t i = 0; i < len; ++i) buf[i] = NT_UPPER[digit[i]];
            buf[len] = '\0';
            visit(buf, len, ctx);
            uint32_t i = 0;
            while (i < len && ++digit[i] == 5u) digit[i++] = 0;
            if (i == len) break;
        }
    }
}

/* Per-length and whole-space totals of one enumeration. */
typedef struct {
    const BeamModel *m;
    double   by_len[8];        /* sum over length L of P(s) / pi(L) */
    double   total;            /* sum of P(s) */
    double   neg_plogp;        /* sum of -P(s) ln P(s) */
    double   sum_p2;           /* sum of P(s)^2 */
    uint32_t finite[8];        /* strings of length L scoring finitely */
    uint32_t visited[8];
} Tally;

static void tally_one(const char *seq, uint32_t len, void *ctx) {
    Tally *t = (Tally *)ctx;
    const double lp = beam_model_log_pseq(t->m, seq);
    t->visited[len]++;
    if (!isfinite(lp)) return;
    t->finite[len]++;
    const double p = exp(lp);
    t->by_len[len] += exp(lp - beam_model_log_length_prior(t->m, len));
    t->total     += p;
    t->neg_plogp -= p * lp;
    t->sum_p2    += p * p;
}

/* A corpus of three and four residues, so lengths 1, 2 and 5 are ones the
 * corpus never produced: their mass comes from the length prior alone, and
 * the strings in them from the context prior alone. */
static BeamModel *nt_tiny(uint32_t order) {
    BeamModel *m = nt_model(order, 5);
    if (!m) return NULL;
    const char *tiny[] = { "TGT", "TGC", "GCC", "TCA", "TGTA", "GCCT" };
    const double tw[] = { 5, 3, 2, 4, 3, 2 };
    if (beam_model_observe(m, tiny, tw, 6) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) {
        beam_model_destroy(m);
        return NULL;
    }
    return m;
}

/*
 * The strongest check available: sum P(s) over EVERY nucleotide sequence the
 * model can read, all 3,905 strings of length 1 to 5. It exercises the
 * context packing, the backward lag, the sink weights, the partition
 * function and the length prior at radix 6 in one assertion. An error in any
 * of the dimensions that follow from the alphabet shows up here and nowhere
 * cheaper.
 *
 * Under the hierarchical prior every one of those strings has positive
 * probability, so there is nothing to skip: each must score finitely, each
 * length must sum to 1 given its length, and the lot to 1 outright.
 */
static void test_the_partition_normalises_over_nucleotides(void) {
    BeamModel *m = nt_tiny(3);
    ASSERT_MSG(m != NULL, "build");

    Tally t;
    memset(&t, 0, sizeof t);
    t.m = m;
    for_every_nucleotide_string(5, tally_one, &t);

    static char msg[256];
    uint32_t want = 1;
    for (uint32_t len = 1; len <= 5; ++len) {
        want *= 5u;
        ASSERT_MSG(t.visited[len] == want, "the enumeration missed strings");
        if (t.finite[len] != want) {
            snprintf(msg, sizeof msg, "only %u of the %u strings of length %u "
                     "score finitely", (unsigned)t.finite[len], (unsigned)want,
                     (unsigned)len);
            beam_model_destroy(m);
            ASSERT_MSG(false, msg);
        }
        if (fabs(t.by_len[len] - 1.0) > 1e-9) {
            snprintf(msg, sizeof msg, "the %u strings of length %u sum to "
                     "%.17g given their length", (unsigned)want, (unsigned)len,
                     t.by_len[len]);
            beam_model_destroy(m);
            ASSERT_MSG(false, msg);
        }
    }
    const double total = t.total;
    beam_model_destroy(m);
    ASSERT_MSG(fabs(total - 1.0) < 1e-9,
               "the 3,905 readable nucleotide sequences must sum to 1");
    PASS();
}

/*
 * The dynamic programs, checked against enumeration rather than against
 * themselves. The path-count sweep, the entropy and the Hill numbers share
 * the trellis, the liveness frontier and the edge weights with every other
 * analytic; scoring alone would not catch a frontier that mis-decodes a
 * state at radix 6, and these do, because each is compared with a sum over
 * the same 3,905 strings computed from `log_pseq` and nothing else.
 */
static void test_the_analytics_match_enumeration_for_nucleotides(void) {
    BeamModel *m = nt_tiny(3);
    ASSERT_MSG(m != NULL, "build");

    Tally t;
    memset(&t, 0, sizeof t);
    t.m = m;
    for_every_nucleotide_string(5, tally_one, &t);

    /* Every string of a length is a path, so the count is the alphabet's,
     * 5^L, and not the corpus's; it is compared with the enumeration rather
     * than with 5^L so that it still tests the sweep if the model changes. */
    double support = 0.0;
    for (uint32_t len = 1; len <= 5; ++len) {
        ASSERT_MSG(beam_model_path_count(m, len) == (double)t.finite[len],
                   "the path count must equal the sequences that actually score");
        support += (double)t.finite[len];
    }
    ASSERT_MSG(beam_model_path_count_total(m) == support,
               "and the total support must be their sum");

    double h = 0.0, d2 = 0.0;
    ASSERT_MSG(beam_model_entropy(m, &h) == BEAM_OK, "entropy");
    ASSERT_MSG(beam_model_hill_number(m, 2.0, &d2) == BEAM_OK, "hill");
    const double want_h = t.neg_plogp, want_d2 = 1.0 / t.sum_p2;
    beam_model_destroy(m);

    static char msg[256];
    if (!(fabs(h - want_h) <= 1e-9 * want_h)) {
        snprintf(msg, sizeof msg, "entropy %.17g nats against the enumerated "
                 "%.17g", h, want_h);
        ASSERT_MSG(false, msg);
    }
    if (!(fabs(d2 - want_d2) <= 1e-9 * want_d2)) {
        snprintf(msg, sizeof msg, "Hill number of order 2 is %.17g against "
                 "the enumerated %.17g", d2, want_d2);
        ASSERT_MSG(false, msg);
    }
    PASS();
}

/*
 * What still scores -INFINITY. The enumeration above shows every readable
 * string has mass; this is the other half, that the only strings without it
 * are the ones the model cannot read at all.
 */
static void test_only_an_unreadable_sequence_scores_minus_infinity(void) {
    BeamModel *m = nt_tiny(3);
    ASSERT_MSG(m != NULL, "build");

    ASSERT_MSG(beam_model_log_pseq(m, "") == -INFINITY,
               "the empty string is not a sequence");
    ASSERT_MSG(beam_model_log_pseq(m, "TGTAC") > -INFINITY,
               "max_length itself is readable");
    ASSERT_MSG(beam_model_log_pseq(m, "TGTACG") == -INFINITY,
               "one past max_length is not");
    ASSERT_MSG(beam_model_log_pseq(m, "TGRT") == -INFINITY,
               "an ambiguity code is outside the alphabet");
    ASSERT_MSG(beam_model_log_pseq(m, "TG.T") == -INFINITY,
               "the sentinel's display character is not a residue");
    ASSERT_MSG(beam_model_log_pseq(m, "TGUT") == -INFINITY,
               "U is not a DNA residue");
    beam_model_destroy(m);
    PASS();
}

static void test_lower_case_scores_as_upper_case(void) {
    BeamModel *m = nt_model(3, 12);
    ASSERT_MSG(m != NULL, "new");
    const char *seqs[] = { "TGTGCCAGCAGT", "TGTGCCAGCAGC" };
    const double ab[] = { 3.0, 1.0 };
    ASSERT_MSG(beam_model_observe(m, seqs, ab, 2) == BEAM_OK, "observe");
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalize");

    const double upper = beam_model_log_pseq(m, "TGTGCCAGCAGT");
    const double lower = beam_model_log_pseq(m, "tgtgccagcagt");
    const double mixed = beam_model_log_pseq(m, "TgTgCcAgCaGt");
    ASSERT_MSG(isfinite(upper), "the upper-case form must score");
    /* Bit-identical, not merely close: case folding happens in the decode
     * table, so the two encode to the same indices and take the same path. */
    ASSERT_MSG(lower == upper, "lower case must be the same sequence");
    ASSERT_MSG(mixed == upper, "mixed case too");
    beam_model_destroy(m);
    PASS();
}

/* Training on lower case and scoring upper must also agree, which is the
 * direction that catches a fold applied at scoring but not at observation. */
static void test_lower_case_trains_as_upper_case(void) {
    const char *upper_seqs[] = { "TGTGCCAGCAGT", "TGTGCCAGCAGC" };
    const char *lower_seqs[] = { "tgtgccagcagt", "tgtgccagcagc" };
    const double ab[] = { 3.0, 1.0 };

    BeamModel *u = nt_model(3, 12);
    BeamModel *l = nt_model(3, 12);
    ASSERT_MSG(u && l, "new");
    ASSERT_MSG(beam_model_observe(u, upper_seqs, ab, 2) == BEAM_OK, "observe u");
    ASSERT_MSG(beam_model_observe(l, lower_seqs, ab, 2) == BEAM_OK, "observe l");
    ASSERT_MSG(beam_model_finalize(u) == BEAM_OK, "finalize u");
    ASSERT_MSG(beam_model_finalize(l) == BEAM_OK, "finalize l");

    ASSERT_MSG(beam_model_log_pseq(u, "TGTGCCAGCAGT")
               == beam_model_log_pseq(l, "TGTGCCAGCAGT"),
               "a corpus in lower case must build the same model");
    beam_model_destroy(u);
    beam_model_destroy(l);
    PASS();
}

/* N is a residue, so a read carrying one is data rather than an error. The
 * amino-acid alphabet has no equivalent: 'X' is rejected. */
static void test_N_is_a_residue_and_other_codes_are_not(void) {
    BeamModel *m = nt_model(3, 12);
    ASSERT_MSG(m != NULL, "new");
    const char *seqs[] = { "TGTGCCAGCAGN", "TGTGCCAGCAGT" };
    const double ab[] = { 2.0, 2.0 };
    ASSERT_MSG(beam_model_observe(m, seqs, ab, 2) == BEAM_OK, "observe");
    ASSERT_MSG(beam_model_n_observed(m) == 2, "the N-bearing read is data");
    ASSERT_MSG(beam_model_n_skipped(m) == 0, "and is not skipped");
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalize");
    ASSERT_MSG(isfinite(beam_model_log_pseq(m, "TGTGCCAGCAGN")),
               "and it scores");

    /* R is 'A or G', which the model cannot represent, so it is refused. */
    const char *bad[] = { "TGTGCCAGCAGR" };
    const double one[] = { 1.0 };
    ASSERT_MSG(beam_model_observe(m, bad, one, 1) == BEAM_OK, "observe returns");
    ASSERT_MSG(beam_model_n_skipped(m) == 1, "an ambiguity code is skipped");
    beam_model_destroy(m);
    PASS();
}

static void test_generation_emits_only_this_alphabet(void) {
    BeamModel *m = nt_model(3, 12);
    ASSERT_MSG(m != NULL, "new");
    const char *seqs[] = { "TGTGCCAGCAGT", "TGTGCCAGCAGC", "TGTGCCAGTTCA" };
    const double ab[] = { 4.0, 2.0, 1.0 };
    ASSERT_MSG(beam_model_observe(m, seqs, ab, 3) == BEAM_OK, "observe");
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalize");

    const uint32_t n = 64, stride = 13;   /* max_length + 1 */
    char *out = (char *)calloc(n, stride);
    ASSERT_MSG(out != NULL, "alloc");
    BeamRng rng;
    beam_rng_seed(&rng, 7u);
    ASSERT_MSG(beam_model_sample(m, n, 0u, &rng, out, NULL) == BEAM_OK,
               "sample");
    for (uint32_t k = 0; k < n; ++k)
        for (const char *p = out + (size_t)k * stride; *p; ++p)
            ASSERT_MSG(strchr(NT_UPPER, *p) != NULL,
                       "generation must emit only this alphabet's residues");
    free(out);
    beam_model_destroy(m);
    PASS();
}

/*
 * The carried requirement from the bite that added the alphabet to the shape
 * check: two models can agree on order, length and depth and still have rows
 * of different widths, and before that check existed the loops would have
 * walked the operand off the end of its allocation. It could not be tested
 * until nucleotide models could be built.
 */
static void test_model_algebra_refuses_across_alphabets(void) {
    BeamModelConfig ac;
    beam_model_config_default(&ac);
    ac.order = 3u;                 /* matched to the nt model below, so the */
    ac.max_length = 12u;           /* alphabet is the ONLY thing differing  */
    BeamModel *aa = beam_model_new(&ac);
    BeamModel *nt = nt_model(3, 12);
    ASSERT_MSG(aa && nt, "both models must build");

    /* Identical order and max_length -- everything the shape check compares
     * besides the alphabet. */
    ASSERT_MSG(beam_model_config(aa)->order == beam_model_config(nt)->order
               && beam_model_config(aa)->max_length
                  == beam_model_config(nt)->max_length,
               "the two agree on every other shape field");

    ASSERT_MSG(beam_model_add(aa, nt) != BEAM_OK, "add must refuse");
    ASSERT_MSG(beam_model_add(nt, aa) != BEAM_OK, "and in both directions");
    ASSERT_MSG(beam_model_subtract(aa, nt) != BEAM_OK, "subtract must refuse");
    ASSERT_MSG(beam_model_posterior(aa, nt, 1.0) != BEAM_OK,
               "posterior must refuse");
    beam_model_destroy(aa);
    beam_model_destroy(nt);
    PASS();
}

/* The file has to carry the alphabet, because alphabet size determines every
 * dimension of the payload behind it. A round trip that came back as amino
 * acid would not merely score differently -- it would read the table at the
 * wrong stride. */
static void test_a_nucleotide_model_round_trips_through_a_file(void) {
    BeamModel *m = nt_model(3, 12);
    ASSERT_MSG(m != NULL, "new");
    const char *seqs[] = { "TGTGCCAGCAGT", "TGTGCCAGCAGC", "TGTGCCAGTTCA" };
    const double ab[] = { 4.0, 2.0, 1.0 };
    ASSERT_MSG(beam_model_observe(m, seqs, ab, 3) == BEAM_OK, "observe");
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalize");

    const char *path = "build/test_nt_roundtrip.tmp";
    ASSERT_MSG(beam_model_save(m, path) == BEAM_OK, "save");
    BeamModel *back = beam_model_load(path);
    ASSERT_MSG(back != NULL, "load");

    ASSERT_MSG(beam_model_alphabet(back) == BEAM_ALPHABET_NT,
               "the file must remember that it is nucleotide");
    ASSERT_MSG(beam_model_n_residues(back) == 5 && beam_model_radix(back) == 6,
               "and rebuild the dimensions that follow from it");
    ASSERT_MSG(beam_model_n_states(back) == beam_model_n_states(m),
               "the trellis must be the same size");
    for (unsigned k = 0; k < 3u; ++k)
        ASSERT_MSG(beam_model_log_pseq(back, seqs[k])
                   == beam_model_log_pseq(m, seqs[k]),
                   "every score must round-trip bit-identically");
    beam_model_destroy(back);
    beam_model_destroy(m);
    remove(path);
    PASS();
}

/* Overwrite the alphabet field of a saved model and repair the CRC, so what
 * the loader is left to catch is the mislabelling itself rather than a
 * checksum that no longer matches. */
static bool relabel_alphabet(const char *path, uint32_t expect_was,
                             uint32_t alphabet) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
    const long end = ftell(f);
    rewind(f);
    unsigned char *b = (unsigned char *)malloc((size_t)end);
    if (!b || fread(b, 1, (size_t)end, f) != (size_t)end) {
        free(b); fclose(f); return false;
    }
    fclose(f);

    /* The third word of the format-8 header, after the magic and the format
     * number. Checked before it is overwritten: had the layout moved, this
     * would otherwise corrupt some other field and the loader's refusal
     * would prove nothing about the alphabet. */
    const size_t at = 2u * sizeof(uint32_t);
    uint32_t was = 0;
    if ((size_t)end >= at + sizeof was) memcpy(&was, b + at, sizeof was);
    if ((size_t)end < at + sizeof was || was != expect_was) {
        free(b); return false;
    }
    memcpy(b + at, &alphabet, sizeof alphabet);
    const uint32_t crc = beam_crc32c_update(0u, b,
                                         (size_t)end - sizeof(uint32_t));
    memcpy(b + end - (long)sizeof crc, &crc, sizeof crc);

    f = fopen(path, "wb");
    if (!f) { free(b); return false; }
    const bool ok = fwrite(b, 1, (size_t)end, f) == (size_t)end;
    fclose(f);
    free(b);
    return ok;
}

/*
 * A nucleotide file must not be readable as an amino-acid one. The two agree
 * on order and length in the header, so before the alphabet field the
 * loader had nothing to tell them apart and would have built a radix-21 model
 * over a radix-6 payload.
 *
 * The CRC is repaired after the relabelling on purpose. Leaving it broken
 * would have the checksum reject the file and prove nothing about the
 * alphabet; repaired, the only thing left to notice is that the payload is
 * the wrong size for the dimensions the header now claims.
 */
static void test_a_nucleotide_file_relabelled_as_amino_acid_is_rejected(void) {
    BeamModel *m = nt_model(3, 12);
    ASSERT_MSG(m != NULL, "new");
    const char *seqs[] = { "TGTGCCAGCAGT", "TGTGCCAGCAGC" };
    const double ab[] = { 2.0, 1.0 };
    ASSERT_MSG(beam_model_observe(m, seqs, ab, 2) == BEAM_OK, "observe");
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalize");

    const char *path = "build/test_nt_relabelled.tmp";
    ASSERT_MSG(beam_model_save(m, path) == BEAM_OK, "save");
    ASSERT_MSG(relabel_alphabet(path, (uint32_t)BEAM_ALPHABET_NT,
                                (uint32_t)BEAM_ALPHABET_AA),
               "relabel as amino acid and repair the checksum");

    ASSERT_MSG(beam_model_load(path) == NULL,
               "a nucleotide payload must not load as an amino-acid model "
               "even when the checksum agrees");
    beam_model_destroy(m);
    remove(path);
    PASS();
}

/* A small amino-acid model stands in for any: the refusal is about the id
 * in the header, and every full-space sweep a default-sized model pays for
 * on finalize would only slow the test down. */
static BeamModel *aa_small(void) {
    BeamModelConfig c;
    beam_model_config_default(&c);
    c.order = 3u;
    c.max_length = 12u;
    return beam_model_new(&c);
}

static void test_an_unknown_alphabet_in_a_file_is_refused(void) {
    BeamModel *m = aa_small();
    const char *seqs[] = { "CASSLGQAYEQY" };
    const double ab[] = { 1.0 };
    ASSERT_MSG(m && beam_model_observe(m, seqs, ab, 1) == BEAM_OK
               && beam_model_finalize(m) == BEAM_OK, "build");

    const char *path = "build/test_unknown_alphabet.tmp";
    ASSERT_MSG(beam_model_save(m, path) == BEAM_OK, "save");

    /* An id no build knows, with the checksum repaired so the refusal has to
     * come from the alphabet check rather than from the CRC. Every table
     * dimension behind it is unreadable, so guessing is not an option. */
    ASSERT_MSG(relabel_alphabet(path, (uint32_t)BEAM_ALPHABET_AA, 99u), "write a wild id");

    ASSERT_MSG(beam_model_load(path) == NULL,
               "an unknown alphabet must be refused, not guessed at");
    beam_model_destroy(m);
    remove(path);
    PASS();
}

/* The two alphabets must not share anything mutable. Building one after the
 * other, and scoring both, is the cheapest check that no descriptor, table or
 * cached dimension leaked between them. */
static void test_the_two_alphabets_do_not_interfere(void) {
    BeamModel *aa1 = aa_small();
    const char *aa_seqs[] = { "CASSLGQAYEQY", "CASSLGQAYEQF" };
    const double ab[] = { 3.0, 1.0 };
    ASSERT_MSG(aa1 && beam_model_observe(aa1, aa_seqs, ab, 2) == BEAM_OK
               && beam_model_finalize(aa1) == BEAM_OK, "aa before");
    const double before = beam_model_log_pseq(aa1, "CASSLGQAYEQY");

    BeamModel *nt = nt_model(3, 12);
    const char *nt_seqs[] = { "TGTGCCAGCAGT", "TGTGCCAGCAGC" };
    ASSERT_MSG(nt && beam_model_observe(nt, nt_seqs, ab, 2) == BEAM_OK
               && beam_model_finalize(nt) == BEAM_OK, "nt in between");

    BeamModel *aa2 = aa_small();
    ASSERT_MSG(aa2 && beam_model_observe(aa2, aa_seqs, ab, 2) == BEAM_OK
               && beam_model_finalize(aa2) == BEAM_OK, "aa after");
    const double after = beam_model_log_pseq(aa2, "CASSLGQAYEQY");

    ASSERT_MSG(before == after,
               "an amino-acid model must score identically either side of a "
               "nucleotide one");
    /* And the amino-acid model must still refuse a nucleotide-only string's
     * lower case, which its own alphabet does not accept. */
    ASSERT_MSG(beam_model_log_pseq(aa1, "cassnpqayeqy") == -INFINITY,
               "the amino-acid alphabet stays upper case only");

    beam_model_destroy(aa1);
    beam_model_destroy(aa2);
    beam_model_destroy(nt);
    PASS();
}

/*
 * Every order the nucleotide alphabet allows, built, finalized and scored.
 *
 * Amino acids cap at order 4, so every per-level array of the prior -- the
 * strengths, the lower-level count tables, the level-k parent of a row --
 * is only ever indexed up to 4 by them. Nucleotides go to 9. An array sized
 * or indexed by the amino-acid ceiling rather than BEAM_MAX_ORDER is
 * invisible to every amino-acid test and walks off its end here; that is
 * how a segfault of exactly this shape once reached the backoff hierarchy
 * the prior replaced.
 *
 * The unseen string is the point of scoring. Its top-level row was never
 * observed, so its score is resolved through every level of the hierarchy
 * down to where the corpus has counts, which is the path an out-of-range
 * level index would take.
 */
static bool order_builds_and_scores(uint32_t order, uint32_t max_length,
                                    const char *const *seqs, uint32_t n,
                                    const char *unseen, char *msg, size_t cap) {
    static const double ab[] = { 4.0, 2.0, 1.0 };
    BeamModelConfig c;
    beam_model_config_default_for(BEAM_ALPHABET_NT, &c);
    c.order = (uint8_t)order;
    c.max_length = (uint8_t)max_length;

    BeamModel *m = beam_model_new(&c);
    if (!m) {
        snprintf(msg, cap, "order %u did not build", (unsigned)order);
        return false;
    }
    if (beam_model_observe(m, seqs, ab, n) != BEAM_OK
        || beam_model_finalize(m) != BEAM_OK) {
        snprintf(msg, cap, "order %u did not finalize: %s", (unsigned)order,
                 beam_error_message());
        beam_model_destroy(m);
        return false;
    }
    const double seen = beam_model_log_pseq(m, seqs[0]);
    const double other = beam_model_log_pseq(m, unseen);
    beam_model_destroy(m);
    if (!(isfinite(seen) && isfinite(other) && other < seen)) {
        snprintf(msg, cap, "order %u: an observed sequence scores %.17g and "
                 "an unseen one %.17g; both must be finite and the unseen one "
                 "lower", (unsigned)order, seen, other);
        return false;
    }
    return true;
}

static void test_every_nucleotide_order_finalizes_and_scores(void) {
    static char msg[256];

    /* Long enough that every order here also sees contexts made entirely of
     * residues, not only of residues and end padding. */
    const char *eight[] = { "TGTGCCAG", "TGTGCCAC", "TGTGCTCA" };
    for (uint32_t order = 2; order <= 7; ++order)
        if (!order_builds_and_scores(order, 8, eight, 3, "NNACGTNN",
                                     msg, sizeof msg))
            ASSERT_MSG(false, msg);

    /*
     * Orders 8 and the ceiling, at length 2. Finalize sweeps the partition
     * function over every state of every layer, and at 6^9 states that is
     * seconds per unit of max_length. A longer corpus would buy nothing
     * here: at these orders every context of any sequence a unit test can
     * afford includes end padding, so length 2 reaches the same levels of
     * the hierarchy that length 8 would.
     */
    const char *two[] = { "TG", "TC", "A" };
    for (uint32_t order = 8; order <= 9; ++order)
        if (!order_builds_and_scores(order, 2, two, 3, "NN", msg, sizeof msg))
            ASSERT_MSG(false, msg);

    BeamModelConfig c;
    beam_model_config_default_for(BEAM_ALPHABET_NT, &c);
    c.order = 9;
    c.max_length = 2;
    BeamModel *probe = beam_model_new(&c);
    ASSERT_MSG(probe != NULL, "the nucleotide ceiling order must build");
    /* Asserted so a change to the ceiling is seen here and not only as a
     * loop bound. */
    const uint32_t states = beam_model_n_states(probe);
    beam_model_destroy(probe);
    ASSERT_MSG(states == 10077696u, "6^9 states");
    PASS();
}

/*
 * And the model that comes out is still a distribution above order four,
 * where the hierarchy has levels no amino-acid model reaches. A level
 * resolved to the wrong parent, or a strength read from the wrong slot,
 * leaves a perfectly buildable model whose probabilities no longer sum to
 * 1, which nothing above would notice.
 *
 * Order 5 against strings of at most five residues, so every context row the
 * enumeration touches is padded with sentinels at one end or the other, and
 * the rows at full width are exactly the ones the corpus never saw.
 */
static void test_the_partition_normalises_above_order_four(void) {
    BeamModel *m = nt_tiny(5);
    ASSERT_MSG(m != NULL, "build");

    Tally t;
    memset(&t, 0, sizeof t);
    t.m = m;
    for_every_nucleotide_string(5, tally_one, &t);
    beam_model_destroy(m);

    static char msg[256];
    uint32_t want = 1;
    for (uint32_t len = 1; len <= 5; ++len) {
        want *= 5u;
        ASSERT_MSG(t.finite[len] == want,
                   "every readable string must score at order 5");
        if (fabs(t.by_len[len] - 1.0) > 1e-9) {
            snprintf(msg, sizeof msg, "an order-5 nucleotide model sums to "
                     "%.17g over length %u", t.by_len[len], (unsigned)len);
            ASSERT_MSG(false, msg);
        }
    }
    ASSERT_MSG(fabs(t.total - 1.0) < 1e-9,
               "and to 1 over every length together");
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
    RUN_TEST(test_the_widest_table_still_fits_a_uint32_row_index);

    printf("\n[mapping]\n");
    RUN_TEST(test_every_byte_maps_exactly_as_specified);
    RUN_TEST(test_no_character_encodes_to_the_sentinel);
    RUN_TEST(test_residue_indices_are_dense);
    RUN_TEST(test_index_and_character_round_trip);

    printf("\n[per-alphabet]\n");
    RUN_TEST(test_nucleotides_fold_case_and_amino_acids_do_not);
    RUN_TEST(test_ambiguity_beyond_N_is_still_rejected);
    RUN_TEST(test_the_alphabets_overlap_with_different_indices);

    printf("\n[registry]\n");
    RUN_TEST(test_the_registry_resolves_ids_and_names);
    RUN_TEST(test_the_registry_refuses_what_it_does_not_know);
    RUN_TEST(test_alphabet_names_are_distinct);

    printf("\n[configuration]\n");
    RUN_TEST(test_the_amino_acid_defaults_are_unchanged);
    RUN_TEST(test_per_alphabet_defaults_come_from_the_descriptor);
    RUN_TEST(test_a_zeroed_config_still_means_amino_acid);
    RUN_TEST(test_defaults_refuse_an_unknown_alphabet);

    printf("\n[validation]\n");
    RUN_TEST(test_the_order_bound_is_the_alphabet_s_own);
    RUN_TEST(test_validation_refuses_an_unknown_alphabet);
    RUN_TEST(test_max_length_is_free_up_to_the_ceiling);
    RUN_TEST(test_every_config_refusal_names_its_own_rule);
    RUN_TEST(test_prior_strengths_past_the_order_are_ignored);

    printf("\n[construction]\n");
    RUN_TEST(test_a_model_reports_the_alphabet_it_was_built_with);
    RUN_TEST(test_construction_refuses_an_unknown_alphabet);
    RUN_TEST(test_a_model_longer_than_the_old_ceiling_builds_and_scores);

    printf("\n[nucleotide models]\n");
    RUN_TEST(test_a_nucleotide_model_builds_and_scores);
    RUN_TEST(test_the_partition_normalises_over_nucleotides);
    RUN_TEST(test_the_analytics_match_enumeration_for_nucleotides);
    RUN_TEST(test_only_an_unreadable_sequence_scores_minus_infinity);
    RUN_TEST(test_lower_case_scores_as_upper_case);
    RUN_TEST(test_lower_case_trains_as_upper_case);
    RUN_TEST(test_N_is_a_residue_and_other_codes_are_not);
    RUN_TEST(test_generation_emits_only_this_alphabet);
    RUN_TEST(test_model_algebra_refuses_across_alphabets);
    RUN_TEST(test_a_nucleotide_model_round_trips_through_a_file);
    RUN_TEST(test_a_nucleotide_file_relabelled_as_amino_acid_is_rejected);
    RUN_TEST(test_an_unknown_alphabet_in_a_file_is_refused);
    RUN_TEST(test_the_two_alphabets_do_not_interfere);
    RUN_TEST(test_every_nucleotide_order_finalizes_and_scores);
    RUN_TEST(test_the_partition_normalises_above_order_four);

    printf("\n==============================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
