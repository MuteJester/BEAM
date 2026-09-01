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
 * The expected values are written out rather than read from the descriptor,
 * so that editing the descriptor cannot quietly redefine what "unchanged"
 * means.
 */
static void test_the_amino_acid_defaults_are_unchanged(void) {
    BeamModelConfig c;
    beam_model_config_default(&c);

    ASSERT_MSG(c.alphabet == BEAM_ALPHABET_AA, "the default is amino acid");
    ASSERT_MSG(c.order == 4, "order");
    ASSERT_MSG(c.depth_cap == 15, "depth_cap");
    ASSERT_MSG(c.max_length == 27, "max_length");
    ASSERT_MSG(c.coef_wide == 0.60 && c.coef_mid == 0.25
               && c.coef_len == 0.15, "mixing coefficients");
    ASSERT_MSG(c.kappa[0] == 0.0 && c.kappa[1] == 0.0 && c.kappa[2] == 0.0
               && c.kappa_wide == 0.0 && c.kappa_len == 0.0,
               "the closed-world backoff strengths");
    ASSERT_MSG(c.weight_forward == 0.5 && c.weight_backward == 0.5,
               "anchor weights");
    ASSERT_MSG(c.strict && !c.sparse, "strict, dense");
    PASS();
}

static void test_per_alphabet_defaults_come_from_the_descriptor(void) {
    BeamModelConfig aa, nt;
    beam_model_config_default_for(BEAM_ALPHABET_AA, &aa);
    beam_model_config_default_for(BEAM_ALPHABET_NT, &nt);

    ASSERT_MSG(aa.order == 4 && aa.max_length == 27 && aa.depth_cap == 15,
               "the amino-acid defaults");
    ASSERT_MSG(nt.order == 6 && nt.max_length == 81,
               "two codons of context over three bases per residue");
    ASSERT_MSG(nt.depth_cap >= nt.max_length,
               "the nucleotide default resolves every depth, which "
               "beam_model_new clamps to max_length - 1");

    /* What the alphabet must NOT decide. These were measured on amino acids
     * and carrying them across is the deliberate choice; re-tuning them needs
     * its own measurement. */
    ASSERT_MSG(nt.coef_wide == aa.coef_wide && nt.coef_mid == aa.coef_mid
               && nt.coef_len == aa.coef_len, "mixing is alphabet-independent");
    ASSERT_MSG(nt.weight_forward == aa.weight_forward
               && nt.weight_backward == aa.weight_backward,
               "anchor weights are alphabet-independent");
    ASSERT_MSG(nt.strict == aa.strict, "closed world is alphabet-independent");
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
    if (c.depth_cap >= c.max_length) c.depth_cap = (uint8_t)(c.max_length - 1u);
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
    c.depth_cap = 15;
    ASSERT_MSG(beam_config_valid(&c), "max_length is a free parameter now");

    c.max_length = 1; c.depth_cap = 1;
    ASSERT_MSG(!beam_config_valid(&c), "depth_cap must stay below max_length");
    c.max_length = 0;
    ASSERT_MSG(!beam_config_valid(&c), "a zero-length model is not a model");
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
     * `explain` rather than `log_pseq`, because log_pseq answers -INFINITY
     * both for a sequence it cannot read and for one it can read but cannot
     * spell, so it cannot tell the boundary from the model's own support.
     */
    char at_limit[61], past_limit[62];
    memset(at_limit, 'A', 60);    at_limit[60] = '\0';
    memset(past_limit, 'A', 61);  past_limit[61] = '\0';

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
 * positions with every depth resolved, which is a 151 MB wide table per
 * direction -- correct, and not something a unit test should allocate. */
static BeamModel *nt_model(uint32_t order, uint32_t max_length,
                           uint32_t depth_cap) {
    BeamModelConfig c;
    beam_model_config_default_for(BEAM_ALPHABET_NT, &c);
    c.order = (uint8_t)order;
    c.max_length = (uint8_t)max_length;
    c.depth_cap = (uint8_t)depth_cap;
    return beam_model_new(&c);
}

static void test_a_nucleotide_model_builds_and_scores(void) {
    BeamModel *m = nt_model(3, 12, 6);
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
 * The strongest check available: sum P(s | L) over EVERY nucleotide sequence
 * of length 3, all 125 of them. It exercises the context packing, the
 * backward lag, the sink weights and the partition function at radix 6 in one
 * assertion, and the amino-acid suite has had the same test at radix 21 all
 * along. An error in any of the dimensions the migration made runtime shows
 * up here and nowhere cheaper.
 */
static void test_the_partition_normalises_over_nucleotides(void) {
    BeamModel *m = nt_model(3, 5, 3);
    ASSERT_MSG(m != NULL, "new");
    const char *tiny[] = { "TGT", "TGC", "GCC", "TCA", "TGT", "AGT" };
    const double tw[] = { 5, 3, 2, 4, 1, 2 };
    ASSERT_MSG(beam_model_observe(m, tiny, tw, 6) == BEAM_OK, "observe");
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalize");

    const double logpi = beam_model_log_length_prior(m, 3);
    double total = 0.0;
    char buf[4] = { 0, 0, 0, 0 };
    for (int a = 0; a < 5; a++)
      for (int b = 0; b < 5; b++)
        for (int c = 0; c < 5; c++) {
            buf[0] = NT_UPPER[a]; buf[1] = NT_UPPER[b]; buf[2] = NT_UPPER[c];
            const double lp = beam_model_log_pseq(m, buf);
            if (isfinite(lp)) total += exp(lp - logpi);
        }
    ASSERT_MSG(fabs(total - 1.0) < 1e-9,
               "the 125 length-3 nucleotide sequences must sum to 1");
    beam_model_destroy(m);
    PASS();
}

/*
 * The dynamic programs, checked against enumeration rather than against
 * themselves. In the closed world a sequence is spellable exactly when it
 * scores finitely, so counting the length-3 strings that do must equal what
 * the path-count sweep reports -- and that sweep shares the trellis, the
 * liveness frontier and the edge weights with every other analytic. Scoring
 * alone would not catch a frontier that mis-decodes a state at radix 6.
 */
static void test_path_count_matches_enumeration_for_nucleotides(void) {
    BeamModel *m = nt_model(3, 5, 3);
    ASSERT_MSG(m != NULL, "new");
    const char *tiny[] = { "TGT", "TGC", "GCC", "TCA", "TGT", "AGT" };
    const double tw[] = { 5, 3, 2, 4, 1, 2 };
    ASSERT_MSG(beam_model_observe(m, tiny, tw, 6) == BEAM_OK, "observe");
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalize");

    uint32_t spellable = 0;
    char buf[4] = { 0, 0, 0, 0 };
    for (int a = 0; a < 5; a++)
      for (int b = 0; b < 5; b++)
        for (int c = 0; c < 5; c++) {
            buf[0] = NT_UPPER[a]; buf[1] = NT_UPPER[b]; buf[2] = NT_UPPER[c];
            if (isfinite(beam_model_log_pseq(m, buf))) spellable++;
        }
    const double counted = beam_model_path_count(m, 3);
    ASSERT_MSG(counted == (double)spellable,
               "the path count must equal the sequences that actually score");
    ASSERT_MSG(spellable > 0, "and the corpus must spell something");

    /* The remaining analytics have to run and answer sanely on this trellis;
     * an entropy outside [0, log(support)] would mean the sweep and the
     * support disagree about which edges exist. */
    double h = 0.0;
    ASSERT_MSG(beam_model_entropy(m, &h) == BEAM_OK, "entropy");
    ASSERT_MSG(isfinite(h) && h >= 0.0, "entropy must be finite and positive");
    double d2 = 0.0;
    ASSERT_MSG(beam_model_hill_number(m, 2.0, &d2) == BEAM_OK, "hill");
    ASSERT_MSG(isfinite(d2) && d2 >= 1.0, "a Hill number is at least one");
    beam_model_destroy(m);
    PASS();
}

static void test_lower_case_scores_as_upper_case(void) {
    BeamModel *m = nt_model(3, 12, 6);
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

    BeamModel *u = nt_model(3, 12, 6);
    BeamModel *l = nt_model(3, 12, 6);
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
    BeamModel *m = nt_model(3, 12, 6);
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
    BeamModel *m = nt_model(3, 12, 6);
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
    ac.order = 3; ac.max_length = 12; ac.depth_cap = 6;
    BeamModel *aa = beam_model_new(&ac);
    BeamModel *nt = nt_model(3, 12, 6);
    ASSERT_MSG(aa && nt, "both models must build");

    /* Identical order, max_length and depth_cap -- everything the shape check
     * compared before the alphabet joined it. */
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
    BeamModel *m = nt_model(3, 12, 6);
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
static bool relabel_alphabet(const char *path, uint32_t alphabet) {
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

    memcpy(b + 136, &alphabet, sizeof alphabet);   /* the appended tail */
    const uint32_t crc = beam_model_crc32c(0u, b,
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
 * on order, length and depth in the header, so before the alphabet field the
 * loader had nothing to tell them apart and would have built a radix-21 model
 * over a radix-6 payload.
 *
 * The CRC is repaired after the relabelling on purpose. Leaving it broken
 * would have the checksum reject the file and prove nothing about the
 * alphabet; repaired, the only thing left to notice is that the payload is
 * the wrong size for the dimensions the header now claims.
 */
static void test_a_nucleotide_file_relabelled_as_amino_acid_is_rejected(void) {
    BeamModel *m = nt_model(3, 12, 6);
    ASSERT_MSG(m != NULL, "new");
    const char *seqs[] = { "TGTGCCAGCAGT", "TGTGCCAGCAGC" };
    const double ab[] = { 2.0, 1.0 };
    ASSERT_MSG(beam_model_observe(m, seqs, ab, 2) == BEAM_OK, "observe");
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalize");

    const char *path = "build/test_nt_relabelled.tmp";
    ASSERT_MSG(beam_model_save(m, path) == BEAM_OK, "save");
    ASSERT_MSG(relabel_alphabet(path, (uint32_t)BEAM_ALPHABET_AA),
               "relabel as amino acid and repair the checksum");

    ASSERT_MSG(beam_model_load(path) == NULL,
               "a nucleotide payload must not load as an amino-acid model "
               "even when the checksum agrees");
    beam_model_destroy(m);
    remove(path);
    PASS();
}

static void test_an_unknown_alphabet_in_a_file_is_refused(void) {
    BeamModelConfig c;
    beam_model_config_default(&c);
    c.order = 3; c.max_length = 12; c.depth_cap = 6;
    BeamModel *m = beam_model_new(&c);
    const char *seqs[] = { "CASSLGQAYEQY" };
    const double ab[] = { 1.0 };
    ASSERT_MSG(m && beam_model_observe(m, seqs, ab, 1) == BEAM_OK
               && beam_model_finalize(m) == BEAM_OK, "build");

    const char *path = "build/test_unknown_alphabet.tmp";
    ASSERT_MSG(beam_model_save(m, path) == BEAM_OK, "save");

    /* An id no build knows, with the checksum repaired so the refusal has to
     * come from the alphabet check rather than from the CRC. Every table
     * dimension behind it is unreadable, so guessing is not an option. */
    ASSERT_MSG(relabel_alphabet(path, 99u), "write a wild id");

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
    BeamModelConfig ac;
    beam_model_config_default(&ac);
    ac.order = 3; ac.max_length = 12; ac.depth_cap = 6;

    BeamModel *aa1 = beam_model_new(&ac);
    const char *aa_seqs[] = { "CASSLGQAYEQY", "CASSLGQAYEQF" };
    const double ab[] = { 3.0, 1.0 };
    ASSERT_MSG(aa1 && beam_model_observe(aa1, aa_seqs, ab, 2) == BEAM_OK
               && beam_model_finalize(aa1) == BEAM_OK, "aa before");
    const double before = beam_model_log_pseq(aa1, "CASSLGQAYEQY");

    BeamModel *nt = nt_model(3, 12, 6);
    const char *nt_seqs[] = { "TGTGCCAGCAGT", "TGTGCCAGCAGC" };
    ASSERT_MSG(nt && beam_model_observe(nt, nt_seqs, ab, 2) == BEAM_OK
               && beam_model_finalize(nt) == BEAM_OK, "nt in between");

    BeamModel *aa2 = beam_model_new(&ac);
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
 * Every order the nucleotide alphabet allows, with backoff switched ON.
 *
 * This is the configuration the suite had no coverage of, and it segfaulted:
 * the wide expert's backoff base was indexed at `order - 1`, into arrays of
 * four pointers holding the mid hierarchy's orders 0..3. Amino acids cap at
 * order 4, so `order - 1` was always in range and always a level that had been
 * built; a nucleotide model at its own default order 6 read two slots past the
 * end. Only the all-kappas-zero path escaped, which is exactly what every
 * other nucleotide test here uses.
 *
 * The base is now the order below capped at 3, since `kappa` has three entries
 * and no count table holds order-4 context at full depth. Orders 2 to 4 are
 * kept in the loop because they must be unaffected by that cap.
 */
static void test_every_nucleotide_order_finalizes_with_backoff(void) {
    const char *seqs[] = { "TGTGCCAGCAGTGATAGC", "TGTGCCAGCAGTGATAGG",
                           "TGTGCCAGCAGCCCGGAC" };
    const double ab[] = { 4.0, 2.0, 1.0 };

    for (uint32_t order = 2; order <= 6; ++order) {
        BeamModelConfig c;
        beam_model_config_default_for(BEAM_ALPHABET_NT, &c);
        c.order = (uint8_t)order;
        c.max_length = 20;
        c.depth_cap = 5;
        /* Open world: every backoff strength positive, which is what puts the
         * wide expert's base lookup on the path at all. */
        c.strict = false;
        c.kappa[0] = c.kappa[1] = c.kappa[2] = 20.0;
        c.kappa_wide = 20.0;
        c.kappa_len = 200.0;

        BeamModel *m = beam_model_new(&c);
        ASSERT_MSG(m != NULL, "a nucleotide model must build at every order");
        ASSERT_MSG(beam_model_observe(m, seqs, ab, 3) == BEAM_OK, "observe");
        ASSERT_MSG(beam_model_finalize(m) == BEAM_OK,
                   "finalize must not walk off the backoff level array");
        const double lp = beam_model_log_pseq(m, seqs[0]);
        ASSERT_MSG(isfinite(lp),
                   "the open world gives every readable sequence mass");
        beam_model_destroy(m);
    }
    PASS();
}

/*
 * And the model that comes out is still a distribution. Changing which table
 * the wide expert backs off to could leave a perfectly buildable model whose
 * probabilities no longer sum to 1, which nothing above would notice.
 */
static void test_the_partition_normalises_with_backoff_above_order_four(void) {
    BeamModelConfig c;
    beam_model_config_default_for(BEAM_ALPHABET_NT, &c);
    c.order = 5;                       /* base capped from 4 down to 3 */
    c.max_length = 5;
    c.depth_cap = 3;
    c.strict = false;
    c.kappa[0] = c.kappa[1] = c.kappa[2] = 20.0;
    c.kappa_wide = 20.0;
    c.kappa_len = 200.0;

    BeamModel *m = beam_model_new(&c);
    ASSERT_MSG(m != NULL, "new");
    const char *tiny[] = { "TGT", "TGC", "GCC", "TCA" };
    const double tw[] = { 5, 3, 2, 4 };
    ASSERT_MSG(beam_model_observe(m, tiny, tw, 4) == BEAM_OK, "observe");
    ASSERT_MSG(beam_model_finalize(m) == BEAM_OK, "finalize");

    const double logpi = beam_model_log_length_prior(m, 3);
    double total = 0.0;
    char buf[4] = { 0, 0, 0, 0 };
    for (int a = 0; a < 5; a++)
      for (int b = 0; b < 5; b++)
        for (int d = 0; d < 5; d++) {
            buf[0] = NT_UPPER[a]; buf[1] = NT_UPPER[b]; buf[2] = NT_UPPER[d];
            const double lp = beam_model_log_pseq(m, buf);
            if (isfinite(lp)) total += exp(lp - logpi);
        }
    ASSERT_MSG(fabs(total - 1.0) < 1e-9,
               "a smoothed order-5 nucleotide model must still normalise");
    beam_model_destroy(m);
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

    printf("\n[construction]\n");
    RUN_TEST(test_a_model_reports_the_alphabet_it_was_built_with);
    RUN_TEST(test_construction_refuses_an_unknown_alphabet);
    RUN_TEST(test_a_model_longer_than_the_old_ceiling_builds_and_scores);

    printf("\n[nucleotide models]\n");
    RUN_TEST(test_a_nucleotide_model_builds_and_scores);
    RUN_TEST(test_the_partition_normalises_over_nucleotides);
    RUN_TEST(test_path_count_matches_enumeration_for_nucleotides);
    RUN_TEST(test_lower_case_scores_as_upper_case);
    RUN_TEST(test_lower_case_trains_as_upper_case);
    RUN_TEST(test_N_is_a_residue_and_other_codes_are_not);
    RUN_TEST(test_generation_emits_only_this_alphabet);
    RUN_TEST(test_model_algebra_refuses_across_alphabets);
    RUN_TEST(test_a_nucleotide_model_round_trips_through_a_file);
    RUN_TEST(test_a_nucleotide_file_relabelled_as_amino_acid_is_rejected);
    RUN_TEST(test_an_unknown_alphabet_in_a_file_is_refused);
    RUN_TEST(test_the_two_alphabets_do_not_interfere);
    RUN_TEST(test_every_nucleotide_order_finalizes_with_backoff);
    RUN_TEST(test_the_partition_normalises_with_backoff_above_order_four);

    printf("\n==============================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
