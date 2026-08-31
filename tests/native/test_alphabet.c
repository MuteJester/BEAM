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

/*
 * The safety gate of this step. The kernels still resolve the radix through
 * the preprocessor, so a nucleotide model would allocate radix-6 tables and
 * then be indexed at radix 21 -- reading off the end of the allocation rather
 * than merely giving a wrong answer. Construction must refuse until the
 * kernels read the model, and this test is what stops the refusal being
 * removed before then.
 */
static void test_construction_refuses_an_alphabet_the_kernels_cannot_read(void)
{
    BeamModelConfig c = valid_for(BEAM_ALPHABET_NT);
    ASSERT_MSG(beam_config_valid(&c),
               "the nucleotide config is valid; it is the build that refuses");

    BeamModel *m = beam_model_new(&c);
    ASSERT_MSG(m == NULL, "a nucleotide model must not build yet");

    const char *msg = beam_error_message();
    ASSERT_MSG(msg && strstr(msg, "nt") != NULL,
               "the refusal must name the alphabet it refused");
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
    RUN_TEST(test_construction_refuses_an_alphabet_the_kernels_cannot_read);
    RUN_TEST(test_construction_refuses_an_unknown_alphabet);
    RUN_TEST(test_a_model_longer_than_the_old_ceiling_builds_and_scores);

    printf("\n==============================================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
