#include <math.h>
#include <stdio.h>
#include <string.h>
#include "beam/model.h"
#include "beam/fingerprint.h"
static int pass_count = 0, fail_count = 0;
#include "test_utils.h"

static BeamModel *make_model(bool reverse) {
    BeamModelConfig cfg;
    beam_model_config_default_for(BEAM_ALPHABET_NT,&cfg);
    cfg.order = 2; cfg.max_length = 3;
    const char *seq[] = {"ACA","CCA","TGA","ANT","TNT"};
    const double pa[] = {30,20,10,5,2}, qa[] = {2,5,10,20,30};
    BeamModel *m = beam_model_new(&cfg);
    if (!m) return NULL;
    if (beam_model_observe(m,seq,reverse ? qa : pa,5) != BEAM_OK ||
        beam_model_finalize(m) != BEAM_OK) { beam_model_destroy(m); return NULL; }
    return m;
}

static void test_fingerprint_partition_matches_sequence_enumeration(void) {
    BeamModel *a = make_model(false), *b = make_model(true);
    ASSERT_MSG(a && b,"models build");
    BeamFingerprintConfig cfg = {8,3,3,25,120,1e-8,1e-10};
    BeamFingerprintResult *r = NULL;
    ASSERT_MSG(beam_fingerprint_search(a,b,3,&cfg,&r) == BEAM_OK,"search succeeds");
    double p[15] = {0}, q[15] = {0};
    const char alphabet[] = "ACGTN";
    for (unsigned code = 0; code < 125; ++code) {
        unsigned copy = code, digits[3]; char s[4] = {0};
        for (int i = 2; i >= 0; --i) { digits[i]=copy%5; s[i]=alphabet[digits[i]]; copy/=5; }
        unsigned at = 0;
        while (r->nodes[at].yes >= 0) {
            const BeamFingerprintNode *n = r->nodes+at;
            at = digits[n->position] == (unsigned)n->residue ? (unsigned)n->yes : (unsigned)n->no;
        }
        p[at] += exp(beam_model_log_pseq(a,s)-beam_model_log_length_prior(a,3));
        q[at] += exp(beam_model_log_pseq(b,s)-beam_model_log_length_prior(b,3));
    }
    double mass_p = 0, mass_q = 0, h2 = 0;
    for (unsigned i = 0; i < r->n_nodes; ++i) {
        if (r->nodes[i].yes >= 0) continue;
        ASSERT_MSG(fabs(p[i]-r->nodes[i].p)<1e-12,"P agrees with independently scored sequences");
        ASSERT_MSG(fabs(q[i]-r->nodes[i].q)<1e-12,"Q agrees with independently scored sequences");
        mass_p += p[i]; mass_q += q[i];
        const double d = sqrt(p[i])-sqrt(q[i]); h2 += .5*d*d;
    }
    ASSERT_MSG(fabs(mass_p-1)<1e-12 && fabs(mass_q-1)<1e-12,"complete partition");
    ASSERT_MSG(fabs(h2-r->squared_distance)<1e-12,"distance agrees");
    ASSERT_MSG(r->queries<=cfg.max_queries && r->n_leaves<=cfg.max_leaves,"bounded search");
    beam_fingerprint_free(r); beam_model_destroy(a); beam_model_destroy(b);
    PASS();
}

static void test_fingerprint_refusals_and_root_only(void) {
    BeamModel *a = make_model(false);
    ASSERT_MSG(a,"model builds");
    BeamFingerprintConfig cfg = {8,3,2,16,50,1e-5,1e-7};
    BeamFingerprintResult *r = NULL;
    ASSERT_MSG(beam_fingerprint_search(NULL,a,3,&cfg,&r)!=BEAM_OK && !r,"null refused");
    cfg.lookahead=4;
    ASSERT_MSG(beam_fingerprint_search(a,a,3,&cfg,&r)!=BEAM_OK && !r,"bad lookahead refused");
    cfg.lookahead=2;
    const uint64_t budget = beam_model_memory_budget();
    beam_model_set_memory_budget(1);
    BeamError err = beam_fingerprint_search(a,a,3,&cfg,&r);
    beam_model_set_memory_budget(budget);
    ASSERT_MSG(err==BEAM_ERR_BUDGET && !r,"cache budget refused before allocation");
    cfg.max_leaves=1;
    ASSERT_MSG(beam_fingerprint_search(a,a,3,&cfg,&r)==BEAM_OK,"root-only succeeds");
    ASSERT_MSG(r->n_leaves==1 && r->queries==0 && r->squared_distance==0,"root-only exact");
    beam_fingerprint_free(r); beam_fingerprint_free(NULL); beam_model_destroy(a);
    PASS();
}

int main(void) {
    beam_model_set_threads(2);
    RUN_TEST(test_fingerprint_partition_matches_sequence_enumeration);
    RUN_TEST(test_fingerprint_refusals_and_root_only);
    printf("%d passed, %d failed\n",pass_count,fail_count);
    return fail_count ? 1 : 0;
}
