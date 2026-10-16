#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "beam/model.h"
#include "beam/distance.h"
#include "beam/internal.h"
static int pass_count=0, fail_count=0;
#include "test_utils.h"

static unsigned ipow(unsigned b,unsigned e) { unsigned n=1; while(e--) n*=b; return n; }

static BeamModel *model(bool second, unsigned max_length, unsigned order, bool sparse_lengths) {
    BeamModelConfig cfg;
    beam_model_config_default_for(BEAM_ALPHABET_NT,&cfg);
    cfg.order=order; cfg.max_length=max_length;
    if (sparse_lengths) cfg.alpha_length=0;
    const char *strings[]={"A","CN","ACA","NTGC","G","NT","CNA","CGNT"};
    const double a[]={31,7,21,10,2,5,3,1}, b[]={2,5,3,1,31,7,21,10};
    BeamModel *m=beam_model_new(&cfg);
    if (m && (beam_model_observe(m,strings,second ? b : a,8)!=BEAM_OK || beam_model_finalize(m)!=BEAM_OK)) {
        beam_model_destroy(m); m=NULL;
    }
    return m;
}

static void direct_block(const BeamModel *m,unsigned L,unsigned start,unsigned width,double *out) {
    const char symbols[]="ACGTN";
    const unsigned size=ipow(5,width);
    memset(out,0,size*sizeof(double));
    char seq[8]={0};
    for (unsigned code=0;code<ipow(5,L);++code) {
        unsigned v=code, word=0;
        for (unsigned j=L;j-- >0;) { seq[j]=symbols[v%5]; v/=5; }
        seq[L]=0;
        for (unsigned j=start;j<start+width;++j) word=word*5+(unsigned)(strchr(symbols,seq[j])-symbols);
        out[word]+=exp(beam_model_log_pseq(m,seq)-beam_model_log_length_prior(m,L));
    }
}

static void test_local_distance_matches_enumeration_all_widths(void) {
    BeamModel *p=model(false,4,2,false),*q=model(true,4,2,false);
    ASSERT_MSG(p && q,"fixtures build");
    for (unsigned width=1;width<=3;++width) {
        BeamLocalDistanceConfig cfg={width,UINT32_MAX};
        BeamLocalDistanceResult *r=NULL;
        ASSERT_MSG(beam_local_distance(p,q,&cfg,&r)==BEAM_OK,"exact comparison succeeds");
        double expected=0, length_total=0, pattern_total=0;
        unsigned at=0;
        for (unsigned L=1;L<=4;++L) {
            const unsigned w=width<L ? width : L, nwin=L-w+1;
            const double pp=exp(beam_model_log_length_prior(p,L)),qq=exp(beam_model_log_length_prior(q,L));
            const double dl=sqrt(pp)-sqrt(qq), lc=.5*dl*dl;
            length_total+=lc;
            ASSERT_MSG(fabs(r->lengths[L-1].length_contribution-lc)<1e-14,"length term agrees");
            for (unsigned i=0;i<nwin;++i) {
                double pm[125],qm[125],sum=0;
                direct_block(p,L,i,w,pm);direct_block(q,L,i,w,qm);
                for (unsigned u=0;u<ipow(5,w);++u) {
                    const double d=sqrt(pm[u])-sqrt(qm[u]);
                    sum+=sqrt(pp*qq)*d*d/(2*nwin);
                }
                const BeamDistanceWindow *window=r->windows+at++;
                ASSERT_MSG(window->length==L && window->position==i && window->width==w,"window coordinates correct");
                ASSERT_MSG(fabs(window->contribution-sum)<3e-13,"window matches independently enumerated sequences");
                pattern_total+=sum;
            }
        }
        expected=length_total+pattern_total;
        ASSERT_MSG(fabs(r->squared_distance-expected)<5e-13,"total metric matches exhaustive sequence enumeration");
        ASSERT_MSG(fabs(r->pattern_squared_distance-pattern_total)<5e-13,"pattern sum agrees");
        ASSERT_MSG(fabs(r->reported_pattern_squared_distance-r->pattern_squared_distance)<5e-13 && r->remainder<5e-13,"all returned terms reconstruct total");
        ASSERT_MSG(r->n_windows==at && r->width==width,"complete output");
        for (unsigned j=0;j<r->n_patterns;++j) {
            const BeamDistancePattern *v=r->patterns+j;
            double pm[125],qm[125];
            direct_block(p,v->length,v->position,v->width,pm);
            direct_block(q,v->length,v->position,v->width,qm);
            ASSERT_MSG(fabs(v->p-pm[v->word_index])<3e-13 && fabs(v->q-qm[v->word_index])<3e-13,"returned term probabilities agree");
            if (j) ASSERT_MSG(r->patterns[j-1].contribution>=v->contribution,"global descending contribution order");
        }
        beam_local_distance_free(r);
    }
    beam_model_destroy(p);beam_model_destroy(q);PASS();
}

static void test_local_distance_topk_symmetry_identity_and_threads(void) {
    BeamModel *p=model(false,4,2,false),*q=model(true,4,2,false);
    ASSERT_MSG(p && q,"fixtures build");
    BeamLocalDistanceConfig all={0,UINT32_MAX}, few={0,7}, none={0,0};
    BeamLocalDistanceResult *full=NULL,*small=NULL,*swap=NULL,*empty=NULL,*self=NULL;
    beam_model_set_threads(1);
    ASSERT_MSG(beam_local_distance(p,q,&all,&full)==BEAM_OK,"all terms");
    beam_model_set_threads(3);
    ASSERT_MSG(beam_local_distance(p,q,&few,&small)==BEAM_OK,"bounded terms");
    ASSERT_MSG(beam_local_distance(q,p,&few,&swap)==BEAM_OK,"swapped models");
    ASSERT_MSG(beam_local_distance(p,q,&none,&empty)==BEAM_OK,"no retained terms");
    ASSERT_MSG(beam_local_distance(p,p,&all,&self)==BEAM_OK,"identity");
    ASSERT_MSG(small->width==3 && small->n_patterns==7,"default resolves to order plus one");
    ASSERT_MSG(full->squared_distance==small->squared_distance && full->squared_distance==swap->squared_distance && full->squared_distance==empty->squared_distance,"same totals for K, thread count and swap");
    ASSERT_MSG(self->squared_distance==0 && self->n_patterns==0,"identity exactly zero");
    ASSERT_MSG(empty->n_patterns==0 && empty->remainder==empty->pattern_squared_distance,"K=0 retains exact remainder");
    double retained=0;
    for (unsigned i=0;i<7;++i) {
        const BeamDistancePattern *a=full->patterns+i,*b=small->patterns+i,*c=swap->patterns+i;
        ASSERT_MSG(a->length==b->length && a->position==b->position && a->word_index==b->word_index && a->contribution==b->contribution,"bounded heap returns exact global top terms");
        ASSERT_MSG(c->length==b->length && c->position==b->position && c->word_index==b->word_index && c->p==b->q && c->q==b->p && c->contribution==b->contribution,"swap preserves coordinate ranks and reverses probabilities");
        retained+=b->contribution;
    }
    ASSERT_MSG(fabs(retained+small->remainder-small->pattern_squared_distance)<1e-14,"selected plus omitted exact accounting");
    beam_local_distance_free(full);beam_local_distance_free(small);beam_local_distance_free(swap);
    beam_local_distance_free(empty);beam_local_distance_free(self);
    beam_model_destroy(p);beam_model_destroy(q);PASS();
}

static void test_local_distance_different_maximum_lengths(void) {
    BeamModel *p=model(false,4,2,false),*q=model(true,2,2,false);
    ASSERT_MSG(p && q,"unequal length models build");
    BeamLocalDistanceConfig cfg={0,100}; BeamLocalDistanceResult *r=NULL;
    ASSERT_MSG(beam_local_distance(p,q,&cfg,&r)==BEAM_OK,"different maxima accepted");
    ASSERT_MSG(r->n_lengths==4,"union of supported lengths");
    for (unsigned i=2;i<4;++i) {
        const BeamDistanceLength *v=r->lengths+i;
        ASSERT_MSG(v->q==0 && v->pattern_contribution==0 && fabs(v->length_contribution-.5*v->p)<1e-14,"one-sided support contributes only length separation");
    }
    for (unsigned i=0;i<r->n_patterns;++i) ASSERT_MSG(r->patterns[i].length<=2,"no undefined probabilities returned");
    beam_local_distance_free(r);beam_model_destroy(p);beam_model_destroy(q);PASS();
}

static void test_local_distance_disjoint_length_support(void) {
    BeamModelConfig cfg;beam_model_config_default_for(BEAM_ALPHABET_NT,&cfg);
    cfg.order=2;cfg.max_length=4;
    BeamModel *p=beam_model_new(&cfg),*q=beam_model_new(&cfg);
    const char *a[]={"A"},*b[]={"ACGT"};
    ASSERT_MSG(p && q && beam_model_observe(p,a,NULL,1)==BEAM_OK && beam_model_observe(q,b,NULL,1)==BEAM_OK,"disjoint fixtures");
    ASSERT_MSG(beam_model_finalize(p)==BEAM_OK && beam_model_finalize(q)==BEAM_OK,"finalize disjoint fixtures");
    /* The current smoothing prior makes all in-range lengths positive.
     * Pin valid point-mass priors to exercise the general zero-support path. */
    for (unsigned L=1;L<=4;++L) {
        p->log_pi[L]=L==1 ? 0 : -INFINITY;
        q->log_pi[L]=L==4 ? 0 : -INFINITY;
    }
    BeamLocalDistanceConfig dc={0,UINT32_MAX};BeamLocalDistanceResult *r=NULL;
    ASSERT_MSG(beam_local_distance(p,q,&dc,&r)==BEAM_OK,"disjoint supports succeed");
    ASSERT_MSG(fabs(r->squared_distance-1)<2e-14 && fabs(r->length_squared_distance-1)<2e-14 && r->pattern_squared_distance==0,"maximal length separation");
    ASSERT_MSG(r->n_lengths==2 && r->n_patterns==0,"skip jointly unsupported lengths and cap huge K to zero");
    beam_local_distance_free(r);beam_model_destroy(p);beam_model_destroy(q);PASS();
}

static void test_local_distance_tie_order(void) {
    BeamModelConfig cfg;beam_model_config_default_for(BEAM_ALPHABET_NT,&cfg);
    cfg.order=2;cfg.max_length=1;cfg.alpha_auto=false;cfg.alpha_length_auto=false;
    BeamModel *p=beam_model_new(&cfg),*q=beam_model_new(&cfg);
    const char *s[]={"A","C"};const double pw[]={30,2},qw[]={2,30};
    ASSERT_MSG(p && q && beam_model_observe(p,s,pw,2)==BEAM_OK && beam_model_observe(q,s,qw,2)==BEAM_OK,"tie fixtures");
    ASSERT_MSG(beam_model_finalize(p)==BEAM_OK && beam_model_finalize(q)==BEAM_OK,"finalize tie fixtures");
    BeamLocalDistanceConfig dc={1,1};BeamLocalDistanceResult *r=NULL;
    ASSERT_MSG(beam_local_distance(p,q,&dc,&r)==BEAM_OK,"tied ranks succeed");
    ASSERT_MSG(r->n_patterns==1 && r->patterns[0].word_index==0,"ties retain lexicographically first word");
    beam_local_distance_free(r);beam_model_destroy(p);beam_model_destroy(q);PASS();
}

static void test_local_distance_refusals(void) {
    BeamModel *p=model(false,4,2,false),*q=model(true,4,3,false);
    ASSERT_MSG(p && q,"models build");
    BeamLocalDistanceConfig cfg={0,4};BeamLocalDistanceResult *r=NULL;
    ASSERT_MSG(beam_local_distance(NULL,p,&cfg,&r)!=BEAM_OK && !r,"null model");
    ASSERT_MSG(beam_local_distance(p,p,NULL,&r)!=BEAM_OK && !r,"null config");
    ASSERT_MSG(beam_local_distance(p,p,&cfg,NULL)!=BEAM_OK,"null output");
    ASSERT_MSG(beam_local_distance(p,q,&cfg,&r)!=BEAM_OK && !r,"different orders refused");
    cfg.width=4;
    ASSERT_MSG(beam_local_distance(p,p,&cfg,&r)!=BEAM_OK && !r,"too wide refused");
    cfg.width=0;
    const uint64_t budget=beam_model_memory_budget();beam_model_set_memory_budget(1);
    BeamError err=beam_local_distance(p,p,&cfg,&r);beam_model_set_memory_budget(budget);
    ASSERT_MSG(err==BEAM_ERR_BUDGET && !r,"refuse before allocations");
    BeamModelConfig config;beam_model_config_default_for(BEAM_ALPHABET_NT,&config);config.order=2;config.max_length=4;
    BeamModel *unbuilt=beam_model_new(&config);
    ASSERT_MSG(unbuilt && beam_local_distance(p,unbuilt,&cfg,&r)==BEAM_ERR_NOT_FINALIZED && !r,"unfinalized refused");
    beam_model_destroy(unbuilt);beam_model_destroy(p);beam_model_destroy(q);beam_local_distance_free(NULL);PASS();
}

int main(void) {
    beam_model_set_threads(2);
    RUN_TEST(test_local_distance_matches_enumeration_all_widths);
    RUN_TEST(test_local_distance_topk_symmetry_identity_and_threads);
    RUN_TEST(test_local_distance_different_maximum_lengths);
    RUN_TEST(test_local_distance_disjoint_length_support);
    RUN_TEST(test_local_distance_tie_order);
    RUN_TEST(test_local_distance_refusals);
    printf("%d passed, %d failed\n",pass_count,fail_count);
    return fail_count ? 1 : 0;
}
