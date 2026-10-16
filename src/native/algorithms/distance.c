/** Streaming local-block marginals of the complete normalized sequence law. */
#include "beam/internal.h"
#include "beam/model_build.h"
#include "beam/distance.h"
#include <float.h>
#include <limits.h>

typedef struct {
    uint32_t n[BEAM_MAX_LENGTH_CEILING+1];
    size_t off[BEAM_MAX_LENGTH_CEILING+2];
    double *beta, *current, *next, *block;
} DistanceCache;

typedef struct { double sum, correction; } StableSum;
static void sum_add(StableSum *s, double x) {
    double t = s->sum+x;
    s->correction += fabs(s->sum) >= fabs(x) ? (s->sum-t)+x : (x-t)+s->sum;
    s->sum=t;
}
static double sum_value(const StableSum *s) { return s->sum+s->correction; }

static bool add_bytes(uint64_t *total, uint64_t count, uint64_t size) {
    if (size && count > (UINT64_MAX-*total)/size) return false;
    *total += count*size;
    return *total <= SIZE_MAX;
}

static uint32_t smaller(uint32_t a, uint32_t b) { return a < b ? a : b; }
static uint64_t power(uint32_t base, uint32_t exponent) {
    uint64_t n=1;
    for (uint32_t i=0; i<exponent; ++i) {
        if (n>UINT64_MAX/base) return UINT64_MAX;
        n*=base;
    }
    return n;
}
static double log_prior(const BeamModel *m, uint32_t L) {
    return L < m->n_len_slots ? m->log_pi[L] : -INFINITY;
}

static void cache_free(DistanceCache *c) {
    free(c->beta); free(c->current); free(c->next); free(c->block);
    memset(c,0,sizeof(*c));
}

static bool cache_bytes(const BeamModel *m, uint32_t L, uint64_t *bytes) {
    uint64_t states=0;
    for (uint32_t i=0; i<=L; ++i)
        if (!add_bytes(&states,power(m->n_residues,smaller(i,m->order)),1)) return false;
    return add_bytes(bytes,states,sizeof(double))
        && add_bytes(bytes,2*power(m->n_residues,smaller(L,m->order)),sizeof(double))
        && add_bytes(bytes,power(m->n_residues,smaller(L,m->order+1)),sizeof(double));
}

/* The sentinel-padded base-(R+1) state corresponding to a reachable base-R
 * context. No arrays of invalid sentinel-internal contexts are carried. */
static uint32_t original_state(uint32_t row, uint32_t width, const BeamModel *m) {
    uint32_t state=m->n_states-beam_ipow_u32(m->n_alphabet,width), place=1;
    for (uint32_t j=0; j<width; ++j) {
        state+=(row % m->n_residues)*place;
        row/=m->n_residues;
        place*=m->n_alphabet;
    }
    return state;
}

static BeamError cache_new(const BeamModel *m, uint32_t L, double *weights,
                           DistanceCache *c) {
    const uint32_t R=m->n_residues;
    for (uint32_t i=0; i<=L; ++i) {
        c->n[i]=beam_ipow_u32(R,smaller(i,m->order));
        c->off[i+1]=c->off[i]+c->n[i];
    }
    c->beta=malloc(c->off[L+1]*sizeof(double));
    c->current=calloc(c->n[L],sizeof(double));
    c->next=malloc((size_t)c->n[L]*sizeof(double));
    c->block=malloc((size_t)power(R,smaller(L,m->order+1))*sizeof(double));
    if (!c->beta || !c->current || !c->next || !c->block)
        return BEAM_EALLOC("local-distance backward cache and rolling marginals");
    beam_tail_weights(m,L,weights);
    for (uint32_t s=0; s<c->n[L]; ++s)
        c->beta[c->off[L]+s]=weights[original_state(s,smaller(L,m->order),m)];
    for (uint32_t i=L; i-- >0;) {
        beam_layer_weights(m,i,L,weights,NULL);
        const double *after=c->beta+c->off[i+1];
        double *before=c->beta+c->off[i];
        const uint32_t shift=c->n[i+1]/R, width=smaller(i,m->order);
        bool invalid=false;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(||:invalid) num_threads(beam_threads())
#endif
        for (int64_t s=0; s<c->n[i]; ++s) {
            const double *row=weights+(size_t)original_state((uint32_t)s,width,m)*R;
            const uint32_t base=((uint32_t)s % shift)*R;
            double maximum=-INFINITY;
            StableSum total={0};
            for (uint32_t x=0; x<R; ++x) {
                double v=row[x]+after[base+x];
                if (!isfinite(v) && v!=-INFINITY) invalid=true;
                if (v>maximum) maximum=v;
            }
            if (isfinite(maximum)) {
                for (uint32_t x=0; x<R; ++x) sum_add(&total,exp(row[x]+after[base+x]-maximum));
                before[s]=maximum+log(sum_value(&total));
            } else before[s]=maximum;
        }
        if (invalid) return BEAM_ERANGE("local-distance non-finite backward weight");
    }
    if (!isfinite(c->beta[0])) return BEAM_ERANGE("local-distance supported length has no finite partition");
    c->current[0]=1;
    return BEAM_OK;
}

static BeamError advance(const BeamModel *m, uint32_t L, uint32_t i,
                          uint32_t requested_width, double *weights, DistanceCache *c) {
    const uint32_t R=m->n_residues, shift=c->n[i+1]/R;
    const uint32_t context_width=smaller(i,m->order);
    const uint64_t block_size=(uint64_t)c->n[i]*R;
    const double *before=c->beta+c->off[i], *after=c->beta+c->off[i+1];
    bool invalid=false;
    beam_layer_weights(m,i,L,weights,NULL);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(||:invalid) num_threads(beam_threads())
#endif
    for (int64_t s=0; s<c->n[i]; ++s) {
        const double *row=weights+(size_t)original_state((uint32_t)s,context_width,m)*R;
        const uint32_t base=((uint32_t)s % shift)*R;
        for (uint32_t x=0; x<R; ++x) {
            double v=0;
            if (c->current[s]>0 && before[s]!=-INFINITY && row[x]!=-INFINITY && after[base+x]!=-INFINITY)
                v=c->current[s]*exp(row[x]+after[base+x]-before[s]);
            if (!isfinite(v) || v<0) invalid=true;
            c->block[(size_t)s*R+x]=v;
        }
    }
    if (invalid) return BEAM_ERANGE("local-distance invalid forward marginal");
    /* The destination is the suffix of each context-plus-residue word.
     * Each destination owns its reduction, independent of thread count. */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
    for (int64_t t=0; t<c->n[i+1]; ++t) {
        StableSum s={0};
        for (uint64_t b=(uint64_t)t; b<block_size; b+=c->n[i+1]) sum_add(&s,c->block[b]);
        c->next[t]=sum_value(&s);
    }
    StableSum mass={0};
    for (uint32_t t=0; t<c->n[i+1]; ++t) sum_add(&mass,c->next[t]);
    double total=sum_value(&mass);
    if (!isfinite(total) || fabs(total-1)>1e-9)
        return BEAM_ERANGE("local-distance marginal lost probability mass");
    for (uint32_t t=0; t<c->n[i+1]; ++t) c->next[t]/=total;
    double *temp=c->current; c->current=c->next; c->next=temp;
    if (i+1>=requested_width) {
        const uint64_t words=power(R,requested_width);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(||:invalid) num_threads(beam_threads())
#endif
        for (int64_t u=0; u<(int64_t)words; ++u) {
            StableSum s={0};
            for (uint64_t b=(uint64_t)u; b<block_size; b+=words) sum_add(&s,c->block[b]);
            double probability=sum_value(&s)/total;
            if (!isfinite(probability) || probability<0 || probability>1+1e-10) invalid=true;
            c->block[u]=probability>1 ? 1 : probability;
        }
        if (invalid) return BEAM_ERANGE("local-distance invalid block probability");
    }
    return BEAM_OK;
}

/* Negative means a ranks ahead of b. Heap root is the WORST retained term. */
static int pattern_order(const void *av, const void *bv) {
    const BeamDistancePattern *a=av, *b=bv;
    if (a->contribution!=b->contribution) return a->contribution>b->contribution ? -1 : 1;
    if (a->length!=b->length) return a->length<b->length ? -1 : 1;
    if (a->position!=b->position) return a->position<b->position ? -1 : 1;
    if (a->word_index!=b->word_index) return a->word_index<b->word_index ? -1 : 1;
    return 0;
}
static void offer(BeamLocalDistanceResult *r, uint32_t capacity, BeamDistancePattern v) {
    if (!capacity || !(v.contribution>0)) return;
    BeamDistancePattern *a=r->patterns;
    if (r->n_patterns<capacity) {
        uint32_t i=r->n_patterns++;
        while (i && pattern_order(&v,&a[(i-1)/2])>0) {
            a[i]=a[(i-1)/2]; i=(i-1)/2;
        }
        a[i]=v;
    } else if (pattern_order(&v,a)<0) {
        uint32_t i=0;
        while ((uint64_t)i*2+1<r->n_patterns) {
            uint32_t child=2*i+1;
            if (child+1<r->n_patterns && pattern_order(&a[child+1],&a[child])>0) ++child;
            if (pattern_order(&v,&a[child])>=0) break;
            a[i]=a[child]; i=child;
        }
        a[i]=v;
    }
}

/* The retained array already is a worst-first heap. In-place heap sort keeps
 * the memory contract independent of libc qsort's optional temporary array. */
static void sort_patterns(BeamDistancePattern *a, uint32_t n) {
    while (n>1) {
        BeamDistancePattern value=a[n-1];
        a[--n]=a[0];
        uint32_t i=0;
        while ((uint64_t)i*2+1<n) {
            uint32_t child=2*i+1;
            if (child+1<n && pattern_order(&a[child+1],&a[child])>0) ++child;
            if (pattern_order(&value,&a[child])>=0) break;
            a[i]=a[child]; i=child;
        }
        a[i]=value;
    }
}

void beam_local_distance_free(BeamLocalDistanceResult *r) {
    if (!r) return;
    free(r->lengths); free(r->windows); free(r->patterns); free(r);
}

BeamError beam_local_distance(const BeamModel *p, const BeamModel *q,
                              const BeamLocalDistanceConfig *config,
                              BeamLocalDistanceResult **out) {
    if (!out) return BEAM_ENULL("local-distance output");
    *out=NULL;
    if (!p || !q || !config) return BEAM_ENULL("local-distance models or config");
    if (!p->finalized || !q->finalized) return BEAM_ENOTBUILT();
    if (p->cfg.alphabet!=q->cfg.alphabet || p->order!=q->order)
        return BEAM_ERANGE("local-distance requires the same alphabet and model order");
    const uint32_t width=config->width ? config->width : p->order+1;
    if (width>p->order+1) return BEAM_ERANGE("local-distance width must be in 1..order+1, or 0 for the default");
    const uint32_t maximum=p->max_len>q->max_len ? p->max_len : q->max_len;
    uint32_t nlengths=0, nwindows=0, cap=config->top_k;
    uint64_t nterms=0, cache_peak=0;
    for (uint32_t L=1; L<=maximum; ++L) {
        const double lp=log_prior(p,L), lq=log_prior(q,L);
        if ((!isfinite(lp) && lp!=-INFINITY) || (!isfinite(lq) && lq!=-INFINITY))
            return BEAM_ERANGE("local-distance non-finite length prior");
        if (lp==-INFINITY && lq==-INFINITY) continue;
        ++nlengths;
        const uint32_t n=L-smaller(width,L)+1;
        nwindows+=n;
        if (lp==-INFINITY || lq==-INFINITY) continue;
        const uint64_t words=power(p->n_residues,smaller(width,L));
        if (words>UINT64_MAX/n || nterms>UINT64_MAX-words*n) nterms=UINT64_MAX;
        else nterms+=words*n;
        uint64_t cache=0;
        if (!cache_bytes(p,L,&cache) || !cache_bytes(q,L,&cache))
            return BEAM_ERANGE("local-distance cache exceeds addressable size");
        if (cache>cache_peak) cache_peak=cache;
    }
    if (nterms<cap) cap=(uint32_t)nterms;
    if (p==q) cap=0; /* identity has no strictly positive coordinates */
    uint64_t output=sizeof(BeamLocalDistanceResult), need=0;
    if (!add_bytes(&output,nlengths,sizeof(BeamDistanceLength)) ||
        !add_bytes(&output,nwindows,sizeof(BeamDistanceWindow)) ||
        !add_bytes(&output,cap,sizeof(BeamDistancePattern)))
        return BEAM_ERANGE("local-distance outputs exceed addressable size");
    need=output;
    if (!add_bytes(&need,cache_peak,1) ||
        (cache_peak && !add_bytes(&need,(uint64_t)p->n_states*p->n_residues,sizeof(double))) ||
        !add_bytes(&need,2,sizeof(DistanceCache)) || !add_bytes(&need,1024,1))
        return BEAM_ERANGE("local-distance buffers exceed addressable size");
    BEAM_BUDGET_OR_REFUSE("local-distance cached marginals and outputs",maximum,need);
    BeamLocalDistanceResult *r=calloc(1,sizeof(*r));
    DistanceCache pc={0}, qc={0};
    double *weights=NULL;
    BeamError err=BEAM_OK;
    if (!r) return BEAM_EALLOC("local-distance result");
    r->width=width; r->scratch_bytes=need; r->output_bytes=output;
    r->lengths=calloc(nlengths ? nlengths : 1,sizeof(*r->lengths));
    r->windows=calloc(nwindows ? nwindows : 1,sizeof(*r->windows));
    if (cap) r->patterns=malloc((size_t)cap*sizeof(*r->patterns));
    if (cache_peak) weights=malloc((size_t)p->n_states*p->n_residues*sizeof(double));
    if (!r->lengths || !r->windows || (cap && !r->patterns) || (cache_peak && !weights)) {
        err=BEAM_EALLOC("local-distance output or edge layer"); goto done;
    }
    StableSum lengths={0}, patterns={0};
    for (uint32_t L=1; L<=maximum; ++L) {
        const double lp=log_prior(p,L), lq=log_prior(q,L);
        if (lp==-INFINITY && lq==-INFINITY) continue;
        const uint32_t w=smaller(width,L), nwin=L-w+1;
        const double diff=exp(lp/2)-exp(lq/2), lc=.5*diff*diff;
        BeamDistanceLength *length=r->lengths+r->n_lengths++;
        *length=(BeamDistanceLength){L,exp(lp),exp(lq),lc,0};
        sum_add(&lengths,lc);
        if (lp==-INFINITY || lq==-INFINITY) {
            for (uint32_t i=0; i<nwin; ++i)
                r->windows[r->n_windows++]=(BeamDistanceWindow){L,i,w,0};
            continue;
        }
        err=cache_new(p,L,weights,&pc); if (err!=BEAM_OK) goto done;
        err=cache_new(q,L,weights,&qc); if (err!=BEAM_OK) goto done;
        const double coefficient=exp((lp+lq)/2)/(2*nwin);
        const uint64_t words=power(p->n_residues,w);
        StableSum length_patterns={0};
        for (uint32_t i=0; i<L; ++i) {
            err=advance(p,L,i,w,weights,&pc); if (err!=BEAM_OK) goto done;
            err=advance(q,L,i,w,weights,&qc); if (err!=BEAM_OK) goto done;
            if (i+1<w) continue;
            StableSum window={0};
            for (uint64_t u=0; u<words; ++u) {
                const double delta=sqrt(pc.block[u])-sqrt(qc.block[u]);
                const double term=coefficient*delta*delta;
                sum_add(&window,term);
                offer(r,cap,(BeamDistancePattern){L,i+1-w,w,u,pc.block[u],qc.block[u],term});
            }
            const double wc=sum_value(&window);
            r->windows[r->n_windows++]=(BeamDistanceWindow){L,i+1-w,w,wc};
            sum_add(&length_patterns,wc);
        }
        length->pattern_contribution=sum_value(&length_patterns);
        sum_add(&patterns,length->pattern_contribution);
        cache_free(&pc); cache_free(&qc);
    }
    r->length_squared_distance=sum_value(&lengths);
    r->pattern_squared_distance=sum_value(&patterns);
    r->squared_distance=r->length_squared_distance+r->pattern_squared_distance;
    sort_patterns(r->patterns,r->n_patterns);
    StableSum reported={0};
    for (uint32_t i=0; i<r->n_patterns; ++i) sum_add(&reported,r->patterns[i].contribution);
    r->reported_pattern_squared_distance=sum_value(&reported);
    r->remainder=r->pattern_squared_distance-r->reported_pattern_squared_distance;
    if (r->remainder<0 && r->remainder>-32*DBL_EPSILON*r->pattern_squared_distance) r->remainder=0;
    if (!isfinite(r->squared_distance) || r->remainder<0) {
        err=BEAM_ERANGE("local-distance non-finite or inconsistent attribution total"); goto done;
    }
    *out=r; r=NULL;
done:
    cache_free(&pc); cache_free(&qc); free(weights); beam_local_distance_free(r);
    return err;
}
