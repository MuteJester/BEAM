/** Cached compact trellises and bounded gapped-mask contrast search.
 * Raw edge potentials and the delayed backward sink are retained exactly.
 * Base-R reachable contexts replace the sentinel-padded base-(R+1) states.
 * Both halves of each conditional sweep use log space. No sequence sampling.
 */
#include "beam/internal.h"
#include "beam/model_build.h"
#include "beam/fingerprint.h"
#include <float.h>
#include <limits.h>

typedef struct {
    uint32_t L, R, k, max_states;
    uint32_t *n;
    size_t *off;
    double *edges, *beta, *cur, *next, *tail;
    double log_z;
} Cache;

typedef struct {
    uint32_t n;
    uint32_t pos[BEAM_FINGERPRINT_LOOKAHEAD];
    uint32_t aa[BEAM_FINGERPRINT_LOOKAHEAD];
    double p[BEAM_FINGERPRINT_LOOKAHEAD], q[BEAM_FINGERPRINT_LOOKAHEAD];
    double gain;
} Proposal;

typedef struct {
    bool done;
    Proposal best[BEAM_FINGERPRINT_LOOKAHEAD];
} LeafWork;

static double h(double p, double q) {
    const double d = sqrt(p) - sqrt(q);
    return 0.5 * d * d;
}

static void cache_free(Cache *c) {
    free(c->n); free(c->off); free(c->edges); free(c->beta);
    free(c->cur); free(c->next); free(c->tail);
    memset(c, 0, sizeof(*c));
}

/* Every dimension is bounded by the already validated model. */
static uint64_t cache_bytes(const BeamModel *m, uint32_t L) {
    uint64_t states = 0, edges = 0;
    uint32_t last = 1;
    for (uint32_t i = 0; i <= L; ++i) {
        const uint32_t n = beam_ipow_u32(m->n_residues, i < m->order ? i : m->order);
        states += n;
        if (i < L) edges += (uint64_t)n * m->n_residues;
        last = n;
    }
    return (edges + states + 3ull * last) * sizeof(double)
         + ((uint64_t)L + 2u) * sizeof(size_t)
         + ((uint64_t)L + 1u) * sizeof(uint32_t);
}

static uint32_t original_state(uint32_t row, uint32_t width,
                               const BeamModel *m) {
    uint32_t state = m->n_states - beam_ipow_u32(m->n_alphabet, width);
    uint32_t place = 1;
    for (uint32_t j = 0; j < width; ++j) {
        state += (row % m->n_residues) * place;
        row /= m->n_residues;
        place *= m->n_alphabet;
    }
    return state;
}

static BeamError cache_new(const BeamModel *m, uint32_t L, Cache *c) {
    c->L = L; c->R = m->n_residues; c->k = m->order;
    c->n = calloc((size_t)L + 1u, sizeof(*c->n));
    c->off = calloc((size_t)L + 2u, sizeof(*c->off));
    if (!c->n || !c->off) return BEAM_EALLOC("fingerprint cache dimensions");
    for (uint32_t i = 0; i <= L; ++i) {
        c->n[i] = beam_ipow_u32(c->R, i < c->k ? i : c->k);
        c->off[i+1] = c->off[i] + c->n[i];
    }
    c->max_states = c->n[L];
    c->edges = malloc(c->off[L] * c->R * sizeof(double));
    c->beta = malloc(c->off[L+1] * sizeof(double));
    c->cur = malloc((size_t)c->max_states * sizeof(double));
    c->next = malloc((size_t)c->max_states * sizeof(double));
    c->tail = malloc((size_t)c->max_states * sizeof(double));
    double *w = malloc((size_t)m->n_states * c->R * sizeof(double));
    if (!c->edges || !c->beta || !c->cur || !c->next || !c->tail || !w) {
        free(w); return BEAM_EALLOC("fingerprint cached edges and workspace");
    }
    for (uint32_t i = 0; i < L; ++i) {
        beam_layer_weights(m, i, L, w, NULL);
        const uint32_t width = i < c->k ? i : c->k;
        double *dest = c->edges + c->off[i] * c->R;
        for (uint32_t s = 0; s < c->n[i]; ++s)
            memcpy(dest + (size_t)s*c->R,
                   w + (size_t)original_state(s, width, m)*c->R,
                   c->R*sizeof(double));
        for (size_t j = 0; j < (size_t)c->n[i]*c->R; ++j)
            if (!isfinite(dest[j]) && dest[j] != -INFINITY) {
                free(w); return BEAM_ERANGE("fingerprint non-finite edge weight");
            }
    }
    beam_tail_weights(m, L, w);
    const uint32_t width = L < c->k ? L : c->k;
    for (uint32_t s = 0; s < c->n[L]; ++s)
        c->tail[s] = w[original_state(s, width, m)];
    free(w);
    return BEAM_OK;
}

/* Safe log-sum. Skip impossible edges before forming -inf - -inf. */
static double add_logs(double a, double b) {
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;
    if (a < b) { double t = a; a = b; b = t; }
    return a + log1p(exp(b-a));
}

static BeamError query(Cache *c, const uint32_t *mask, bool root,
                        double *mass, double *rows) {
    const uint32_t L = c->L, R = c->R;
    memcpy(c->beta+c->off[L], c->tail, (size_t)c->n[L]*sizeof(double));
    for (uint32_t layer = L; layer-- > 0;) {
        const double *edge = c->edges + c->off[layer]*R;
        const double *after = c->beta+c->off[layer+1];
        double *before = c->beta+c->off[layer];
        const uint32_t shift = c->n[layer+1]/R;
        const uint32_t allowed = mask[layer];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t s = 0; s < c->n[layer]; ++s) {
            const uint32_t base = ((uint32_t)s % shift)*R;
            double mx = -INFINITY, sum = 0;
            for (uint32_t mm = allowed; mm; mm &= mm-1u) {
                const uint32_t x = (uint32_t)__builtin_ctz(mm);
                const double v = edge[(size_t)s*R+x] + after[base+x];
                if (v > mx) mx = v;
            }
            if (mx != -INFINITY) {
                for (uint32_t mm = allowed; mm; mm &= mm-1u) {
                    const uint32_t x = (uint32_t)__builtin_ctz(mm);
                    sum += exp(edge[(size_t)s*R+x]+after[base+x]-mx);
                }
            }
            before[s] = mx == -INFINITY ? mx : mx + log(sum);
        }
    }
    const double z = c->beta[0];
    if (root) c->log_z = z;
    memset(rows, 0, (size_t)L*R*sizeof(double));
    if (z == -INFINITY) { *mass = 0; return BEAM_OK; }
    if (!isfinite(z) || !isfinite(c->log_z))
        return BEAM_ERANGE("fingerprint conditional sweep: non-finite partition");
    *mass = root ? 1.0 : exp(z-c->log_z);
    if (!isfinite(*mass) || *mass > 1.0+1e-10)
        return BEAM_ERANGE("fingerprint conditional sweep: invalid event mass");
    if (*mass > 1.0) *mass = 1.0;
    c->cur[0] = 0;
    for (uint32_t i = 0; i < L; ++i) {
        const double *edge = c->edges+c->off[i]*R;
        const double *after = c->beta+c->off[i+1];
        const uint32_t nnext = c->n[i+1], shift = nnext/R;
        const uint32_t sources = c->n[i]/shift;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int64_t t = 0; t < nnext; ++t) {
            const uint32_t x = (uint32_t)t % R, rest = (uint32_t)t/R;
            double v = -INFINITY;
            if ((mask[i] & (1u<<x)) && after[t] != -INFINITY) {
                for (uint32_t a = 0; a < sources; ++a) {
                    const uint32_t s = a*shift+rest;
                    v = add_logs(v, c->cur[s]+edge[(size_t)s*R+x]);
                }
            }
            c->next[t] = v;
        }
        /* One residue per worker, fixed state order: thread reproducibility. */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(beam_threads())
#endif
        for (int32_t x = 0; x < (int32_t)R; ++x) {
            double total = 0;
            for (uint32_t t = (uint32_t)x; t < nnext; t += R)
                total += exp(c->next[t]+after[t]-z);
            rows[(size_t)i*R+(uint32_t)x] = total;
        }
        double total = 0;
        for (uint32_t x = 0; x < R; ++x) total += rows[(size_t)i*R+x];
        if (!isfinite(total) || fabs(total-1.0) > 1e-9)
            return BEAM_ERANGE("fingerprint conditional sweep: marginal mass lost");
        for (uint32_t x = 0; x < R; ++x) rows[(size_t)i*R+x] /= total;
        double *tmp = c->cur; c->cur = c->next; c->next = tmp;
    }
    return BEAM_OK;
}

static void node_mask(const BeamFingerprintResult *r, uint32_t node,
                       uint32_t R, uint32_t *mask) {
    const uint32_t all = (1u<<R)-1u;
    for (uint32_t i = 0; i < r->length; ++i) mask[i] = all;
    int32_t at = (int32_t)node;
    while (r->nodes[at].parent >= 0) {
        const int32_t up = r->nodes[at].parent;
        const BeamFingerprintNode *parent = r->nodes+up;
        const uint32_t bit = 1u << parent->residue;
        mask[parent->position] &= parent->yes == at ? bit : (all^bit);
        at = up;
    }
}

static bool better(const Proposal *a, const Proposal *b) {
    if (!b->n) return true;
    const double va = a->gain/a->n, vb = b->gain/b->n;
    if (va > vb+1e-14) return true;
    if (vb > va+1e-14) return false;
    if (a->n != b->n) return a->n < b->n;
    for (uint32_t i = 0; i < a->n; ++i) {
        if (a->pos[i] != b->pos[i]) return a->pos[i] < b->pos[i];
        if (a->aa[i] != b->aa[i]) return a->aa[i] < b->aa[i];
    }
    return false;
}

/* Sorted bounded list. Half the slots use gain, half use one candidate per
 * final position to retain supported but zero-gain routes. Exact duplicates
 * are removed before probing. This is exploration, not exhaustive search. */
static void offer(Proposal *list, uint32_t cap, const Proposal *p) {
    if (!cap) return;
    for (uint32_t i = 0; i < cap; ++i) {
        if (better(p, list+i)) {
            memmove(list+i+1, list+i, (cap-i-1u)*sizeof(*list));
            list[i] = *p; return;
        }
    }
}

static bool same(const Proposal *a, const Proposal *b) {
    return a->n == b->n
        && memcmp(a->pos,b->pos,a->n*sizeof(uint32_t)) == 0
        && memcmp(a->aa,b->aa,a->n*sizeof(uint32_t)) == 0;
}

static BeamError examine(Cache *p, Cache *q, BeamFingerprintResult *r,
                         const BeamFingerprintConfig *cfg, uint32_t node,
                         LeafWork *work, Proposal *front, Proposal *pool,
                         Proposal *diverse, uint32_t *base, uint32_t *mask,
                         double *pr, double *qr) {
    const BeamFingerprintNode *leaf = r->nodes+node;
    work->done = true;
    uint32_t depth = cfg->lookahead;
    if (depth > cfg->max_depth-leaf->depth) depth = cfg->max_depth-leaf->depth;
    if (!depth || leaf->p == 0 || leaf->q == 0 || (leaf->p+leaf->q)*0.5 < cfg->min_mass)
        return BEAM_OK;
    const uint32_t B = cfg->beam_width, L = r->length, R = p->R;
    node_mask(r, node, R, base);
    memset(front, 0, B*sizeof(*front));
    uint32_t nfront = 1;
    for (uint32_t d = 0; d < depth; ++d) {
        memset(pool, 0, B*sizeof(*pool));
        memset(diverse, 0, L*sizeof(*diverse));
        for (uint32_t f = 0; f < nfront; ++f) {
            if (r->queries >= cfg->max_queries) { r->budget_exhausted = true; break; }
            const Proposal path = front[f];
            memcpy(mask, base, L*sizeof(*mask));
            for (uint32_t t = 0; t < path.n; ++t) mask[path.pos[t]] = 1u<<path.aa[t];
            double pm, qm;
            const bool root = node == 0 && d == 0;
            BeamError err = query(p, mask, root, &pm, pr);
            if (err != BEAM_OK) return err;
            err = query(q, mask, root, &qm, qr);
            if (err != BEAM_OK) return err;
            r->queries++;
            if (pm == 0 || qm == 0 || (pm+qm)*0.5 < cfg->min_mass) continue;
            /* Canonical increasing positions within a look-ahead chain avoid
             * visiting every permutation of the same positive conjunction. */
            const uint32_t begin = path.n ? path.pos[path.n-1]+1u : 0u;
            for (uint32_t i = begin; i < L; ++i) {
                if (__builtin_popcount(mask[i]) < 2) continue;
                for (uint32_t aa = 0; aa < R; ++aa) {
                    if (!(mask[i] & (1u<<aa))) continue;
                    const double py = pm*pr[(size_t)i*R+aa];
                    const double qy = qm*qr[(size_t)i*R+aa];
                    if ((py+qy)*0.5 < cfg->min_mass) continue;
                    Proposal child = path;
                    child.pos[d] = i; child.aa[d] = aa;
                    child.p[d] = py; child.q[d] = qy; child.n = d+1;
                    const double gain = h(py,qy)+h(pm-py,qm-qy)-h(pm,qm);
                    if (gain < -1e-10) return BEAM_ERANGE("fingerprint negative split gain");
                    child.gain += gain > 0 ? gain : 0;
                    if (child.gain >= cfg->min_gain && better(&child, &work->best[d]))
                        work->best[d] = child;
                    offer(pool, B, &child);
                    /* Prefer support on each position for the exploratory half. */
                    Proposal *v = diverse+i;
                    if (!v->n || py+qy > v->p[d]+v->q[d]+1e-14 ||
                        (fabs(py+qy-v->p[d]-v->q[d]) <= 1e-14 && better(&child,v)))
                        *v = child;
                }
            }
        }
        nfront = 0;
        const uint32_t ranked = (B+1u)/2u;
        for (uint32_t j = 0; j < ranked && pool[j].n; ++j) front[nfront++] = pool[j];
        for (uint32_t i = 0; i < L && nfront < B; ++i) {
            if (!diverse[i].n) continue;
            bool found = false;
            for (uint32_t j = 0; j < nfront; ++j) if (same(front+j,diverse+i)) found = true;
            if (!found) front[nfront++] = diverse[i];
        }
        for (uint32_t i = 0; i < B && pool[i].n && nfront < B; ++i) {
            bool found = false;
            for (uint32_t j = 0; j < nfront; ++j) if (same(front+j,pool+i)) found = true;
            if (!found) front[nfront++] = pool[i];
        }
        if (r->budget_exhausted || !nfront) break;
    }
    return BEAM_OK;
}

void beam_fingerprint_free(BeamFingerprintResult *r) {
    if (r) { free(r->nodes); free(r); }
}

BeamError beam_fingerprint_search(const BeamModel *p, const BeamModel *q,
                                  uint32_t L, const BeamFingerprintConfig *cfg,
                                  BeamFingerprintResult **out) {
    if (!out) return BEAM_ENULL("fingerprint output");
    *out = NULL;
    if (!p || !q || !cfg) return BEAM_ENULL("fingerprint models or config");
    if (!p->finalized || !q->finalized) return BEAM_ENOTBUILT();
    if (!beam_length_live(p,L)) return BEAM_EBADLEN(p,L);
    if (!beam_length_live(q,L)) return BEAM_EBADLEN(q,L);
    if (p->cfg.alphabet != q->cfg.alphabet) return BEAM_ERANGE("fingerprint alphabets differ");
    if (!cfg->max_leaves || cfg->max_leaves > INT32_MAX/2u ||
        cfg->max_depth > L || !cfg->lookahead || cfg->lookahead > BEAM_FINGERPRINT_LOOKAHEAD ||
        !cfg->beam_width || !cfg->max_queries || !isfinite(cfg->min_mass) ||
        cfg->min_mass < 0 || cfg->min_mass > 1 || !isfinite(cfg->min_gain) || cfg->min_gain <= 0)
        return BEAM_ERANGE("fingerprint: invalid leaf/depth/lookahead/beam/query or threshold budget");
    const uint32_t nmax = 2u*cfg->max_leaves-1u, R = p->n_residues;
    uint64_t need = cache_bytes(p,L)+cache_bytes(q,L);
    const uint64_t ps = (uint64_t)p->n_states*R, qs = (uint64_t)q->n_states*R;
    need += (ps > qs ? ps : qs)*sizeof(double);
    need += (uint64_t)nmax*(sizeof(BeamFingerprintNode)+sizeof(LeafWork));
    need += (2ull*cfg->beam_width+L)*sizeof(Proposal);
    need += 2ull*L*sizeof(uint32_t)+2ull*L*R*sizeof(double);
    need += sizeof(BeamFingerprintResult)+2u*sizeof(Cache);
    if (need > SIZE_MAX) return BEAM_ERANGE("fingerprint buffers exceed addressable size");
    BEAM_BUDGET_OR_REFUSE("fingerprint cached search",L,need);

    Cache pc = {0}, qc = {0};
    BeamFingerprintResult *r = calloc(1,sizeof(*r));
    LeafWork *work = calloc(nmax,sizeof(*work));
    Proposal *front = calloc(cfg->beam_width,sizeof(*front));
    Proposal *pool = calloc(cfg->beam_width,sizeof(*pool));
    Proposal *diverse = calloc(L,sizeof(*diverse));
    uint32_t *base = malloc(L*sizeof(*base)), *mask = malloc(L*sizeof(*mask));
    double *pr = malloc((size_t)L*R*sizeof(double)), *qr = malloc((size_t)L*R*sizeof(double));
    if (r) r->nodes = calloc(nmax,sizeof(*r->nodes));
    BeamError err = BEAM_OK;
    if (!r || !r->nodes || !work || !front || !pool || !diverse || !base || !mask || !pr || !qr) {
        err = BEAM_EALLOC("fingerprint search state"); goto done;
    }
    r->length = L; r->n_nodes = r->n_leaves = 1; r->scratch_bytes = need;
    r->nodes[0] = (BeamFingerprintNode){-1,-1,-1,-1,-1,0,1,1};
    if (cfg->max_leaves == 1 || cfg->max_depth == 0) goto success;
    err = cache_new(p,L,&pc); if (err != BEAM_OK) goto done;
    err = cache_new(q,L,&qc); if (err != BEAM_OK) goto done;
    while (r->n_leaves < cfg->max_leaves) {
        Proposal best = {0}; uint32_t chosen = 0;
        for (uint32_t i = 0; i < r->n_nodes; ++i) {
            if (r->nodes[i].yes >= 0) continue;
            if (!work[i].done && r->queries < cfg->max_queries) {
                err = examine(&pc,&qc,r,cfg,i,work+i,front,pool,diverse,base,mask,pr,qr);
                if (err != BEAM_OK) goto done;
            }
            for (uint32_t d = 0; d < cfg->lookahead; ++d) {
                const Proposal *v = &work[i].best[d];
                if (v->n && v->n <= cfg->max_leaves-r->n_leaves && better(v,&best)) {
                    best = *v; chosen = i;
                }
            }
        }
        if (!best.n) break;
        for (uint32_t j = 0; j < best.n; ++j) {
            BeamFingerprintNode *node = r->nodes+chosen;
            uint32_t yes = r->n_nodes++, no = r->n_nodes++;
            const double py = fmin(node->p,best.p[j]), qy = fmin(node->q,best.q[j]);
            node->position = (int32_t)best.pos[j]; node->residue = (int32_t)best.aa[j];
            node->yes = (int32_t)yes; node->no = (int32_t)no;
            r->nodes[yes] = (BeamFingerprintNode){(int32_t)chosen,-1,-1,-1,-1,node->depth+1,py,qy};
            r->nodes[no] = (BeamFingerprintNode){(int32_t)chosen,-1,-1,-1,-1,node->depth+1,node->p-py,node->q-qy};
            chosen = yes; r->n_leaves++;
        }
    }
    r->budget_exhausted = r->budget_exhausted || r->queries >= cfg->max_queries;
    for (uint32_t i = 0; i < r->n_nodes; ++i)
        if (r->nodes[i].yes < 0) r->squared_distance += h(r->nodes[i].p,r->nodes[i].q);
success:
    *out = r; r = NULL;
done:
    cache_free(&pc); cache_free(&qc); beam_fingerprint_free(r);
    free(work); free(front); free(pool); free(diverse); free(base); free(mask); free(pr); free(qr);
    return err;
}
