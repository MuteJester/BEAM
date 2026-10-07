#include "bench_fixtures.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "beam/errors.h"
#include "bench_clock.h"
#include "bench_json.h"

typedef struct Corpus {
    char  **seqs;
    size_t  n;
} Corpus;

static BenchWorkload g_w;
static FILE        *g_jsonl = NULL;
static Corpus       g_corpus[BENCH_N_VARIANTS];
static BeamModel   *g_model[BENCH_N_VARIANTS];
static bool         g_tried[BENCH_N_VARIANTS];
static char         g_error[512];

const BenchWorkload *bench_workload(void) { return &g_w; }
const char *bench_fixture_error(void) { return g_error; }

/* ── Synthetic repertoire ─────────────────────────────────────── */

static uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static uint32_t rnd_below(uint64_t *s, uint32_t n) { return (uint32_t)(splitmix64(s) % n); }

/* Approximately N(mu, sd) by summing uniforms (Irwin-Hall, 12 terms). */
static double rnd_normal(uint64_t *s, double mu, double sd) {
    double acc = 0.0;
    for (int i = 0; i < 12; ++i) acc += (double)(splitmix64(s) >> 11) * 0x1.0p-53;
    return mu + sd * (acc - 6.0);
}

static const char AA[] = "ACDEFGHIKLMNPQRSTVWY";
static const char *V_PREFIX[] = {"CASS", "CASS", "CASS", "CASS", "CASR", "CAST", "CAWS", "CSAR", "CASG"};
static const char *J_SUFFIX[] = {"EQYF", "YEQYF", "TEAFF", "NTIYF", "GELFF", "QPQHF", "TDTQYF", "SYNEQFF",
                                 "NQPQHF", "YGYTF"};
#define N_PREFIX (sizeof V_PREFIX / sizeof V_PREFIX[0])
#define N_SUFFIX (sizeof J_SUFFIX / sizeof J_SUFFIX[0])

static const char *codon_of(char aa) {
    switch (aa) {
        case 'A': return "GCC"; case 'C': return "TGT"; case 'D': return "GAC"; case 'E': return "GAG";
        case 'F': return "TTC"; case 'G': return "GGC"; case 'H': return "CAC"; case 'I': return "ATC";
        case 'K': return "AAG"; case 'L': return "CTG"; case 'M': return "ATG"; case 'N': return "AAC";
        case 'P': return "CCC"; case 'Q': return "CAG"; case 'R': return "CGG"; case 'S': return "AGC";
        case 'T': return "ACC"; case 'V': return "GTG"; case 'W': return "TGG"; default:  return "TAC";
    }
}

/* One CDR3-like amino-acid string of at most `max_aa` residues into `out`. */
static void synth_aa(uint64_t *s, uint32_t max_aa, char *out) {
    long t = lround(rnd_normal(s, 15.0, 2.5));
    if (t < 5) t = 5;
    if (t > (long)max_aa) t = (long)max_aa;
    const char *pre = V_PREFIX[rnd_below(s, N_PREFIX)];
    const char *suf = J_SUFFIX[rnd_below(s, N_SUFFIX)];
    const size_t lp = strlen(pre), ls = strlen(suf);
    size_t k = 0;
    if ((size_t)t >= lp + ls + 1) {
        memcpy(out, pre, lp); k = lp;
        const size_t mid = (size_t)t - lp - ls;
        for (size_t i = 0; i < mid; ++i) out[k++] = AA[rnd_below(s, 20)];
        memcpy(out + k, suf, ls); k += ls;
    } else {
        out[k++] = 'C';
        while (k < (size_t)t) out[k++] = AA[rnd_below(s, 20)];
    }
    out[k] = '\0';
}

static bool synth_corpus(Corpus *c, uint32_t n, uint64_t seed) {
    const bool nt = g_w.alphabet == BEAM_ALPHABET_NT;
    const uint32_t max_aa = nt ? g_w.max_length / 3u : g_w.max_length;
    if (max_aa < 1u) {
        snprintf(g_error, sizeof g_error, "max_length %u is too short for a synthetic corpus", g_w.max_length);
        return false;
    }
    c->seqs = (char **)calloc(n, sizeof(char *));
    if (!c->seqs) return false;
    uint64_t s = seed;
    char aa[256];
    for (uint32_t i = 0; i < n; ++i) {
        synth_aa(&s, max_aa, aa);
        const size_t la = strlen(aa);
        const size_t len = nt ? 3u * la : la;
        char *q = (char *)malloc(len + 1u);
        if (!q) return false;
        if (nt) {
            for (size_t j = 0; j < la; ++j) memcpy(q + 3u * j, codon_of(aa[j]), 3u);
            q[len] = '\0';
        } else {
            memcpy(q, aa, la + 1u);
        }
        c->seqs[i] = q;
        c->n = i + 1u;
    }
    return true;
}

/* ── File corpus ──────────────────────────────────────────────── */

static bool push_seq(Corpus *c, size_t *cap, const char *s, size_t len) {
    if (c->n == *cap) {
        size_t nc = *cap ? *cap * 2u : 4096u;
        char **ns = (char **)realloc(c->seqs, nc * sizeof(char *));
        if (!ns) return false;
        c->seqs = ns;
        *cap = nc;
    }
    char *q = (char *)malloc(len + 1u);
    if (!q) return false;
    memcpy(q, s, len);
    q[len] = '\0';
    c->seqs[c->n++] = q;
    return true;
}

static bool file_corpus(Corpus *all, const char *path, uint32_t cap_n) {
    FILE *f = fopen(path, "r");
    if (!f) {
        snprintf(g_error, sizeof g_error, "cannot open corpus file %s", path);
        return false;
    }
    char line[4096];
    size_t cap = 0;
    while (fgets(line, sizeof line, f)) {
        size_t len = strcspn(line, "\t, \r\n");
        if (len == 0) continue;
        if (!push_seq(all, &cap, line, len)) { fclose(f); return false; }
        if (cap_n && all->n >= cap_n) break;
    }
    fclose(f);
    if (all->n == 0) {
        snprintf(g_error, sizeof g_error, "corpus file %s holds no sequences", path);
        return false;
    }
    return true;
}

/* Variant 1 of a file corpus is every other sequence: a different model of
 * the same configuration, built from data the primary also saw. */
static bool split_odd(const Corpus *all, Corpus *odd) {
    odd->seqs = (char **)calloc(all->n / 2u + 1u, sizeof(char *));
    if (!odd->seqs) return false;
    for (size_t i = 1; i < all->n; i += 2) odd->seqs[odd->n++] = all->seqs[i];
    return true;
}

static void emit_corpus_record(void) {
    if (!g_jsonl) return;
    int first = 1;
    fputc('{', g_jsonl);
    BJ_STR(g_jsonl, &first, "type", "corpus");
    BJ_STR(g_jsonl, &first, "source", g_w.corpus_file ? g_w.corpus_file : "synthetic");
    BJ_U64(g_jsonl, &first, "seed", g_w.seed);
    for (uint32_t v = 0; v < BENCH_N_VARIANTS; ++v) {
        size_t total = 0;
        for (size_t i = 0; i < g_corpus[v].n; ++i) total += strlen(g_corpus[v].seqs[i]);
        char key[32];
        snprintf(key, sizeof key, "n_sequences_%u", v);
        BJ_U64(g_jsonl, &first, key, g_corpus[v].n);
        snprintf(key, sizeof key, "mean_length_%u", v);
        BJ_NUM(g_jsonl, &first, key, g_corpus[v].n ? (double)total / (double)g_corpus[v].n : 0.0);
    }
    fputs("}\n", g_jsonl);
    fflush(g_jsonl);
}

bool bench_fixtures_init(const BenchWorkload *w, FILE *jsonl) {
    g_w = *w;
    g_jsonl = jsonl;
    g_error[0] = '\0';
    bool ok;
    if (w->corpus_file) {
        ok = file_corpus(&g_corpus[0], w->corpus_file, w->corpus_size) && split_odd(&g_corpus[0], &g_corpus[1]);
    } else {
        const uint32_t n = w->corpus_size ? w->corpus_size : 20000u;
        ok = synth_corpus(&g_corpus[0], n, w->seed) && synth_corpus(&g_corpus[1], n, w->seed ^ 0xA5A5A5A5A5A5A5A5ULL);
    }
    if (!ok) {
        if (!g_error[0]) snprintf(g_error, sizeof g_error, "out of memory building the corpus");
        return false;
    }
    emit_corpus_record();
    return true;
}

void bench_workload_config(BeamModelConfig *cfg) {
    beam_model_config_default_for(g_w.alphabet, cfg);
    cfg->order = g_w.order;
    cfg->max_length = g_w.max_length;
}

const char *const *bench_corpus(uint32_t variant, size_t *n) {
    if (variant >= BENCH_N_VARIANTS) variant = 0;
    if (n) *n = g_corpus[variant].n;
    return (const char *const *)g_corpus[variant].seqs;
}

const char *bench_typical_sequence(void) {
    const Corpus *c = &g_corpus[BENCH_VARIANT_PRIMARY];
    for (size_t i = 0; i < c->n; ++i) {
        const size_t len = strlen(c->seqs[i]);
        if (len >= 2u && len + 1u <= g_w.max_length) return c->seqs[i];
    }
    return c->n ? c->seqs[0] : "";
}

static void emit_fixture_record(uint32_t v, const BeamModel *m, uint64_t observe_ns, uint64_t finalize_ns) {
    if (!g_jsonl) return;
    int first = 1;
    fputc('{', g_jsonl);
    BJ_STR(g_jsonl, &first, "type", "fixture");
    BJ_U64(g_jsonl, &first, "variant", v);
    BJ_STR(g_jsonl, &first, "alphabet", g_w.alphabet == BEAM_ALPHABET_NT ? "nt" : "aa");
    BJ_U64(g_jsonl, &first, "order", g_w.order);
    BJ_U64(g_jsonl, &first, "max_length", g_w.max_length);
    if (m) {
        BJ_U64(g_jsonl, &first, "n_states", beam_model_n_states(m));
        BJ_U64(g_jsonl, &first, "n_residues", beam_model_n_residues(m));
        BJ_U64(g_jsonl, &first, "radix", beam_model_radix(m));
        BJ_U64(g_jsonl, &first, "n_cells", beam_model_n_cells(m));
        BJ_U64(g_jsonl, &first, "n_nonzero_cells", beam_model_n_nonzero_cells(m));
        BJ_U64(g_jsonl, &first, "n_observed", beam_model_n_observed(m));
        BJ_U64(g_jsonl, &first, "n_skipped", beam_model_n_skipped(m));
        BJ_U64(g_jsonl, &first, "memory_bytes", beam_model_memory_bytes(m));
        BJ_U64(g_jsonl, &first, "observe_ns", observe_ns);
        BJ_U64(g_jsonl, &first, "finalize_ns", finalize_ns);
    } else {
        BJ_STR(g_jsonl, &first, "error", g_error);
    }
    fputs("}\n", g_jsonl);
    fflush(g_jsonl);
}

BeamModel *bench_fixture(uint32_t variant) {
    if (variant >= BENCH_N_VARIANTS) return NULL;
    if (g_model[variant] || g_tried[variant]) return g_model[variant];
    g_tried[variant] = true;

    BeamModelConfig cfg;
    bench_workload_config(&cfg);
    BeamModel *m = beam_model_new(&cfg);
    if (!m) {
        snprintf(g_error, sizeof g_error, "beam_model_new: %s", beam_error_message());
        emit_fixture_record(variant, NULL, 0, 0);
        return NULL;
    }
    size_t n = 0;
    const char *const *seqs = bench_corpus(variant, &n);
    const uint64_t t0 = bench_now_ns();
    BeamError e = beam_model_observe(m, seqs, NULL, (uint32_t)n);
    const uint64_t t1 = bench_now_ns();
    if (e == BEAM_OK) e = beam_model_finalize(m);
    const uint64_t t2 = bench_now_ns();
    if (e != BEAM_OK) {
        snprintf(g_error, sizeof g_error, "fixture %u: %s", variant, beam_error_message());
        beam_model_destroy(m);
        emit_fixture_record(variant, NULL, 0, 0);
        return NULL;
    }
    g_model[variant] = m;
    emit_fixture_record(variant, m, t1 - t0, t2 - t1);
    return m;
}

void bench_fixtures_drop_caches(void) {
    for (uint32_t v = 0; v < BENCH_N_VARIANTS; ++v)
        if (g_model[v]) beam_model_drop_caches(g_model[v]);
}

void bench_fixtures_teardown(void) {
    for (uint32_t v = 0; v < BENCH_N_VARIANTS; ++v) {
        if (g_model[v]) beam_model_destroy(g_model[v]);
        g_model[v] = NULL;
        g_tried[v] = false;
    }
    /* Variant 1 of a file corpus borrows its strings from variant 0. */
    const bool borrowed = g_w.corpus_file != NULL;
    for (uint32_t v = 0; v < BENCH_N_VARIANTS; ++v) {
        if (!(borrowed && v == 1))
            for (size_t i = 0; i < g_corpus[v].n; ++i) free(g_corpus[v].seqs[i]);
        free(g_corpus[v].seqs);
        g_corpus[v].seqs = NULL;
        g_corpus[v].n = 0;
    }
}
