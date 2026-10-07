#ifndef BENCH_JSON_H
#define BENCH_JSON_H

/*
 * Minimal JSON-lines writer. Every record the harness emits is one line,
 * flushed as soon as it is written, so a run that crashes or is killed still
 * leaves every finished result on disk and names the benchmark it died in.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>

static inline void bench_json_str(FILE *f, const char *s) {
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p; ++p) {
        switch (*p) {
            case '"':  fputs("\\\"", f); break;
            case '\\': fputs("\\\\", f); break;
            case '\n': fputs("\\n", f); break;
            case '\r': fputs("\\r", f); break;
            case '\t': fputs("\\t", f); break;
            default:
                if (*p < 0x20) fprintf(f, "\\u%04x", *p);
                else fputc(*p, f);
        }
    }
    fputc('"', f);
}

/* Non-finite doubles have no JSON spelling; they become null. */
static inline void bench_json_num(FILE *f, double v) {
    if (isfinite(v)) fprintf(f, "%.17g", v);
    else fputs("null", f);
}

static inline void bench_json_key(FILE *f, const char *k, int *first) {
    if (!*first) fputc(',', f);
    *first = 0;
    bench_json_str(f, k);
    fputc(':', f);
}

#define BJ_STR(f, first, k, v) (bench_json_key((f), (k), (first)), bench_json_str((f), (v)))
#define BJ_NUM(f, first, k, v) (bench_json_key((f), (k), (first)), bench_json_num((f), (double)(v)))
#define BJ_U64(f, first, k, v) (bench_json_key((f), (k), (first)), fprintf((f), "%llu", (unsigned long long)(v)))
#define BJ_I64(f, first, k, v) (bench_json_key((f), (k), (first)), fprintf((f), "%lld", (long long)(v)))
#define BJ_BOOL(f, first, k, v) (bench_json_key((f), (k), (first)), fputs((v) ? "true" : "false", (f)))

#endif /* BENCH_JSON_H */
