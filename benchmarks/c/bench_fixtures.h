#ifndef BENCH_FIXTURES_H
#define BENCH_FIXTURES_H

/*
 * The workload every benchmark runs against: one training corpus and the
 * models built from it, for exactly the alphabet, order and max_length the
 * run asked for. Nothing here falls back to a different configuration -- a
 * fixture that cannot be built is reported as a failure, never substituted.
 *
 * The corpus is either a file (one sequence per line; the first tab-, comma-
 * or space-separated field is used) or a deterministic synthetic repertoire:
 * CDR3-like strings with a V-side prefix, a random junction and a J-side
 * suffix, lengths around 15 residues, never longer than max_length. For the
 * nucleotide alphabet the same amino-acid strings are written as codons.
 *
 * Trellis analytics cost the same on a 30-sequence and a 50,000-sequence
 * model (the hierarchical prior gives every string support, so the trellis is
 * full either way); the corpus size matters for observe, finalize, add,
 * subtract, posterior and save/load, which is why it is a parameter.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "beam/model.h"

typedef struct BenchWorkload {
    BeamAlphabet alphabet;
    uint8_t      order;
    uint8_t      max_length;
    uint32_t     length;        /* the per-length argument, <= max_length */
    uint32_t     corpus_size;   /* synthetic size, or a cap on a file (0 = all) */
    const char  *corpus_file;   /* NULL for the synthetic corpus */
    uint64_t     seed;
} BenchWorkload;

/* Model variants: two models of the same configuration from different data,
 * for the benchmarks that need a pair (mixture, add, subtract, posterior). */
enum { BENCH_VARIANT_PRIMARY = 0, BENCH_VARIANT_SECOND = 1, BENCH_N_VARIANTS = 2 };

/* Load or generate the corpora. Returns false (with a message on stderr)
 * if no usable corpus results. `jsonl` receives one "corpus" record. */
bool bench_fixtures_init(const BenchWorkload *w, FILE *jsonl);
void bench_fixtures_teardown(void);

/* The finalized model for a variant, built on first use (outside any timed
 * window) and reported as a "fixture" record. NULL if the build failed; the
 * reason is in `bench_fixture_error`. */
BeamModel *bench_fixture(uint32_t variant);
const char *bench_fixture_error(void);

/* A model configuration matching the workload, for benchmarks that build
 * their own model. */
void bench_workload_config(BeamModelConfig *cfg);

/* The corpus a variant is trained on. */
const char *const *bench_corpus(uint32_t variant, size_t *n);

/* A training sequence of length 2..max_length-1 (room for an insertion),
 * the same one every call. */
const char *bench_typical_sequence(void);

/* Release every built fixture's rebuildable caches, so each benchmark starts
 * from the same state whatever ran before it. */
void bench_fixtures_drop_caches(void);

const BenchWorkload *bench_workload(void);

#endif /* BENCH_FIXTURES_H */
