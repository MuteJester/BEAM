#include "bench_framework.h"
#include "beam/crc32c.h"
#include "beam/rng.h"

#define CRC_BYTES (64u * 1024u)

static unsigned char *crc_buffer(BenchContext *ctx) {
    unsigned char *buf = (unsigned char *)bench_scratch(ctx, CRC_BYTES);
    if (buf)
        for (uint32_t i = 0; i < CRC_BYTES; ++i) buf[i] = (unsigned char)(i * 2654435761u >> 24);
    return buf;
}

BEAM_BENCHMARK(core, crc32c_hw, "CRC32C over a 64 KiB buffer (hardware path when available)") {
    unsigned char *buf = crc_buffer(ctx);
    if (!buf) return;
    bench_set_items(ctx, CRC_BYTES, "bytes");
    uint32_t c = 0;
    BEAM_TIMED_BLOCK(ctx) {
        c ^= beam_crc32c(buf, CRC_BYTES);
    }
    BENCH_KEEP(c);
}

BEAM_BENCHMARK(core, crc32c_sw, "CRC32C over a 64 KiB buffer, portable software path") {
    unsigned char *buf = crc_buffer(ctx);
    if (!buf) return;
    bench_set_items(ctx, CRC_BYTES, "bytes");
    uint32_t c = 0;
    BEAM_TIMED_BLOCK(ctx) {
        c ^= beam_crc32c_software(0, buf, CRC_BYTES);
    }
    BENCH_KEEP(c);
}

BEAM_BENCHMARK(core, crc32c_update, "Incremental CRC32C: 64 updates of 1 KiB") {
    unsigned char *buf = crc_buffer(ctx);
    if (!buf) return;
    bench_set_items(ctx, CRC_BYTES, "bytes");
    uint32_t c = 0;
    BEAM_TIMED_BLOCK(ctx) {
        for (uint32_t i = 0; i < 64u; ++i) c = beam_crc32c_update(c, buf + i * 1024u, 1024u);
    }
    BENCH_KEEP(c);
}

BEAM_BENCHMARK(core, rng_next, "xoshiro256++ next: 1,000 draws") {
    BeamRng rng;
    beam_rng_seed(&rng, 123456789ULL);
    bench_set_items(ctx, 1000, "draws");
    uint64_t sink = 0;
    BEAM_TIMED_BLOCK(ctx) {
        for (int i = 0; i < 1000; ++i) sink += beam_rng_next(&rng);
    }
    BENCH_KEEP(sink);
}

BEAM_BENCHMARK(core, rng_double, "Uniform double in [0, 1): 1,000 draws") {
    BeamRng rng;
    beam_rng_seed(&rng, 987654321ULL);
    bench_set_items(ctx, 1000, "draws");
    double sink = 0.0;
    BEAM_TIMED_BLOCK(ctx) {
        for (int i = 0; i < 1000; ++i) sink += beam_rng_double(&rng);
    }
    BENCH_KEEP(sink);
}

BEAM_BENCHMARK(core, rng_bounded, "Bounded integer draw below 100: 1,000 draws") {
    BeamRng rng;
    beam_rng_seed(&rng, 13579ULL);
    bench_set_items(ctx, 1000, "draws");
    uint64_t sink = 0;
    BEAM_TIMED_BLOCK(ctx) {
        for (int i = 0; i < 1000; ++i) sink += beam_rng_bounded(&rng, 100);
    }
    BENCH_KEEP(sink);
}

BEAM_BENCHMARK(core, rng_seed, "SplitMix64 state expansion: 1,000 seedings") {
    BeamRng rng;
    bench_set_items(ctx, 1000, "seedings");
    uint64_t sink = 0;
    BEAM_TIMED_BLOCK(ctx) {
        for (uint64_t i = 0; i < 1000u; ++i) {
            beam_rng_seed(&rng, i);
            sink += rng.s[0];
        }
    }
    BENCH_KEEP(sink);
}

BEAM_BENCHMARK(core, error_lookup, "Thread-local error message read and clear") {
    const char *msg = NULL;
    BEAM_TIMED_BLOCK(ctx) {
        msg = beam_error_message();
        beam_clear_error();
    }
    BENCH_KEEP(msg);
}

BEAM_BENCHMARK(core, error_lifecycle, "Thread-local error set (formatted), read, message, clear") {
    BeamError err = BEAM_OK;
    const char *msg = NULL;
    BEAM_TIMED_BLOCK(ctx) {
        beam_set_error(BEAM_ERR_INVALID_ARG, "benchmark error %d", 42);
        err = beam_last_error();
        msg = beam_error_message();
        beam_clear_error();
    }
    BENCH_KEEP(err);
    BENCH_KEEP(msg);
}
