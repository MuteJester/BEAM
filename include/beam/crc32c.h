/**
 * @file crc32c.h
 * @brief CRC-32C (Castagnoli) checksum, polynomial 0x1EDC6F41.
 *
 * `beam_crc32c_update` picks the fastest path the running CPU offers: the
 * SSE4.2 instruction where it exists, and a table otherwise. The choice is
 * made at run time inside `core/crc32c.c`, not at link time, and there is
 * nothing for a caller to select.
 *
 * This header used to say the acceleration "can be substituted at link time",
 * which was never how it worked: the fast path was a separate symbol in
 * `io/model_file.c` that dispatched at run time and left the entry points here
 * on the slow path.
 */
#ifndef BEAM_CRC32C_H
#define BEAM_CRC32C_H

#include "beam/common.h"

/** Compute CRC-32C over a byte buffer. */
uint32_t beam_crc32c(const void *data, size_t len);

/** Update a running CRC-32C with additional data, fastest path available. */
uint32_t beam_crc32c_update(uint32_t crc, const void *data, size_t len);

/**
 * The portable table implementation, bypassing any hardware path.
 *
 * Exposed for one reason: on a machine with SSE4.2 this is the only way to
 * reach the fallback, and a fallback nothing exercises is a fallback that
 * breaks on the first machine without the instruction. The test drives both
 * this and `beam_crc32c_update` against a reference it derives from the
 * polynomial itself.
 */
uint32_t beam_crc32c_software(uint32_t crc, const void *data, size_t len);

#endif /* BEAM_CRC32C_H */
