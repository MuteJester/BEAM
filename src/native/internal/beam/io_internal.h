/**
 * @file io_internal.h
 * @brief Internals of the model-file format, shared with its tests.
 *
 * Internal to src/native/. Not a public header and not installed.
 *
 * This exists because `beam_model_crc32c` had been declared in the error-macro
 * header, where it has nothing to do with anything else in the file. That went
 * unnoticed until the header was renamed to `beam/error_macros.h`: the
 * declaration went with it, `tests/native/test_alphabet.c` did not follow, and
 * the call fell back to an implicit `int` declaration. It happened to keep
 * working because `uint32_t` and `int` are returned in the same register --
 * and it is a hard error, not a warning, on GCC 14 and Clang 16.
 *
 * A checksum in an errors header is the kind of thing that tells a reader the
 * file boundaries are not meaningful. Giving it an address of its own is the
 * fix; putting it back would only hide the next instance.
 */
#ifndef BEAM_INTERNAL_IO_H
#define BEAM_INTERNAL_IO_H

#include <stddef.h>
#include <stdint.h>

/**
 * CRC-32C over `data`, continuing from `crc`.
 *
 * The model file's integrity check. Distinct from the public
 * `include/beam/crc32c.h` entry points: this one is the incremental form the
 * reader and writer thread through a whole file, and the tests pin it against
 * known vectors.
 */
uint32_t beam_model_crc32c(uint32_t crc, const void *data, size_t len);

#endif /* BEAM_INTERNAL_IO_H */
