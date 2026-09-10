/**
 * @file guards.h
 * @brief Validating the numbers a Python caller supplies, before they size anything.
 *
 * Internal to src/native/bindings/python/. Not installed.
 *
 * TWO FAILURES THIS EXISTS TO PREVENT, both observed in this library.
 *
 * `PyArg_ParseTuple`'s "I" TRUNCATES. It takes a Python int modulo 2^32 with
 * no error, so `length_derivatives(q, 2**32 + 2)` was answered as order 2 --
 * a well-formed table for a question nobody asked, returned with no
 * indication. A guard taking the already-parsed C integer cannot catch that;
 * the truncation happened before it was called. So these take the PyObject*.
 *
 * A COUNT THAT PARSES CAN STILL NOT FIT. `sample(2**28)` claimed 3.5 GB for
 * its result before the library's own budget check ran, and reported a
 * `MemoryError` carrying an empty string -- no quantity, nothing the caller
 * could act on, and machine-dependent. A refusal has to name what was asked
 * for and what the limit is, in the units of the argument.
 *
 * Both return -1 with a Python exception set, 0 on success, so a call site
 * reads `if (beam_py_count(...) < 0) return NULL;`.
 */
#ifndef BEAM_PY_GUARDS_H
#define BEAM_PY_GUARDS_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

/**
 * A count argument, parsed without truncation.
 *
 * Accepts any Python integer in [0, UINT32_MAX] and rejects everything else
 * by name -- negative, non-integral, or too large to be what the caller
 * meant. `2**32` raises rather than becoming 0.
 */
int beam_py_count(PyObject *o, const char *name, uint32_t *out);

/**
 * The same, with an inclusive upper bound the caller's domain imposes.
 */
int beam_py_count_max(PyObject *o, const char *name, uint32_t hi,
                      uint32_t *out);

/**
 * Refuse a request whose result would not fit the memory budget.
 *
 * `bytes` is what the binding is about to allocate. `n` and `unit` describe
 * the request in the caller's own terms, so the message says "400000000 draws
 * of up to 12 characters" rather than a byte count they cannot map back to
 * anything they typed.
 */
int beam_py_budget(uint64_t bytes, uint32_t n, const char *unit);

/**
 * What `n` returned sequences cost at PEAK: the flat buffer, the `str`
 * objects built from it, and the list holding them -- all three alive at
 * once, because the buffer is freed only after the list is complete.
 *
 * A binding that returns a list of strings does not hand back the flat buffer
 * it filled -- it hands back one `str` object per row and a list to hold them,
 * and that is the larger figure by four times:
 *
 *     2,000,000 draws of ~12 characters
 *       C buffer, n * stride            30.0 MB   <- what was being checked
 *       list of pointers                16.0 MB
 *       the str objects                106.0 MB
 *       measured RSS growth            144.5 MB
 *
 * A compact-ASCII `str` is a 40-byte PyASCIIObject plus its bytes plus a NUL,
 * and the list holds an 8-byte pointer to each. So the honest per-row cost is
 * the buffer row AND both of those. Charging only the buffer meant a request
 * was admitted at roughly a quarter of what it would actually take, which is
 * how `sample(2**28)` passed a 4.29 GB budget and then asked CPython for
 * about 16 GB.
 *
 * IT DELIBERATELY OVER-CHARGES, because it must assume every row is as long
 * as the model allows and real rows are shorter, so their `str` objects land
 * in smaller pymalloc size classes. Measured against the true high-water mark,
 * one process per point:
 *
 *     stride 15, ~12-char rows    charged 174 MB   peak 157 MB   1.11x
 *     stride 28, ~22-char rows    charged 232 MB   peak 164 MB   1.42x
 *     stride 28, ~12-char rows    charged 232 MB   peak 180 MB   1.29x
 *
 * Over by a tenth to a half, never under. The figure it replaced was 30 MB
 * against a 157 MB peak -- 0.19x -- so a request was admitted at a fifth of
 * its cost. Refusing a little early is the side of that trade to be on.
 *
 * The 40 and the 16 are CPython's layout, not portable constants. Both are
 * measured rather than assumed: `sys.getsizeof` reports a flat 41-byte
 * overhead at every ASCII length on this build, and a 55-byte object occupies
 * a 64-byte block. A wrong value here moves a refusal threshold; it can never
 * make a buffer short, because nothing is allocated from this.
 */
uint64_t beam_py_sequence_bytes(uint32_t n, uint32_t stride);

/**
 * Fail an allocation the way the rest of the library fails.
 *
 * `PyErr_NoMemory()` raises MemoryError with an EMPTY string. It says nothing
 * about what could not be allocated or how large it was, and -- because the
 * exception hierarchy is reached by name from BEAM.errors -- it is not a
 * `BeamError`, so `except BeamError` does not catch it. errors.py has carried
 * a note about that gap, and a test has pinned it, since before this existed.
 *
 * This raises `BEAM.errors.OutOfMemory`, which IS both, with a message naming
 * the thing and its size. Falls back to a plain MemoryError if the module
 * cannot be imported, so a broken install degrades to today's behaviour
 * rather than to a confusing one.
 *
 * Always returns NULL, so a call site reads `return beam_py_nomem(...)`.
 */
void *beam_py_nomem(const char *what, uint64_t bytes);

/**
 * Allocate `bytes`, refusing first if that would exceed the memory budget.
 *
 * One call in place of "check, allocate, test, PyErr_NoMemory". `n` and `unit`
 * describe the request in the caller's terms for the refusal message; pass 0
 * and NULL where the size is the model's rather than the caller's, and the
 * budget check is skipped -- an allocation the caller cannot influence should
 * not be refused for being large, only reported when it fails.
 */
void *beam_py_alloc(uint64_t bytes, const char *what,
                    uint32_t n, const char *unit);

#endif /* BEAM_PY_GUARDS_H */
