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

#endif /* BEAM_PY_GUARDS_H */
