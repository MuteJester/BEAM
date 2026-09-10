/**
 * @file guards.c
 * @brief Validating the numbers a Python caller supplies. See guards.h.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

#include "beam/common.h"
#include "beam/model.h"
#include "guards.h"

int beam_py_count(PyObject *o, const char *name, uint32_t *out) {
    return beam_py_count_max(o, name, UINT32_MAX, out);
}

int beam_py_count_max(PyObject *o, const char *name, uint32_t hi,
                      uint32_t *out) {
    if (!PyLong_Check(o)) {
        /* An exact integer, not something merely convertible: accepting a
         * float here would silently truncate 2.9 to 2, which is the same
         * class of quiet answer this file exists to stop. */
        PyErr_Format(PyExc_TypeError, "%s must be an integer, not %s",
                     name, Py_TYPE(o)->tp_name);
        return -1;
    }
    int overflow = 0;
    const long long v = PyLong_AsLongLongAndOverflow(o, &overflow);
    if (overflow || v > (long long)hi || v < 0) {
        PyObject *r = PyObject_Repr(o);
        PyErr_Format(PyExc_ValueError, "%s must be between 0 and %u, got %S",
                     name, (unsigned)hi, r ? r : o);
        Py_XDECREF(r);
        return -1;
    }
    *out = (uint32_t)v;
    return 0;
}

/* BEAM.errors.<name>, or `fallback` if the module cannot be reached. Same
 * approach as common.c's own lookup: a broken install degrades to a builtin
 * rather than to a confusing failure. */
static PyObject *beam_py_error_class(const char *name, PyObject *fallback) {
    PyObject *mod = PyImport_ImportModule("BEAM.errors");
    if (!mod) { PyErr_Clear(); Py_INCREF(fallback); return fallback; }
    PyObject *cls = PyObject_GetAttrString(mod, name);
    Py_DECREF(mod);
    if (!cls) { PyErr_Clear(); Py_INCREF(fallback); return fallback; }
    return cls;
}

int beam_py_budget(uint64_t bytes, uint32_t n, const char *unit) {
    const uint64_t budget = beam_model_memory_budget();
    if (bytes <= budget) return 0;
    /*
     * `BudgetExceeded`, which is what the C layer raises for the same refusal
     * through `set_beam_error`. Raising a plain ValueError here would give one
     * failure two types depending on which layer noticed it first -- and a
     * caller writing `except BudgetExceeded` would catch the deep one and miss
     * the shallow one, which is the worse half of the pair to miss.
     */
    PyObject *cls = beam_py_error_class("BudgetExceeded", PyExc_MemoryError);
    PyErr_Format(cls,
                 "%u %s needs %llu bytes and the memory budget is %llu; "
                 "raise it with BEAM.set_memory_budget, or ask for fewer",
                 (unsigned)n, unit,
                 (unsigned long long)bytes, (unsigned long long)budget);
    Py_DECREF(cls);
    return -1;
}

/* PyASCIIObject header, then the characters and a NUL, then the list slot.
 * `stride` already covers the characters and the NUL. */
#define BEAM_PY_STR_HEADER 40u

/*
 * pymalloc hands out size classes, not exact sizes: anything up to 512 bytes
 * is rounded up to a multiple of 16. A 15-character row is a 55-byte object
 * occupying a 64-byte block, and charging 55 under-reported the real cost by
 * 13% -- measured, 126.0 MB charged against 144.5 MB of resident growth for
 * 2,000,000 draws. Rounding brings the charge to 144.0 MB against that 144.5,
 * which is the side of the estimate to be on.
 */
#define BEAM_PYMALLOC_ALIGN 16u

uint64_t beam_py_sequence_bytes(uint32_t n, uint32_t stride) {
    const uint64_t obj = (uint64_t)stride + BEAM_PY_STR_HEADER;
    const uint64_t block = (obj + (BEAM_PYMALLOC_ALIGN - 1u))
                         & ~(uint64_t)(BEAM_PYMALLOC_ALIGN - 1u);
    /* The flat buffer is still alive while the objects are built from it --
     * it is freed only after the list is complete -- so the peak holds both,
     * and the peak is what runs a machine out of memory. */
    return (uint64_t)n * (uint64_t)stride
         + (uint64_t)n * (block + sizeof(PyObject *));
}

void *beam_py_nomem(const char *what, uint64_t bytes) {
    PyObject *cls = beam_py_error_class("OutOfMemory", PyExc_MemoryError);
    PyErr_Format(cls, "out of memory allocating %s (%llu bytes)",
                 what, (unsigned long long)bytes);
    Py_DECREF(cls);
    return NULL;
}

void *beam_py_alloc(uint64_t bytes, const char *what,
                    uint32_t n, const char *unit) {
    if (unit && beam_py_budget(bytes, n, unit) < 0) return NULL;
    if (bytes > (uint64_t)PY_SSIZE_T_MAX) return beam_py_nomem(what, bytes);
    void *p = PyMem_Malloc((size_t)(bytes ? bytes : 1u));
    if (!p) return beam_py_nomem(what, bytes);
    return p;
}
