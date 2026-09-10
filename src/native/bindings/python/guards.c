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

int beam_py_budget(uint64_t bytes, uint32_t n, const char *unit) {
    const uint64_t budget = beam_model_memory_budget();
    if (bytes <= budget) return 0;
    PyErr_Format(PyExc_ValueError,
                 "%u %s needs %llu bytes and the memory budget is %llu; "
                 "raise it with BEAM.set_memory_budget, or ask for fewer",
                 (unsigned)n, unit,
                 (unsigned long long)bytes, (unsigned long long)budget);
    return -1;
}
