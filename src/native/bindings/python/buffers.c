/**
 * @file buffers.c
 * @brief Buffer-protocol helpers for the array-passing entry points.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>

#include "beam/common.h"
#include "beam/model.h"
#include "beam/publicness.h"

/* Capsule tag. Carried over verbatim so a model pickled or passed between the
 * two packages during the transition is still recognised; it is renamed with
 * everything else in the rename step. */
#include "beam/python_bindings.h"

/* ── Buffer helpers for the publicness batch API ───────────── */

/* The publicness entry points move one PMF of up to 65,536 doubles per atom,
 * which is well past the point where a Python list round-trip makes sense.
 * They take the buffer protocol instead. Every array is float64 so the format
 * check stays portable across platforms; a complex128 array is handed over as
 * its float64 view, and integral quantities (multiplicities, bin edges) as
 * exact float64 integers. */

int get_f64_buffer(PyObject *obj, Py_buffer *view, int writable,
                          const char *what) {
    int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;
    if (writable) flags |= PyBUF_WRITABLE;

    if (PyObject_GetBuffer(obj, view, flags) != 0) {
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError,
                     "%s must be a C-contiguous%s float64 array",
                     what, writable ? ", writable" : "");
        return 0;
    }
    if (view->itemsize != (Py_ssize_t)sizeof(double) ||
        !view->format || strcmp(view->format, "d") != 0) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_TypeError, "%s must have dtype float64", what);
        return 0;
    }
    return 1;
}
void release_f64_buffers(Py_buffer *views, int n) {
    for (int i = 0; i < n; i++) PyBuffer_Release(&views[i]);
}
/* Acquire several float64 buffers at once, releasing what was already taken
 * if any of them fails. */
int get_f64_buffers(PyObject **objs, Py_buffer *views,
                           const int *writable, const char **names, int n) {
    for (int i = 0; i < n; i++) {
        if (!get_f64_buffer(objs[i], &views[i], writable[i], names[i])) {
            release_f64_buffers(views, i);
            return 0;
        }
    }
    return 1;
}
Py_ssize_t f64_buffer_count(const Py_buffer *view) {
    return view->len / (Py_ssize_t)sizeof(double);
}
int get_u32_buffer(PyObject *obj, Py_buffer *view, const char *what) {
    if (PyObject_GetBuffer(obj, view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) != 0) {
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError,
                     "%s must be a C-contiguous uint32 array", what);
        return 0;
    }
    if (view->itemsize != (Py_ssize_t)sizeof(uint32_t) ||
        !view->format || strcmp(view->format, "I") != 0) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_TypeError, "%s must have dtype uint32", what);
        return 0;
    }
    return 1;
}
