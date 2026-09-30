/**
 * @file scoring_bindings.c
 * @brief Scalar and batch scoring, and the per-residue decomposition.
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
#include "guards.h"

PyObject *py_log_pseq(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; const char *seq;
    if (!PyArg_ParseTuple(args, "Os", &cap, &seq)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    return PyFloat_FromDouble(beam_model_log_pseq(m, seq));
}
PyObject *py_log_pseq_batch(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *seqs;
    if (!PyArg_ParseTuple(args, "OO", &cap, &seqs)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;

    Py_ssize_t n = 0;
    const char **v = edge_borrow_strings(seqs, &n);
    if (!v) return NULL;
    double *out = (double *)PyMem_Malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!out) { PyMem_Free(v);
                return beam_py_nomem("the score buffer",
                                     (uint64_t)n * sizeof(double)); }

    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_log_pseq_batch(m, v, (uint32_t)n, out);
    Py_END_ALLOW_THREADS
    PyMem_Free(v);
    if (err != BEAM_OK) { PyMem_Free(out); return set_beam_error(err); }

    PyObject *list = PyList_New(n);
    if (!list) { PyMem_Free(out); return NULL; }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *f = PyFloat_FromDouble(out[i]);
        if (!f) { Py_DECREF(list); PyMem_Free(out); return NULL; }
        PyList_SET_ITEM(list, i, f);
    }
    PyMem_Free(out);
    return list;
}
/* log_pseq_packed(capsule, data, offsets, out): data any bytes-like buffer
 * (or None when every sequence is empty), offsets C-contiguous int64 with
 * n + 1 entries, out a writable float64 array of n. Scores in place, without
 * the GIL. */
PyObject *py_log_pseq_packed(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *data_obj, *off_obj, *out_obj;
    if (!PyArg_ParseTuple(args, "OOOO", &cap, &data_obj, &off_obj, &out_obj))
        return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;

    Py_buffer data = {0}, off, out;
    int have_data = 0;
    if (data_obj != Py_None) {
        if (PyObject_GetBuffer(data_obj, &data, PyBUF_SIMPLE) != 0) {
            PyErr_Clear();
            PyErr_SetString(PyExc_TypeError, "data must be a bytes-like buffer");
            return NULL;
        }
        have_data = 1;
    }
    if (PyObject_GetBuffer(off_obj, &off, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) != 0) {
        PyErr_Clear();
        if (have_data) PyBuffer_Release(&data);
        PyErr_SetString(PyExc_TypeError, "offsets must be a C-contiguous int64 array");
        return NULL;
    }
    if (off.itemsize != 8 || !off.format ||
        (strcmp(off.format, "q") != 0 && strcmp(off.format, "l") != 0) ||
        off.len < 8) {
        PyBuffer_Release(&off);
        if (have_data) PyBuffer_Release(&data);
        PyErr_SetString(PyExc_TypeError,
                        "offsets must be int64 with at least one entry");
        return NULL;
    }
    if (!get_f64_buffer(out_obj, &out, 1, "out")) {
        PyBuffer_Release(&off);
        if (have_data) PyBuffer_Release(&data);
        return NULL;
    }
    const uint64_t n = (uint64_t)(off.len / 8) - 1u;
    if ((uint64_t)f64_buffer_count(&out) != n) {
        PyBuffer_Release(&out); PyBuffer_Release(&off);
        if (have_data) PyBuffer_Release(&data);
        PyErr_Format(PyExc_ValueError, "out holds %zd scores, offsets describe %llu",
                     f64_buffer_count(&out), (unsigned long long)n);
        return NULL;
    }

    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_log_pseq_packed(m, have_data ? (const char *)data.buf : NULL,
                                     have_data ? (uint64_t)data.len : 0u,
                                     (const int64_t *)off.buf, n, (double *)out.buf);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&out); PyBuffer_Release(&off);
    if (have_data) PyBuffer_Release(&data);
    if (err != BEAM_OK) return set_beam_error(err);
    Py_RETURN_NONE;
}

PyObject *py_explain(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; const char *seq;
    if (!PyArg_ParseTuple(args, "Os", &cap, &seq)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;

    double fwd[BEAM_MAX_LENGTH_CEILING], bwd[BEAM_MAX_LENGTH_CEILING];
    uint32_t len = 0;
    BeamError err = beam_model_explain(m, seq, fwd, bwd, &len);
    if (err != BEAM_OK) return set_beam_error(err);

    PyObject *lf = PyList_New(len), *lb = PyList_New(len);
    if (!lf || !lb) { Py_XDECREF(lf); Py_XDECREF(lb); return NULL; }
    for (uint32_t i = 0; i < len; i++) {
        PyList_SET_ITEM(lf, i, PyFloat_FromDouble(fwd[i]));
        PyList_SET_ITEM(lb, i, PyFloat_FromDouble(bwd[i]));
    }
    return Py_BuildValue("(NN)", lf, lb);
}
