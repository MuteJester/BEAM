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

PyObject *py_log_pseq(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; const char *seq;
    if (!PyArg_ParseTuple(args, "Os", &cap, &seq)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    return PyFloat_FromDouble(beam_model_log_pseq(m, seq));
}
PyObject *py_prism_factor(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *seqs;
    unsigned int omega = 1u;
    if (!PyArg_ParseTuple(args, "OO|I", &cap, &seqs, &omega)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;

    Py_ssize_t n = 0;
    const char **v = edge_borrow_strings(seqs, &n);
    if (!v) return NULL;
    double *out = (double *)PyMem_Malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!out) { PyMem_Free(v); return PyErr_NoMemory(); }

    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_prism_factor_batch(m, v, (uint32_t)n, (uint32_t)omega, out);
    Py_END_ALLOW_THREADS
    PyMem_Free(v);
    if (err != BEAM_OK) { PyMem_Free(out); return set_beam_error(err); }

    PyObject *list = PyList_New(n);
    if (!list) { PyMem_Free(out); return NULL; }
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *o = PyFloat_FromDouble(out[i]);
        if (!o) { Py_DECREF(list); PyMem_Free(out); return NULL; }
        PyList_SET_ITEM(list, i, o);
    }
    PyMem_Free(out);
    return list;
}

PyObject *py_prism_explain(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    const char *seq;
    unsigned int omega = 1u;
    if (!PyArg_ParseTuple(args, "Os|I", &cap, &seq, &omega)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;

    double sub[BEAM_MAX_LENGTH_CEILING], ins[BEAM_MAX_LENGTH_CEILING];
    double del[BEAM_MAX_LENGTH_CEILING + 1];
    double lb = 0.0, lq = 0.0, lp = 0.0;
    uint32_t len = 0;
    BeamError err = beam_prism_factor(m, seq, (uint32_t)omega, &lb, &lq,
                                      &lp, sub, ins, del, &len);
    if (err != BEAM_OK) return set_beam_error(err);

    PyObject *ls = PyList_New(len), *li = PyList_New(len),
             *ld = PyList_New((Py_ssize_t)len + 1);
    if (!ls || !li || !ld) { Py_XDECREF(ls); Py_XDECREF(li); Py_XDECREF(ld);
                             return NULL; }
    for (uint32_t i = 0; i < len; ++i) {
        PyList_SET_ITEM(ls, i, PyFloat_FromDouble(sub[i]));
        PyList_SET_ITEM(li, i, PyFloat_FromDouble(ins[i]));
    }
    for (uint32_t j = 0; j <= len; ++j)
        PyList_SET_ITEM(ld, j, PyFloat_FromDouble(del[j]));

    return Py_BuildValue("{s:d,s:d,s:d,s:N,s:N,s:N,s:I}",
                         "log_b", lb, "log_q", lq, "log_p0", lp,
                         "sub", ls, "ins", li, "del", ld, "length", len);
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
    if (!out) { PyMem_Free(v); return PyErr_NoMemory(); }

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
