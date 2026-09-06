/**
 * @file common.c
 * @brief Error mapping, capsule ownership, and argument conversion.
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
#include "beam/alphabet.h"
#include "beam/publicness.h"

/* Capsule tag. Carried over verbatim so a model pickled or passed between the
 * two packages during the transition is still recognised; it is renamed with
 * everything else in the rename step. */
#include "beam/python_bindings.h"

/*
 * Map a C error code onto a Python exception.
 *
 * Adapted from LZGraphs' version. The gene-data, graph-operation and
 * convergence branches referred to exception types that only exist in that
 * package and to failures this model cannot produce, so they fold into the
 * generic cases rather than being carried across as dead code. The message
 * itself always comes from `beam_error_message()`, which is where the useful
 * detail lives -- a bare code reaches Python as "invalid argument" and is
 * worth nothing to a caller.
 */
PyObject *set_beam_error(BeamError err) {
    const char *msg = beam_error_message();
    const int has_msg = msg && msg[0];

    switch (err) {
        case BEAM_ERR_BUDGET:
            /*
             * MemoryError, the same class BEAM_ERR_ALLOC produces, so this
             * commit changes what is DISTINGUISHABLE without changing what is
             * caught: `except MemoryError` keeps working everywhere it did.
             * The Python hierarchy will give this its own subclass, at which
             * point "raise your budget" becomes catchable apart from "buy
             * more RAM" -- which is the point of splitting the code out.
             */
        case BEAM_ERR_ALLOC:
            /* PyErr_NoMemory() sets MemoryError with no message, which is
             * right for a bare allocation failure and wrong for a refusal
             * that has something to say -- "this configuration needs 41 GB
             * and the budget is 4" tells a caller what to change, and losing
             * it leaves them a blank MemoryError to guess at. Keep the type,
             * keep the detail. */
            if (has_msg) {
                PyErr_SetString(PyExc_MemoryError, msg);
                return NULL;
            }
            return PyErr_NoMemory();

        case BEAM_ERR_NULL_ARG:
        case BEAM_ERR_EMPTY_INPUT:
        case BEAM_ERR_INVALID_SEQUENCE:
        case BEAM_ERR_INVALID_VARIANT:
        case BEAM_ERR_LENGTH_MISMATCH:
        case BEAM_ERR_PARAM_OUT_OF_RANGE:
            PyErr_SetString(PyExc_ValueError, has_msg ? msg : "invalid argument");
            return NULL;

        case BEAM_ERR_NOT_BUILT:
        case BEAM_ERR_NO_LIVE_PATHS:
        case BEAM_ERR_HAS_CYCLES:
            PyErr_SetString(PyExc_RuntimeError, has_msg ? msg : "model state error");
            return NULL;

        case BEAM_ERR_IO_OPEN:
        case BEAM_ERR_IO_READ:
        case BEAM_ERR_IO_WRITE:
            PyErr_SetString(PyExc_OSError, has_msg ? msg : "I/O error");
            return NULL;

        case BEAM_ERR_IO_CORRUPT:
        case BEAM_ERR_IO_VERSION:
            /*
             * OSError, with the other I/O codes, because that is the contract
             * `beam_model_load` has always had: every one of its failures was
             * a hard-coded OSError, and three tests plus any user writing
             * `except OSError` around a load depend on it.
             *
             * These two codes mapped to ValueError and nothing noticed,
             * because until `load` learned to name its failures they were
             * never produced -- dead cases in a switch. Making them live
             * surfaced the disagreement, and the tested contract wins over
             * the unreachable one.
             */
            PyErr_SetString(PyExc_OSError,
                            has_msg ? msg : "corrupt or unsupported model file");
            return NULL;

        default:
            PyErr_SetString(PyExc_RuntimeError, has_msg ? msg : "unknown error");
            return NULL;
    }
}
void capsule_destructor(PyObject *capsule) {
    BeamModel *m = (BeamModel *)PyCapsule_GetPointer(capsule, BEAM_CAPSULE_NAME);
    if (m) beam_model_destroy(m);
}
BeamModel *edge_from_capsule(PyObject *capsule) {
    return (BeamModel *)PyCapsule_GetPointer(capsule, BEAM_CAPSULE_NAME);
}
char **edge_borrow_strings(PyObject *seq, Py_ssize_t *out_n) {
    if (!PyList_Check(seq)) {
        PyErr_SetString(PyExc_TypeError, "expected a list of strings");
        return NULL;
    }
    Py_ssize_t n = PyList_GET_SIZE(seq);
    const char **v = (const char **)PyMem_Malloc((size_t)(n > 0 ? n : 1) * sizeof(char *));
    if (!v) { PyErr_NoMemory(); return NULL; }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyList_GET_ITEM(seq, i);
        if (!PyUnicode_Check(item)) {
            PyMem_Free(v);
            PyErr_Format(PyExc_TypeError, "sequence %zd is not a string", i);
            return NULL;
        }
        v[i] = PyUnicode_AsUTF8(item);
        if (!v[i]) { PyMem_Free(v); return NULL; }
    }
    *out_n = n;
    return v;
}
double *edge_borrow_abundances(PyObject *abund, Py_ssize_t n, bool *out_ok) {
    *out_ok = true;
    if (abund == Py_None) return NULL;
    PyObject *fast = PySequence_Fast(abund, "abundances must be a sequence");
    if (!fast) { *out_ok = false; return NULL; }
    Py_ssize_t an = PySequence_Fast_GET_SIZE(fast);
    if (an != n) {
        Py_DECREF(fast);
        PyErr_Format(PyExc_ValueError,
                     "abundances has %zd entries but sequences has %zd", an, n);
        *out_ok = false;
        return NULL;
    }
    double *w = (double *)PyMem_Malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!w) { Py_DECREF(fast); PyErr_NoMemory(); *out_ok = false; return NULL; }
    for (Py_ssize_t i = 0; i < n; i++) {
        w[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(fast, i));
        if (PyErr_Occurred()) {
            PyMem_Free(w); Py_DECREF(fast); *out_ok = false; return NULL;
        }
    }
    Py_DECREF(fast);
    return w;
}
int edge_parse_config(PyObject *args, BeamModelConfig *cfg,
                             PyObject **seqs, PyObject **abund) {
    int order, depth_cap, max_length, strict, sparse;
    double cw, cm, cl, k0, k1, k2, kw, kl, wf, wb;
    /* The alphabet is appended rather than placed first, where it belongs,
     * so every existing position in the tuple keeps its meaning. */
    const char *alphabet = NULL;
    const char *fmt = seqs ? "OOiii(ddd)(ddd)dd(dd)pps" :
                             "iii(ddd)(ddd)dd(dd)pps";
    int ok;
    if (seqs)
        ok = PyArg_ParseTuple(args, fmt, seqs, abund, &order, &depth_cap,
                              &max_length, &cw, &cm, &cl, &k0, &k1, &k2,
                              &kw, &kl, &wf, &wb, &strict, &sparse, &alphabet);
    else
        ok = PyArg_ParseTuple(args, fmt, &order, &depth_cap, &max_length,
                              &cw, &cm, &cl, &k0, &k1, &k2, &kw, &kl, &wf, &wb,
                              &strict, &sparse, &alphabet);
    if (!ok) return 0;
    const BeamAlphabetInfo *a = beam_alphabet_by_name(alphabet);
    if (!a) {
        PyErr_Format(PyExc_ValueError, "unknown alphabet '%s'", alphabet);
        return 0;
    }
    beam_model_config_default_for(a->id, cfg);
    /*
     * Range-checked BEFORE the cast. These three were cast straight to
     * uint8_t, so `order = 258` wrapped to 2 -- a LEGAL order, which then
     * passed `beam_config_valid` and produced a fully working model at an
     * order the caller never asked for, scoring wrongly and saying nothing.
     * `max_length = 300` wrapped to 44 the same way. The Python facade
     * validates these properly, but `_cbeam` is an importable boundary and
     * must not be the one place where a nonsense argument silently becomes a
     * plausible one.
     */
    if (order < 0 || order > 255 || depth_cap < 0 || depth_cap > 255
        || max_length < 0 || max_length > 255) {
        PyErr_Format(PyExc_ValueError,
                     "order, depth_cap and max_length must each be in 0..255 "
                     "(got %d, %d, %d); a larger value would wrap to an "
                     "unrelated but legal setting",
                     order, depth_cap, max_length);
        return 0;
    }
    cfg->order = (uint8_t)order;
    cfg->depth_cap = (uint8_t)depth_cap;
    cfg->max_length = (uint8_t)max_length;
    cfg->coef_wide = cw; cfg->coef_mid = cm; cfg->coef_len = cl;
    cfg->kappa[0] = k0; cfg->kappa[1] = k1; cfg->kappa[2] = k2;
    cfg->kappa_wide = kw; cfg->kappa_len = kl;
    cfg->weight_forward = wf; cfg->weight_backward = wb;
    cfg->strict = (bool)strict;
    cfg->sparse = (bool)sparse;
    return 1;
}
PyObject *edge_doubles_to_bytes(const double *v, size_t n) {
    return PyBytes_FromStringAndSize((const char *)v,
                                     (Py_ssize_t)(n * sizeof(double)));
}
PyObject *edge_moment_list(const BeamModel *m, double q,
                                  unsigned int order, int cumulants) {
    const unsigned int n = cumulants ? order : order + 1u;
    double *buf = (double *)PyMem_Calloc(n ? n : 1u, sizeof(double));
    if (!buf) return PyErr_NoMemory();
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = cumulants ? beam_model_surprisal_cumulants(m, q, order, buf)
                    : beam_model_surprisal_moments(m, q, order, buf);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) { PyMem_Free(buf); return set_beam_error(err); }
    PyObject *list = PyList_New(n);
    if (!list) { PyMem_Free(buf); return NULL; }
    for (unsigned int i = 0; i < n; i++)
        PyList_SET_ITEM(list, i, PyFloat_FromDouble(buf[i]));
    PyMem_Free(buf);
    return list;
}
