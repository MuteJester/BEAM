/**
 * @file algorithm_bindings.c
 * @brief Sampling and best-first top-k.
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

PyObject *py_sample(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; unsigned int n, length; unsigned long long seed;
    if (!PyArg_ParseTuple(args, "OIIK", &cap, &n, &length, &seed)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    const BeamModelConfig *c = beam_model_config(m);
    const uint32_t stride = (uint32_t)c->max_length + 1u;

    char *buf = (char *)PyMem_Calloc((size_t)n * stride + 1u, 1);
    if (!buf) return PyErr_NoMemory();
    BeamRng rng;
    beam_rng_seed(&rng, (uint64_t)seed);
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_sample(m, n, length, &rng, buf, NULL);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) { PyMem_Free(buf); return set_beam_error(err); }

    PyObject *list = PyList_New(n);
    if (!list) { PyMem_Free(buf); return NULL; }
    for (unsigned int i = 0; i < n; i++) {
        PyObject *s = PyUnicode_FromString(buf + (size_t)i * stride);
        if (!s) { Py_DECREF(list); PyMem_Free(buf); return NULL; }
        PyList_SET_ITEM(list, i, s);
    }
    PyMem_Free(buf);
    return list;
}
PyObject *py_top_k(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; unsigned int k, length;
    if (!PyArg_ParseTuple(args, "OII", &cap, &k, &length)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    const unsigned int stride = (unsigned int)beam_model_config(m)->max_length + 1u;

    char *buf = (char *)PyMem_Calloc((size_t)k * stride + 1u, 1);
    double *lp = (double *)PyMem_Calloc(k ? k : 1u, sizeof(double));
    if (!buf || !lp) { PyMem_Free(buf); PyMem_Free(lp); return PyErr_NoMemory(); }
    uint32_t n = 0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_top_k(m, k, length, buf, lp, &n);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) { PyMem_Free(buf); PyMem_Free(lp); return set_beam_error(err); }

    PyObject *seqs = PyList_New(n), *probs = PyList_New(n);
    if (!seqs || !probs) {
        Py_XDECREF(seqs); Py_XDECREF(probs);
        PyMem_Free(buf); PyMem_Free(lp);
        return NULL;
    }
    for (uint32_t i = 0; i < n; i++) {
        PyObject *str = PyUnicode_FromString(buf + (size_t)i * stride);
        if (!str) { Py_DECREF(seqs); Py_DECREF(probs); PyMem_Free(buf); PyMem_Free(lp); return NULL; }
        PyList_SET_ITEM(seqs, i, str);
        PyList_SET_ITEM(probs, i, PyFloat_FromDouble(lp[i]));
    }
    PyMem_Free(buf); PyMem_Free(lp);
    return Py_BuildValue("(NN)", seqs, probs);
}
