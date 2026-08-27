/**
 * @file analytics_bindings.c
 * @brief Spectra, Mellin, moments, diversity, ranges, depth attribution, topology.
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

PyObject *py_spectrum_pair(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; unsigned int length, bins; double spacing;
    if (!PyArg_ParseTuple(args, "OIId", &cap, &length, &bins, &spacing))
        return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    if (bins < 2) {
        PyErr_SetString(PyExc_ValueError, "bins must be at least 2");
        return NULL;
    }
    double *cw = (double *)PyMem_Malloc((size_t)bins * sizeof(double));
    double *gw = (double *)PyMem_Malloc((size_t)bins * sizeof(double));
    if (!cw || !gw) { PyMem_Free(cw); PyMem_Free(gw); return PyErr_NoMemory(); }
    double ds = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_spectrum_pair(m, length, bins, spacing, cw, gw, &ds);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) { PyMem_Free(cw); PyMem_Free(gw); return set_beam_error(err); }

    PyObject *cb = edge_doubles_to_bytes(cw, bins);
    PyObject *gb = edge_doubles_to_bytes(gw, bins);
    PyMem_Free(cw); PyMem_Free(gw);
    if (!cb || !gb) { Py_XDECREF(cb); Py_XDECREF(gb); return NULL; }
    return Py_BuildValue("(NNd)", cb, gb, ds);
}
PyObject *py_spectra_by_length(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; int counting; unsigned int max_length, bins; double spacing;
    if (!PyArg_ParseTuple(args, "OpIId", &cap, &counting, &max_length, &bins,
                          &spacing))
        return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    if (bins < 2) {
        PyErr_SetString(PyExc_ValueError, "bins must be at least 2");
        return NULL;
    }
    unsigned int rows = max_length ? max_length + 1u
                                   : (unsigned int)beam_model_config(m)->max_length + 1u;
    size_t n = (size_t)rows * (size_t)bins;
    double *w = (double *)PyMem_Calloc(n ? n : 1, sizeof(double));
    if (!w) return PyErr_NoMemory();
    double ds = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_spectra_by_length(m, counting ? true : false,
                                           max_length, bins, spacing, w, &ds);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) { PyMem_Free(w); return set_beam_error(err); }
    PyObject *b = edge_doubles_to_bytes(w, n);
    PyMem_Free(w);
    if (!b) return NULL;
    return Py_BuildValue("(NIId)", b, rows, bins, ds);
}
PyObject *py_length_derivatives(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double q; unsigned int order;
    if (!PyArg_ParseTuple(args, "OdI", &cap, &q, &order)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    unsigned int rows = (unsigned int)beam_model_config(m)->max_length + 1u;
    unsigned int cols = order + 1u;
    size_t n = (size_t)rows * (size_t)cols;
    double *out = (double *)PyMem_Calloc(n, sizeof(double));
    if (!out) return PyErr_NoMemory();
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_length_derivatives(m, q, order, out);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) { PyMem_Free(out); return set_beam_error(err); }
    PyObject *b = edge_doubles_to_bytes(out, n);
    PyMem_Free(out);
    if (!b) return NULL;
    return Py_BuildValue("(NII)", b, rows, cols);
}
PyObject *py_path_count(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; unsigned int L;
    if (!PyArg_ParseTuple(args, "OI", &cap, &L)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    double v;
    Py_BEGIN_ALLOW_THREADS
    v = beam_model_path_count(m, L);
    Py_END_ALLOW_THREADS
    return PyFloat_FromDouble(v);
}
PyObject *py_entropy(PyObject *self, PyObject *arg) {
    (void)self;
    BeamModel *m = edge_from_capsule(arg);
    if (!m) return NULL;
    double h = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_entropy(m, &h);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) return set_beam_error(err);
    return PyFloat_FromDouble(h);
}
PyObject *py_hill_number(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double alpha;
    if (!PyArg_ParseTuple(args, "Od", &cap, &alpha)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    double d = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_hill_number(m, alpha, &d);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) return set_beam_error(err);
    return PyFloat_FromDouble(d);
}
PyObject *py_spectrum(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; int counting; unsigned int length, bins; double spacing;
    if (!PyArg_ParseTuple(args, "OpIId", &cap, &counting, &length, &bins, &spacing))
        return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    if (bins < 2) {
        PyErr_SetString(PyExc_ValueError, "bins must be at least 2");
        return NULL;
    }
    double *w = (double *)PyMem_Malloc((size_t)bins * sizeof(double));
    if (!w) return PyErr_NoMemory();
    double ds = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_spectrum(m, counting ? true : false, length, bins,
                                  spacing, w, &ds);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) { PyMem_Free(w); return set_beam_error(err); }

    PyObject *list = PyList_New(bins);
    if (!list) { PyMem_Free(w); return NULL; }
    for (unsigned int i = 0; i < bins; i++)
        PyList_SET_ITEM(list, i, PyFloat_FromDouble(w[i]));
    PyMem_Free(w);
    return Py_BuildValue("(Nd)", list, ds);
}
PyObject *py_log_mellin(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double q;
    if (!PyArg_ParseTuple(args, "Od", &cap, &q)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    double v = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_log_mellin(m, q, &v);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) return set_beam_error(err);
    return PyFloat_FromDouble(v);
}
PyObject *py_surprisal_moments(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double q; unsigned int order;
    if (!PyArg_ParseTuple(args, "OdI", &cap, &q, &order)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    return edge_moment_list(m, q, order, 0);
}
PyObject *py_surprisal_cumulants(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double q; unsigned int order;
    if (!PyArg_ParseTuple(args, "OdI", &cap, &q, &order)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    return edge_moment_list(m, q, order, 1);
}
PyObject *py_surprisal_range(PyObject *self, PyObject *arg) {
    (void)self;
    BeamModel *m = edge_from_capsule(arg);
    if (!m) return NULL;
    double lo = 0.0, hi = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_surprisal_range(m, &lo, &hi);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) return set_beam_error(err);
    return Py_BuildValue("(dd)", lo, hi);
}
PyObject *py_length_marginals(PyObject *self, PyObject *arg) {
    (void)self;
    BeamModel *m = edge_from_capsule(arg);
    if (!m) return NULL;
    const unsigned int n = (unsigned int)beam_model_config(m)->max_length + 1u;
    double *c = (double *)PyMem_Calloc(n, sizeof(double));
    double *g = (double *)PyMem_Calloc(n, sizeof(double));
    double *s = (double *)PyMem_Calloc(n, sizeof(double));
    double *v = (double *)PyMem_Calloc(n, sizeof(double));
    if (!c || !g || !s || !v) {
        PyMem_Free(c); PyMem_Free(g); PyMem_Free(s); PyMem_Free(v);
        return PyErr_NoMemory();
    }
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_length_marginals(m, c, g, s, v);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) {
        PyMem_Free(c); PyMem_Free(g); PyMem_Free(s); PyMem_Free(v);
        return set_beam_error(err);
    }

    PyObject *lc = PyList_New(n), *lg = PyList_New(n);
    PyObject *ls = PyList_New(n), *lv = PyList_New(n);
    if (!lc || !lg || !ls || !lv) {
        Py_XDECREF(lc); Py_XDECREF(lg); Py_XDECREF(ls); Py_XDECREF(lv);
        PyMem_Free(c); PyMem_Free(g); PyMem_Free(s); PyMem_Free(v);
        return NULL;
    }
    for (unsigned int i = 0; i < n; i++) {
        PyList_SET_ITEM(lc, i, PyFloat_FromDouble(c[i]));
        PyList_SET_ITEM(lg, i, PyFloat_FromDouble(g[i]));
        PyList_SET_ITEM(ls, i, PyFloat_FromDouble(s[i]));
        PyList_SET_ITEM(lv, i, PyFloat_FromDouble(v[i]));
    }
    PyMem_Free(c); PyMem_Free(g); PyMem_Free(s); PyMem_Free(v);
    return Py_BuildValue("(NNNN)", lc, lg, ls, lv);
}
PyObject *py_surprisal_tilt(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double q;
    if (!PyArg_ParseTuple(args, "Od", &cap, &q)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    double lm = 0.0, mean = 0.0, var = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_surprisal_tilt(m, q, &lm, &mean, &var);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) return set_beam_error(err);
    return Py_BuildValue("(ddd)", lm, mean, var);
}
PyObject *py_surprisal_by_depth(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double q;
    if (!PyArg_ParseTuple(args, "Od", &cap, &q)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    const unsigned int n = (unsigned int)beam_model_config(m)->max_length;
    double *buf = (double *)PyMem_Calloc(n, sizeof(double));
    if (!buf) return PyErr_NoMemory();
    double konst = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_surprisal_by_depth(m, q, buf, &konst);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) { PyMem_Free(buf); return set_beam_error(err); }
    PyObject *list = PyList_New(n);
    if (!list) { PyMem_Free(buf); return NULL; }
    for (unsigned int i = 0; i < n; i++)
        PyList_SET_ITEM(list, i, PyFloat_FromDouble(buf[i]));
    PyMem_Free(buf);
    return Py_BuildValue("(Nd)", list, konst);
}
PyObject *py_topology(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; unsigned int length;
    if (!PyArg_ParseTuple(args, "OI", &cap, &length)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    uint32_t *layers = (uint32_t *)PyMem_Calloc(length + 1u, sizeof(uint32_t));
    if (!layers) return PyErr_NoMemory();
    double nodes = 0.0, edges = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_topology(m, length, &nodes, &edges, layers);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) { PyMem_Free(layers); return set_beam_error(err); }
    PyObject *list = PyList_New(length + 1u);
    if (!list) { PyMem_Free(layers); return NULL; }
    for (unsigned int i = 0; i <= length; i++)
        PyList_SET_ITEM(list, i, PyLong_FromUnsignedLong(layers[i]));
    PyMem_Free(layers);
    return Py_BuildValue("(ddN)", nodes, edges, list);
}
