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
#include "guards.h"
#include "beam/dp.h"   /* BEAM_MAX_MOMENT */

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
    /* `bins` is the caller's: 2**31 of them asked for 34 GB and reported a
     * MemoryError carrying nothing. */
    double *cw = (double *)beam_py_alloc((uint64_t)bins * sizeof(double) * 2u,
                                         "the spectrum histograms",
                                         bins, "bins");
    if (!cw) return NULL;
    double *gw = (double *)beam_py_alloc((uint64_t)bins * sizeof(double),
                                         "the generated histogram", 0u, NULL);
    if (!gw) { PyMem_Free(cw); return NULL; }
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
    if (!w) return beam_py_nomem("the per-length weights",
                                 (uint64_t)n * sizeof(double));
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
    PyObject *cap, *order_obj; double q; unsigned int order;
    if (!PyArg_ParseTuple(args, "OdO", &cap, &q, &order_obj)) return NULL;
    /* "I" truncated: order 2**32 + 2 was answered as order 2, a well-formed
     * table for a question nobody asked. The library's own check never saw
     * the real number. */
    if (beam_py_count_max(order_obj, "order", BEAM_MAX_MOMENT, &order) < 0)
        return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    unsigned int rows = (unsigned int)beam_model_config(m)->max_length + 1u;
    unsigned int cols = order + 1u;
    size_t n = (size_t)rows * (size_t)cols;
    double *out = (double *)PyMem_Calloc(n, sizeof(double));
    if (!out) return beam_py_nomem("the derivative table",
                                   (uint64_t)n * sizeof(double));
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
    /*
     * This entry point reports failure as 0.0 with the error set, and 0.0 is
     * also a legitimate answer, so the return value cannot distinguish them.
     * Without this the memory-budget refusal reached Python as "this length
     * has no paths" -- a refusal that looks like an answer is worse than a
     * crash. `beam_model_path_count` clears the error on entry, so a stale
     * one from an earlier call cannot trip this.
     */
    if (beam_last_error() != BEAM_OK) return set_beam_error(beam_last_error());
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
    double *w = (double *)beam_py_alloc((uint64_t)bins * sizeof(double),
                                       "the spectrum histogram", bins, "bins");
    if (!w) return NULL;
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
/* The orders arrive as a float64 buffer and the values leave as bytes, the
 * facade's to wrap. The buffer is held across the GIL-free call, so the
 * facade hands over a private copy: an array the caller still holds could be
 * written by another thread while the C reads it. */
PyObject *py_log_mellin_many(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *qobj;
    if (!PyArg_ParseTuple(args, "OO", &cap, &qobj)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    Py_buffer qv;
    if (!get_f64_buffer(qobj, &qv, 0, "orders")) return NULL;
    const Py_ssize_t nq = f64_buffer_count(&qv);
    if ((uint64_t)nq > UINT32_MAX) {
        PyBuffer_Release(&qv);
        PyErr_Format(PyExc_ValueError,
                     "%zd orders in one call; the limit is %u",
                     nq, (unsigned)UINT32_MAX);
        return NULL;
    }
    double *out = (double *)beam_py_alloc((uint64_t)nq * sizeof(double),
                                          "the Mellin values", (uint32_t)nq,
                                          "orders");
    if (!out) { PyBuffer_Release(&qv); return NULL; }
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_log_mellin_many(m, (const double *)qv.buf, (size_t)nq,
                                     out);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&qv);
    if (err != BEAM_OK) { PyMem_Free(out); return set_beam_error(err); }
    PyObject *res = edge_doubles_to_bytes(out, (size_t)nq);
    PyMem_Free(out);
    return res;
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
        return beam_py_nomem("the four per-length arrays",
                             4ull * (uint64_t)n * sizeof(double));
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
    if (!buf) return beam_py_nomem("the analytic buffer",
                                   (uint64_t)n * sizeof(double));
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
    /*
     * Validate the length BEFORE sizing anything from it. Sized first, a
     * nonsense length raised a bare `MemoryError` with an EMPTY message --
     * the allocator's verdict on a 2 GB request -- instead of the C layer's
     * precise "length N is not one this model generates". The query bindings
     * already refuse an out-of-range length before anything is sized; these
     * two did not.
     */
    {
        const BeamModelConfig *bcfg = beam_model_config(m);
        if (bcfg && length > bcfg->max_length) {
            PyErr_Format(PyExc_ValueError,
                         "length %u is above this model's max_length of %u",
                         length, (unsigned)bcfg->max_length);
            return NULL;
        }
    }
    uint32_t *layers = (uint32_t *)PyMem_Calloc(length + 1u, sizeof(uint32_t));
    if (!layers) return beam_py_nomem("the layer index",
                                      (uint64_t)(length + 1u) * sizeof(uint32_t));
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

/* ── Positional marginals and expected cell usage ───────────── */

/*
 * The clamp crosses the boundary as a bytes object of exactly `length` bytes,
 * one per position, rather than as a dict. Building the byte array is the
 * facade's job because that is where the alphabet lives, and a fixed-width
 * buffer means this function has no parsing to get wrong and no residue name
 * to look up.
 */
PyObject *py_positional_marginals(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *clamp_obj;
    unsigned int length;
    int allow_empty = 0;
    if (!PyArg_ParseTuple(args, "OIO|p", &cap, &length, &clamp_obj,
                          &allow_empty))
        return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;

    const uint8_t *clamp = NULL;
    if (clamp_obj != Py_None) {
        char *buf; Py_ssize_t n;
        if (PyBytes_AsStringAndSize(clamp_obj, &buf, &n) != 0) return NULL;
        if (n != (Py_ssize_t)length) {
            PyErr_Format(PyExc_ValueError,
                         "clamp must hold one byte per position: expected %u, "
                         "got %zd", length, n);
            return NULL;
        }
        clamp = (const uint8_t *)buf;
    }

    /*
     * Validate the length BEFORE sizing anything from it. Sized first, a
     * nonsense length raised a bare `MemoryError` with an EMPTY message --
     * the allocator's verdict on a 2 GB request -- instead of the C layer's
     * precise "length N is not one this model generates". The query bindings
     * already refuse an out-of-range length before anything is sized; these
     * two did not.
     */
    {
        const BeamModelConfig *bcfg = beam_model_config(m);
        if (bcfg && length > bcfg->max_length) {
            PyErr_Format(PyExc_ValueError,
                         "length %u is above this model's max_length of %u",
                         length, (unsigned)bcfg->max_length);
            return NULL;
        }
    }

    const unsigned int R = (unsigned int)beam_model_n_residues(m);
    const size_t n = (size_t)length * R;
    double *out = (double *)PyMem_Calloc(n ? n : 1u, sizeof(double));
    if (!out) return beam_py_nomem("the positional marginals",
                                   (uint64_t)n * sizeof(double));
    double logz = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_positional_marginals(m, length, clamp, out, &logz);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) {
        PyMem_Free(out);
        /*
         * A clamp of probability zero is the one failure here that a caller
         * may legitimately want as a VALUE rather than as an exception:
         * `log_clamp_probability` asks exactly "how much mass does this keep",
         * and "none" is the answer to that question rather than a failure to
         * answer it. It is opt-in, so the marginals themselves -- which have
         * no conditional to return -- still fail loudly.
         */
        if (allow_empty && err == BEAM_ERR_NO_LIVE_PATHS)
            return Py_BuildValue("(y#Id)", "", (Py_ssize_t)0, R, -INFINITY);
        /*
         * Otherwise report it as a ValueError rather than the RuntimeError
         * BEAM_ERR_NO_LIVE_PATHS maps to by default. At this entry point the
         * length has already been checked live, so the only way the trellis
         * can be empty is that the caller's clamp emptied it, and that is an
         * argument problem.
         */
        if (err == BEAM_ERR_NO_LIVE_PATHS) {
            const char *msg = beam_error_message();
            PyErr_SetString(PyExc_ValueError,
                            (msg && msg[0]) ? msg
                                            : "the clamp selects no sequence");
            return NULL;
        }
        return set_beam_error(err);
    }
    PyObject *b = edge_doubles_to_bytes(out, n);
    PyMem_Free(out);
    if (!b) return NULL;
    return Py_BuildValue("(NId)", b, R, logz);
}

/**
 * `positional_banded(cap, length, counting, bins, spacing, edges) ->
 *  (marginals, band_mass, position_mass, n_residues, n_bands, spacing)`
 *
 * `edges` is a buffer of float64 band edges in nats, ascending. The three
 * outputs come back as byte blocks the facade reshapes: the marginals are
 * `n_bands x length x n_residues`, the masses `n_bands` and
 * `n_bands x length`.
 *
 * `position_mass` is returned always rather than on request. It is the only
 * thing that tells a caller whether the grid resolved their bands -- the
 * positions of one band agree to the last bits when it did and visibly
 * disagree when it did not -- and a diagnostic nobody can reach is one nobody
 * uses.
 */
PyObject *py_positional_banded(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *edges_obj;
    unsigned int length, bins;
    int counting = 0;
    double spacing = 0.0;
    if (!PyArg_ParseTuple(args, "OIpIdO", &cap, &length, &counting, &bins,
                          &spacing, &edges_obj))
        return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;

    /* The length before anything is sized from it; see the note on
     * py_positional_marginals for what sizing first used to report. */
    {
        const BeamModelConfig *bcfg = beam_model_config(m);
        if (bcfg && length > bcfg->max_length) {
            PyErr_Format(PyExc_ValueError,
                         "length %u is above this model's max_length of %u",
                         length, (unsigned)bcfg->max_length);
            return NULL;
        }
    }
    if (length < 1u) {
        PyErr_SetString(PyExc_ValueError,
                        "banded marginals are per-length and need a length of "
                        "at least 1");
        return NULL;
    }

    char *ebuf; Py_ssize_t ebytes;
    if (PyBytes_AsStringAndSize(edges_obj, &ebuf, &ebytes) != 0) return NULL;
    if (ebytes % (Py_ssize_t)sizeof(double) != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "the band edges must be a whole number of float64s");
        return NULL;
    }
    const size_t n_edges = (size_t)ebytes / sizeof(double);
    if (n_edges < 2u) {
        PyErr_Format(PyExc_ValueError,
                     "%zu band edge(s) define no band; two ascending edges "
                     "define one", n_edges);
        return NULL;
    }
    const size_t n_bands = n_edges - 1u;
    const unsigned int R = (unsigned int)beam_model_n_residues(m);

    /* Sized in size_t and checked, because the edge count is the caller's. */
    if (n_bands > SIZE_MAX / length / (R ? R : 1u) / sizeof(double)) {
        PyErr_Format(PyExc_ValueError,
                     "%zu bands at length %u would not fit in memory",
                     n_bands, length);
        return NULL;
    }
    const size_t n_marg = n_bands * length * R;
    const size_t n_pm   = n_bands * length;

    double *marg = (double *)PyMem_Calloc(n_marg ? n_marg : 1u, sizeof(double));
    double *bmas = (double *)PyMem_Calloc(n_bands, sizeof(double));
    double *pmas = (double *)PyMem_Calloc(n_pm ? n_pm : 1u, sizeof(double));
    if (!marg || !bmas || !pmas) {
        PyMem_Free(marg); PyMem_Free(bmas); PyMem_Free(pmas);
        return beam_py_nomem("the banded positional marginals",
                             (uint64_t)(n_marg + n_bands + n_pm)
                                 * sizeof(double));
    }

    double used = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_positional_banded(m, length, (bool)counting, bins,
                                       spacing, (const double *)ebuf,
                                       (uint32_t)n_edges, marg, bmas, pmas,
                                       &used);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) {
        PyMem_Free(marg); PyMem_Free(bmas); PyMem_Free(pmas);
        return set_beam_error(err);
    }

    PyObject *a = edge_doubles_to_bytes(marg, n_marg);
    PyObject *b = edge_doubles_to_bytes(bmas, n_bands);
    PyObject *c = edge_doubles_to_bytes(pmas, n_pm);
    PyMem_Free(marg); PyMem_Free(bmas); PyMem_Free(pmas);
    if (!a || !b || !c) { Py_XDECREF(a); Py_XDECREF(b); Py_XDECREF(c);
                          return NULL; }
    return Py_BuildValue("(NNNInd)", a, b, c, R, (Py_ssize_t)n_bands, used);
}


/*
 * The six blocks come back as a flat list of `(rows, counts)` byte pairs in
 * `[anchor][expert]` order, which the facade reshapes. Flat rather than nested
 * because the reshaping is naming, and naming belongs on the Python side where
 * the names can be documented next to what they mean.
 */
PyObject *py_expected_counts(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; unsigned int length;
    if (!PyArg_ParseTuple(args, "OI", &cap, &length)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;

    BeamExpectedCounts ec;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_expected_counts(m, length, &ec);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) return set_beam_error(err);

    PyObject *list = PyList_New(6);
    if (!list) { beam_model_expected_counts_free(&ec); return NULL; }
    /*
     * Release each C block as soon as its bytes objects exist, rather than
     * freeing all six at the end. A `bytes` is a full copy, so holding the C
     * side until the loop finishes means both copies of the whole answer are
     * resident at once -- 1,338 MB on the foundation model where 669 MB is
     * what the answer actually costs. Only the block being converted is ever
     * duplicated this way.
     */
    for (unsigned int d = 0, k = 0; d < 2u; ++d) {
        for (unsigned int e = 0; e < 3u; ++e, ++k) {
            BeamCountBlock *blk = &ec.block[d][e];
            PyObject *rows = PyBytes_FromStringAndSize(
                (const char *)blk->rows,
                (Py_ssize_t)((size_t)blk->n_rows * sizeof(uint32_t)));
            PyObject *vals = edge_doubles_to_bytes(
                blk->counts, (size_t)blk->n_rows * ec.n_residues);
            beam_count_block_free(blk);
            if (!rows || !vals) {
                Py_XDECREF(rows); Py_XDECREF(vals); Py_DECREF(list);
                beam_model_expected_counts_free(&ec);
                return NULL;
            }
            PyObject *pair = Py_BuildValue("(NN)", rows, vals);
            if (!pair) {
                Py_DECREF(list);
                beam_model_expected_counts_free(&ec);
                return NULL;
            }
            PyList_SET_ITEM(list, k, pair);
        }
    }
    const unsigned int R = ec.n_residues;
    const double total = ec.expected_length;
    const unsigned long long ws = (unsigned long long)ec.working_set_bytes;
    beam_model_expected_counts_free(&ec);
    return Py_BuildValue("(NIdK)", list, R, total, ws);
}
