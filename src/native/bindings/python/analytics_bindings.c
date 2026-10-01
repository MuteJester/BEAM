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
/* log_gamma_ratio(w, z_re, z_im) -> bytes of nz * nw complex128, row i the
 * ratio ln G(w_k) - ln G(w_k + z_i) at every w: the layout of a (nz, nw)
 * complex128 array. Not a model's: a function the contour kernels call. */
PyObject *py_log_gamma_ratio(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *wobj, *reobj, *imobj;
    if (!PyArg_ParseTuple(args, "OOO", &wobj, &reobj, &imobj)) return NULL;
    Py_buffer wv, rv, iv;
    if (!get_f64_buffer(wobj, &wv, 0, "w")) return NULL;
    if (!get_f64_buffer(reobj, &rv, 0, "real parts")) { PyBuffer_Release(&wv); return NULL; }
    if (!get_f64_buffer(imobj, &iv, 0, "imaginary parts")) {
        PyBuffer_Release(&wv); PyBuffer_Release(&rv);
        return NULL;
    }
    const Py_ssize_t nw = f64_buffer_count(&wv), nz = f64_buffer_count(&rv);
    if (f64_buffer_count(&iv) != nz) {
        PyBuffer_Release(&wv); PyBuffer_Release(&rv); PyBuffer_Release(&iv);
        PyErr_Format(PyExc_ValueError, "%zd real parts but %zd imaginary parts",
                     nz, f64_buffer_count(&iv));
        return NULL;
    }
    const uint64_t cells = (uint64_t)nw * (uint64_t)nz;
    double *out = (double *)beam_py_alloc(cells * 4u * sizeof(double),
                                          "the log-gamma ratios", 0u, "cells");
    if (!out) {
        PyBuffer_Release(&wv); PyBuffer_Release(&rv); PyBuffer_Release(&iv);
        return NULL;
    }
    double *ore = out, *oim = out + cells, *il = out + 2u * cells;
    Py_BEGIN_ALLOW_THREADS
    beam_log_gamma_ratio((const double *)wv.buf, (size_t)nw,
                         (const double *)rv.buf, (const double *)iv.buf,
                         (size_t)nz, ore, oim);
    for (uint64_t x = 0; x < cells; ++x) { il[2 * x] = ore[x]; il[2 * x + 1] = oim[x]; }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&wv); PyBuffer_Release(&rv); PyBuffer_Release(&iv);
    PyObject *res = edge_doubles_to_bytes(il, (size_t)cells * 2u);
    PyMem_Free(out);
    return res;
}
/* Several float64 buffers at once; on failure, those taken are released. */
static int take_f64s(PyObject **objs, Py_buffer *bufs, const char **names, int n) {
    for (int i = 0; i < n; ++i) {
        if (!get_f64_buffer(objs[i], &bufs[i], 0, names[i])) {
            for (int j = 0; j < i; ++j) PyBuffer_Release(&bufs[j]);
            return 0;
        }
    }
    return 1;
}

static void release_f64s(Py_buffer *bufs, int n) {
    for (int i = 0; i < n; ++i) PyBuffer_Release(&bufs[i]);
}

/* beta_contour_sums(zr, zi, lgr, lgi, lmr, lmi, wt, tt, dph, w, lnns) -> bytes
 * of 3 * nk doubles: the values' sums, then the two rounding sums. */
PyObject *py_beta_contour_sums(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *o[11];
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOO", &o[0], &o[1], &o[2], &o[3], &o[4],
                          &o[5], &o[6], &o[7], &o[8], &o[9], &o[10])) return NULL;
    static const char *names[11] = {"zr", "zi", "lgr", "lgi", "lmr", "lmi", "wt",
                                    "tt", "dph", "w", "lnns"};
    Py_buffer b[11];
    if (!take_f64s(o, b, names, 11)) return NULL;
    const Py_ssize_t nt = f64_buffer_count(&b[0]), nk = f64_buffer_count(&b[9]);
    for (int i = 1; i < 9; ++i)
        if (f64_buffer_count(&b[i]) != nt) {
            release_f64s(b, 11);
            PyErr_Format(PyExc_ValueError, "%s has %zd entries, not %zd",
                         names[i], f64_buffer_count(&b[i]), nt);
            return NULL;
        }
    if (f64_buffer_count(&b[10]) != nk) {
        release_f64s(b, 11);
        PyErr_Format(PyExc_ValueError, "lnns has %zd entries, not %zd",
                     f64_buffer_count(&b[10]), nk);
        return NULL;
    }
    double *out = (double *)beam_py_alloc((uint64_t)nk * 3u * sizeof(double),
                                          "the contour sums", 0u, "parameters");
    if (!out) { release_f64s(b, 11); return NULL; }
    Py_BEGIN_ALLOW_THREADS
    beam_beta_contour_sums((size_t)nt, (const double *)b[0].buf, (const double *)b[1].buf,
                           (const double *)b[2].buf, (const double *)b[3].buf,
                           (const double *)b[4].buf, (const double *)b[5].buf,
                           (const double *)b[6].buf, (const double *)b[7].buf,
                           (const double *)b[8].buf, (size_t)nk,
                           (const double *)b[9].buf, (const double *)b[10].buf,
                           out, out + nk, out + 2 * nk);
    Py_END_ALLOW_THREADS
    release_f64s(b, 11);
    PyObject *res = edge_doubles_to_bytes(out, (size_t)nk * 3u);
    PyMem_Free(out);
    return res;
}

/* log_gamma_ratio_shift(w, d, z_re, z_im) -> bytes of nz * nd complex128, row
 * i the shift [ln G(w-d_k) - ln G(w-d_k+z_i)] - [ln G(w) - ln G(w+z_i)] at
 * every d: the layout of a (nz, nd) complex128 array. */
PyObject *py_log_gamma_ratio_shift(PyObject *self, PyObject *args) {
    (void)self;
    double w;
    PyObject *o[3];
    if (!PyArg_ParseTuple(args, "dOOO", &w, &o[0], &o[1], &o[2])) return NULL;
    static const char *names[3] = {"d", "real parts", "imaginary parts"};
    Py_buffer b[3];
    if (!take_f64s(o, b, names, 3)) return NULL;
    const Py_ssize_t nd = f64_buffer_count(&b[0]), nz = f64_buffer_count(&b[1]);
    if (f64_buffer_count(&b[2]) != nz) {
        release_f64s(b, 3);
        PyErr_Format(PyExc_ValueError, "%zd real parts but %zd imaginary parts",
                     nz, f64_buffer_count(&b[2]));
        return NULL;
    }
    const uint64_t cells = (uint64_t)nd * (uint64_t)nz;
    double *out = (double *)beam_py_alloc((cells ? cells : 1u) * 4u * sizeof(double),
                                          "the log-gamma shifts", 0u, "cells");
    if (!out) { release_f64s(b, 3); return NULL; }
    double *ore = out, *oim = out + cells, *il = out + 2u * cells;
    Py_BEGIN_ALLOW_THREADS
    beam_log_gamma_ratio_shift(w, (const double *)b[0].buf, (size_t)nd,
                               (const double *)b[1].buf, (const double *)b[2].buf,
                               (size_t)nz, ore, oim);
    for (uint64_t x = 0; x < cells; ++x) { il[2 * x] = ore[x]; il[2 * x + 1] = oim[x]; }
    Py_END_ALLOW_THREADS
    release_f64s(b, 3);
    PyObject *res = edge_doubles_to_bytes(il, (size_t)cells * 2u);
    PyMem_Free(out);
    return res;
}

/* cohort_private_sums(zr, zi, lbr, lbi, lmr, lmi, wt, tt, dph, w, d, lnns) ->
 * bytes of 3 * nk doubles: the values' sums, then the two rounding sums. */
PyObject *py_cohort_private_sums(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *o[11];
    double w;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOdOO", &o[0], &o[1], &o[2], &o[3], &o[4],
                          &o[5], &o[6], &o[7], &o[8], &w, &o[9], &o[10])) return NULL;
    static const char *names[11] = {"zr", "zi", "lbr", "lbi", "lmr", "lmi", "wt",
                                    "tt", "dph", "d", "lnns"};
    Py_buffer b[11];
    if (!take_f64s(o, b, names, 11)) return NULL;
    const Py_ssize_t nt = f64_buffer_count(&b[0]), nk = f64_buffer_count(&b[9]);
    for (int i = 1; i < 9; ++i)
        if (f64_buffer_count(&b[i]) != nt) {
            release_f64s(b, 11);
            PyErr_Format(PyExc_ValueError, "%s has %zd entries, not %zd",
                         names[i], f64_buffer_count(&b[i]), nt);
            return NULL;
        }
    if (f64_buffer_count(&b[10]) != nk) {
        release_f64s(b, 11);
        PyErr_Format(PyExc_ValueError, "lnns has %zd entries, not %zd",
                     f64_buffer_count(&b[10]), nk);
        return NULL;
    }
    double *out = (double *)beam_py_alloc((uint64_t)(nk ? nk : 1) * 3u * sizeof(double),
                                          "the contour sums", 0u, "parameters");
    if (!out) { release_f64s(b, 11); return NULL; }
    Py_BEGIN_ALLOW_THREADS
    beam_cohort_private_sums((size_t)nt, (const double *)b[0].buf, (const double *)b[1].buf,
                             (const double *)b[2].buf, (const double *)b[3].buf,
                             (const double *)b[4].buf, (const double *)b[5].buf,
                             (const double *)b[6].buf, (const double *)b[7].buf,
                             (const double *)b[8].buf, w, (size_t)nk,
                             (const double *)b[9].buf, (const double *)b[10].buf,
                             out, out + nk, out + 2 * nk);
    Py_END_ALLOW_THREADS
    release_f64s(b, 11);
    PyObject *res = edge_doubles_to_bytes(out, (size_t)nk * 3u);
    PyMem_Free(out);
    return res;
}

/* ztp_bin_reduce(lam, t_bins, edges, allowed, tau) -> bytes of
 * 2 nb + 3 + (nb + 1)^2 doubles: the tilted zero-truncated-Poisson bin
 * moments summed over the sequences. See math/ztp_moments.c. */
PyObject *py_ztp_bin_reduce(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *o[4];
    double tau;
    if (!PyArg_ParseTuple(args, "OOOOd", &o[0], &o[1], &o[2], &o[3], &tau))
        return NULL;
    static const char *names[4] = {"lam", "t_bins", "edges", "allowed"};
    Py_buffer b[4];
    if (!take_f64s(o, b, names, 4)) return NULL;
    const Py_ssize_t n = f64_buffer_count(&b[0]), nb = f64_buffer_count(&b[2]);
    if (f64_buffer_count(&b[1]) != nb || f64_buffer_count(&b[3]) != nb) {
        release_f64s(b, 4);
        PyErr_Format(PyExc_ValueError, "t_bins and allowed must each have %zd "
                     "entries, one per bin", nb);
        return NULL;
    }
    if (nb < 1 || nb > (Py_ssize_t)BEAM_ZTP_MAX_BINS) {
        release_f64s(b, 4);
        PyErr_Format(PyExc_ValueError, "%zd abundance bins; between 1 and %u "
                     "are supported", nb, (unsigned)BEAM_ZTP_MAX_BINS);
        return NULL;
    }
    const uint64_t width = 2u * (uint64_t)nb + 3u + (uint64_t)(nb + 1) * (nb + 1);
    double *out = (double *)beam_py_alloc(width * sizeof(double),
                                          "the bin moments", 0u, "bins");
    if (!out) { release_f64s(b, 4); return NULL; }
    Py_BEGIN_ALLOW_THREADS
    beam_ztp_bin_reduce((const double *)b[0].buf, (size_t)n,
                        (const double *)b[1].buf, (const double *)b[2].buf,
                        (const double *)b[3].buf, (size_t)nb, tau, out);
    Py_END_ALLOW_THREADS
    release_f64s(b, 4);
    PyObject *res = edge_doubles_to_bytes(out, (size_t)width);
    PyMem_Free(out);
    return res;
}

/* ztp_bin_expect(lam, edges, total) -> bytes of nb doubles. */
PyObject *py_ztp_bin_expect(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *o[2];
    double total;
    if (!PyArg_ParseTuple(args, "OOd", &o[0], &o[1], &total)) return NULL;
    static const char *names[2] = {"lam", "edges"};
    Py_buffer b[2];
    if (!take_f64s(o, b, names, 2)) return NULL;
    const Py_ssize_t n = f64_buffer_count(&b[0]), nb = f64_buffer_count(&b[1]);
    if (nb < 1 || nb > (Py_ssize_t)BEAM_ZTP_MAX_BINS) {
        release_f64s(b, 2);
        PyErr_Format(PyExc_ValueError, "%zd abundance bins; between 1 and %u "
                     "are supported", nb, (unsigned)BEAM_ZTP_MAX_BINS);
        return NULL;
    }
    double *out = (double *)beam_py_alloc((uint64_t)nb * sizeof(double),
                                          "the expected bin counts", 0u, "bins");
    if (!out) { release_f64s(b, 2); return NULL; }
    Py_BEGIN_ALLOW_THREADS
    beam_ztp_bin_expect((const double *)b[0].buf, (size_t)n,
                        (const double *)b[1].buf, (size_t)nb, total, out);
    Py_END_ALLOW_THREADS
    release_f64s(b, 2);
    PyObject *res = edge_doubles_to_bytes(out, (size_t)nb);
    PyMem_Free(out);
    return res;
}

/* beta_contour_tail(zr, zi, lgr, ts, w) -> bytes of nk doubles. */
PyObject *py_beta_contour_tail(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *o[5];
    if (!PyArg_ParseTuple(args, "OOOOO", &o[0], &o[1], &o[2], &o[3], &o[4])) return NULL;
    static const char *names[5] = {"zr", "zi", "lgr", "ts", "w"};
    Py_buffer b[5];
    if (!take_f64s(o, b, names, 5)) return NULL;
    const Py_ssize_t nt = f64_buffer_count(&b[0]), nk = f64_buffer_count(&b[4]);
    for (int i = 1; i < 4; ++i)
        if (f64_buffer_count(&b[i]) != nt) {
            release_f64s(b, 5);
            PyErr_Format(PyExc_ValueError, "%s has %zd entries, not %zd",
                         names[i], f64_buffer_count(&b[i]), nt);
            return NULL;
        }
    double *out = (double *)beam_py_alloc((uint64_t)(nk ? nk : 1) * sizeof(double),
                                          "the contour tails", 0u, "parameters");
    if (!out) { release_f64s(b, 5); return NULL; }
    Py_BEGIN_ALLOW_THREADS
    beam_beta_contour_tail((size_t)nt, (const double *)b[0].buf, (const double *)b[1].buf,
                           (const double *)b[2].buf, (const double *)b[3].buf,
                           (size_t)nk, (const double *)b[4].buf, out);
    Py_END_ALLOW_THREADS
    release_f64s(b, 5);
    PyObject *res = edge_doubles_to_bytes(out, (size_t)nk);
    PyMem_Free(out);
    return res;
}

/* log_mellin_complex(cap, re, im) -> bytes of 2*nq doubles, interleaved
 * (re, im) per order: the memory layout of numpy's complex128. */
PyObject *py_log_mellin_complex(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *reobj, *imobj;
    if (!PyArg_ParseTuple(args, "OOO", &cap, &reobj, &imobj)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    Py_buffer rv, iv;
    if (!get_f64_buffer(reobj, &rv, 0, "real parts")) return NULL;
    if (!get_f64_buffer(imobj, &iv, 0, "imaginary parts")) {
        PyBuffer_Release(&rv);
        return NULL;
    }
    const Py_ssize_t nq = f64_buffer_count(&rv);
    if (f64_buffer_count(&iv) != nq) {
        PyBuffer_Release(&rv); PyBuffer_Release(&iv);
        PyErr_Format(PyExc_ValueError,
                     "%zd real parts but %zd imaginary parts",
                     nq, f64_buffer_count(&iv));
        return NULL;
    }
    if ((uint64_t)nq > UINT32_MAX) {
        PyBuffer_Release(&rv); PyBuffer_Release(&iv);
        PyErr_Format(PyExc_ValueError,
                     "%zd orders in one call; the limit is %u",
                     nq, (unsigned)UINT32_MAX);
        return NULL;
    }
    double *out = (double *)beam_py_alloc((uint64_t)nq * 2u * sizeof(double),
                                          "the complex Mellin values",
                                          (uint32_t)nq, "orders");
    if (!out) { PyBuffer_Release(&rv); PyBuffer_Release(&iv); return NULL; }
    double *ore = out, *oim = out + nq;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_log_mellin_complex(m, (const double *)rv.buf,
                                        (const double *)iv.buf, (size_t)nq,
                                        ore, oim);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&rv); PyBuffer_Release(&iv);
    if (err != BEAM_OK) { PyMem_Free(out); return set_beam_error(err); }
    /* Interleave in a second buffer so the result is complex128 as-is. */
    double *il = (double *)beam_py_alloc((uint64_t)nq * 2u * sizeof(double),
                                         "the complex Mellin values",
                                         (uint32_t)nq, "orders");
    if (!il) { PyMem_Free(out); return NULL; }
    for (Py_ssize_t j = 0; j < nq; ++j) {
        il[2 * j] = ore[j];
        il[2 * j + 1] = oim[j];
    }
    PyMem_Free(out);
    PyObject *res = edge_doubles_to_bytes(il, (size_t)nq * 2u);
    PyMem_Free(il);
    return res;
}
/* log_mellin_line(cap, re, im0, dim, n) -> bytes of 2n doubles,
 * interleaved (re, im): numpy complex128. */
PyObject *py_log_mellin_line(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double re, im0, dim; Py_ssize_t n;
    if (!PyArg_ParseTuple(args, "Odddn", &cap, &re, &im0, &dim, &n)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    if (n < 0) {
        PyErr_Format(PyExc_ValueError, "n must be non-negative, got %zd", n);
        return NULL;
    }
    if ((uint64_t)n > UINT32_MAX) {
        PyErr_Format(PyExc_ValueError, "%zd orders in one call; the limit is %u",
                     n, (unsigned)UINT32_MAX);
        return NULL;
    }
    double *out = (double *)beam_py_alloc((uint64_t)n * 4u * sizeof(double),
                                          "the line Mellin values",
                                          (uint32_t)n, "orders");
    if (!out) return NULL;
    double *ore = out, *oim = out + n, *il = out + 2 * n;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_log_mellin_line(m, re, im0, dim, (size_t)n, ore, oim);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) { PyMem_Free(out); return set_beam_error(err); }
    for (Py_ssize_t j = 0; j < n; ++j) { il[2 * j] = ore[j]; il[2 * j + 1] = oim[j]; }
    PyObject *res = edge_doubles_to_bytes(il, (size_t)n * 2u);
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

    PyObject *list = PyList_New(2);
    if (!list) { beam_model_expected_counts_free(&ec); return NULL; }
    /*
     * Release each C block as soon as its bytes objects exist, rather than
     * freeing both at the end. A `bytes` is a full copy, so holding the C
     * side until the loop finishes means both copies of the whole answer are
     * resident at once -- 1,338 MB on the foundation model where 669 MB is
     * what the answer actually costs. Only the block being converted is ever
     * duplicated this way.
     */
    for (unsigned int d = 0; d < 2u; ++d) {
        {
            BeamCountBlock *blk = &ec.block[d];
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
            PyList_SET_ITEM(list, d, pair);
        }
    }
    const unsigned int R = ec.n_residues;
    const double total = ec.expected_length;
    const unsigned long long ws = (unsigned long long)ec.working_set_bytes;
    beam_model_expected_counts_free(&ec);
    return Py_BuildValue("(NIdK)", list, R, total, ws);
}
