/**
 * @file publicness_bindings.c
 * @brief The Poisson-binomial occupancy entry points.
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

/* ── publicness_moments(p, depths, mult, mean, var) → None ── */

PyObject *py_publicness_moments(PyObject *self, PyObject *args) {
    (void)self;
    static const int writable[5] = {0, 0, 0, 1, 1};
    static const char *names[5] = {"probabilities", "depths", "multiplicities",
                                   "mean output", "variance output"};
    PyObject *objs[5];
    Py_buffer views[5];
    Py_ssize_t n_atoms, n_groups;
    BeamError err;

    if (!PyArg_ParseTuple(args, "OOOOO", &objs[0], &objs[1], &objs[2],
                          &objs[3], &objs[4])) return NULL;
    if (!get_f64_buffers(objs, views, writable, names, 5)) return NULL;

    n_atoms = f64_buffer_count(&views[0]);
    n_groups = f64_buffer_count(&views[1]);
    if (f64_buffer_count(&views[2]) != n_groups ||
        f64_buffer_count(&views[3]) != n_atoms ||
        f64_buffer_count(&views[4]) != n_atoms) {
        release_f64_buffers(views, 5);
        PyErr_SetString(PyExc_ValueError,
            "depths and multiplicities must be the same length, and both "
            "outputs must match the probability count");
        return NULL;
    }
    if (n_atoms > UINT32_MAX || n_groups > UINT32_MAX) {
        release_f64_buffers(views, 5);
        PyErr_SetString(PyExc_OverflowError, "batch exceeds uint32 limit");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    err = beam_publicness_moments((const double *)views[0].buf,
                                 (uint32_t)n_atoms,
                                 (const double *)views[1].buf,
                                 (const double *)views[2].buf,
                                 (uint32_t)n_groups,
                                 (double *)views[3].buf,
                                 (double *)views[4].buf);
    Py_END_ALLOW_THREADS

    release_f64_buffers(views, 5);
    if (err != BEAM_OK) return set_beam_error(err);
    Py_RETURN_NONE;
}
/* ── publicness_pgf(p, depths, mult, out) → None ──────────── */

PyObject *py_publicness_pgf(PyObject *self, PyObject *args) {
    (void)self;
    static const int writable[4] = {0, 0, 0, 1};
    static const char *names[4] = {"probabilities", "depths", "multiplicities",
                                   "generating-function output"};
    PyObject *objs[4];
    Py_buffer views[4];
    Py_ssize_t n_atoms, n_groups, out_count, k_fft;
    BeamError err;

    if (!PyArg_ParseTuple(args, "OOOO", &objs[0], &objs[1], &objs[2],
                          &objs[3])) return NULL;
    if (!get_f64_buffers(objs, views, writable, names, 4)) return NULL;

    n_atoms = f64_buffer_count(&views[0]);
    n_groups = f64_buffer_count(&views[1]);
    out_count = f64_buffer_count(&views[3]);
    /* The output is the float64 view of an (n_atoms, k_fft) complex array. */
    k_fft = (n_atoms > 0 && out_count % (2 * n_atoms) == 0)
                ? out_count / (2 * n_atoms) : -1;

    if (f64_buffer_count(&views[2]) != n_groups || k_fft < 2) {
        release_f64_buffers(views, 4);
        PyErr_SetString(PyExc_ValueError,
            "depths and multiplicities must be the same length, and the "
            "output must be the float64 view of an (n_atoms, k_fft) "
            "complex128 array");
        return NULL;
    }
    if (n_atoms > UINT32_MAX || n_groups > UINT32_MAX || k_fft > UINT32_MAX) {
        release_f64_buffers(views, 4);
        PyErr_SetString(PyExc_OverflowError, "batch exceeds uint32 limit");
        return NULL;
    }

    /* Release the GIL: this is n_atoms * n_groups * k_fft/2 complex powers,
     * which reaches tens of seconds on a foundation-scale cohort. */
    Py_BEGIN_ALLOW_THREADS
    err = beam_publicness_pgf((const double *)views[0].buf, (uint32_t)n_atoms,
                             (const double *)views[1].buf,
                             (const double *)views[2].buf,
                             (uint32_t)n_groups, (uint32_t)k_fft,
                             (double *)views[3].buf);
    Py_END_ALLOW_THREADS

    release_f64_buffers(views, 4);
    if (err != BEAM_OK) return set_beam_error(err);
    Py_RETURN_NONE;
}
/* ── publicness_accumulate(pmf, mean, var, weight, edges,
 *                          counts, retained, sigma, floor, tol) → None ── */

PyObject *py_publicness_accumulate(PyObject *self, PyObject *args) {
    (void)self;
    static const int writable[7] = {0, 0, 0, 0, 0, 1, 1};
    static const char *names[7] = {"pmf", "mean", "variance", "weight",
                                   "bin edges", "count output",
                                   "retained-mass output"};
    PyObject *objs[7];
    Py_buffer views[7];
    Py_ssize_t n_atoms, k_fft, n_bins, pmf_count;
    double tail_sigma, tail_floor, mass_tol;
    BeamError err;

    if (!PyArg_ParseTuple(args, "OOOOOOOddd", &objs[0], &objs[1], &objs[2],
                          &objs[3], &objs[4], &objs[5], &objs[6],
                          &tail_sigma, &tail_floor, &mass_tol)) return NULL;
    if (!get_f64_buffers(objs, views, writable, names, 7)) return NULL;

    n_atoms = f64_buffer_count(&views[1]);
    pmf_count = f64_buffer_count(&views[0]);
    k_fft = (n_atoms > 0 && pmf_count % n_atoms == 0) ? pmf_count / n_atoms : -1;
    n_bins = f64_buffer_count(&views[4]) - 1;

    if (k_fft < 1 || n_bins < 1 ||
        f64_buffer_count(&views[2]) != n_atoms ||
        f64_buffer_count(&views[3]) != n_atoms ||
        f64_buffer_count(&views[6]) != n_atoms ||
        f64_buffer_count(&views[5]) != n_bins) {
        release_f64_buffers(views, 7);
        PyErr_SetString(PyExc_ValueError,
            "pmf must be (n_atoms, k_fft); mean, variance, weight and the "
            "retained-mass output must each have n_atoms entries; and the "
            "count output must have one entry fewer than the bin edges");
        return NULL;
    }
    if (n_atoms > UINT32_MAX || k_fft > UINT32_MAX || n_bins > UINT32_MAX) {
        release_f64_buffers(views, 7);
        PyErr_SetString(PyExc_OverflowError, "batch exceeds uint32 limit");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    err = beam_publicness_accumulate((const double *)views[0].buf,
                                    (uint32_t)n_atoms, (uint32_t)k_fft,
                                    (const double *)views[1].buf,
                                    (const double *)views[2].buf,
                                    (const double *)views[3].buf,
                                    (const double *)views[4].buf,
                                    (uint32_t)n_bins,
                                    tail_sigma, tail_floor, mass_tol,
                                    (double *)views[5].buf,
                                    (double *)views[6].buf);
    Py_END_ALLOW_THREADS

    release_f64_buffers(views, 7);
    if (err != BEAM_OK) return set_beam_error(err);
    Py_RETURN_NONE;
}
/* ── publicness_accumulate_grouped(..., groups, ..., matrix, ...) → None ── */

PyObject *py_publicness_accumulate_grouped(
    PyObject *self, PyObject *args) {
    (void)self;
    static const int writable[7] = {0, 0, 0, 0, 0, 1, 1};
    static const char *names[7] = {
        "pmf", "mean", "variance", "weight", "bin edges",
        "grouped count output", "retained-mass output"
    };
    PyObject *objs[7], *group_obj;
    Py_buffer views[7], group_view;
    Py_ssize_t n_atoms, k_fft, n_bins, n_groups_out, pmf_count;
    double tail_sigma, tail_floor, mass_tol;
    BeamError err;

    if (!PyArg_ParseTuple(args, "OOOOOOOOddd",
                          &objs[0], &objs[1], &objs[2], &objs[3],
                          &group_obj, &objs[4], &objs[5], &objs[6],
                          &tail_sigma, &tail_floor, &mass_tol))
        return NULL;
    if (!get_f64_buffers(objs, views, writable, names, 7)) return NULL;
    if (!get_u32_buffer(group_obj, &group_view, "output groups")) {
        release_f64_buffers(views, 7);
        return NULL;
    }

    n_atoms = f64_buffer_count(&views[1]);
    pmf_count = f64_buffer_count(&views[0]);
    k_fft = (n_atoms > 0 && pmf_count % n_atoms == 0)
        ? pmf_count / n_atoms : -1;
    n_bins = f64_buffer_count(&views[4]) - 1;
    n_groups_out = n_bins > 0 && f64_buffer_count(&views[5]) % n_bins == 0
        ? f64_buffer_count(&views[5]) / n_bins : -1;

    if (k_fft < 1 || n_bins < 1 || n_groups_out < 1 ||
        f64_buffer_count(&views[2]) != n_atoms ||
        f64_buffer_count(&views[3]) != n_atoms ||
        group_view.len / (Py_ssize_t)sizeof(uint32_t) != n_atoms ||
        f64_buffer_count(&views[6]) != n_atoms) {
        PyBuffer_Release(&group_view);
        release_f64_buffers(views, 7);
        PyErr_SetString(PyExc_ValueError,
            "pmf must be (n_atoms, k_fft); mean, variance, weight, groups "
            "and retained mass must each have n_atoms entries; and grouped "
            "counts must be a whole number of rows by n_bins");
        return NULL;
    }
    if (n_atoms > UINT32_MAX || k_fft > UINT32_MAX ||
        n_bins > UINT32_MAX || n_groups_out > UINT32_MAX) {
        PyBuffer_Release(&group_view);
        release_f64_buffers(views, 7);
        PyErr_SetString(PyExc_OverflowError, "batch exceeds uint32 limit");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    err = beam_publicness_accumulate_grouped(
        (const double *)views[0].buf,
        (uint32_t)n_atoms, (uint32_t)k_fft,
        (const double *)views[1].buf,
        (const double *)views[2].buf,
        (const double *)views[3].buf,
        (const uint32_t *)group_view.buf, (uint32_t)n_groups_out,
        (const double *)views[4].buf, (uint32_t)n_bins,
        tail_sigma, tail_floor, mass_tol,
        (double *)views[5].buf, (double *)views[6].buf);
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&group_view);
    release_f64_buffers(views, 7);
    if (err != BEAM_OK) return set_beam_error(err);
    Py_RETURN_NONE;
}
