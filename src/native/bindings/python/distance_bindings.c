#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "beam/python_bindings.h"
#include "beam/distance.h"
#include "guards.h"

PyObject *py_local_distance(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *pa, *qa, *width, *top_k;
    BeamLocalDistanceConfig config;
    if (!PyArg_ParseTuple(args,"OOOO",&pa,&qa,&width,&top_k)) return NULL;
    if (beam_py_count(width,"width",&config.width)<0 ||
        beam_py_count(top_k,"top_k",&config.top_k)<0) return NULL;
    BeamModel *p=edge_from_capsule(pa), *q=edge_from_capsule(qa);
    if (!p || !q) return NULL;
    BeamLocalDistanceResult *r=NULL;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err=beam_local_distance(p,q,&config,&r);
    Py_END_ALLOW_THREADS
    if (err!=BEAM_OK) return set_beam_error(err);
    /* Native scratch has been freed. Reserve retained native output plus the
     * raw lists/tuples/scalars AND the Python facade's result objects, attribute
     * dictionaries and decoded words while those rows coexist. The deliberately
     * generous bound also includes allocator rounding; it is not measured RSS. */
    const uint64_t materialization=r->output_bytes+16384
        +((uint64_t)r->n_lengths+r->n_windows+r->n_patterns)*1024;
    if (materialization>(uint64_t)PY_SSIZE_T_MAX ||
        beam_py_budget(materialization,r->n_patterns,"local-distance attribution rows")<0) {
        beam_local_distance_free(r);
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_OverflowError,"local-distance result exceeds Python addressable size");
        return NULL;
    }
    if (materialization>r->scratch_bytes) r->scratch_bytes=materialization;
    PyObject *lengths=PyList_New(r->n_lengths), *windows=PyList_New(r->n_windows);
    PyObject *patterns=PyList_New(r->n_patterns), *out=NULL;
    if (!lengths || !windows || !patterns) goto done;
    for (uint32_t i=0; i<r->n_lengths; ++i) {
        const BeamDistanceLength *v=r->lengths+i;
        PyObject *row=Py_BuildValue("(Idddd)",v->length,v->p,v->q,
                                     v->length_contribution,v->pattern_contribution);
        if (!row) goto done;
        PyList_SET_ITEM(lengths,i,row);
    }
    for (uint32_t i=0; i<r->n_windows; ++i) {
        const BeamDistanceWindow *v=r->windows+i;
        PyObject *row=Py_BuildValue("(IIId)",v->length,v->position,v->width,v->contribution);
        if (!row) goto done;
        PyList_SET_ITEM(windows,i,row);
    }
    for (uint32_t i=0; i<r->n_patterns; ++i) {
        const BeamDistancePattern *v=r->patterns+i;
        PyObject *row=Py_BuildValue("(IIIKddd)",v->length,v->position,v->width,
                                   (unsigned long long)v->word_index,v->p,v->q,v->contribution);
        if (!row) goto done;
        PyList_SET_ITEM(patterns,i,row);
    }
    out=Py_BuildValue("{s:I,s:d,s:d,s:d,s:d,s:d,s:K,s:O,s:O,s:O}",
        "width",r->width,"squared_distance",r->squared_distance,
        "length_squared_distance",r->length_squared_distance,
        "pattern_squared_distance",r->pattern_squared_distance,
        "reported_pattern_squared_distance",r->reported_pattern_squared_distance,
        "remainder",r->remainder,"scratch_bytes",(unsigned long long)r->scratch_bytes,
        "lengths",lengths,"windows",windows,"patterns",patterns);
done:
    Py_XDECREF(lengths); Py_XDECREF(windows); Py_XDECREF(patterns);
    beam_local_distance_free(r);
    return out;
}
