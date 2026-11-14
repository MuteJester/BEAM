"""Where every parallel region gets its thread count.

`gil_regions` proves a plan is RESOLVED at the Python boundary. It cannot
prove the region underneath NAMES it, and one region did not:
`occupancy/poisson_binomial/pgf.c` carried no `num_threads` clause at all, so
`py_publicness_pgf` ran at one thread per core however narrow a team the
caller's plan had resolved -- while the gate reported the plan "in force" the
whole time. Wrapping an entry point and applying a plan are different things.

These tests pin the half that was missing, on synthetic sources, because the
real tree's counts move whenever a region is added.
"""
from pathlib import Path

import pytest

import check_lib

ROOT = Path(__file__).resolve().parents[2]


def _tree(tmp_path: Path, body: str, name: str = "a.c") -> Path:
    d = tmp_path / "src" / "native"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(body, encoding="utf-8")
    return tmp_path


# The real tree
# ════════════════════════════════════════════════════════════════════════


def test_every_real_region_names_a_permitted_count():
    missing, forbidden, n = check_lib.thread_count_clauses(ROOT)
    assert missing == [], f"regions with no thread count: {missing}"
    assert forbidden == [], f"regions naming something else: {forbidden}"
    assert n > 80, f"only {n} clauses found; the scan must have lost its way"


# The rules
# ════════════════════════════════════════════════════════════════════════


def test_a_region_with_no_clause_is_reported(tmp_path):
    """pgf.c's defect, which nothing caught for as long as it existed."""
    root = _tree(tmp_path, """
void f(void) {
    #pragma omp parallel for schedule(static) if(n >= 4u)
    for (int i = 0; i < n; ++i) work(i);
}
""")
    missing, forbidden, n = check_lib.thread_count_clauses(root)
    assert len(missing) == 1 and "a.c" in missing[0]
    assert forbidden == [] and n == 0


def test_a_clause_on_the_continuation_line_is_found(tmp_path):
    """Most of them are: a regex over raw lines would call these missing."""
    root = _tree(tmp_path, """
void f(void) {
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:failed) \\
            num_threads(beam_threads())
    for (int i = 0; i < n; ++i) work(i);
}
""")
    missing, forbidden, n = check_lib.thread_count_clauses(root)
    assert missing == [] and forbidden == [] and n == 1


def test_a_cast_does_not_make_a_new_form(tmp_path):
    root = _tree(tmp_path, """
void f(void) {
    #pragma omp parallel for num_threads((int)conc)
    for (int i = 0; i < n; ++i) work(i);
    #pragma omp parallel for num_threads(conc)
    for (int i = 0; i < n; ++i) work(i);
}
""")
    missing, forbidden, n = check_lib.thread_count_clauses(root)
    assert missing == [] and forbidden == [] and n == 2


def test_a_nested_parenthesis_is_not_truncated(tmp_path):
    """`conc ? (int)conc : 1` contains parens; a regex stopping at the first
    `)` reads it as `conc ? (int` and reports a forbidden form."""
    root = _tree(tmp_path, """
void f(void) {
    #pragma omp parallel num_threads(conc ? (int)conc : 1)
    { work(); }
}
""")
    missing, forbidden, n = check_lib.thread_count_clauses(root)
    assert missing == [] and forbidden == [] and n == 1


@pytest.mark.parametrize("expr", ["omp_get_max_threads()",
                                  "omp_get_num_procs()",
                                  "16",
                                  "cores / 2"])
def test_asking_the_machine_directly_is_forbidden(tmp_path, expr):
    """The whole failure plan.h exists to stop: measured between 1.7x and
    3814x the optimum, and bimodal, so unpredictable as well as slow."""
    root = _tree(tmp_path, """
void f(void) {
    #pragma omp parallel for num_threads(%s)
    for (int i = 0; i < n; ++i) work(i);
}
""" % expr)
    missing, forbidden, n = check_lib.thread_count_clauses(root)
    assert missing == []
    assert len(forbidden) == 1 and n == 1


def test_an_omp_for_inside_a_parallel_is_not_asked_for_a_clause(tmp_path):
    """`#pragma omp for` takes no num_threads -- the language rejects it."""
    root = _tree(tmp_path, """
void f(void) {
    #pragma omp parallel num_threads(beam_threads())
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) work(i);
    }
    #pragma omp critical(x)
    { tally(); }
    #pragma omp atomic
    total += 1;
}
""")
    missing, forbidden, n = check_lib.thread_count_clauses(root)
    assert missing == [] and forbidden == [] and n == 1


def test_a_pragma_named_only_in_a_comment_is_not_a_region(tmp_path):
    root = _tree(tmp_path, """
/* This explains the #pragma omp parallel for region below, which has no
 * num_threads in this sentence. */
void f(void) {
    #pragma omp parallel for num_threads(nth)
    for (int i = 0; i < n; ++i) work(i);
}
""")
    missing, forbidden, n = check_lib.thread_count_clauses(root)
    assert missing == [] and forbidden == [] and n == 1
