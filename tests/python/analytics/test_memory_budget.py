"""The memory budget: one number, enforced two ways.

The budget is the most scratch BEAM will hold at once for one operation. Every
per-length dynamic program in `dp/`, `math/` and `algorithms/` must therefore

  - refuse, BEFORE allocating, when one program does not fit, and
  - divide the budget across lengths that run concurrently rather than letting
    the thread count multiply the footprint.

Fifteen of eighteen such files did neither. `calloc` on Linux grants an absurd
request lazily and fails on first write, by which time the OOM killer is
choosing its victim from the whole machine rather than from this process, so an
unguarded sweep does not fail -- it takes the machine down.

Three things are pinned here, and the third is the one that keeps the other two
honest:

  1. every heavy entry point refuses under a budget below one sweep;
  2. the refusal is DIAGNOSABLE -- it names the budget as the cause and names
     the way out, because a bare "out of memory" sends the user looking at
     their corpus for a problem that is in their settings;
  3. a reduced budget does not change a single answer. A guard that returns
     different numbers is not a guard, and this is what would catch one
     computed from a different formula than the allocation it protects.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

import BEAM
from BEAM import BudgetExceeded

#: Small enough that no sweep on any model fits, large enough to be a
#: plausible setting rather than a degenerate zero.
STARVED = 2048

#: Far below the 4 GB default, comfortably above one sweep of the test model.
#: Concurrency drops; answers must not.
REDUCED = 64 << 20


@pytest.fixture(scope="module")
def model():
    """max_length is the corpus's longest (12). Every length up to it is live
    under the prior, and the moment drivers below run a whole Mellin pass
    before they reach the refusal they are pinned to, fifty times over; two
    lengths the corpus never had would add a third to each pass."""
    return BEAM.BeamModel(
        ["CASSLGGYEQYF", "CASSLGGTEQYF", "CASRLGGYEQYF", "CASSPGGYEQYF",
         "CASSLGGYEQF", "CASRLGGTEQF"],
        order=3, max_length=12)


def _refusing_calls(m):
    """The drivers above, including two `_calls` does not carry."""
    d = dict(_calls(m))
    d["length_marginals"] = lambda: m.length_marginals()
    # A moment sweep of order 12 holds thirteen moments a state, more than a
    # Mellin pass of one order (one vector of eight): so a budget between the
    # two lets the Mellin pass through and stops the moment sweep itself.
    d["surprisal_moments"] = lambda: m.surprisal_moments(q=1.0, order=MOMENT_ORDER)
    d["length_derivatives"] = lambda: m.length_moments(q=1.0, order=MOMENT_ORDER)
    return d


#: See `_refusing_calls`.
MOMENT_ORDER = 12


def _moment_sweep_bytes(m):
    """`beam_jet_sweep_bytes` (internal/beam/dp.h) at MOMENT_ORDER + 1
    moments; a budget one byte short of it stops the moment sweep."""
    S, R, nm = m.n_states, 20, MOMENT_ORDER + 1
    return S * R * 8 + S * 8 + 2 * nm * S * 8 + S


def _calls(m):
    """Every entry point that runs a per-length dynamic program."""
    return {
        "path_count":           lambda: m.path_count(12),
        "path_count_total":     lambda: m.path_count(),
        "entropy":              lambda: m.entropy(),
        "hill_number":          lambda: m.hill_number(2.0),
        "log_mellin":           lambda: m.log_mellin(0.5),
        "surprisal_moments":    lambda: m.surprisal_moments(q=1.0, order=2),
        "surprisal_cumulants":  lambda: m.surprisal_cumulants(q=1.0, order=2),
        "surprisal_range":      lambda: m.surprisal_range(),
        "cgf_jet":              lambda: m.cgf_jet(0.4),
        "positional_marginals": lambda: m.positional_marginals(12),
        "expected_counts":      lambda: m.expected_counts(),
        "top_k_sequences":      lambda: m.top_k_sequences(3),
        "sample":               lambda: m.sample(4, seed=1),
        "topology":             lambda: m.topology(12),
        "spectrum":             lambda: m.spectrum(bins=64),
    }


def _same(a, b):
    """Exact equality over whatever shape an entry point returns.

    Results here are dicts, tuples, arrays, floats and strings in a mix, so
    this walks the structure rather than assuming one. Equality is EXACT: these
    are deterministic sweeps whose reductions are order-independent, and a
    tolerance would be the place a real drift went to hide.
    """
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and np.array_equal(a, b)
    return a == b


@pytest.fixture
def restore_budget():
    saved = BEAM.memory_budget()
    yield
    BEAM.set_memory_budget(saved)
    assert BEAM.memory_budget() == saved


#: Named rather than derived from `_calls`, so that deleting an entry point
#: from the dict is a test that disappears rather than one that silently stops
#: checking anything.
GUARDED = ["path_count", "path_count_total", "entropy", "hill_number",
           "log_mellin", "surprisal_moments", "surprisal_cumulants",
           "surprisal_range", "cgf_jet", "positional_marginals",
           "top_k_sequences", "sample", "topology", "spectrum"]


@pytest.mark.parametrize("name", GUARDED)
def test_every_heavy_entry_point_refuses_under_a_starved_budget(
        model, restore_budget, name):
    """A sweep that does not fit must refuse rather than allocate and hope.

    Parametrised one-per-entry-point deliberately: a single test looping over
    all of them reports the first failure and hides the rest, and the whole
    point of this file is that the guard is present in EVERY one of these
    paths rather than in most of them.
    """
    call = _calls(model)[name]
    BEAM.set_memory_budget(STARVED)
    # `BudgetExceeded`, not bare `Exception`. This test used to catch anything
    # at all and assert on the words in the message, because a budget refusal
    # and a genuine out-of-memory shared an error code and therefore a Python
    # class -- the type system had nothing to offer for the refusal this repo
    # has invested the most engineering in. Catching the type is the point of
    # having split BEAM_ERR_BUDGET out.
    with pytest.raises(BudgetExceeded) as exc:
        call()
    text = str(exc.value)
    # Names the cause...
    assert "memory budget" in text, f"{name} refused without naming why: {text}"
    # ...and the way out. A refusal the user cannot act on is a crash with
    # better manners.
    assert "set_memory_budget" in text, f"{name} named no remedy: {text}"


def test_scoring_one_sequence_is_not_budgeted(model, restore_budget):
    """The guards are targeted, not blanket.

    `log_pseq` walks one sequence through the trellis and allocates nothing
    that scales with the state space, so it must still work under a budget
    that refuses every sweep. A guard broad enough to catch this one would be
    refusing calls that cannot exhaust anything.
    """
    want = model.log_pseq("CASSLGGYEQYF")
    BEAM.set_memory_budget(STARVED)
    assert model.log_pseq("CASSLGGYEQYF") == want


def test_a_reduced_budget_changes_no_answer(model, restore_budget):
    """The budget changes the schedule and what is refused. Never a number.

    This is the assertion that would catch a guard whose sizing expression
    drifted from the allocation it guards, or a concurrency bound that altered
    the order terms are summed in. Every quantity is compared exactly: these
    are deterministic sweeps and the reductions are order-independent, so
    'close enough' would be hiding something.
    """
    calls = _calls(model)
    before = {k: fn() for k, fn in calls.items()}

    BEAM.set_memory_budget(REDUCED)
    after = {k: fn() for k, fn in calls.items()}

    for k in calls:
        assert _same(before[k], after[k]), \
            f"{k} moved when only the budget changed"


def test_the_refusal_reports_a_figure_in_a_useful_unit(model, restore_budget):
    """A refusal has to say how much, in units the reader can compare.

    Both halves were printed in gigabytes at first, so every message about a
    small budget read "needs about 0.00 GB and the budget is 0.00 GB" -- the
    message existed to report a figure and reported nothing, in exactly the
    case a user hits deliberately.
    """
    BEAM.set_memory_budget(STARVED)
    with pytest.raises(BudgetExceeded) as exc:
        model.path_count(12)
    text = str(exc.value)
    assert "0.00 GB" not in text, f"unit did not scale: {text}"
    assert any(u in text for u in ("bytes", "kB", "MB", "GB")), text
    # And the two figures must differ: needing exactly the budget would not be
    # a refusal, so a message showing the same number twice is a formatting
    # bug rather than a diagnosis.
    assert text.count("2.05 kB") == 1, f"budget and need printed alike: {text}"


def test_the_reported_figure_matches_an_independent_calculation(
        model, restore_budget):
    """The guard's size must be the allocation's size, checked from outside.

    Everything above would still pass if a sizing expression drifted to half
    the truth: a 2 kB budget refuses either way. That is exactly the defect
    that matters, because a guard computed from a different formula than the
    allocation it protects is not a guard -- and the difference only shows up
    on the machine that runs out of memory.

    So the figure is recomputed here from the model's own shape. The
    path-counting sweep holds one residue-mask word and three state vectors per
    state, and no layer of edge weights, which is what makes it the one sweep
    small enough to state in a line:

        n_states * (sizeof(uint32) + 3 * sizeof(double))

    Anything but agreement means the guard and the allocation have parted.
    """
    want = model.n_states * (4 + 3 * 8)

    BEAM.set_memory_budget(STARVED)
    with pytest.raises(BudgetExceeded) as exc:
        model.path_count(12)

    text = str(exc.value)
    figure = float(re.search(r"needs about ([\d.]+) kB", text).group(1)) * 1e3
    # The message rounds to two places, so allow the rounding and nothing more.
    assert abs(figure - want) <= 10.0, (
        f"guard reports {figure:.0f} B, the allocation needs {want} B: the "
        f"sizing expression and the allocation have drifted apart\n{text}")


#: Entry points that reach an up-front driver refusal, paired with the budget
#: that gets them there.
#:
#: The first version of this list claimed to cover "the returns that leaked".
#: It did not, and the claim was never measured. At a 2 kB budget both moment
#: drivers refuse inside `beam_mellin_terms`, which they call first, so they
#: never reach their own guard -- three of the seven fixed returns were
#: exercised and four were not. That is the same mistake as the leaks
#: themselves: treating sites that look alike as though they behave alike.
#:
#: The middle tier is what separates them. One byte short of an order-12
#: moment sweep, the Mellin pass fits and the moment sweep does not, so
#: `moments.c` refuses on its own account. The message
#: is asserted for exactly this reason -- it is the only evidence available
#: from Python about WHICH return ran.
REFUSING_DRIVERS = [
    # (entry point, budget, the refusal that must be reached)
    ("hill_number",         2048,    "a tilted forward sweep"),      # diversity.c
    ("entropy",             2048,    "an expected-weight sweep"),    # diversity.c
    ("log_mellin",          2048,    "a tilted per-length sweep"),   # mellin.c
    ("length_marginals",    2048,    "a per-length marginal sweep"), # marginals.c
    ("cgf_jet",             2048,    "a tilted cumulant sweep"),     # tilt.c
    ("surprisal_moments",   None,    "a moment sweep"),              # moments.c
    ("length_derivatives",  None,    "a moment sweep"),              # moments.c (via length_moments)
]
# None: one byte short of the moment sweep itself (`_moment_sweep_bytes`), since
# the Mellin pass the moment drivers run first no longer fits below ~1.8 MB.


@pytest.mark.parametrize("name,budget,expected", REFUSING_DRIVERS,
                         ids=[r[0] for r in REFUSING_DRIVERS])
def test_a_refusal_can_be_repeated_without_growing(
        model, restore_budget, name, budget, expected):
    """The refusal path is reachable, repeatable, and frees what it allocated.

    Five of these seven returns leaked their per-length work arrays. The
    sites written by hand freed correctly; the ones generated by a uniform
    script did not, because the cleanup obligation differs at every one and a
    script cannot see that.

    The suite could not have caught it: no test lowered the budget far enough
    to execute a refusal RETURN, so a completely clean ASan+LSan run over all
    201 C tests still missed them. This test exists to make those returns
    execute. Under a plain run it pins that the refusal is deterministic and
    does not degrade; run the suite under LSan and it becomes the leak test.
    """
    call = _refusing_calls(model)[name]
    if budget is None:
        budget = _moment_sweep_bytes(model) - 1
    BEAM.set_memory_budget(budget)

    first = None
    for _ in range(50):                 # enough that a per-call leak is visible
        with pytest.raises(BudgetExceeded) as exc:
            call()
        text = str(exc.value)
        if first is None:
            first = text
        else:
            assert text == first, (
                f"{name}: the refusal message changed between identical calls, "
                f"which means the path is carrying state across them")
    assert "memory budget" in first
    # WHICH return ran, not merely that one did. Without this the moment
    # drivers pass by refusing inside beam_mellin_terms, leaving their own
    # returns -- two of the seven that leaked -- untouched.
    assert expected in first, (
        f"{name} at a {budget}-byte budget reached {first.split(' at length')[0]!r}, "
        f"not {expected!r}: this test is no longer exercising the return it names")
