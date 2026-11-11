"""The gate's inventory comparison.

Four ids are invisible to the gate today because it compares a SET of bare
names: two Python and two C test names each appear in two files, so deleting
either copy changes nothing the gate can see.
"""


def test_a_duplicate_name_is_not_collapsed():
    from check_lib import compare
    assert compare(["a", "a", "b"], ["a", "b"]) is False


def test_order_still_does_not_matter():
    from check_lib import compare
    assert compare(["b", "a"], ["a", "b"]) is True


def test_collected_ids_keep_duplicates_and_stay_whole():
    """The dedupe that hides a test is in check.py's set comprehensions, not
    in check(). Two files holding the same test name must survive collection,
    and the path must survive with them."""
    from check_lib import collect_ids
    text = "tests/a.py::test_x\ntests/b.py::test_x\nnot an id\n"
    assert collect_ids(text) == ["tests/a.py::test_x", "tests/b.py::test_x"]


def test_the_compared_python_inventory_keeps_paths():
    """The gate split ids on `::` and compared bare names, so a test moving
    between files was invisible. Task 10's bijection proof depends on this."""
    from check_lib import python_inventory
    want, got = python_inventory("tests/a.py::test_x\n", "tests/b.py::test_x\n")
    assert want == ["tests/a.py::test_x"]
    assert got == ["tests/b.py::test_x"]


def test_a_moved_file_is_a_difference():
    from check_lib import compare, python_inventory
    want, got = python_inventory("tests/model/test_x.py::test_a\n",
                                 "tests/suites/model/test_x.py::test_a\n")
    assert compare(want, got) is False


# ── The report has to be able to NAME what the comparison saw ──────────────
#
# `compare` became a multiset test so that deleting one of two same-named
# tests is a difference. The diagnostic printed beneath the FAIL stayed a
# membership test, so for exactly that case it printed nothing at all: the
# name is still in `got`, once. A maintainer facing a FAIL with no `missing`
# and no `added` line is the person most likely to re-record the baseline,
# which is the single failure this gate exists to prevent.

def test_a_lost_duplicate_is_named_in_the_difference():
    from check_lib import inventory_difference
    want = ["a", "a", "b"]
    got = ["a", "b"]
    missing, extra = inventory_difference(got, want)
    assert missing == ["a"]
    assert extra == []


def test_an_added_duplicate_is_named_in_the_difference():
    from check_lib import inventory_difference
    missing, extra = inventory_difference(["a", "a", "b"], ["a", "b"])
    assert missing == []
    assert extra == ["a"]


def test_identical_multisets_differ_in_nothing():
    from check_lib import inventory_difference
    assert inventory_difference(["b", "a"], ["a", "b"]) == ([], [])


def test_a_moved_id_is_named_on_both_sides():
    from check_lib import inventory_difference
    missing, extra = inventory_difference(
        ["tests/suites/x.py::t"], ["tests/python/x.py::t"])
    assert missing == ["tests/python/x.py::t"]
    assert extra == ["tests/suites/x.py::t"]
