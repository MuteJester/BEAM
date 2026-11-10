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
