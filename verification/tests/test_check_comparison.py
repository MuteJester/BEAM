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
