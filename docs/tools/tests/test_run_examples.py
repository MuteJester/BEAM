"""The documentation example runner.

Each test builds a small docs/ layout in a temporary directory, writes
example programs into it and runs them through `run_examples.run_all`, so
nothing here touches the real site or needs BEAM.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import run_examples  # noqa: E402

HELPERS = TOOLS.parent / "examples" / "_docs.py"


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A docs/ tree with the real figure/data helpers and an empty src/."""
    root = tmp_path / "docs"
    (root / "examples").mkdir(parents=True)
    (root / "examples" / "_docs.py").write_text(HELPERS.read_text())
    src = tmp_path / "src"
    src.mkdir()
    monkeypatch.delenv("BEAM_PRETRAINED_REGISTRY", raising=False)

    def example(key: str, code: str) -> Path:
        path = root / "examples" / f"{key}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
        return path

    def run(**kw):
        return run_examples.run_all(root, src, **kw)

    return root, example, run


def test_output_and_display_code_are_recorded(site):
    root, example, run = site
    example("intro/hello", 'import math  # hide\nx = 1.5\nprint("hello", x)\n')
    summary = run()
    assert summary.ran == ["intro/hello"] and not summary.failed
    out = root / "src" / "generated" / "intro" / "hello"
    assert (out / "output.txt").read_text() == "hello 1.5\n"
    assert (out / "code.py").read_text() == 'x = 1.5\nprint("hello", x)\n'


def test_hidden_blocks_are_left_out_of_the_displayed_code(site):
    root, example, run = site
    example("intro/blocks", "# --- hide\nimport os\n# --- show\nprint(1)\n")
    run()
    code = (root / "src" / "generated" / "intro" / "blocks" / "code.py").read_text()
    assert code == "print(1)\n"


def test_figure_is_written_in_a_light_and_a_dark_variant(site):
    root, example, run = site
    example("plots/curve", (
        "from _docs import figure\n"
        "with figure('curve') as (fig, ax):\n"
        "    ax.plot([1, 2, 3], [1, 4, 9])\n"
        "    ax.set_xlabel('depth')\n"))
    assert not run().failed
    figs = root / "public" / "figures" / "plots" / "curve"
    light, dark = figs / "curve-light.svg", figs / "curve-dark.svg"
    assert light.exists() and dark.exists()
    assert light.read_text() != dark.read_text()


def test_data_is_written_as_json(site):
    root, example, run = site
    example("viz/grid", (
        "import numpy as np\nfrom _docs import save_data\n"
        "save_data('grid', {'x': np.arange(3), 'y': np.float64(0.5)})\n"))
    assert not run().failed
    data = json.loads((root / "public" / "data" / "viz" / "grid" / "grid.json").read_text())
    assert data == {"x": [0, 1, 2], "y": 0.5}


def test_an_unchanged_example_is_not_run_again(site):
    _, example, run = site
    example("intro/hello", "print('hi')\n")
    run()
    summary = run()
    assert summary.ran == [] and summary.skipped == ["intro/hello"]


def test_a_changed_example_runs_again(site):
    _, example, run = site
    example("intro/hello", "print('hi')\n")
    run()
    example("intro/hello", "print('hello')\n")
    assert run().ran == ["intro/hello"]


def test_force_runs_everything(site):
    _, example, run = site
    example("intro/hello", "print('hi')\n")
    run()
    assert run(force=True).ran == ["intro/hello"]


def test_a_different_model_runs_examples_again(site, tmp_path, monkeypatch):
    _, example, run = site
    example("intro/hello", "print('hi')\n")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"schema": 1, "models": {"m": {"sha256": "a" * 64}}}))
    monkeypatch.setenv("BEAM_PRETRAINED_REGISTRY", str(registry))
    run()
    registry.write_text(json.dumps({"schema": 1, "models": {"m": {"sha256": "b" * 64}}}))
    assert run().ran == ["intro/hello"]


def test_only_selects_examples_by_glob(site):
    _, example, run = site
    example("intro/a", "print(1)\n")
    example("other/b", "print(2)\n")
    assert run(only="intro/*").ran == ["intro/a"]


def test_check_passes_when_numbers_agree_to_tolerance(site):
    root, example, run = site
    value = root / "examples" / "num" / "value.txt"
    value.parent.mkdir(parents=True)
    value.write_text("0.1000000000000")
    example("num/drift", "from pathlib import Path\n"
                         "print(Path(__file__).with_name('value.txt').read_text())\n")
    run()
    value.write_text("0.1000000000001")
    summary = run(check=True)
    assert summary.mismatched == [] and not summary.failed


def test_check_fails_and_names_the_example_when_a_number_moves(site):
    root, example, run = site
    value = root / "examples" / "num" / "value.txt"
    value.parent.mkdir(parents=True)
    value.write_text("0.1")
    example("num/drift", "from pathlib import Path\n"
                         "print(Path(__file__).with_name('value.txt').read_text())\n")
    run()
    value.write_text("0.1001")
    summary = run(check=True)
    assert [m.key for m in summary.mismatched] == ["num/drift"]
    assert "0.1001" in summary.mismatched[0].detail
    out = root / "src" / "generated" / "num" / "drift" / "output.txt"
    assert out.read_text() == "0.1\n"          # --check never rewrites


def test_check_fails_when_text_changes(site):
    _, example, run = site
    example("t/words", "print('alpha')\n")
    run()
    example("t/words", "print('beta')\n")
    assert [m.key for m in run(check=True).mismatched] == ["t/words"]


def test_a_failing_example_is_reported_with_its_traceback(site):
    root, example, run = site
    example("bad/boom", "raise ValueError('no such sequence')\n")
    summary = run()
    assert [f.key for f in summary.failed] == ["bad/boom"]
    assert "ValueError: no such sequence" in summary.failed[0].detail
    assert not (root / "src" / "generated" / "bad" / "boom" / "output.txt").exists()


def test_main_exits_non_zero_on_failure(site, capsys):
    root, example, _ = site
    example("bad/boom", "raise SystemExit(3)\n")
    code = run_examples.main(["--root", str(root), "--src", str(root.parent / "src")])
    assert code == 1
    assert "bad/boom" in capsys.readouterr().out


def test_examples_can_import_the_package_source(site, tmp_path):
    root, example, run = site
    pkg = root.parent / "src" / "toypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("VALUE = 42\n")
    example("intro/imp", "import toypkg\nprint(toypkg.VALUE)\n")
    run()
    assert (root / "src" / "generated" / "intro" / "imp" / "output.txt").read_text() == "42\n"
