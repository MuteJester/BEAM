"""Run the documentation's example programs and record what they produce.

    python docs/tools/run_examples.py                 # run new or changed examples
    python docs/tools/run_examples.py --only 'depth/*'
    python docs/tools/run_examples.py --force         # run everything
    python docs/tools/run_examples.py --check         # re-run, compare, change nothing

Every example is a Python file `docs/examples/<section>/<name>.py` (files
starting with `_` are helpers). It runs in its own process, with the package
source on the path, and the site shows exactly what it produced:

    docs/src/generated/<section>/<name>/code.py      the code as displayed
    docs/src/generated/<section>/<name>/output.txt   what it printed
    docs/src/generated/<section>/<name>/meta.json    what it ran against
    docs/public/figures/<section>/<name>/*.svg       figures (light and dark)
    docs/public/data/<section>/<name>/*.json         data for interactive figures

Lines ending in `# hide`, and blocks between `# --- hide` and `# --- show`,
run but are left out of the displayed code.

An example is run again only when its source, the helpers or the models it
can load (the registry in $BEAM_PRETRAINED_REGISTRY) changed: some examples
take minutes. `--check` re-runs everything and compares with what was recorded: the
printed output, the interactive data, and the data each figure plots
(recorded beside it as <name>.data.json), numbers to a relative 1e-9 and
everything else exactly.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]
REL_TOL = 1e-9
# No absolute tolerance: BEAM's outputs are mostly tiny probabilities and
# error bounds, and an absolute floor would let any of them change unnoticed.
ABS_TOL = 0.0
NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


@dataclass
class Problem:
    key: str
    detail: str


@dataclass
class Summary:
    ran: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[Problem] = field(default_factory=list)
    mismatched: list[Problem] = field(default_factory=list)


def _sha256(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def discover(root: Path) -> dict[str, Path]:
    """Example key ("section/name") -> file, helpers excluded."""
    examples = root / "examples"
    out = {}
    for path in sorted(examples.rglob("*.py")):
        if any(part.startswith("_") for part in path.relative_to(examples).parts):
            continue
        out[path.relative_to(examples).with_suffix("").as_posix()] = path
    return out


def display_code(source: str) -> str:
    """The source as shown on the page: hidden lines and blocks removed."""
    shown, hiding = [], False
    for line in source.splitlines(keepends=True):
        marker = line.strip()
        if marker == "# --- hide":
            hiding = True
        elif marker == "# --- show":
            hiding = False
        elif not hiding and not line.rstrip().endswith("# hide"):
            shown.append(line)
    return "".join(shown)


def model_fingerprint() -> str:
    """The SHA-256s of the models the examples can load, or 'none'."""
    registry = os.environ.get("BEAM_PRETRAINED_REGISTRY")
    if not registry:
        return "none"
    models = json.loads(Path(registry).read_text()).get("models", {})
    return _sha256(",".join(sorted(m["sha256"] for m in models.values())))


def helpers_fingerprint(root: Path) -> str:
    helpers = sorted((root / "examples").glob("_*.py"))
    return _sha256("".join(p.read_text() for p in helpers))


def numbers_agree(expected: str, got: str) -> str | None:
    """None if the two outputs agree, else a description of the first difference."""
    exp_lines, got_lines = expected.splitlines(), got.splitlines()
    if len(exp_lines) != len(got_lines):
        return f"{len(exp_lines)} lines recorded, {len(got_lines)} now"
    for i, (e, g) in enumerate(zip(exp_lines, got_lines), 1):
        if NUMBER.sub("#", e) != NUMBER.sub("#", g):
            return f"line {i}: recorded {e!r}, now {g!r}"
        for a, b in zip(NUMBER.findall(e), NUMBER.findall(g)):
            if not math.isclose(float(a), float(b), rel_tol=REL_TOL, abs_tol=ABS_TOL):
                return f"line {i}: recorded {e!r}, now {g!r}"
    return None


def _values_agree(a, b, where: str = "") -> str | None:
    """None if two JSON values agree (numbers to REL_TOL), else where they differ."""
    if isinstance(a, bool) or isinstance(b, bool) or not (
            isinstance(a, (int, float)) and isinstance(b, (int, float))):
        if type(a) is not type(b):
            return f"{where or 'value'}: {type(a).__name__} recorded, {type(b).__name__} now"
    if isinstance(a, dict):
        if set(a) != set(b):
            return f"{where or 'object'}: keys differ"
        for k in a:
            d = _values_agree(a[k], b[k], f"{where}.{k}" if where else str(k))
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{where or 'list'}: {len(a)} items recorded, {len(b)} now"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _values_agree(x, y, f"{where}[{i}]")
            if d:
                return d
        return None
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        if not math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            return f"{where}: recorded {a!r}, now {b!r}"
        return None
    return None if a == b else f"{where}: recorded {a!r}, now {b!r}"


def json_files_agree(recorded: Path, now: Path) -> str | None:
    """Compare every .json file of two directories; None if all agree."""
    names = {p.name for p in recorded.glob("*.json")} | {p.name for p in now.glob("*.json")}
    for name in sorted(names):
        r, n = recorded / name, now / name
        if not r.exists() or not n.exists():
            return f"{name}: {'not recorded' if not r.exists() else 'no longer written'}"
        d = _values_agree(json.loads(r.read_text()), json.loads(n.read_text()))
        if d:
            return f"{name}: {d}"
    return None


def _execute(path: Path, root: Path, src: Path, figures: Path, data: Path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(src), str(root / "examples")] + [p for p in [env.get("PYTHONPATH")] if p])
    env["DOCS_FIGURES"], env["DOCS_DATA"] = str(figures), str(data)
    env["MPLBACKEND"] = "Agg"
    with tempfile.TemporaryDirectory() as cwd:
        return subprocess.run([sys.executable, str(path)], cwd=cwd, env=env,
                              capture_output=True, text=True)


def run_all(root: Path, src: Path, *, only: str | None = None, force: bool = False,
            check: bool = False) -> Summary:
    root, src = Path(root), Path(src)
    summary = Summary()
    model, helpers = model_fingerprint(), helpers_fingerprint(root)
    for key, path in discover(root).items():
        if only and not fnmatch.fnmatch(key, only):
            continue
        source = path.read_text()
        gen = root / "src" / "generated" / key
        meta_path = gen / "meta.json"
        fingerprint = {"source_sha256": _sha256(source), "helpers_sha256": helpers,
                       "model_sha256": model}
        if not (force or check) and meta_path.exists() and (gen / "output.txt").exists():
            meta = json.loads(meta_path.read_text())
            if all(meta.get(k) == v for k, v in fingerprint.items()):
                summary.skipped.append(key)
                continue

        if check:
            with tempfile.TemporaryDirectory() as scratch:
                fig_now, data_now = Path(scratch) / "f", Path(scratch) / "d"
                proc = _execute(path, root, src, fig_now, data_now)
                files_diff = (
                    json_files_agree(root / "public" / "figures" / key, fig_now)
                    or json_files_agree(root / "public" / "data" / key, data_now)
                ) if proc.returncode == 0 else None
        else:
            proc = _execute(path, root, src, root / "public" / "figures" / key,
                            root / "public" / "data" / key)
        if proc.returncode != 0:
            summary.failed.append(Problem(key, proc.stderr.strip() or
                                          f"exited with status {proc.returncode}"))
            continue

        if check:
            recorded = gen / "output.txt"
            if not recorded.exists():
                summary.mismatched.append(Problem(key, "no recorded output"))
                continue
            diff = numbers_agree(recorded.read_text(), proc.stdout) or files_diff
            if diff:
                summary.mismatched.append(Problem(key, diff))
            else:
                summary.ran.append(key)
            continue

        gen.mkdir(parents=True, exist_ok=True)
        (gen / "output.txt").write_text(proc.stdout)
        (gen / "code.py").write_text(display_code(source))
        meta_path.write_text(json.dumps({**fingerprint, "ran_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S%z")}, indent=2) + "\n")
        summary.ran.append(key)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DOCS, help="the docs/ directory")
    ap.add_argument("--src", type=Path, default=DOCS.parent / "src",
                    help="the package source put on PYTHONPATH")
    ap.add_argument("--only", help="glob over example keys, e.g. 'depth/*'")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)

    started = time.perf_counter()
    s = run_all(a.root, a.src, only=a.only, force=a.force, check=a.check)
    for key in s.ran:
        print(f"{'checked' if a.check else 'ran':8} {key}")
    for key in s.skipped:
        print(f"{'cached':8} {key}")
    for p in s.mismatched:
        print(f"CHANGED  {p.key}: {p.detail}")
    for p in s.failed:
        print(f"FAILED   {p.key}\n{p.detail}")
    print(f"{len(s.ran)} run, {len(s.skipped)} cached, {len(s.mismatched)} changed, "
          f"{len(s.failed)} failed in {time.perf_counter() - started:.1f} s")
    return 1 if (s.failed or s.mismatched) else 0


if __name__ == "__main__":
    sys.exit(main())
