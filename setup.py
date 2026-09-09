"""Build the BEAM C extension.

Sources come from the explicit manifests under build_support/ rather than from
globs, so that a nested source layout cannot silently drop a file. Keep this
file thin: package metadata and dependencies belong in pyproject.toml.

TWO THINGS ABOUT REBUILDS, both of which have produced false test results.

`make` does NOT build this extension. It builds `build/libbeam.a` for the C
tests, from a SEPARATE object tree. Editing a .c file and running `make` leaves
the extension that Python imports exactly as it was.

And `build_ext` decides whether to rebuild by comparing the .so against source
MTIMES, at a granularity that loses to a fast edit-build loop: an edit landing
in the same second as the previous .so is skipped silently, leaving the OLD
binary installed under a clean `git status`. Headers are worse -- they are not
in `sources` at all, so touching one triggers nothing here.

So any A/B experiment that edits C and compares results -- mutation testing
above all -- must pass `--force`, and should verify the .so hash actually
moved before trusting the verdict. A full forced rebuild takes about seven
seconds; a verdict computed against a stale binary is worth less than nothing,
because it is believed.
"""
from pathlib import Path

from setuptools import Extension, setup

HERE = Path(__file__).parent


def manifest(name):
    """Paths listed in a build_support manifest, comments and blanks removed."""
    path = HERE / "build_support" / name
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if not (HERE / line).is_file():
            raise FileNotFoundError(f"{path.name} lists a missing source: {line}")
        out.append(line)
    if not out:
        raise ValueError(f"{path.name} lists no sources")
    return out


setup(
    ext_modules=[Extension(
        "BEAM._cbeam",
        sources=manifest("python_binding_sources.txt") + manifest("native_sources.txt"),
        include_dirs=["include", "src/native/internal"],
        extra_compile_args=["-O2", "-std=c11", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        libraries=["m"],
    )],
)
