"""Build the BEAM C extension.

Sources come from the explicit manifests under build_support/ rather than from
globs, so that a nested source layout cannot silently drop a file. Keep this
file thin: package metadata and dependencies belong in pyproject.toml.
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
        include_dirs=["include", "lib/model"],
        extra_compile_args=["-O2", "-std=c11", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        libraries=["m"],
    )],
)
