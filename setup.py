"""Build the BEAM C extension."""
from setuptools import setup, Extension
from pathlib import Path
import glob

sources = ["src/BEAM/_cbeam.c"] + sorted(
    glob.glob("lib/model/*.c") + glob.glob("lib/core/*.c")
    + glob.glob("lib/occupancy/*.c"))

setup(
    ext_modules=[Extension(
        "BEAM._cbeam",
        sources=sources,
        include_dirs=["include", "lib/model"],
        extra_compile_args=["-O2", "-std=c11", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        libraries=["m"],
    )],
)
