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
binary installed under a clean `git status`.

Headers used to be worse still: they are not in `sources`, so touching one
triggered nothing at all. `depends` below fixes that half -- see the note there
for what it cost to find. The mtime-granularity half remains.

So any A/B experiment that edits C and compares results -- mutation testing
above all -- must pass `--force`, and should verify the .so hash actually
moved before trusting the verdict. A full forced rebuild takes about seven
seconds; a verdict computed against a stale binary is worth less than nothing,
because it is believed.
"""
import os
import platform
import sys
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


def headers():
    """Every header the extension compiles against, for `depends`.

    A GLOB, where `sources` is a manifest, and the asymmetry is deliberate.

    A source missing from its manifest is missing code: the extension builds
    and behaves differently, silently, which is what the manifests exist to
    prevent. A header missing from `depends` is a missed REBUILD -- precisely
    the bug this list fixes -- so a list that picks up new headers by itself is
    the safer one, and a hand-maintained list would reintroduce the failure the
    first time someone added a header and forgot.

    What it cost to find: editing only `dp.h` and `model_state.h` and running
    `build_ext` rebuilt nothing, leaving an .so nineteen minutes older than the
    headers compiled into it, and every check that runs against the extension
    testing the previous binary. `make` never had this problem -- its CFLAGS
    carry `-MMD -MP` and it includes the generated .d files -- so the two build
    systems disagreed about what "up to date" meant.
    """
    out = []
    for d in ("include", "src/native/internal"):
        out += [str(p.relative_to(HERE)) for p in sorted((HERE / d).rglob("*.h"))]
    if not out:
        raise ValueError("no headers found; depends= would be silently empty")
    return out


def target_is_x86():
    """Whether the extension is being compiled FOR x86.

    `-mpopcnt` exists only there. On macOS the target can differ from the
    machine doing the build (cibuildwheel sets ARCHFLAGS to choose it), so
    ARCHFLAGS wins when it names an architecture.
    """
    archflags = os.environ.get("ARCHFLAGS", "")
    if "-arch" in archflags:
        return "x86_64" in archflags
    return platform.machine().lower() in ("x86_64", "amd64", "i386", "i686")


def libomp_prefix():
    """A libomp installation for Apple clang, or None.

    Apple's compiler accepts OpenMP pragmas only through `-Xpreprocessor
    -fopenmp` and ships no runtime, so the headers and library must come from
    elsewhere: BEAM_LIBOMP_PREFIX if set, else Homebrew's usual locations.
    """
    candidates = [os.environ.get("BEAM_LIBOMP_PREFIX"),
                  "/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]
    for prefix in filter(None, candidates):
        if (Path(prefix) / "include" / "omp.h").is_file():
            return prefix
    return None


def platform_flags():
    """Compile and link flags for this platform: (compile, link)."""
    compile_args = ["-O2", "-std=c11"]
    link_args = []
    if sys.platform.startswith("linux"):
        compile_args.append("-fopenmp")
        link_args.append("-fopenmp")
    elif sys.platform == "darwin":
        prefix = libomp_prefix()
        if prefix:
            compile_args += ["-Xpreprocessor", "-fopenmp", f"-I{prefix}/include"]
            link_args += [f"-L{prefix}/lib", "-lomp"]
        else:
            # Every OpenMP use in the C core sits behind `#ifdef _OPENMP`, so
            # this builds a correct, single-threaded extension.
            print("BEAM: libomp not found (brew install libomp); "
                  "building without multithreading", file=sys.stderr)
    else:
        sys.exit("BEAM builds on Linux and macOS. On Windows, install it "
                 "inside WSL: https://learn.microsoft.com/windows/wsl/install")
    if target_is_x86():
        compile_args.append("-mpopcnt")
    return compile_args, link_args


COMPILE_ARGS, LINK_ARGS = platform_flags()

setup(
    ext_modules=[Extension(
        "BEAM._cbeam",
        sources=manifest("python_binding_sources.txt") + manifest("native_sources.txt"),
        depends=headers(),
        include_dirs=["include", "src/native/internal"],
        extra_compile_args=COMPILE_ARGS,
        extra_link_args=LINK_ARGS,
        libraries=["m"],
    )],
)
