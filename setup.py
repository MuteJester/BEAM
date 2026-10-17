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
import shutil
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

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


def llvm_prefix():
    """The LLVM installation holding clang-cl, or None (Windows only).

    BEAM_LLVM_PREFIX wins; otherwise the directory above clang-cl on PATH.
    """
    if os.environ.get("BEAM_LLVM_PREFIX"):
        return Path(os.environ["BEAM_LLVM_PREFIX"])
    exe = shutil.which("clang-cl")
    return Path(exe).resolve().parent.parent if exe else None


def windows_flags():
    """clang-cl flags for Windows: (compile, link, library dirs, libomp.dll).

    Microsoft's cl.exe cannot compile the C core, which uses GCC vector
    extensions; clang-cl can, and produces objects for the Microsoft linker and
    runtime that CPython itself uses. OpenMP comes from LLVM's libomp; its DLL
    is copied beside the extension (see BuildExt), where Windows looks for it.
    """
    prefix = llvm_prefix()
    if prefix is None or not (prefix / "bin" / "clang-cl.exe").is_file():
        sys.exit("BEAM on Windows is compiled with clang-cl, from LLVM "
                 "(https://releases.llvm.org); none was found on PATH. Set "
                 "BEAM_LLVM_PREFIX to the LLVM directory, or install BEAM "
                 "inside WSL: https://learn.microsoft.com/windows/wsl/install")
    # -ffp-contract=off: as on Linux and macOS, no fused multiply-add.
    compile_args = ["/O2", "/std:c11", "/clang:-ffp-contract=off"]
    link_args, library_dirs, dll = [], [], None
    lib, bin_ = prefix / "lib", prefix / "bin"
    if (lib / "libomp.lib").is_file() and (bin_ / "libomp.dll").is_file():
        compile_args.append("-fopenmp")
        library_dirs.append(str(lib))
        link_args.append("libomp.lib")
        dll = bin_ / "libomp.dll"
    else:
        print("BEAM: LLVM's libomp not found; building without multithreading",
              file=sys.stderr)
    if target_is_x86():
        compile_args.append("-mpopcnt")
    return compile_args, link_args, library_dirs, dll


def platform_flags():
    """Compile and link flags for this platform: (compile, link)."""
    # -ffp-contract=off: the C core relies on no fused multiply-add, so that
    # its SIMD and scalar copies, and mirrored computations, agree to the bit.
    # GCC already behaves so under -std=c11; Clang (Apple's compiler) contracts
    # by default and must be told.
    compile_args = ["-O2", "-std=c11", "-ffp-contract=off"]
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
        sys.exit("BEAM builds on Linux, macOS and Windows. On another system, "
                 "build it from source with GCC or Clang and OpenMP.")
    if target_is_x86():
        compile_args.append("-mpopcnt")
    return compile_args, link_args


WINDOWS = sys.platform == "win32"
if WINDOWS:
    COMPILE_ARGS, LINK_ARGS, LIBRARY_DIRS, OMP_DLL = windows_flags()
    # M_PI and friends are behind _USE_MATH_DEFINES in the Microsoft headers.
    MACROS = [("_USE_MATH_DEFINES", None), ("_CRT_SECURE_NO_WARNINGS", None)]
    LIBRARIES = []
else:
    COMPILE_ARGS, LINK_ARGS = platform_flags()
    LIBRARY_DIRS, OMP_DLL, MACROS, LIBRARIES = [], None, [], ["m"]


class BuildExt(build_ext):
    """On Windows: compile with clang-cl, and ship libomp.dll beside the module."""

    def build_extensions(self):
        if WINDOWS and self.compiler.compiler_type == "msvc":
            if not self.compiler.initialized:
                self.compiler.initialize()
            self.compiler.cc = str(llvm_prefix() / "bin" / "clang-cl.exe")
        super().build_extensions()
        if OMP_DLL is not None:
            dest = Path(self.get_ext_fullpath("BEAM._cbeam")).parent
            shutil.copy2(OMP_DLL, dest / OMP_DLL.name)

setup(
    ext_modules=[Extension(
        "BEAM._cbeam",
        sources=manifest("python_binding_sources.txt") + manifest("native_sources.txt"),
        depends=headers(),
        include_dirs=["include", "src/native/internal"],
        define_macros=MACROS,
        extra_compile_args=COMPILE_ARGS,
        extra_link_args=LINK_ARGS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES,
    )],
    cmdclass={"build_ext": BuildExt},
)
