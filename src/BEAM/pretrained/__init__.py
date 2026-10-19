"""Published BEAM models: list them, download once, verify, load.

    import BEAM
    BEAM.pretrained_models()                     # what is available
    model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")   # a ready BeamModel

The models are released separately from the code, under GitHub releases
tagged ``models-vN`` and archived on Zenodo with a DOI. ``registry.json``,
shipped inside the package, names each file with its size and SHA-256. A file
is downloaded once into a cache directory and checked against that hash
before it is ever loaded, and checked again on every later load, so a
truncated or altered file is never read as a model.

The cache is ``$BEAM_CACHE_DIR`` if set, else ``$XDG_CACHE_HOME/beam-model``
(``~/.cache/beam-model`` on Linux, ``~/Library/Caches/beam-model`` on macOS,
``%LOCALAPPDATA%\\beam-model`` on Windows).
Once a model is cached, loading it needs no network.

``$BEAM_PRETRAINED_REGISTRY`` points at a different registry file, for a
mirror or for models of your own.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

__all__ = ["PretrainedModel", "pretrained_models", "load_pretrained",
           "pretrained_path", "MODEL_FORMAT"]

#: The model-file format this build reads. A saved model carries it in bytes
#: 4-8 of its header; the test suite checks this matches what ``save`` writes.
MODEL_FORMAT = 9

_MAGIC = 0x4245414D
_CHUNK = 1 << 20


@dataclass(frozen=True)
class PretrainedModel:
    """One published model, as the registry describes it."""
    name: str
    release: str
    file: str
    bytes: int
    sha256: str
    format: int
    description: str
    urls: tuple
    doi: str | None = None
    citation: str | None = None
    license: str | None = None
    built_with: str | None = None

    def __str__(self):
        size = f"{self.bytes / 1e6:.0f} MB"
        doi = f", doi:{self.doi}" if self.doi else ""
        return f"{self.name} ({self.release}, {size}{doi}): {self.description}"


def _registry_path() -> Path:
    override = os.environ.get("BEAM_PRETRAINED_REGISTRY")
    return Path(override) if override else Path(__file__).with_name("registry.json")


def _registry() -> dict[str, PretrainedModel]:
    data = json.loads(_registry_path().read_text(encoding="utf-8"))   # JSON is UTF-8
    if data.get("schema") != 1:
        raise ValueError(f"{_registry_path()}: unsupported registry schema "
                         f"{data.get('schema')!r}")
    out = {}
    for name, m in data["models"].items():
        out[name] = PretrainedModel(
            name=name, release=m["release"], file=m["file"],
            bytes=int(m["bytes"]), sha256=m["sha256"].lower(),
            format=int(m["format"]), description=m["description"],
            urls=tuple(m["urls"]), doi=m.get("doi"),
            citation=m.get("citation"), license=m.get("license"),
            built_with=m.get("built_with"))
    return out


def pretrained_models() -> list[PretrainedModel]:
    """Every published model this version of BEAM knows about."""
    return list(_registry().values())


def cache_dir() -> Path:
    """Where downloaded models are kept."""
    if os.environ.get("BEAM_CACHE_DIR"):
        return Path(os.environ["BEAM_CACHE_DIR"])
    if os.environ.get("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]) / "beam-model"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "beam-model"
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "beam-model"
    return Path.home() / ".cache" / "beam-model"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _lookup(name: str) -> PretrainedModel:
    models = _registry()
    if name not in models:
        known = ", ".join(sorted(models)) or "none in this version"
        raise KeyError(f"no published BEAM model named {name!r}; "
                       f"available: {known}")
    m = models[name]
    if m.format != MODEL_FORMAT:
        raise ValueError(
            f"model {name!r} ({m.release}) is file format {m.format}, and "
            f"this BEAM reads format {MODEL_FORMAT}: upgrade beam-model "
            f"(pip install -U beam-model)")
    return m


def _download(m: PretrainedModel, dest: Path, progress: bool) -> None:
    """Fetch into a temporary file beside `dest`, verify, then rename.

    The rename is atomic, so `dest` either does not exist or is a complete,
    verified file -- also when two processes download at once.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in m.urls:
        fd, tmp = tempfile.mkstemp(prefix=f".{m.file}.", dir=dest.parent)
        try:
            with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as r:
                h, done = hashlib.sha256(), 0
                for block in iter(lambda: r.read(_CHUNK), b""):
                    out.write(block)
                    h.update(block)
                    done += len(block)
                    if progress:
                        print(f"\rBEAM: downloading {m.name} "
                              f"{done / 1e6:6.1f} / {m.bytes / 1e6:.1f} MB",
                              end="", file=sys.stderr, flush=True)
            if progress:
                print(file=sys.stderr)
            if done != m.bytes or h.hexdigest() != m.sha256:
                raise OSError(f"got {done} bytes with SHA-256 {h.hexdigest()}, "
                              f"expected {m.bytes} bytes with {m.sha256}")
            os.replace(tmp, dest)
            return
        except Exception as e:                      # try the next mirror
            errors.append(f"  {url}: {e}")
            if os.path.exists(tmp):
                os.unlink(tmp)
    raise OSError(f"could not download BEAM model {m.name!r}:\n" + "\n".join(errors))


def pretrained_path(name: str, *, progress: bool = True) -> Path:
    """The local path of a published model, downloading it if needed."""
    m = _lookup(name)
    dest = cache_dir() / m.name / m.release / m.file
    if dest.exists() and _sha256(dest) != m.sha256:
        dest.unlink()                               # damaged: fetch again
    if not dest.exists():
        _download(m, dest, progress)
    with open(dest, "rb") as f:
        magic, fmt = struct.unpack("<II", f.read(8))
    if magic != _MAGIC or fmt != m.format:
        raise ValueError(f"{dest}: header says magic {magic:#x} format {fmt}, "
                         f"registry says format {m.format}")
    return dest


def load_pretrained(name: str, *, progress: bool = True):
    """Download (once) and load a published model as a :class:`BeamModel`."""
    from .._model import BeamModel
    return BeamModel.from_file(str(pretrained_path(name, progress=progress)))
