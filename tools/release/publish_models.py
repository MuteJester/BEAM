"""Publish BEAM models as a `models-vN` release, then register them.

    python tools/release/publish_models.py prepare  models/models-v1.json --file human_tcrb_cdr3_aa_foundation=/path/human_tcrb_cdr3_aa_foundation.beam
    python tools/release/publish_models.py github   models/models-v1.json
    python tools/release/publish_models.py register models/models-v1.json --zenodo-record 1234567 --doi 10.5281/zenodo.1234567

The spec (models/models-vN.json) describes the release and each model in
words; everything that can be read from a model file is read from it, never
typed: size, SHA-256, file format, configuration, when it was built and how
much data went in. Which code built it is given in the spec ("built_with"),
since a model file does not record it.

prepare   loads every model with THIS build of BEAM and refuses one that does
          not load or is not the current file format, then writes
          dist/<release>/: the model files, MODELS.json, SHA256SUMS.txt and
          MODEL_CARD.md.
github    creates the GitHub release <release> with those files. Refuses if
          it exists. Never marked "latest".
register  adds the models to src/BEAM/pretrained/registry.json, with the
          Zenodo record as the primary URL and the GitHub release as mirror.

See RELEASING.md for where the Zenodo upload fits between github and register.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = "MuteJester/BEAM"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_spec(path: Path) -> dict:
    spec = json.loads(path.read_text())
    for key in ("release", "title", "license", "citation", "models"):
        if key not in spec:
            sys.exit(f"{path}: missing '{key}'")
    for name, m in spec["models"].items():
        for key in ("file", "description", "built_from", "built_with", "intended_use",
                    "limitations"):
            if not m.get(key):
                sys.exit(f"{path}: model '{name}' needs '{key}'")
    return spec


def prepare(spec: dict, sources: dict[str, Path], dist: Path) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    import BEAM
    from BEAM import BeamModel
    from BEAM.pretrained import MODEL_FORMAT

    missing = set(spec["models"]) - set(sources)
    if missing:
        sys.exit(f"give --file NAME=PATH for: {', '.join(sorted(missing))}")
    if dist.exists():
        sys.exit(f"{dist} exists; remove it to prepare again")
    dist.mkdir(parents=True)

    entries = {}
    for name, m in spec["models"].items():
        src = sources[name]
        with open(src, "rb") as f:
            magic, fmt = struct.unpack("<II", f.read(8))
        if magic != 0x4245414D:
            sys.exit(f"{src}: not a BEAM model file")
        if fmt != MODEL_FORMAT:
            sys.exit(f"{src}: file format {fmt}, but this BEAM reads format "
                     f"{MODEL_FORMAT}; rebuild the model with this version")
        print(f"loading {name} from {src} ...", flush=True)
        model = BeamModel.from_file(str(src))
        meta = model.metadata(lengths=False)
        cfg = model.config
        dest = dist / m["file"]
        shutil.copyfile(src, dest)
        digest = sha256(dest)
        if digest != sha256(src):
            sys.exit(f"{dest}: copy differs from {src}")
        entries[name] = {
            "release": spec["release"], "file": m["file"],
            "bytes": dest.stat().st_size, "sha256": digest, "format": fmt,
            "description": m["description"],
            # metadata()'s beam_version is the version READING the file, so
            # the build's provenance comes from the spec; the file itself
            # stores when it was built.
            "built_with": m["built_with"],
            "built_at": meta.get("created"),
            "config": {k: cfg[k] for k in ("alphabet", "order", "max_length")},
            # n_observed counts the sequence records observed (distinct
            # sequences when the corpus was merged first, as the foundation
            # was); n_sequences is the abundance-weighted total.
            "n_observed": int(model.n_observed),
            "total_weight": float(model.total_weight),
            "license": spec["license"], "citation": spec["citation"],
        }
        print(f"  {name}: {entries[name]['bytes']:,} bytes, sha256 {digest}")

    (dist / "MODELS.json").write_text(json.dumps(
        {"release": spec["release"], "title": spec["title"],
         "created": dt.date.today().isoformat(),
         "beam_version_checked": BEAM.__version__, "models": entries},
        indent=2) + "\n")
    (dist / "SHA256SUMS.txt").write_text("".join(
        f"{e['sha256']}  {e['file']}\n" for e in entries.values()))
    (dist / "MODEL_CARD.md").write_text(model_card(spec, entries))
    print(f"prepared {dist}")


def model_card(spec: dict, entries: dict) -> str:
    out = [f"# {spec['title']}\n",
           f"Release `{spec['release']}` of the published BEAM models. "
           f"Load one with:\n",
           "```python\nimport BEAM\nmodel = BEAM.load_pretrained(\"NAME\")\n```\n",
           f"**License:** {spec['license']}  \n**Cite:** {spec['citation']}\n"]
    for name, e in entries.items():
        m = spec["models"][name]
        c = e["config"]
        out += [f"\n## `{name}`\n", f"{m['description']}\n",
                f"- **Built from:** {m['built_from']}",
                f"- **Intended use:** {m['intended_use']}",
                f"- **Limitations:** {m['limitations']}",
                f"- **Model:** alphabet `{c['alphabet']}`, order {c['order']}, "
                f"maximum length {c['max_length']}; "
                f"{e['n_observed']:,} sequence records observed, total "
                f"abundance weight {e['total_weight']:,.0f}",
                f"- **File:** `{e['file']}`, {e['bytes']:,} bytes, format "
                f"{e['format']}, built {e['built_at']} with {e['built_with']}",
                f"- **SHA-256:** `{e['sha256']}`\n"]
    return "\n".join(out)


def github(spec: dict, dist: Path, dry_run: bool) -> None:
    release = spec["release"]
    files = [str(dist / "MODELS.json"), str(dist / "SHA256SUMS.txt"),
             str(dist / "MODEL_CARD.md")]
    files += [str(dist / m["file"]) for m in spec["models"].values()]
    if subprocess.run(["gh", "release", "view", release, "--repo", REPO],
                      capture_output=True).returncode == 0:
        sys.exit(f"release {release} already exists; models are never replaced, "
                 f"publish a new models-vN instead")
    cmd = ["gh", "release", "create", release, "--repo", REPO,
           "--title", spec["title"], "--notes-file", str(dist / "MODEL_CARD.md"),
           "--latest=false", "--target", "main", *files]
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def register(spec: dict, dist: Path, registry: Path, record: str, doi: str) -> None:
    built = json.loads((dist / "MODELS.json").read_text())["models"]
    data = json.loads(registry.read_text())
    for name, e in built.items():
        urls = [f"https://zenodo.org/records/{record}/files/{e['file']}?download=1",
                f"https://github.com/{REPO}/releases/download/{spec['release']}/{e['file']}"]
        data["models"][name] = {
            "release": e["release"], "file": e["file"], "bytes": e["bytes"],
            "sha256": e["sha256"], "format": e["format"],
            "description": e["description"], "urls": urls, "doi": doi,
            "citation": e["citation"], "license": e["license"],
            "built_with": e["built_with"]}
    registry.write_text(json.dumps(data, indent=2) + "\n")
    print(f"registered {', '.join(built)} in {registry}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "github", "register"])
    ap.add_argument("spec", type=Path)
    ap.add_argument("--file", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--dist", type=Path, help="default: dist/<release>")
    ap.add_argument("--registry", type=Path,
                    default=ROOT / "src/BEAM/pretrained/registry.json")
    ap.add_argument("--zenodo-record")
    ap.add_argument("--doi")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    spec = load_spec(a.spec)
    dist = a.dist or ROOT / "dist" / spec["release"]
    if a.command == "prepare":
        sources = {}
        for item in a.file:
            name, _, path = item.partition("=")
            sources[name] = Path(path).expanduser().resolve()
        prepare(spec, sources, dist)
    elif a.command == "github":
        github(spec, dist, a.dry_run)
    else:
        if not (a.zenodo_record and a.doi):
            sys.exit("register needs --zenodo-record and --doi")
        register(spec, dist, a.registry, a.zenodo_record, a.doi)


if __name__ == "__main__":
    main()
