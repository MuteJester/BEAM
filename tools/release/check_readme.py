"""Run every ```python block in the README, in order, as one script.

    python tools/release/check_readme.py README.md               # CI
    python tools/release/check_readme.py README.md --pretrained  # before a release

By default the published "foundation" model is replaced by a small stand-in
built here, and the example input files the README names are generated, so the
check needs no network and runs in seconds. It proves the examples run against
the current API; it does not check the numbers in their comments.

With --pretrained the real published model is downloaded and used, and the
value of every top-level expression is printed beside its line, so the numbers
quoted in the README can be compared with what BEAM now returns.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import sys
import tempfile
from pathlib import Path

AA = "ACDEFGHIKLMNPQRSTVWY"
STARTS = ["CASS", "CASR", "CSAR", "CASG", "CAWS", "CSVE"]
ENDS = ["YEQYF", "TEQYF", "GELFF", "NTEAFF", "TDTQYF"]


def blocks(readme: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", readme.read_text(), re.S)


def cdr3s(rng: random.Random, n: int) -> list[str]:
    return [rng.choice(STARTS) + "".join(rng.choices(AA, k=rng.randint(2, 6)))
            + rng.choice(ENDS) for _ in range(n)]


def example_files(workdir: Path, rng: random.Random) -> None:
    """The input files the README's examples read."""
    for name in ("donor_A.tsv", "donor_B.tsv"):
        rows = [f"{s}\t{rng.randint(1, 20)}" for s in cdr3s(rng, 1500)]
        (workdir / name).write_text("junction_aa\tduplicate_count\n" + "\n".join(rows) + "\n")
    (workdir / "cdr3.tsv").write_text(
        "\n".join(f"{s}\t{rng.randint(1, 5)}" for s in cdr3s(rng, 500)) + "\n")


def stand_in_foundation(workdir: Path, rng: random.Random) -> None:
    """Register a small model under the name "foundation".

    Order 2 keeps the exact occupancy integrals to seconds; the examples only
    need a model with the published one's interface.
    """
    from BEAM import BeamModel
    from BEAM.pretrained import MODEL_FORMAT

    server = workdir / "server"
    server.mkdir()
    BeamModel(cdr3s(rng, 3000), order=2, max_length=20).save(str(server / "foundation"))
    (f,) = list(server.iterdir())
    data = f.read_bytes()
    registry = workdir / "registry.json"
    registry.write_text(json.dumps({"schema": 1, "models": {"foundation": {
        "release": "stand-in", "file": f.name, "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(), "format": MODEL_FORMAT,
        "description": "README stand-in", "urls": [f.as_uri()]}}}))
    os.environ["BEAM_PRETRAINED_REGISTRY"] = str(registry)
    os.environ["BEAM_CACHE_DIR"] = str(workdir / "cache")


def run(code: list[str], namespace: dict, show: bool) -> None:
    for i, block in enumerate(code, 1):
        tree = ast.parse(block)
        for node in tree.body:
            src = ast.get_source_segment(block, node)
            if show and isinstance(node, ast.Expr):
                value = eval(compile(ast.Expression(node.value), f"<block {i}>", "eval"),
                             namespace)
                print(f"{src}\n    -> {value!r}"[:400])
            else:
                exec(compile(ast.Module([node], []), f"<block {i}>", "exec"), namespace)
        print(f"block {i}: ok", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("readme", type=Path)
    ap.add_argument("--pretrained", action="store_true",
                    help="use the real published models and print every value")
    args = ap.parse_args()
    readme = args.readme.resolve()
    code = blocks(readme)
    if not code:
        sys.exit(f"{readme}: no python blocks found")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        os.chdir(work)
        namespace = {"__name__": "__readme__"}
        rng = random.Random(7)
        example_files(work, rng)
        if not args.pretrained:
            stand_in_foundation(work, rng)
        namespace["sequences"] = cdr3s(rng, 400)
        namespace["counts"] = [rng.randint(1, 9) for _ in namespace["sequences"]]
        run(code, namespace, show=args.pretrained)
    print(f"{readme.name}: all {len(code)} python blocks ran", file=sys.stderr)


if __name__ == "__main__":
    main()
