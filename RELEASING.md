# Releasing BEAM

Two things are released, on separate schedules:

- **the package** (`beam-model` on PyPI), from a `vX.Y.Z` tag, by
  `.github/workflows/release.yml`;
- **the published models**, from a `models-vN` release made by
  `tools/release/publish_models.py`, archived on Zenodo with their own DOI.

Models change rarely and the code often, so a code release does not re-upload
the models; the package's `src/BEAM/pretrained/registry.json` says which model
files it uses.

## Versions

`BEAM.__version__` in `src/BEAM/__init__.py` is the only place the package
version is written. Between releases it carries a `.devN` suffix
(`1.1.0.dev0`), so a build from `main` never claims to be a release.

Versions follow [Semantic Versioning](https://semver.org/):

| Change | Version |
|---|---|
| Bug fix, faster code, same results and API | patch, `1.0.1` |
| New functions or options; old code still works | minor, `1.1.0` |
| A public function removed or changed incompatibly | major, `2.0.0` |
| A new model **file format** that this release cannot read back from older files | major |

The model file format is part of the API. A release that writes a new format
must still read the previous one, or it is a major release, and the published
models must be re-issued as a new `models-vN` in the new format.

## Releasing the package

1. **Prepare on `main`** (by pull request, so CI runs):
   - `src/BEAM/__init__.py`: `__version__ = "X.Y.Z"` (drop `.devN`).
   - `CITATION.cff`: `version: X.Y.Z` and `date-released: YYYY-MM-DD`.
   - `CHANGELOG.md`: rename the section to `## [X.Y.Z] - YYYY-MM-DD`.
   - Run the README examples against the real models and check the numbers
     quoted in the README still hold:
     `python tools/release/check_readme.py README.md --pretrained`
   - Run the verification gate: `python verification/check.py`.
2. **Tag the merged commit and push the tag:**
   ```bash
   git tag -a vX.Y.Z -m "BEAM X.Y.Z"
   git push origin vX.Y.Z
   ```
3. **Approve the PyPI deployment.** The workflow tests everything, builds
   15 wheels and the source distribution, smoke-tests each and creates a
   draft GitHub Release; then it waits on the protected `pypi` environment.
   Open the run in the Actions tab and approve. A PyPI version can never be
   uploaded twice, so this is the point of no return.
4. The release becomes public, and Zenodo archives it and mints its DOI.
   A last job then installs the published version from PyPI on clean Linux
   and macOS machines and runs the smoke test and the README examples. If
   it fails, the release that users get is broken: fix forward with a
   patch release.
5. **Open the next cycle on `main`:** `__version__ = "X.Y+1.0.dev0"` and a
   new `## [Unreleased]` section in `CHANGELOG.md`.

Release candidates work the same way with a tag like `v1.1.0rc1` (and
`__version__ = "1.1.0rc1"`); they are published as pre-releases and pip
installs them only when asked for explicitly.

### If something fails

- **Before the PyPI step** (guard, tests, wheels, draft): fix the problem on
  `main`, delete the tag locally and on GitHub
  (`git tag -d vX.Y.Z && git push origin :refs/tags/vX.Y.Z`), delete the
  draft release if one was created, and tag again.
- **A transient failure** (a runner or network error): use "Re-run failed
  jobs". Every step is safe to repeat; the draft step refuses to modify a
  release that is already public.
- **After PyPI accepted the upload:** that version is final. Fix forward with
  a patch release.

## Releasing models

Models are built where the data lives, then published with
`tools/release/publish_models.py`, driven by a spec file in
`models/` (for example `models/models-v1.json`), which names each model's
file and describes how it was built.

1. **Prepare:** `python tools/release/publish_models.py prepare models/models-vN.json`
   loads each model with the current BEAM, refuses one that does not load or
   is not the current file format, and writes `dist/models-vN/`: the model
   files, a `MODEL_CARD.md`, `MODELS.json` and `SHA256SUMS.txt`.
2. **GitHub:** `python tools/release/publish_models.py github models/models-vN.json`
   creates the `models-vN` release with those files. It is never marked
   "latest", so the package release stays the one GitHub shows first.
3. **Zenodo:** create a new upload at <https://zenodo.org/uploads/new>
   (type *Dataset*), add every file from `dist/models-vN/`, copy the title,
   authors and description from `MODEL_CARD.md`, link it to the paper and
   the repository, and publish. Note the record number and DOI.
4. **Register:** `python tools/release/publish_models.py register models/models-vN.json --zenodo-record NNNN --doi 10.5281/zenodo.NNNN`
   writes the models into `src/BEAM/pretrained/registry.json`, Zenodo first
   and GitHub as the mirror. Commit that by pull request; the next package
   release ships it.

## One-time setup (already done for this repository)

- PyPI: a Trusted Publisher for project `beam-model`, repository
  `MuteJester/BEAM`, workflow `release.yml`, environment `pypi`.
- GitHub: the `pypi` environment, limited to `v*` tags, with a required
  reviewer.
- Zenodo: the GitHub integration switched on for `MuteJester/BEAM`.
