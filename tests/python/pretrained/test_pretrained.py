"""The published-model loader, against a local registry: no network.

What matters to a user is that a model is either loaded exactly as published
or refused, never loaded damaged; that the second load needs no network; and
that one dead mirror does not stop a download.
"""
from __future__ import annotations

import hashlib
import json
import struct

import numpy as np
import pytest

import BEAM
from BEAM import BeamModel
from BEAM.pretrained import MODEL_FORMAT

SEQS = ["CASSLGQ", "CASSPGT", "CASRLAG", "CSARDRG", "CASSQET", "CAWSVGQ"] * 3


@pytest.fixture
def published(tmp_path, monkeypatch):
    """A tiny model 'published' as a file:// URL, with its registry entry."""
    src = tmp_path / "server"
    src.mkdir()
    model = BeamModel(SEQS, order=2, max_length=10)
    model.save(str(src / "tiny"))
    (f,) = list(src.iterdir())
    data = f.read_bytes()
    entry = {
        "release": "models-v0", "file": f.name, "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(), "format": MODEL_FORMAT,
        "description": "test model", "urls": [f.as_uri()],
    }
    registry = tmp_path / "registry.json"

    def write(**changes):
        registry.write_text(json.dumps(
            {"schema": 1, "models": {"tiny": {**entry, **changes}}}))

    write()
    monkeypatch.setenv("BEAM_PRETRAINED_REGISTRY", str(registry))
    monkeypatch.setenv("BEAM_CACHE_DIR", str(tmp_path / "cache"))
    return {"model": model, "file": f, "entry": entry, "write": write}


def test_model_format_matches_what_save_writes(published):
    magic, fmt = struct.unpack("<II", published["file"].read_bytes()[:8])
    assert magic == 0x4245414D
    assert fmt == MODEL_FORMAT


def test_shipped_registry_is_readable(monkeypatch):
    monkeypatch.delenv("BEAM_PRETRAINED_REGISTRY", raising=False)
    for m in BEAM.pretrained_models():
        assert m.format == MODEL_FORMAT, f"{m.name} cannot be read by this build"
        assert len(m.sha256) == 64 and m.bytes > 0 and m.urls


def test_listed(published):
    (m,) = BEAM.pretrained_models()
    assert m.name == "tiny" and m.bytes == published["entry"]["bytes"]


def test_loads_exactly_the_published_model(published):
    loaded = BEAM.load_pretrained("tiny", progress=False)
    probe = ["CASSLGQ", "CASRLAG", "CWWWWWW"]
    assert np.array_equal(loaded.log_pseq_batch(probe),
                          published["model"].log_pseq_batch(probe))


def test_second_load_needs_no_network(published):
    BEAM.load_pretrained("tiny", progress=False)
    published["file"].unlink()                      # the "server" is gone
    BEAM.load_pretrained("tiny", progress=False)


def test_damaged_cache_is_fetched_again(published):
    path = BEAM.pretrained_path("tiny", progress=False)
    path.write_bytes(path.read_bytes()[:-10])       # truncated on disk
    BEAM.load_pretrained("tiny", progress=False)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == published["entry"]["sha256"]


def test_wrong_bytes_are_refused_and_not_kept(published):
    published["write"](sha256="0" * 64)
    with pytest.raises(OSError, match="SHA-256"):
        BEAM.load_pretrained("tiny", progress=False)
    cached = list((published["file"].parent.parent / "cache").rglob("*"))
    assert not [p for p in cached if p.is_file()]


def test_falls_back_to_the_next_mirror(published, tmp_path):
    dead = (tmp_path / "missing.beam").as_uri()
    published["write"](urls=[dead, published["file"].as_uri()])
    BEAM.load_pretrained("tiny", progress=False)


def test_unknown_name_lists_what_exists(published):
    with pytest.raises(KeyError, match="available: tiny"):
        BEAM.load_pretrained("nonexistent")


def test_newer_format_asks_for_an_upgrade(published):
    published["write"](format=MODEL_FORMAT + 1)
    with pytest.raises(ValueError, match="upgrade beam-model"):
        BEAM.load_pretrained("tiny", progress=False)
