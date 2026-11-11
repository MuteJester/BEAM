"""Shared engineering: what every suite draws on, in one place.

The corpora and the constants in :mod:`_engineering.corpora`, the built
fixtures and the paths in :mod:`_engineering.fixtures`. Imported by name --
``from _engineering.corpora import make_corpus`` -- and not by a bare module
name that happens to resolve.

This layer holds nothing that computes an expected value. The independent
references live in ``tests/oracles/`` and must share no computation with the
implementation or with each other, so nothing in that layer imports anything
from this one; a bug in a shared helper could otherwise make an
implementation and its oracle agree wrongly, which is the one failure the
oracles exist to catch.
"""
