"""The independent references, which owe nothing to the code they check.

Each module here recomputes a quantity from its definition -- by enumeration,
by direct summation, or by exact convolution -- so that a test comparing the
two is comparing a dynamic program against arithmetic. That is the only
comparison that can find an error in the dynamic program.

They may share corpora. They must never share computation, with the
implementation or with each other: a bug in a shared helper would make the
implementation and its oracle agree wrongly, and a passing suite would be the
evidence for it. Nothing here imports ``_engineering``.

Both modules build their models with ``BEAM.BeamModel`` and read per-sequence
probabilities from it, which is deliberate and is not shared computation in
the sense above: what they recompute is the aggregate -- the mixture's
entropy, a spectrum's expectation, a repertoire's shared mass -- and that
arithmetic is theirs alone.
"""
