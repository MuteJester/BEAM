# The BEAM foundation is byte-identical to the EDGE foundation

Recorded 2026-08-27, before Phase 4 of the reorganization.

## What was compared

| | EDGE-era | BEAM |
|---|---|---|
| host | ig03 | ig05 |
| file | `/home/bcrlab/thomas/edge_data/edge_full_strict.edge` | `~/beam_work/models/beam_full_strict.beam` |
| built by | LZGraphs, 2026-08-26 | BEAM commit 9117dd3, 2026-08-27 |
| compiler | gcc on ig03 | gcc 11.5.0 on ig05 |
| size | 701,713,116 | 701,713,116 |

Corpus verified identical first: the abundance parquet is
`f2b94d0a818c5b16225fecfbaf3ff1ef97061f96d06e2eb600ebbb29c3e70cb0` on the
local machine and on ig05.

Configuration was not chosen for the rebuild, it was read off the EDGE model:
order 4, depth_cap 15, max_length 27, strict, sparse, coefficients
(0.6, 0.25, 0.15), weights (0.5, 0.5), all five backoff strengths zero.

## Result

Bytes 5 through size-4, which is everything except the four magic bytes and
the trailing CRC that covers them:

    a9831c12a6e56792a59a39a64a823f31e5595c0a3141652b026ac13e1f261003   EDGE
    a9831c12a6e56792a59a39a64a823f31e5595c0a3141652b026ac13e1f261003   BEAM

All 701,713,108 payload bytes are identical. The two differences are both
accounted for: the magic was deliberately changed from EDGE's 0x45444745 to
BEAM's 0x4245414D, and the CRC is computed over the header, so it moves with
the magic.

Derived quantities, all matching:

    n_sequences      5,629,439,400
    n_observed         926,909,993
    lengths                 3 .. 27
    n_states                194,481
    nonzero cells        63,136,647
    D0            4.511483539322333e+24

Build time on ig05: 6.3 minutes, 1.36 GiB resident.

## Why this matters for the reorganization

Phases 4 through 9 cut live numerical code into new translation units, where a
changed inlining decision can change a float's last bit without failing any
assertion. This file is the reference those phases are measured against, at
foundation scale rather than on a 2,000-sequence fixture.

Rebuilding the foundation after those phases and comparing the payload hash
again is the check. It is stronger than comparing figures, because a figure
can absorb a last-bit change without visibly moving.

Also worth recording: ig05 (gcc 11.5.0, Python 3.11) reproduces the full
`.private/baseline/check.py` bit-identically, including the saved-model
sha256. Cross-compiler reproducibility is not assumed here, it is measured.
