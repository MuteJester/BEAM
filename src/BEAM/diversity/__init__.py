"""Diversity of repertoires compared at one common sequencing depth.

    >>> from BEAM import depth_matched_hill_profiles
    >>> r = depth_matched_hill_profiles([seqs_a, seqs_b], [templates_a, templates_b],
    ...                                 seed=1)
    >>> r.profiles.shape                 # (2 repertoires, 20 resamples, 31 orders)
    >>> r.resampling_band(5, 95)         # NOT a confidence interval

A Hill profile depends on how deeply a repertoire was sequenced: a donor
sampled at 360,000 templates shows more distinct sequences, and a different
profile, than the same donor at 330,000. Comparing two people's profiles
means first taking them to the same depth. This package does that by
resampling each repertoire to the smallest template total in the set, fitting
a fresh model to every resample, and profiling each with
:meth:`BeamModel.hill_profile`.

## What the numbers are and are not

The depth belongs to the CALL. It is the smallest template total among the
repertoires passed together, so adding or removing one can change every other
repertoire's profiles. Compare results only from one call, or from calls whose
``depth`` agrees.

The spread over resamples is a RESAMPLING band, returned by
:meth:`DepthMatchedProfiles.resampling_band`. Every replicate reweights the
same sequences, so the band measures what reweighting what you have does, not
what a new sample would give: at ``q = 1`` it was about 3% wide where held-out
disjoint blocks of one repertoire spread over about 50%.

See https://mutejester.github.io/BEAM/reference/diversity/.
"""
from .depth_matched import DEFAULT_ORDERS, depth_matched_hill_profiles
from .results import DepthMatchedProfiles, ResamplingBand

__all__ = ["depth_matched_hill_profiles", "DepthMatchedProfiles",
           "ResamplingBand", "DEFAULT_ORDERS"]
