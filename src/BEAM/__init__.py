"""BEAM: Bidirectional Expert-Anchored Model over CDR3 amino-acid sequences.

Every residue is scored twice -- once by an N-anchored expert indexed by its
distance from the front of the sequence, and once by a C-anchored expert
indexed by its distance from the back. The two readings are combined
log-linearly and normalised per length by an exact forward dynamic program
over a layered DAG. Nothing is sampled and nothing is pruned.
"""
from .errors import (
    BeamError,
    BeamTypeError,
    BeamValueError,
    BudgetExceeded,
    InternalError,
    InvalidSequence,
    ModelFileError,
    ModelMismatch,
    NoLivePaths,
    NotFinalized,
    OutOfMemory,
)
from ._model import (
    BeamModel,
    Atoms,
    Builder,
    Calibration,
    Histogram,
    PseqAnalysis,
    effective_threads,
    file_provenance_span,
    memory_budget,
    threads,
    set_memory_budget,
    set_threads,
)
from ._publicness import PublicnessModel
from .analytics.scale import Attribution, ScaleIndex, attribute
from .mixture import (BetaDiversity, EntropyBounds, EntropyEstimate,
                      IncompatibleMembers, Mixture, MixtureError,
                      MixtureTooLarge, SupportSize, UnsupportedOrder)
from .diversity import (DepthMatchedProfiles, ResamplingBand,
                        depth_matched_hill_profiles)
from .likelihood import (AbundanceSpectrum, PairLikelihood, Piece,
                         RepertoireLikelihood, abundance_spectrum,
                         auto_profile_edges, pair_likelihood,
                         repertoire_likelihood)
from .model.fit import AlphaFit, fit_alpha
from .query import BeamQuery, Condition
from .pretrained import (PretrainedModel, load_pretrained, pretrained_models,
                         pretrained_path)
from .query.errors import (ImpossibleCondition, QueryError, QueryTooLarge,
                           UnsupportedQuery)

__all__ = [
    # Exceptions. Exported here because a caller cannot catch what it cannot
    # import, and until now the query classes lived three packages deep.
    "BeamError", "BeamTypeError", "BeamValueError", "BudgetExceeded",
    "InternalError", "InvalidSequence", "ModelFileError", "ModelMismatch", "NoLivePaths",
    "NotFinalized", "OutOfMemory",
    "QueryError", "ImpossibleCondition", "UnsupportedQuery", "QueryTooLarge",
    "MixtureError", "IncompatibleMembers", "UnsupportedOrder",
    "MixtureTooLarge",

    "BeamModel", "Atoms", "Builder", "Calibration", "Histogram",
    "PseqAnalysis", "effective_threads", "file_provenance_span",
    "memory_budget",
    "threads", "set_memory_budget", "set_threads",
    "PublicnessModel",
    "Attribution", "ScaleIndex", "attribute",
    "BeamQuery", "Condition",
    "Mixture", "BetaDiversity", "EntropyBounds", "EntropyEstimate",
    "SupportSize",
    "depth_matched_hill_profiles", "DepthMatchedProfiles", "ResamplingBand",
    "repertoire_likelihood", "pair_likelihood", "RepertoireLikelihood",
    "abundance_spectrum", "auto_profile_edges", "AbundanceSpectrum",
    "PairLikelihood", "Piece",
    "fit_alpha", "AlphaFit",
    "load_pretrained", "pretrained_models", "pretrained_path",
    "PretrainedModel",
]
__version__ = "1.0.0.dev0"
