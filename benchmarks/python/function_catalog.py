"""
Reference notes on BEAM's public C functions: header, signature, category and
a one-line purpose, plus the complexity the author of each entry EXPECTED.

Nothing here is measured, and nothing here decides what a benchmark covers or
how a result is interpreted: coverage is read from the benchmark sources
(bench_sources.py) and every number in a report comes from the harness. The
expected complexities are shown beside measured scaling exponents only as a
claim to check, and an entry may be stale -- the headers are authoritative.
"""

from typing import Any, Dict, List, Optional

FUNCTION_CATALOG: Dict[str, Dict[str, Any]] = {
    "beam_crc32c": {
        "header": "include/beam/crc32c.h",
        "signature": "uint32_t beam_crc32c(const void *buf, size_t len)",
        "category": "Utility & Hardware",
        "algorithmic_class": "Hardware SIMD / CRC32 instruction",
        "time_complexity": "O(N / 8)",
        "space_complexity": "O(1)",
        "purpose": "Calculates hardware-accelerated CRC-32C (Castagnoli) checksum via SSE4.2 / ARM CRC32."
    },
    "beam_crc32c_update": {
        "header": "include/beam/crc32c.h",
        "signature": "uint32_t beam_crc32c_update(uint32_t crc, const void *buf, size_t len)",
        "category": "Utility & Hardware",
        "algorithmic_class": "Incremental Digest Accumulator",
        "time_complexity": "O(N)",
        "space_complexity": "O(1)",
        "purpose": "Updates running CRC32C digest across chunked / streaming buffers."
    },
    "beam_crc32c_software": {
        "header": "include/beam/crc32c.h",
        "signature": "uint32_t beam_crc32c_software(uint32_t crc, const void *buf, size_t len)",
        "category": "Utility & Hardware",
        "algorithmic_class": "Sliced Table-Driven CRC Lookup",
        "time_complexity": "O(N)",
        "space_complexity": "O(1)",
        "purpose": "Portable 8-way sliced lookup-table software fallback for CRC32C when hardware instructions are absent."
    },
    "beam_error_message": {
        "header": "include/beam/errors.h",
        "signature": "const char *beam_error_message(void)",
        "category": "System & Diagnostics",
        "algorithmic_class": "Thread-Local String Read",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns detailed human-readable error description for the most recent failure on calling thread."
    },
    "beam_last_error": {
        "header": "include/beam/errors.h",
        "signature": "BeamError beam_last_error(void)",
        "category": "System & Diagnostics",
        "algorithmic_class": "Thread-Local Enum Read",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns numeric BeamError code for the most recent failure on calling thread."
    },
    "beam_set_error": {
        "header": "include/beam/errors.h",
        "signature": "void beam_set_error(BeamError err, const char *fmt, ...)",
        "category": "System & Diagnostics",
        "algorithmic_class": "Thread-Local Formatted Write",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Records thread-local error status code and formats descriptive message."
    },
    "beam_clear_error": {
        "header": "include/beam/errors.h",
        "signature": "void beam_clear_error(void)",
        "category": "System & Diagnostics",
        "algorithmic_class": "Thread-Local Reset",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Resets calling thread error status to BEAM_OK and clears message buffer."
    },
    "beam_logo_request_default": {
        "header": "include/beam/insights.h",
        "signature": "void beam_logo_request_default(BeamLogoRequest *req)",
        "category": "Sequence Insights",
        "algorithmic_class": "Struct Initialization",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Initializes BeamLogoRequest struct with safe default sampling and visualization options."
    },
    "beam_insights_logos": {
        "header": "include/beam/insights.h",
        "signature": "BeamError beam_insights_logos(const BeamModel *m, const BeamLogoRequest *req, BeamLogoSet *out)",
        "category": "Sequence Insights",
        "algorithmic_class": "Trellis Positional Marginal & Information Matrix",
        "time_complexity": "O(|Σ|^K · L + L · |Σ|)",
        "space_complexity": "O(L · |Σ|)",
        "purpose": "Evaluates exact position-by-position sequence logos, bit heights, and residue probabilities."
    },
    "beam_insights_logo_cost": {
        "header": "include/beam/insights.h",
        "signature": "BeamError beam_insights_logo_cost(const BeamModel *m, const BeamLogoRequest *req, BeamQueryCost *out)",
        "category": "Sequence Insights",
        "algorithmic_class": "Analytical Complexity Predictor",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Predicts FLOP count and memory footprint required for logo generation before running the sweep."
    },
    "beam_insights_logo_set_free": {
        "header": "include/beam/insights.h",
        "signature": "void beam_insights_logo_set_free(BeamLogoSet *set)",
        "category": "Sequence Insights",
        "algorithmic_class": "Heap Deallocation",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Frees heap memory allocated for BeamLogoSet heights and color mappings."
    },
    "beam_mixture_log_inner_product": {
        "header": "include/beam/mixture.h",
        "signature": "BeamError beam_mixture_log_inner_product(const BeamModel *const *models, uint32_t k, uint32_t len, double *out)",
        "category": "Mixture & Divergence",
        "algorithmic_class": "Coupled Trellis Sweep (Summed Log Weights)",
        "time_complexity": "O(|Σ|^K · L)",
        "space_complexity": "O(|Σ|^K)",
        "purpose": "Exact pairwise/multi-model inner product <P_1, ..., P_k> over sequences of length L."
    },
    "beam_mixture_log_inner_product_by_length": {
        "header": "include/beam/mixture.h",
        "signature": "BeamError beam_mixture_log_inner_product_by_length(const BeamModel *const *models, uint32_t k, double *out_by_length, double *out_total)",
        "category": "Mixture & Divergence",
        "algorithmic_class": "Concurrent Multi-Length Trellis Sweep",
        "time_complexity": "O(|Σ|^K · L_max)",
        "space_complexity": "O(|Σ|^K · T)",
        "purpose": "Evaluates exact inner products concurrently across all sequence lengths and their total sum."
    },
    "beam_mixture_log_power_product": {
        "header": "include/beam/mixture.h",
        "signature": "BeamError beam_mixture_log_power_product(const BeamModel *const *models, const double *expo, uint32_t k, uint32_t len, double *out)",
        "category": "Mixture & Divergence",
        "algorithmic_class": "Chernoff Alpha-Divergence Coupled Sweep",
        "time_complexity": "O(|Σ|^K · L)",
        "space_complexity": "O(|Σ|^K)",
        "purpose": "Computes ln sum_s prod P_f(s)^a_f (Bhattacharyya coefficient and Chernoff alpha-divergence)."
    },
    "beam_mixture_log_power_product_by_length": {
        "header": "include/beam/mixture.h",
        "signature": "BeamError beam_mixture_log_power_product_by_length(const BeamModel *const *models, const double *expo, uint32_t k, double *out_by_length, double *out_total)",
        "category": "Mixture & Divergence",
        "algorithmic_class": "Concurrent Power Product Multi-Length Sweep",
        "time_complexity": "O(|Σ|^K · L_max)",
        "space_complexity": "O(|Σ|^K · T)",
        "purpose": "Evaluates power product across all lengths, separating length distribution overlap from sequence affinity."
    },
    "beam_mixture_log_power_product_by_position": {
        "header": "include/beam/mixture.h",
        "signature": "BeamError beam_mixture_log_power_product_by_position(const BeamModel *const *models, const double *expo, uint32_t k, uint32_t len, double *out_by_position, double *out_sink, double *out_const, double *out_total)",
        "category": "Mixture & Divergence",
        "algorithmic_class": "Position-by-Position Ratio Decomposition",
        "time_complexity": "O(|Σ|^K · L)",
        "space_complexity": "O(|Σ|^K · L)",
        "purpose": "Decomposes power product divergence into exact position-specific log-ratio contributions."
    },
    "beam_mixture_gram": {
        "header": "include/beam/mixture.h",
        "signature": "BeamError beam_mixture_gram(const BeamModel *const *models, uint32_t k, double *out)",
        "category": "Mixture & Divergence",
        "algorithmic_class": "Pairwise Inner Product Gram Matrix (Symmetric)",
        "time_complexity": "O(k^2 · |Σ|^K · L_max)",
        "space_complexity": "O(k^2 + |Σ|^K · T)",
        "purpose": "Evaluates full k x k symmetric Gram matrix <P_i, P_j> for Hill order q=2 diversity."
    },
    "beam_mixture_power_gram": {
        "header": "include/beam/mixture.h",
        "signature": "BeamError beam_mixture_power_gram(const BeamModel *const *models, uint32_t k, double alpha, double *out)",
        "category": "Mixture & Divergence",
        "algorithmic_class": "Pairwise Bhattacharyya / Hellinger Matrix",
        "time_complexity": "O(k^2 · |Σ|^K · L_max)",
        "space_complexity": "O(k^2 + |Σ|^K · T)",
        "purpose": "Computes pairwise alpha-power Gram matrix for cohort geometric embeddings."
    },
    "beam_mixture_power_pairs": {
        "header": "include/beam/mixture.h",
        "signature": "BeamError beam_mixture_power_pairs(const BeamModel *const *models, uint32_t k, double alpha, const uint32_t *pairs, uint32_t n_pairs, double *out)",
        "category": "Mixture & Divergence",
        "algorithmic_class": "Sparse Pair Bhattacharyya Evaluator",
        "time_complexity": "O(n_pairs · |Σ|^K · L_max)",
        "space_complexity": "O(n_pairs + |Σ|^K · T)",
        "purpose": "Evaluates power products for arbitrary sparse subsets of cohort pairs without full Gram matrix cost."
    },
    "beam_mixture_support_union": {
        "header": "include/beam/mixture.h",
        "signature": "BeamError beam_mixture_support_union(const BeamModel *const *models, uint32_t n_models, uint32_t len, uint64_t cap, double *out, uint64_t *out_peak)",
        "category": "Mixture & Divergence",
        "algorithmic_class": "Bitmask Trellis Forward Sweep (q=0 Diversity)",
        "time_complexity": "O(|Live Masks| · |Σ|^K · L)",
        "space_complexity": "O(|Live Masks| · |Σ|^K)",
        "purpose": "Computes exact size of sequence support union across cohort members via surviving bitmasks."
    },
    "beam_mixture_support_union_by_length": {
        "header": "include/beam/mixture.h",
        "signature": "BeamError beam_mixture_support_union_by_length(const BeamModel *const *models, uint32_t n_models, uint64_t cap, double *out_by_length, double *out_total, uint64_t *out_peak)",
        "category": "Mixture & Divergence",
        "algorithmic_class": "Multi-Length Support Union Sweep",
        "time_complexity": "O(|Live Masks| · |Σ|^K · L_max)",
        "space_complexity": "O(|Live Masks| · |Σ|^K · T)",
        "purpose": "Computes cohort support union across every sequence length simultaneously."
    },
    "beam_mixture_check_shape": {
        "header": "include/beam/mixture.h",
        "signature": "BeamError beam_mixture_check_shape(const BeamModel *const *models, uint32_t n_models)",
        "category": "Mixture & Divergence",
        "algorithmic_class": "Trellis Geometry Verification",
        "time_complexity": "O(M)",
        "space_complexity": "O(1)",
        "purpose": "Verifies that all mixture members agree on alphabet, Markov order, and max length."
    },
    "beam_model_new": {
        "header": "include/beam/model.h",
        "signature": "BeamModel *beam_model_new(const BeamModelConfig *config)",
        "category": "Model Lifecycle",
        "algorithmic_class": "Trellis State Graph Allocation",
        "time_complexity": "O(|Σ|^K · L_max)",
        "space_complexity": "O(|Σ|^K · L_max)",
        "purpose": "Allocates model trellis, count hash tables, and Dirichlet prior structures."
    },
    "beam_model_destroy": {
        "header": "include/beam/model.h",
        "signature": "void beam_model_destroy(BeamModel *m)",
        "category": "Model Lifecycle",
        "algorithmic_class": "Heap Deallocation",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Releases all memory allocated for a BeamModel instance."
    },
    "beam_model_observe": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_observe(BeamModel *m, const char *const *seqs, const double *weights, uint32_t count)",
        "category": "Model Learning",
        "algorithmic_class": "Sliding-Window K-mer Count Accumulation",
        "time_complexity": "O(N · L)",
        "space_complexity": "O(1)",
        "purpose": "Accumulates sequence observations and abundances into sparse transition tables."
    },
    "beam_model_finalize": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_finalize(BeamModel *m)",
        "category": "Model Learning",
        "algorithmic_class": "Dirichlet Smoothing & Trellis Normalization",
        "time_complexity": "O(|Σ|^K · L_max)",
        "space_complexity": "O(|Σ|^K · L_max)",
        "purpose": "Turns accumulated counts into normalized log-probability tables and partition functions Z(L)."
    },
    "beam_model_is_finalized": {
        "header": "include/beam/model.h",
        "signature": "bool beam_model_is_finalized(const BeamModel *m)",
        "category": "Model Lifecycle",
        "algorithmic_class": "Boolean Flag Query",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Checks if model is finalized and ready for scoring or dynamic programming queries."
    },
    "beam_model_log_pseq": {
        "header": "include/beam/model.h",
        "signature": "double beam_model_log_pseq(const BeamModel *m, const char *seq)",
        "category": "Scoring & Evaluation",
        "algorithmic_class": "Sliding Context Markov Log-Likelihood",
        "time_complexity": "O(L)",
        "space_complexity": "O(1)",
        "purpose": "Evaluates exact log-probability ln P(s) of a single sequence under the model."
    },
    "beam_model_log_pseq_batch": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_log_pseq_batch(const BeamModel *m, const char *const *seqs, uint32_t count, double *out)",
        "category": "Scoring & Evaluation",
        "algorithmic_class": "Vectorized Batch Log-Likelihood",
        "time_complexity": "O(N · L)",
        "space_complexity": "O(1)",
        "purpose": "Evaluates exact log-probabilities for an array of sequences in parallel."
    },
    "beam_model_log_pseq_packed": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_log_pseq_packed(const BeamModel *m, const char *packed, uint64_t n_bytes, const int64_t *offsets, uint64_t count, double *out)",
        "category": "Scoring & Evaluation",
        "algorithmic_class": "Contiguous Memory Packed Scoring",
        "time_complexity": "O(N · L)",
        "space_complexity": "O(1)",
        "purpose": "Scores sequences from a contiguous byte buffer, eliminating pointer array indirection."
    },
    "beam_model_add": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_add(BeamModel *dst, const BeamModel *src)",
        "category": "Count Algebra",
        "algorithmic_class": "Sparse Cell Count Pooling",
        "time_complexity": "O(|Nonzero Cells|)",
        "space_complexity": "O(1)",
        "purpose": "Adds counts from src into dst (training on pooled corpus before prior smoothing)."
    },
    "beam_model_subtract": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_subtract(BeamModel *dst, const BeamModel *src)",
        "category": "Count Algebra",
        "algorithmic_class": "Leave-One-Out Count Subtraction",
        "time_complexity": "O(|Nonzero Cells|)",
        "space_complexity": "O(1)",
        "purpose": "Subtracts counts from dst (enables fast leave-one-out cross validation without retraining)."
    },
    "beam_model_posterior": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_posterior(BeamModel *dst, const BeamModel *prior, double weight)",
        "category": "Count Algebra",
        "algorithmic_class": "Bayesian Empirical Prior Blending",
        "time_complexity": "O(|Cells|)",
        "space_complexity": "O(1)",
        "purpose": "Blends fitted background model into dst as an informative empirical prior."
    },
    "beam_model_temper": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_temper(BeamModel *m, double temperature)",
        "category": "Model Transformation",
        "algorithmic_class": "Log-Probability Scaling & Partition Recalculation",
        "time_complexity": "O(max_length · S · R · max_length)",
        "space_complexity": "O(tables)",
        "purpose": "Raises a finalized model to a power t (P_t proportional to P^t), recomputes Z(L) for every length, and releases the counts as compact does."
    },
    "beam_model_compact": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_compact(BeamModel *m)",
        "category": "Memory Management",
        "algorithmic_class": "Count Table Release",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Releases raw training count tables, keeping only final log-probability tables."
    },
    "beam_model_drop_caches": {
        "header": "include/beam/model.h",
        "signature": "void beam_model_drop_caches(const BeamModel *m)",
        "category": "Memory Management",
        "algorithmic_class": "Trellis Cache Eviction",
        "time_complexity": "O(cached lengths)",
        "space_complexity": "O(1)",
        "purpose": "Frees the per-length live cache (the memoised trellis frontiers analytics build on first use); the next analytic rebuilds what it needs."
    },
    "beam_model_set_prior": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_set_prior(BeamModel *m, const double *alpha, uint32_t n_alpha, double alpha_length)",
        "category": "Model Learning",
        "algorithmic_class": "Dirichlet Prior Parameter Validation",
        "time_complexity": "O(finalize)",
        "space_complexity": "O(tables)",
        "purpose": "Sets the hierarchical prior's strengths (or asks for them to be estimated) and RE-FINALIZES the model from its counts; refused on a model with no observations."
    },
    "beam_model_log_partition": {
        "header": "include/beam/model.h",
        "signature": "double beam_model_log_partition(const BeamModel *m, uint32_t length)",
        "category": "Dynamic Programming",
        "algorithmic_class": "Table read",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns ln Z(L), the partition function finalize already computed for length L: an O(1) read, no sweep (src/native/model/introspection.c)."
    },
    "beam_model_log_partition_by_layer": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_log_partition_by_layer(const BeamModel *m, uint32_t length, double *out)",
        "category": "Dynamic Programming",
        "algorithmic_class": "Frontier Mass Evolution DP",
        "time_complexity": "O(|Σ|^K · L)",
        "space_complexity": "O(L)",
        "purpose": "Computes exact per-layer cumulative log-partition values along sequence generation path."
    },
    "beam_model_log_length_prior": {
        "header": "include/beam/model.h",
        "signature": "double beam_model_log_length_prior(const BeamModel *m, uint32_t length)",
        "category": "Model Introspection",
        "algorithmic_class": "Length Prior Table Lookup",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns log prior probability ln P(length) of a sequence having given length."
    },
    "beam_model_transition_rows": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_transition_rows(const BeamModel *m, uint32_t length, uint32_t position, bool log_space, double *out)",
        "category": "Model Introspection",
        "algorithmic_class": "Position Conditional Transition Tensor Extraction",
        "time_complexity": "O(|Σ|^min(p, K) · |Σ|)",
        "space_complexity": "O(|Σ|^min(p, K) · |Σ|)",
        "purpose": "Extracts conditional transition probability matrix P(x_{p+1} | x_{p-K..p}) at position p."
    },
    "beam_model_path_count": {
        "header": "include/beam/model.h",
        "signature": "double beam_model_path_count(const BeamModel *m, uint32_t length)",
        "category": "Dynamic Programming",
        "algorithmic_class": "Support Path Counting DP",
        "time_complexity": "O(|Σ|^K · L)",
        "space_complexity": "O(|Σ|^K)",
        "purpose": "Computes exact number of valid trellis paths of length L."
    },
    "beam_model_path_count_total": {
        "header": "include/beam/model.h",
        "signature": "double beam_model_path_count_total(const BeamModel *m)",
        "category": "Dynamic Programming",
        "algorithmic_class": "Total Trellis Support Summation",
        "time_complexity": "O(|Σ|^K · L_max)",
        "space_complexity": "O(|Σ|^K)",
        "purpose": "Computes total support cardinality across all sequence lengths up to max_length."
    },
    "beam_model_accumulator_max": {
        "header": "include/beam/model.h",
        "signature": "double beam_model_accumulator_max(const BeamModel *m)",
        "category": "Dynamic Programming",
        "algorithmic_class": "Critical Path DAG Max Search",
        "time_complexity": "O(|Σ|^K · L_max)",
        "space_complexity": "O(|Σ|^K)",
        "purpose": "Evaluates maximum un-rescaled path weight across DAG to establish numerical scaling limits."
    },
    "beam_model_entropy": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_entropy(const BeamModel *m, double *out)",
        "category": "Information Theory",
        "algorithmic_class": "Saddlepoint Jet Sweep (q=1 Diversity)",
        "time_complexity": "O(|Σ|^K · L_max)",
        "space_complexity": "O(|Σ|^K)",
        "purpose": "Exact Shannon entropy H(P) = -sum P(s) ln P(s) over entire sequence space."
    },
    "beam_model_hill_number": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_hill_number(const BeamModel *m, double q, double *out)",
        "category": "Information Theory",
        "algorithmic_class": "Continuous Hill Profile Sweep",
        "time_complexity": "O(|Σ|^K · L_max)",
        "space_complexity": "O(|Σ|^K)",
        "purpose": "Exact Hill diversity number D_q(P) for arbitrary real diversity order q."
    },
    "beam_model_log_mellin": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_log_mellin(const BeamModel *m, double q, double *out)",
        "category": "Information Theory",
        "algorithmic_class": "Tilted Trellis Partition DP",
        "time_complexity": "O(|Σ|^K · L_max)",
        "space_complexity": "O(|Σ|^K)",
        "purpose": "Evaluates ln sum_s P(s)^q at arbitrary real power q."
    },
    "beam_model_log_mellin_many": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_log_mellin_many(const BeamModel *m, const double *q, uint32_t n, double *out)",
        "category": "Information Theory",
        "algorithmic_class": "AVX2 SIMD Vectorized Mellin Sweep",
        "time_complexity": "O(ceil(n / 4) · |Σ|^K · L_max)",
        "space_complexity": "O(n · |Σ|^K)",
        "purpose": "Evaluates Mellin transform across multiple orders concurrently in SIMD lanes."
    },
    "beam_model_log_mellin_complex": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_log_mellin_complex(const BeamModel *m, const double *q_re, const double *q_im, uint32_t n, double *out_re, double *out_im)",
        "category": "Information Theory",
        "algorithmic_class": "Complex Exponential Trellis Sweep",
        "time_complexity": "O(n · |Σ|^K · L_max)",
        "space_complexity": "O(n · |Σ|^K)",
        "purpose": "Evaluates Mellin transform in the complex plane for Mellin-Barnes inversion."
    },
    "beam_model_log_mellin_line": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_log_mellin_line(const BeamModel *m, double sigma, double t0, double dt, uint32_t n, double *out_re, double *out_im)",
        "category": "Information Theory",
        "algorithmic_class": "AVX2 Vertical Contour Line Sweep",
        "time_complexity": "O(ceil(n / 4) · |Σ|^K · L_max)",
        "space_complexity": "O(n · |Σ|^K)",
        "purpose": "Mellin-Barnes contour integration along vertical line sigma + i*t."
    },
    "beam_model_surprisal_moments": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_surprisal_moments(const BeamModel *m, double q, uint32_t order, double *out)",
        "category": "Information Theory",
        "algorithmic_class": "Trigonomentric / Taylor Jet Expansion",
        "time_complexity": "O(|Σ|^K · L_max)",
        "space_complexity": "O(order · |Σ|^K)",
        "purpose": "Exact raw and central moments of surprisal -ln P(s) up to order 4."
    },
    "beam_model_surprisal_tilt": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_surprisal_tilt(const BeamModel *m, double q, double *out_log_mellin, double *out_mean, double *out_variance)",
        "category": "Information Theory",
        "algorithmic_class": "Saddlepoint Jet Sweep Bundle",
        "time_complexity": "O(|Σ|^K · L_max)",
        "space_complexity": "O(|Σ|^K)",
        "purpose": "Simultaneously returns tilted log-partition, mean, and variance for saddlepoint expansion."
    },
    "beam_model_surprisal_cumulants": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_surprisal_cumulants(const BeamModel *m, double q, uint32_t order, double *out)",
        "category": "Information Theory",
        "algorithmic_class": "Log-Mellin Logarithmic Derivative Jet",
        "time_complexity": "O(|Σ|^K · L_max)",
        "space_complexity": "O(order · |Σ|^K)",
        "purpose": "Computes exact cumulants of surprisal (K_1, K_2, K_3, K_4) without finite differencing."
    },
    "beam_model_surprisal_range": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_surprisal_range(const BeamModel *m, double *out_min, double *out_max)",
        "category": "Information Theory",
        "algorithmic_class": "DAG Shortest & Longest Path Search",
        "time_complexity": "O(|Σ|^K · L_max)",
        "space_complexity": "O(|Σ|^K)",
        "purpose": "Finds global minimum and maximum possible surprisal [s_min, s_max] across all admissible sequences."
    },
    "beam_model_surprisal_by_depth": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_surprisal_by_depth(const BeamModel *m, double q, double *out_by_depth, double *out_length_prior)",
        "category": "Information Theory",
        "algorithmic_class": "Layerwise Surprisal Decomposition",
        "time_complexity": "O(|Σ|^K · L_max)",
        "space_complexity": "O(L_max)",
        "purpose": "Decomposes surprisal into positional layer contributions across sequence length."
    },
    "beam_model_spectrum": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_spectrum(const BeamModel *m, bool counting, uint32_t length, uint32_t n_bins, double smooth, double *out_weights, double *out_bin_width)",
        "category": "Density & Spectra",
        "algorithmic_class": "FFT Surprisal Density Transport",
        "time_complexity": "O(n_bins · log(n_bins) + |Σ|^K · L)",
        "space_complexity": "O(n_bins + |Σ|^K)",
        "purpose": "Computes continuous surprisal probability density spectrum via characteristic function FFT inversion."
    },
    "beam_model_spectrum_pair": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_spectrum_pair(const BeamModel *m, uint32_t length, uint32_t n_bins, double smooth, double *out_counting, double *out_generated, double *out_bin_width)",
        "category": "Density & Spectra",
        "algorithmic_class": "Paired FFT Surprisal Transport",
        "time_complexity": "O(n_bins · log(n_bins) + |Σ|^K · L)",
        "space_complexity": "O(2 · n_bins + |Σ|^K)",
        "purpose": "Simultaneously computes counting (uniform sequence) and generated probability spectra."
    },
    "beam_model_spectra_by_length": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_spectra_by_length(const BeamModel *m, bool counting, uint32_t length, uint32_t n_bins, double smooth, double *out_weights, double *out_bin_width)",
        "category": "Density & Spectra",
        "algorithmic_class": "Concurrent Multi-Length Spectrum Sweep",
        "time_complexity": "O(L · (n_bins · log(n_bins) + |Σ|^K))",
        "space_complexity": "O(L · n_bins)",
        "purpose": "Evaluates surprisal density spectra across all sequence lengths in parallel."
    },
    "beam_model_topology": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_topology(const BeamModel *m, uint32_t length, double *out_nodes, double *out_edges, uint32_t *out_layer_nodes)",
        "category": "Model Introspection",
        "algorithmic_class": "Trellis Graph Structural Audit",
        "time_complexity": "O(|Σ|^K · L)",
        "space_complexity": "O(L)",
        "purpose": "Counts live active nodes and transitions per layer in DAG trellis graph."
    },
    "beam_model_length_derivatives": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_length_derivatives(const BeamModel *m, double q, uint32_t order, double *out)",
        "category": "Information Theory",
        "algorithmic_class": "Unmixed Per-Length Surprisal Moments Jet",
        "time_complexity": "O(|Σ|^K · L_max)",
        "space_complexity": "O(order · L_max)",
        "purpose": "Calculates unmixed raw surprisal moments separated by sequence length."
    },
    "beam_model_length_marginals": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_length_marginals(const BeamModel *m, double *out_counting, double *out_generated, double *out_mean, double *out_variance)",
        "category": "Information Theory",
        "algorithmic_class": "Length Prior Marginal Moments Decomposition",
        "time_complexity": "O(|Σ|^K · L_max)",
        "space_complexity": "O(L_max)",
        "purpose": "Computes exact counting and generated length probabilities with conditional mean and variance."
    },
    "beam_model_provenance": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_provenance(const BeamModel *m, int64_t *created_epoch_ns, int64_t *observed_ns, int64_t *finalized_ns)",
        "category": "Model Introspection",
        "algorithmic_class": "Metadata Extraction",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Extracts timestamps, observation durations, and training run telemetry."
    },
    "beam_model_file_provenance_span": {
        "header": "include/beam/model.h",
        "signature": "void beam_model_file_provenance_span(size_t *offset, size_t *bytes)",
        "category": "Model Introspection",
        "algorithmic_class": "Binary Format Header Offset Calculation",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Reports file offset and size of binary provenance header block."
    },
    "beam_model_depth_occupancy": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_depth_occupancy(const BeamModel *m, uint64_t *out_rows, uint64_t *out_cells)",
        "category": "Model Introspection",
        "algorithmic_class": "Dictionary Occupancy Audit",
        "time_complexity": "O(L_max)",
        "space_complexity": "O(1)",
        "purpose": "Counts populated hash table rows and active cells at each trellis layer."
    },
    "beam_model_positional_marginals": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_positional_marginals(const BeamModel *m, uint32_t length, const BeamConstraint *c, double *out_marginals, double *out_log_partition)",
        "category": "Information Theory",
        "algorithmic_class": "Forward-Backward Trellis Marginal Sweep",
        "time_complexity": "O(|Σ|^K · L)",
        "space_complexity": "O(L · |Σ| + |Σ|^K)",
        "purpose": "Calculates exact per-position residue marginal distribution P(x_i = a | L)."
    },
    "beam_model_positional_banded": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_positional_banded(const BeamModel *m, uint32_t length, bool counting, uint32_t n_bins, double smooth, const double *edges, uint32_t n_edges, double *out_marginals, double *out_band_mass, double *out_pos_mass, double *out_bin_width)",
        "category": "Information Theory",
        "algorithmic_class": "Banded Surprisal Marginal Decomposition",
        "time_complexity": "O(n_bins · log(n_bins) + |Σ|^K · L)",
        "space_complexity": "O(n_bands · L · |Σ|)",
        "purpose": "Decomposes positional marginal distributions across discrete surprisal / frequency bands."
    },
    "beam_model_expected_counts": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_expected_counts(const BeamModel *m, uint32_t length, BeamExpectedCounts *out)",
        "category": "Dynamic Programming",
        "algorithmic_class": "Full Forward-Backward Trellis Expectation",
        "time_complexity": "O(|Σ|^K · L)",
        "space_complexity": "O(|Σ|^K · L)",
        "purpose": "Calculates analytical expected traversal counts for every context and edge under model distribution."
    },
    "beam_model_expected_counts_free": {
        "header": "include/beam/model.h",
        "signature": "void beam_model_expected_counts_free(BeamExpectedCounts *ec)",
        "category": "Memory Management",
        "algorithmic_class": "Trellis Expectation Table Deallocation",
        "time_complexity": "O(L)",
        "space_complexity": "O(1)",
        "purpose": "Releases all memory allocated for BeamExpectedCounts structure."
    },
    "beam_count_block_free": {
        "header": "include/beam/model.h",
        "signature": "void beam_count_block_free(BeamCountBlock *block)",
        "category": "Memory Management",
        "algorithmic_class": "Trellis Count Block Deallocation",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Frees an allocated count block within the expectation hierarchy."
    },
    "beam_model_top_k": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_top_k(const BeamModel *m, uint32_t k, uint32_t length, char *out_seqs, double *out_log_p, uint32_t *out_n)",
        "category": "Algorithms & Generation",
        "algorithmic_class": "Exact Trellis Beam Search (A* Priority Queue)",
        "time_complexity": "O(k · log(k) + k · |Σ|)",
        "space_complexity": "O(k · L)",
        "purpose": "Finds exact top-k highest probability sequences of length L under model distribution."
    },
    "beam_model_sample": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_sample(const BeamModel *m, uint32_t n, uint32_t length, BeamRng *rng, char *out_seqs, uint32_t *out_lens)",
        "category": "Algorithms & Generation",
        "algorithmic_class": "Categorical Trellis Random Walk",
        "time_complexity": "O(n · L)",
        "space_complexity": "O(1)",
        "purpose": "Generates n independent sequences sampled exactly according to model probabilities."
    },
    "beam_model_explain": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_explain(const BeamModel *m, const char *seq, double *out_fwd, double *out_bwd, uint32_t *out_len)",
        "category": "Algorithms & Generation",
        "algorithmic_class": "Forward/Backward Sequence Attribution",
        "time_complexity": "O(L)",
        "space_complexity": "O(1)",
        "purpose": "Decomposes sequence log-probability into residue-by-residue forward and backward score contributions."
    },
    "beam_model_save": {
        "header": "include/beam/model.h",
        "signature": "BeamError beam_model_save(const BeamModel *m, const char *path)",
        "category": "Serialization",
        "algorithmic_class": "Binary Trellis Serialization with CRC32C",
        "time_complexity": "O(|Cells|)",
        "space_complexity": "O(1)",
        "purpose": "Saves model in native binary format with hardware-verified CRC32C digest."
    },
    "beam_model_load": {
        "header": "include/beam/model.h",
        "signature": "BeamModel *beam_model_load(const char *path)",
        "category": "Serialization",
        "algorithmic_class": "Binary Trellis Deserialization with Validation",
        "time_complexity": "O(|Cells|)",
        "space_complexity": "O(|Cells|)",
        "purpose": "Loads model from disk, validating header magic, version, and CRC32C digest."
    },
    "beam_model_config": {
        "header": "include/beam/model.h",
        "signature": "const BeamModelConfig *beam_model_config(const BeamModel *m)",
        "category": "Model Introspection",
        "algorithmic_class": "Config Pointer Query",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns pointer to model configuration parameters (alphabet, order, max_length)."
    },
    "beam_model_config_default": {
        "header": "include/beam/model.h",
        "signature": "void beam_model_config_default(BeamModelConfig *cfg)",
        "category": "Model Lifecycle",
        "algorithmic_class": "Default Parameter Initialization",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Initializes BeamModelConfig with default amino acid parameters (K=4, L=27)."
    },
    "beam_model_config_default_for": {
        "header": "include/beam/model.h",
        "signature": "void beam_model_config_default_for(BeamAlphabet alpha, BeamModelConfig *cfg)",
        "category": "Model Lifecycle",
        "algorithmic_class": "Alphabet-Specific Parameter Initialization",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Initializes config tailored for specified alphabet (NT: K=6, L=81; AA: K=4, L=27)."
    },
    "beam_model_total_weight": {
        "header": "include/beam/model.h",
        "signature": "double beam_model_total_weight(const BeamModel *m)",
        "category": "Model Introspection",
        "algorithmic_class": "Weight Sum Query",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns total observation weight accumulated in model."
    },
    "beam_model_n_observed": {
        "header": "include/beam/model.h",
        "signature": "uint64_t beam_model_n_observed(const BeamModel *m)",
        "category": "Model Introspection",
        "algorithmic_class": "Sequence Counter Query",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns count of distinct sequence observations incorporated into model."
    },
    "beam_model_n_skipped": {
        "header": "include/beam/model.h",
        "signature": "uint64_t beam_model_n_skipped(const BeamModel *m)",
        "category": "Model Introspection",
        "algorithmic_class": "Skipped Counter Query",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns count of sequences skipped due to invalid characters or length bounds."
    },
    "beam_model_alphabet": {
        "header": "include/beam/model.h",
        "signature": "BeamAlphabet beam_model_alphabet(const BeamModel *m)",
        "category": "Model Introspection",
        "algorithmic_class": "Enum Query",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns alphabet type (BEAM_ALPHABET_AA or BEAM_ALPHABET_NT)."
    },
    "beam_model_n_residues": {
        "header": "include/beam/model.h",
        "signature": "uint32_t beam_model_n_residues(const BeamModel *m)",
        "category": "Model Introspection",
        "algorithmic_class": "Cardinality Query",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns alphabet residue cardinality (20 for AA, 4 for NT without ambiguity)."
    },
    "beam_model_radix": {
        "header": "include/beam/model.h",
        "signature": "uint32_t beam_model_radix(const BeamModel *m)",
        "category": "Model Introspection",
        "algorithmic_class": "Radix Query",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns alphabet radix (21 for AA, 5 for NT including end-of-string token)."
    },
    "beam_model_n_states": {
        "header": "include/beam/model.h",
        "signature": "uint64_t beam_model_n_states(const BeamModel *m)",
        "category": "Model Introspection",
        "algorithmic_class": "State Count Query",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns total number of trellis states (|Σ|^K)."
    },
    "beam_model_n_cells": {
        "header": "include/beam/model.h",
        "signature": "uint64_t beam_model_n_cells(const BeamModel *m)",
        "category": "Model Introspection",
        "algorithmic_class": "Trellis Cell Query",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns total transition cells across all trellis layers."
    },
    "beam_model_n_nonzero_cells": {
        "header": "include/beam/model.h",
        "signature": "uint64_t beam_model_n_nonzero_cells(const BeamModel *m)",
        "category": "Model Introspection",
        "algorithmic_class": "Sparse Cell Counter",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns count of transition cells with nonzero observed counts."
    },
    "beam_model_memory_bytes": {
        "header": "include/beam/model.h",
        "signature": "size_t beam_model_memory_bytes(const BeamModel *m)",
        "category": "Memory Management",
        "algorithmic_class": "Heap Accounting",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns exact allocated byte count of model structure."
    },
    "beam_model_live_cache_bytes": {
        "header": "include/beam/model.h",
        "signature": "size_t beam_model_live_cache_bytes(const BeamModel *m)",
        "category": "Memory Management",
        "algorithmic_class": "Cache Memory Accounting",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns byte count currently retained by cached DP tables."
    },
    "beam_model_set_threads": {
        "header": "include/beam/model.h",
        "signature": "void beam_model_set_threads(int32_t threads)",
        "category": "System & Runtime",
        "algorithmic_class": "OpenMP Thread Pool Config",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Configures number of OpenMP worker threads for native sweeps (0 = auto)."
    },
    "beam_model_threads": {
        "header": "include/beam/model.h",
        "signature": "int32_t beam_model_threads(void)",
        "category": "System & Runtime",
        "algorithmic_class": "Thread Query",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns user-configured thread count target."
    },
    "beam_model_effective_threads": {
        "header": "include/beam/model.h",
        "signature": "uint32_t beam_model_effective_threads(void)",
        "category": "System & Runtime",
        "algorithmic_class": "Active Thread Query",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns actual number of OpenMP threads that will run based on hardware detection."
    },
    "beam_model_set_memory_budget": {
        "header": "include/beam/model.h",
        "signature": "void beam_model_set_memory_budget(uint64_t bytes)",
        "category": "Memory Management",
        "algorithmic_class": "Safety Budget Setter",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Sets safety ceiling on memory allocation during dynamic programming sweeps."
    },
    "beam_model_memory_budget": {
        "header": "include/beam/model.h",
        "signature": "uint64_t beam_model_memory_budget(void)",
        "category": "Memory Management",
        "algorithmic_class": "Budget Query",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns current memory allocation limit in bytes."
    },
    "beam_publicness_moments": {
        "header": "include/beam/publicness.h",
        "signature": "BeamError beam_publicness_moments(const double *p, uint32_t n_atoms, const double *depths, const double *multiplicities, uint32_t n_groups, double *out_mean, double *out_variance)",
        "category": "Occupancy & Publicness",
        "algorithmic_class": "Poisson-Binomial Analytical Moments",
        "time_complexity": "O(n_atoms · n_groups)",
        "space_complexity": "O(1)",
        "purpose": "Computes exact mean and variance of sequence sharing across cohort sequencing depths."
    },
    "beam_publicness_pgf": {
        "header": "include/beam/publicness.h",
        "signature": "BeamError beam_publicness_pgf(const double *p, uint32_t n_atoms, const double *depths, const double *multiplicities, uint32_t n_groups, uint32_t k_fft, double *out)",
        "category": "Occupancy & Publicness",
        "algorithmic_class": "Complex Roots-of-Unity PGF Evaluation",
        "time_complexity": "O(k_fft · n_atoms · n_groups)",
        "space_complexity": "O(k_fft · n_atoms)",
        "purpose": "Evaluates probability generating function along unit circle for FFT PMF inversion."
    },
    "beam_publicness_accumulate": {
        "header": "include/beam/publicness.h",
        "signature": "BeamError beam_publicness_accumulate(const double *pmfs, uint32_t n_atoms, uint32_t k_fft, const double *means, const double *variances, const double *weights, const double *bin_edges, uint32_t n_bins, double tail_sigma, double tail_floor, double tail_slack, double *out_counts, double *out_residual)",
        "category": "Occupancy & Publicness",
        "algorithmic_class": "Tail-Truncated PMF Binning",
        "time_complexity": "O(n_atoms · n_bins)",
        "space_complexity": "O(1)",
        "purpose": "Truncates PMF numerical tails and bins sharing distributions into publicness spectrum."
    },
    "beam_publicness_accumulate_grouped": {
        "header": "include/beam/publicness.h",
        "signature": "BeamError beam_publicness_accumulate_grouped(const double *pmfs, uint32_t n_atoms, uint32_t k_fft, const double *means, const double *variances, const double *weights, const uint32_t *groups, uint32_t n_groups_out, const double *bin_edges, uint32_t n_bins, double tail_sigma, double tail_floor, double tail_slack, double *out_counts, double *out_residual)",
        "category": "Occupancy & Publicness",
        "algorithmic_class": "Grouped Cohort Publicness Binning",
        "time_complexity": "O(n_atoms · n_bins)",
        "space_complexity": "O(n_groups_out · n_bins)",
        "purpose": "Accumulates publicness sharing spectra partitioned by clinical or demographic cohort groups."
    },
    "beam_query_constraint_new": {
        "header": "include/beam/query.h",
        "signature": "BeamConstraint *beam_query_constraint_new(uint32_t length, uint32_t n_residues)",
        "category": "Query & Constraints",
        "algorithmic_class": "Constraint Bitmask Allocation",
        "time_complexity": "O(L)",
        "space_complexity": "O(L)",
        "purpose": "Allocates unconstrained BeamConstraint admitting all sequences of specified length."
    },
    "beam_query_constraint_free": {
        "header": "include/beam/query.h",
        "signature": "void beam_query_constraint_free(BeamConstraint *c)",
        "category": "Query & Constraints",
        "algorithmic_class": "Constraint Deallocation",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Frees constraint and any attached automata."
    },
    "beam_query_constraint_length": {
        "header": "include/beam/query.h",
        "signature": "uint32_t beam_query_constraint_length(const BeamConstraint *c)",
        "category": "Query & Constraints",
        "algorithmic_class": "Length Getter",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns target sequence length of constraint."
    },
    "beam_query_constraint_residues": {
        "header": "include/beam/query.h",
        "signature": "uint32_t beam_query_constraint_residues(const BeamConstraint *c)",
        "category": "Query & Constraints",
        "algorithmic_class": "Residue Getter",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns residue cardinality of constraint."
    },
    "beam_query_constraint_mask": {
        "header": "include/beam/query.h",
        "signature": "BeamResidueMask beam_query_constraint_mask(const BeamConstraint *c, uint32_t position)",
        "category": "Query & Constraints",
        "algorithmic_class": "Bitmask Read",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns allowed residue bitmask at given sequence position."
    },
    "beam_query_constraint_is_free": {
        "header": "include/beam/query.h",
        "signature": "bool beam_query_constraint_is_free(const BeamConstraint *c)",
        "category": "Query & Constraints",
        "algorithmic_class": "Boolean Status Check",
        "time_complexity": "O(L)",
        "space_complexity": "O(1)",
        "purpose": "Returns true if no position is restricted, allowing engine fast paths."
    },
    "beam_query_constrain_position": {
        "header": "include/beam/query.h",
        "signature": "BeamError beam_query_constrain_position(BeamConstraint *c, uint32_t position, uint32_t residue)",
        "category": "Query & Constraints",
        "algorithmic_class": "Single Residue Mask Narrowing",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Requires position to hold exactly the specified residue."
    },
    "beam_query_constrain_set": {
        "header": "include/beam/query.h",
        "signature": "BeamError beam_query_constrain_set(BeamConstraint *c, uint32_t position, BeamResidueMask allowed)",
        "category": "Query & Constraints",
        "algorithmic_class": "Bitmask Set Narrowing",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Restricts position to residues present in bitmask."
    },
    "beam_query_constrain_exclude": {
        "header": "include/beam/query.h",
        "signature": "BeamError beam_query_constrain_exclude(BeamConstraint *c, uint32_t position, uint32_t residue)",
        "category": "Query & Constraints",
        "algorithmic_class": "Bitmask Bit Clearance",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Forbids specified residue from appearing at position."
    },
    "beam_query_constrain_run": {
        "header": "include/beam/query.h",
        "signature": "BeamError beam_query_constrain_run(BeamConstraint *c, uint32_t start, const uint8_t *residues, uint32_t n)",
        "category": "Query & Constraints",
        "algorithmic_class": "Subsegment Run Mask Application",
        "time_complexity": "O(n)",
        "space_complexity": "O(1)",
        "purpose": "Requires n consecutive positions starting at start to match exact residue array."
    },
    "beam_query_automaton_motif": {
        "header": "include/beam/query.h",
        "signature": "BeamAutomaton *beam_query_automaton_motif(uint32_t length, uint32_t n_residues, const uint8_t *motif, uint32_t n)",
        "category": "Query & Automata",
        "algorithmic_class": "Knuth-Morris-Pratt DFA Compilation",
        "time_complexity": "O(n · |Σ|)",
        "space_complexity": "O(n · |Σ|)",
        "purpose": "Compiles KMP pattern automaton matching motif at any sequence position."
    },
    "beam_query_automaton_count": {
        "header": "include/beam/query.h",
        "signature": "BeamAutomaton *beam_query_automaton_count(uint32_t length, uint32_t n_residues, BeamResidueMask of, uint32_t lo, uint32_t hi)",
        "category": "Query & Automata",
        "algorithmic_class": "Bounded Counter DFA Compilation",
        "time_complexity": "O(hi · |Σ|)",
        "space_complexity": "O(hi · |Σ|)",
        "purpose": "Compiles automaton accepting sequences with between lo and hi residues from mask of."
    },
    "beam_query_automaton_from_masks": {
        "header": "include/beam/query.h",
        "signature": "BeamAutomaton *beam_query_automaton_from_masks(const BeamConstraint *c)",
        "category": "Query & Automata",
        "algorithmic_class": "Positional Mask to DFA Conversion",
        "time_complexity": "O(L · |Σ|)",
        "space_complexity": "O(L · |Σ|)",
        "purpose": "Converts constraint positional mask restrictions into a 2-state layer-indexed automaton."
    },
    "beam_query_automaton_and": {
        "header": "include/beam/query.h",
        "signature": "BeamAutomaton *beam_query_automaton_and(const BeamAutomaton *a, const BeamAutomaton *b)",
        "category": "Query & Automata",
        "algorithmic_class": "Product Automaton Conjunction (A ∩ B)",
        "time_complexity": "O(|Q_a| · |Q_b| · |Σ|)",
        "space_complexity": "O(|Q_a| · |Q_b| · |Σ|)",
        "purpose": "Constructs product automaton accepting sequences accepted by BOTH input automata."
    },
    "beam_query_automaton_or": {
        "header": "include/beam/query.h",
        "signature": "BeamAutomaton *beam_query_automaton_or(const BeamAutomaton *a, const BeamAutomaton *b)",
        "category": "Query & Automata",
        "algorithmic_class": "Product Automaton Disjunction (A ∪ B)",
        "time_complexity": "O(|Q_a| · |Q_b| · |Σ|)",
        "space_complexity": "O(|Q_a| · |Q_b| · |Σ|)",
        "purpose": "Constructs product automaton accepting sequences accepted by EITHER input automaton."
    },
    "beam_query_automaton_not": {
        "header": "include/beam/query.h",
        "signature": "BeamAutomaton *beam_query_automaton_not(const BeamAutomaton *a)",
        "category": "Query & Automata",
        "algorithmic_class": "Automaton State Inversion",
        "time_complexity": "O(|Q|)",
        "space_complexity": "O(|Q| · |Σ|)",
        "purpose": "Constructs complement automaton accepting sequences rejected by input automaton."
    },
    "beam_query_automaton_free": {
        "header": "include/beam/query.h",
        "signature": "void beam_query_automaton_free(BeamAutomaton *a)",
        "category": "Query & Automata",
        "algorithmic_class": "Automaton Deallocation",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Frees memory allocated for a BeamAutomaton."
    },
    "beam_query_automaton_states": {
        "header": "include/beam/query.h",
        "signature": "uint32_t beam_query_automaton_states(const BeamAutomaton *a)",
        "category": "Query & Automata",
        "algorithmic_class": "Automaton State Cardinality Query",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns number of states |Q| in the automaton."
    },
    "beam_query_automaton_accepts": {
        "header": "include/beam/query.h",
        "signature": "bool beam_query_automaton_accepts(const BeamAutomaton *a, const uint8_t *residues, uint32_t n)",
        "category": "Query & Automata",
        "algorithmic_class": "Sequential DFA State Execution",
        "time_complexity": "O(n)",
        "space_complexity": "O(1)",
        "purpose": "Directly tests whether automaton accepts specified sequence of residue indices."
    },
    "beam_query_constrain_automaton": {
        "header": "include/beam/query.h",
        "signature": "BeamError beam_query_constrain_automaton(BeamConstraint *c, BeamAutomaton *a)",
        "category": "Query & Automata",
        "algorithmic_class": "Automaton Constraint Attachment",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Attaches automaton to constraint; transfers ownership of a so caller does not free it."
    },
    "beam_query_constraint_automaton": {
        "header": "include/beam/query.h",
        "signature": "const BeamAutomaton *beam_query_constraint_automaton(const BeamConstraint *c)",
        "category": "Query & Automata",
        "algorithmic_class": "Automaton Borrow Query",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Returns borrowed pointer to automaton attached to constraint (or NULL)."
    },
    "beam_query_expect": {
        "header": "include/beam/query.h",
        "signature": "BeamError beam_query_expect(const BeamModel *m, const BeamConstraint *c, const BeamFunctional *f, BeamQueryMoments *out)",
        "category": "Statistical Query",
        "algorithmic_class": "Conditional Functional Expectation DP",
        "time_complexity": "O(|Σ|^K · L · |Q|)",
        "space_complexity": "O(|Σ|^K · |Q|)",
        "purpose": "Computes exact mean and variance of an additive functional under constraint conditioning."
    },
    "beam_query_entropy": {
        "header": "include/beam/query.h",
        "signature": "BeamError beam_query_entropy(const BeamModel *m, const BeamConstraint *c, double *out_entropy, double *out_variance)",
        "category": "Statistical Query",
        "algorithmic_class": "Constrained Shannon Entropy Jet Sweep",
        "time_complexity": "O(|Σ|^K · L · |Q|)",
        "space_complexity": "O(|Σ|^K · |Q|)",
        "purpose": "Calculates exact conditional Shannon entropy H(S | C) under complex query constraints."
    },
    "beam_query_neighbourhood_free": {
        "header": "include/beam/query.h",
        "signature": "void beam_query_neighbourhood_free(BeamNeighbourhood *nb)",
        "category": "Memory Management",
        "algorithmic_class": "Neighbourhood Heap Deallocation",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Frees memory allocated for 1-edit insertion/deletion neighborhood structure."
    },
    "beam_query_insertions": {
        "header": "include/beam/query.h",
        "signature": "BeamError beam_query_insertions(const BeamModel *m, const char *seq, BeamNeighbourhood *out)",
        "category": "Statistical Query",
        "algorithmic_class": "1-Edit Insertion Neighborhood Sweep",
        "time_complexity": "O(L · |Σ|)",
        "space_complexity": "O(L · |Σ|)",
        "purpose": "Generates all single-residue insertions and computes exact posterior odds and log-probabilities."
    },
    "beam_query_deletions": {
        "header": "include/beam/query.h",
        "signature": "BeamError beam_query_deletions(const BeamModel *m, const char *seq, BeamNeighbourhood *out)",
        "category": "Statistical Query",
        "algorithmic_class": "1-Edit Deletion Neighborhood Sweep",
        "time_complexity": "O(L)",
        "space_complexity": "O(L)",
        "purpose": "Generates all single-residue deletions and computes exact posterior odds and log-probabilities."
    },
    "beam_query_estimate": {
        "header": "include/beam/query.h",
        "signature": "BeamError beam_query_estimate(const BeamModel *m, const BeamConstraint *c, BeamQueryCost *out)",
        "category": "Query Optimization",
        "algorithmic_class": "Analytical Query Cost Estimator",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Estimates state count, edge count, and memory footprint of product graph without running sweep."
    },
    "beam_query_position_entropy": {
        "header": "include/beam/query.h",
        "signature": "BeamError beam_query_position_entropy(const BeamModel *m, const BeamConstraint *c, double *out)",
        "category": "Statistical Query",
        "algorithmic_class": "Per-Position Conditional Entropy Vector",
        "time_complexity": "O(|Σ|^K · L · |Q|)",
        "space_complexity": "O(L · |Σ|)",
        "purpose": "Evaluates positional Shannon entropy H(X_i | C) vector across all positions."
    },
    "beam_query_pair_joint": {
        "header": "include/beam/query.h",
        "signature": "BeamError beam_query_pair_joint(const BeamModel *m, const BeamConstraint *c, uint32_t i, uint32_t j, double *out)",
        "category": "Statistical Query",
        "algorithmic_class": "Pairwise 2D Joint Marginal Sweep",
        "time_complexity": "O(|Σ|^K · L · |Q|)",
        "space_complexity": "O(|Σ|^2)",
        "purpose": "Computes exact |Σ| x |Σ| joint probability matrix P(X_i = a, X_j = b | C)."
    },
    "beam_query_information_matrix": {
        "header": "include/beam/query.h",
        "signature": "BeamError beam_query_information_matrix(const BeamModel *m, const BeamConstraint *c, double *out)",
        "category": "Statistical Query",
        "algorithmic_class": "Pairwise Mutual Information Matrix (L x L)",
        "time_complexity": "O(L^2 · R^2 · S)",
        "space_complexity": "O(L · R + L^2)",
        "purpose": "L x L mutual information matrix I(X_i; X_j | C): one conditional marginal sweep per (position, residue) carrying mass, L*R + 1 sweeps in all (src/native/query/information.c)."
    },
    "beam_query_information_cost": {
        "header": "include/beam/query.h",
        "signature": "BeamError beam_query_information_cost(const BeamModel *m, const BeamConstraint *c, BeamQueryCost *out)",
        "category": "Query Optimization",
        "algorithmic_class": "Analytical Complexity Predictor",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Estimates total operation count and memory requirement for full information matrix calculation."
    },
    "beam_query_log_probability": {
        "header": "include/beam/query.h",
        "signature": "BeamError beam_query_log_probability(const BeamModel *m, const BeamConstraint *c, double *out)",
        "category": "Statistical Query",
        "algorithmic_class": "Conditional Constraint Probability P(C | L)",
        "time_complexity": "O(|Σ|^K · L · |Q|)",
        "space_complexity": "O(|Σ|^K · |Q|)",
        "purpose": "Exact conditional probability ln P(C | L) of satisfying constraint under model."
    },
    "beam_query_marginals": {
        "header": "include/beam/query.h",
        "signature": "BeamError beam_query_marginals(const BeamModel *m, const BeamConstraint *c, double *out_marginals, double *out_log_partition)",
        "category": "Statistical Query",
        "algorithmic_class": "Constrained Forward-Backward Posterior Marginals",
        "time_complexity": "O(|Σ|^K · L · |Q|)",
        "space_complexity": "O(L · |Σ| + |Σ|^K · |Q|)",
        "purpose": "Computes exact per-position posterior residue marginal distribution conditioned on constraint."
    },
    "beam_rng_seed": {
        "header": "include/beam/rng.h",
        "signature": "void beam_rng_seed(BeamRng *rng, uint64_t seed)",
        "category": "Utility & PRNG",
        "algorithmic_class": "SplitMix64 State Expansion",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Seeds xoshiro256++ PRNG state deterministically using SplitMix64 generator."
    },
    "beam_rng_next": {
        "header": "include/beam/rng.h",
        "signature": "uint64_t beam_rng_next(BeamRng *rng)",
        "category": "Utility & PRNG",
        "algorithmic_class": "xoshiro256++ Pseudo-Random Generator",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Generates next 64-bit pseudo-random unsigned integer."
    },
    "beam_rng_double": {
        "header": "include/beam/rng.h",
        "signature": "double beam_rng_double(BeamRng *rng)",
        "category": "Utility & PRNG",
        "algorithmic_class": "IEEE-754 Uniform Float Extraction",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Generates uniform pseudo-random double in half-open interval [0, 1)."
    },
    "beam_rng_bounded": {
        "header": "include/beam/rng.h",
        "signature": "uint32_t beam_rng_bounded(BeamRng *rng, uint32_t upper_bound)",
        "category": "Utility & PRNG",
        "algorithmic_class": "Lemire Fast Bounded Random Integer",
        "time_complexity": "O(1)",
        "space_complexity": "O(1)",
        "purpose": "Generates unbiased uniform random integer in [0, upper_bound) without modulo division."
    },
}


def get_catalog_entry(symbol: str) -> Optional[Dict[str, Any]]:
    return FUNCTION_CATALOG.get(symbol)


def list_all_symbols() -> List[str]:
    return sorted(FUNCTION_CATALOG)
