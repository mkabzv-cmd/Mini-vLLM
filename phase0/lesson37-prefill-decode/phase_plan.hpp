#pragma once

#include <cstddef>
#include <vector>

namespace minitensor::inference {

// Prefill and decode both predict a next token, but they do different work:
//
//   Prefill  processes the complete prompt.
//   Decode   processes one newly appended token while reusing cached history.
enum class InferencePhase {
    Prefill,
    Decode,
};

[[nodiscard]] const char* phase_name(
    InferencePhase phase
) noexcept;

// Describes the work needed to make one next-token prediction.
//
// The "full recomputation" fields describe Lesson 36's real implementation.
// The cache-aware fields describe the schedule that a future KV-cache
// implementation will execute.
struct PredictionStep {
    // Zero means "predict the first new token."
    std::size_t prediction_index = 0;

    InferencePhase cache_aware_phase =
        InferencePhase::Prefill;

    // Lesson 36 sends the entire growing sequence through the model.
    std::size_t full_recomputation_input_tokens = 0;

    // Prefill processes the prompt. Each later decode processes one token.
    std::size_t cache_aware_input_tokens = 0;

    // Absolute sequence position of the first cache-aware input token.
    std::size_t start_position = 0;

    std::size_t cached_tokens_before = 0;
    std::size_t cached_tokens_after = 0;

    // Per attention head and layer, the allocated score matrix has:
    //
    //   [query_tokens, score_key_tokens]
    //
    // During causal prefill, some entries are masked. Therefore this is the
    // key-axis width, not the number of keys visible to every query.
    std::size_t query_tokens = 0;
    std::size_t score_key_tokens = 0;
};

struct WorkTotals {
    // Sum of the token counts sent through the transformer.
    std::size_t model_input_token_positions = 0;

    // Sum of query_tokens * score_key_tokens for one head in one layer.
    // Head count and layer count are deliberately excluded.
    std::size_t attention_score_cells_per_head = 0;
};

struct GenerationPhasePlan {
    std::size_t prompt_tokens = 0;

    // Number of next-token predictions. This is also the maximum number of
    // newly generated tokens if EOS does not stop generation early.
    std::size_t prediction_count = 0;

    // One comparison row for every prediction.
    std::vector<PredictionStep> steps;

    WorkTotals full_recomputation;
    WorkTotals cache_aware;
};

// Builds a work plan only; it does not execute the model or allocate a
// KV cache. A prompt must contain at least one token. A prediction count of
// zero is valid and produces an empty plan with zero work.
[[nodiscard]] GenerationPhasePlan plan_generation_phases(
    std::size_t prompt_tokens,
    std::size_t prediction_count
);

} // namespace minitensor::inference
