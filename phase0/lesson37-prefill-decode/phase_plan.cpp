#include "phase_plan.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace minitensor::inference {

namespace {

[[nodiscard]] std::size_t checked_add(
    std::size_t left,
    std::size_t right,
    const char* description
) {
    if (
        right >
        std::numeric_limits<std::size_t>::max() - left
    ) {
        throw std::overflow_error(
            std::string(description) + " overflowed size_t"
        );
    }

    return left + right;
}

[[nodiscard]] std::size_t checked_multiply(
    std::size_t left,
    std::size_t right,
    const char* description
) {
    if (
        left != 0 &&
        right >
            std::numeric_limits<std::size_t>::max() / left
    ) {
        throw std::overflow_error(
            std::string(description) + " overflowed size_t"
        );
    }

    return left * right;
}

void add_to_total(
    std::size_t value,
    std::size_t& total,
    const char* description
) {
    total = checked_add(total, value, description);
}

} // namespace

const char* phase_name(
    InferencePhase phase
) noexcept {
    switch (phase) {
    case InferencePhase::Prefill:
        return "prefill";
    case InferencePhase::Decode:
        return "decode";
    }

    return "unknown";
}

GenerationPhasePlan plan_generation_phases(
    std::size_t prompt_tokens,
    std::size_t prediction_count
) {
    if (prompt_tokens == 0) {
        throw std::invalid_argument(
            "a generation plan requires at least one "
            "prompt token"
        );
    }

    GenerationPhasePlan plan;
    plan.prompt_tokens = prompt_tokens;
    plan.prediction_count = prediction_count;

    if (prediction_count > plan.steps.max_size()) {
        throw std::length_error(
            "prediction count exceeds the plan's step capacity"
        );
    }

    plan.steps.reserve(prediction_count);

    for (
        std::size_t prediction_index = 0;
        prediction_index < prediction_count;
        ++prediction_index
    ) {
        // Lesson 36 grows its full model input by one token before every
        // prediction after the first.
        const std::size_t full_input_tokens =
            checked_add(
                prompt_tokens,
                prediction_index,
                "full-recomputation sequence length"
            );
        const std::size_t full_attention_cells =
            checked_multiply(
                full_input_tokens,
                full_input_tokens,
                "full-recomputation attention work"
            );

        PredictionStep step;
        step.prediction_index = prediction_index;
        step.full_recomputation_input_tokens =
            full_input_tokens;

        if (prediction_index == 0) {
            // Prefill processes every prompt token and writes its keys and
            // values into the future cache. Its final logit predicts the
            // first new token.
            step.cache_aware_phase =
                InferencePhase::Prefill;
            step.cache_aware_input_tokens =
                prompt_tokens;
            step.start_position = 0;
            step.cached_tokens_before = 0;
            step.cached_tokens_after = prompt_tokens;
            step.query_tokens = prompt_tokens;
            step.score_key_tokens = prompt_tokens;
        } else {
            // The preceding prediction selected one new token. Decode now
            // processes that token at its true absolute position, appends
            // its key/value entries, and predicts the following token.
            step.cache_aware_phase =
                InferencePhase::Decode;
            step.cache_aware_input_tokens = 1;
            step.cached_tokens_before =
                full_input_tokens - 1;
            step.cached_tokens_after =
                full_input_tokens;
            step.start_position =
                step.cached_tokens_before;
            step.query_tokens = 1;
            step.score_key_tokens =
                step.cached_tokens_after;
        }

        const std::size_t cache_aware_attention_cells =
            checked_multiply(
                step.query_tokens,
                step.score_key_tokens,
                "cache-aware attention work"
            );

        add_to_total(
            step.full_recomputation_input_tokens,
            plan.full_recomputation
                .model_input_token_positions,
            "full-recomputation token work"
        );
        add_to_total(
            full_attention_cells,
            plan.full_recomputation
                .attention_score_cells_per_head,
            "full-recomputation attention total"
        );
        add_to_total(
            step.cache_aware_input_tokens,
            plan.cache_aware.model_input_token_positions,
            "cache-aware token work"
        );
        add_to_total(
            cache_aware_attention_cells,
            plan.cache_aware
                .attention_score_cells_per_head,
            "cache-aware attention total"
        );

        plan.steps.push_back(step);
    }

    return plan;
}

} // namespace minitensor::inference
