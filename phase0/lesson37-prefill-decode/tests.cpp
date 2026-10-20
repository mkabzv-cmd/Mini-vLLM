#include "phase_plan.hpp"

#include <array>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

using minitensor::inference::GenerationPhasePlan;
using minitensor::inference::InferencePhase;
using minitensor::inference::PredictionStep;

void require(
    bool condition,
    std::string_view message
) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Exception, typename Function>
void require_throws(
    Function&& function,
    std::string_view expected_message_part
) {
    try {
        std::forward<Function>(function)();
    } catch (const Exception& error) {
        if (
            std::string_view(error.what()).find(
                expected_message_part
            ) == std::string_view::npos
        ) {
            throw std::runtime_error(
                "exception message did not contain '" +
                std::string(expected_message_part) +
                "': " + error.what()
            );
        }

        return;
    } catch (const std::exception& error) {
        throw std::runtime_error(
            "unexpected exception type: " +
            std::string(error.what())
        );
    }

    throw std::runtime_error(
        "expected an exception containing '" +
        std::string(expected_message_part) + "'"
    );
}

void test_hello_eight_prediction_plan() {
    const GenerationPhasePlan plan =
        minitensor::inference::plan_generation_phases(
            2,
            8
        );

    require(
        plan.prompt_tokens == 2 &&
            plan.prediction_count == 8 &&
            plan.steps.size() == 8,
        "wrong plan header for P=2, G=8"
    );

    constexpr std::array<std::size_t, 8>
        expected_full_inputs{
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
        };

    for (
        std::size_t index = 0;
        index < plan.steps.size();
        ++index
    ) {
        const PredictionStep& step = plan.steps[index];

        require(
            step.prediction_index == index,
            "prediction indexes are not sequential"
        );
        require(
            step.full_recomputation_input_tokens ==
                expected_full_inputs[index],
            "wrong full-recomputation input length"
        );

        if (index == 0) {
            require(
                step.cache_aware_phase ==
                    InferencePhase::Prefill &&
                    step.cache_aware_input_tokens == 2 &&
                    step.start_position == 0 &&
                    step.cached_tokens_before == 0 &&
                    step.cached_tokens_after == 2 &&
                    step.query_tokens == 2 &&
                    step.score_key_tokens == 2,
                "wrong prefill step"
            );
        } else {
            const std::size_t expected_cache_after =
                2 + index;

            require(
                step.cache_aware_phase ==
                    InferencePhase::Decode &&
                    step.cache_aware_input_tokens == 1 &&
                    step.start_position ==
                        expected_cache_after - 1 &&
                    step.cached_tokens_before ==
                        expected_cache_after - 1 &&
                    step.cached_tokens_after ==
                        expected_cache_after &&
                    step.query_tokens == 1 &&
                    step.score_key_tokens ==
                        expected_cache_after,
                "wrong decode step"
            );
        }
    }

    require(
        plan.full_recomputation
                .model_input_token_positions == 44 &&
            plan.cache_aware
                .model_input_token_positions == 9,
        "wrong token-work totals for P=2, G=8"
    );
    require(
        plan.full_recomputation
                .attention_score_cells_per_head == 284 &&
            plan.cache_aware
                .attention_score_cells_per_head == 46,
        "wrong attention-work totals for P=2, G=8"
    );
}

void test_zero_predictions_do_no_work() {
    const GenerationPhasePlan plan =
        minitensor::inference::plan_generation_phases(
            4,
            0
        );

    require(
        plan.steps.empty() &&
            plan.full_recomputation
                .model_input_token_positions == 0 &&
            plan.full_recomputation
                .attention_score_cells_per_head == 0 &&
            plan.cache_aware
                .model_input_token_positions == 0 &&
            plan.cache_aware
                .attention_score_cells_per_head == 0,
        "zero predictions should produce zero work"
    );
}

void test_one_prediction_is_prefill_only() {
    const GenerationPhasePlan plan =
        minitensor::inference::plan_generation_phases(
            4,
            1
        );

    require(
        plan.steps.size() == 1,
        "one prediction should produce one step"
    );

    const PredictionStep& step = plan.steps.front();
    require(
        step.cache_aware_phase ==
                InferencePhase::Prefill &&
            step.full_recomputation_input_tokens == 4 &&
            step.cache_aware_input_tokens == 4 &&
            step.cached_tokens_before == 0 &&
            step.cached_tokens_after == 4 &&
            step.query_tokens == 4 &&
            step.score_key_tokens == 4,
        "single prediction was not prefill-only"
    );
    require(
        plan.full_recomputation
                .model_input_token_positions == 4 &&
            plan.cache_aware
                .model_input_token_positions == 4 &&
            plan.full_recomputation
                .attention_score_cells_per_head == 16 &&
            plan.cache_aware
                .attention_score_cells_per_head == 16,
        "single-prefill totals are wrong"
    );
}

void test_small_plan_formulas() {
    const GenerationPhasePlan plan =
        minitensor::inference::plan_generation_phases(
            1,
            3
        );

    require(
        plan.full_recomputation
                .model_input_token_positions == 6 &&
            plan.cache_aware
                .model_input_token_positions == 3,
        "P=1, G=3 token totals are wrong"
    );
    require(
        plan.full_recomputation
                .attention_score_cells_per_head == 14 &&
            plan.cache_aware
                .attention_score_cells_per_head == 6,
        "P=1, G=3 attention totals are wrong"
    );
    require(
        plan.steps[1].start_position == 1 &&
            plan.steps[2].start_position == 2,
        "decode start positions are wrong"
    );
}

void test_phase_names() {
    require(
        std::string_view(
            minitensor::inference::phase_name(
                InferencePhase::Prefill
            )
        ) == "prefill",
        "wrong prefill name"
    );
    require(
        std::string_view(
            minitensor::inference::phase_name(
                InferencePhase::Decode
            )
        ) == "decode",
        "wrong decode name"
    );
}

void test_invalid_and_overflowing_plans() {
    require_throws<std::invalid_argument>(
        [] {
            static_cast<void>(
                minitensor::inference::
                    plan_generation_phases(0, 1)
            );
        },
        "at least one prompt token"
    );

    require_throws<std::overflow_error>(
        [] {
            static_cast<void>(
                minitensor::inference::
                    plan_generation_phases(
                        std::numeric_limits<
                            std::size_t
                        >::max(),
                        1
                    )
            );
        },
        "attention work"
    );
}

} // namespace

int main() {
    try {
        test_hello_eight_prediction_plan();
        test_zero_predictions_do_no_work();
        test_one_prediction_is_prefill_only();
        test_small_plan_formulas();
        test_phase_names();
        test_invalid_and_overflowing_plans();

        std::cout
            << "Lesson 37 prefill/decode tests passed.\n";
    } catch (const std::exception& error) {
        std::cerr
            << "test failure: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
