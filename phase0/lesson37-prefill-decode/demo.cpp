#include "phase_plan.hpp"

#include "../lesson36-autoregressive-generation/generation.hpp"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

[[nodiscard]] std::size_t parse_size(
    std::string_view text,
    std::string_view name
) {
    std::size_t value = 0;
    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const auto [next, error] =
        std::from_chars(begin, end, value);

    if (
        error != std::errc{} ||
        next != end
    ) {
        throw std::invalid_argument(
            std::string(name) +
            " must be a nonnegative decimal integer"
        );
    }

    return value;
}

[[nodiscard]] std::uint32_t parse_token_id(
    std::string_view text
) {
    const std::size_t value =
        parse_size(text, "token ID");

    if (
        value >
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()
        )
    ) {
        throw std::out_of_range(
            "token ID does not fit in uint32_t"
        );
    }

    return static_cast<std::uint32_t>(value);
}

void write_token_ids(
    const std::filesystem::path& output_path,
    const std::vector<std::uint32_t>& token_ids
) {
    std::ofstream output(output_path);

    if (!output) {
        throw std::runtime_error(
            "could not open output token-ID file: " +
            output_path.string()
        );
    }

    for (
        std::size_t index = 0;
        index < token_ids.size();
        ++index
    ) {
        if (index != 0) {
            output << ' ';
        }

        output << token_ids[index];
    }

    output << '\n';

    if (!output) {
        throw std::runtime_error(
            "could not write output token-ID file: " +
            output_path.string()
        );
    }
}

void print_token_ids(
    const std::vector<std::uint32_t>& token_ids
) {
    std::cout << '[';

    for (
        std::size_t index = 0;
        index < token_ids.size();
        ++index
    ) {
        if (index != 0) {
            std::cout << ", ";
        }

        std::cout << token_ids[index];
    }

    std::cout << ']';
}

void validate_plan_matches_execution(
    const minitensor::hf::GenerationResult& result,
    const minitensor::inference::GenerationPhasePlan& plan
) {
    if (
        plan.prompt_tokens !=
            result.stats.prompt_token_count ||
        plan.prediction_count !=
            result.generated_token_ids.size() ||
        plan.prediction_count !=
            result.stats.generated_token_count ||
        plan.steps.size() !=
            result.generated_token_ids.size() ||
        plan.prediction_count !=
            result.stats.forward_pass_count ||
        plan.full_recomputation
                .model_input_token_positions !=
            result.stats.model_input_token_count
    ) {
        throw std::logic_error(
            "Lesson 37 phase plan disagrees with the "
            "Lesson 36 execution statistics"
        );
    }

    if (
        plan.cache_aware.model_input_token_positions >
            plan.full_recomputation
                .model_input_token_positions ||
        plan.cache_aware
                .attention_score_cells_per_head >
            plan.full_recomputation
                .attention_score_cells_per_head
    ) {
        throw std::logic_error(
            "cache-aware plan unexpectedly exceeds "
            "full-recomputation work"
        );
    }
}

void print_phase_table(
    const minitensor::hf::GenerationResult& result,
    const minitensor::inference::GenerationPhasePlan& plan
) {
    std::cout
        << "\nExecution comparison\n"
        << "IMPORTANT: Lesson 37 still executes every prediction by "
           "rerunning the full prefix.\n"
        << "The phase and cache-aware columns describe the future KV-cache "
           "schedule only.\n\n";

    std::cout
        << std::left
        << std::setw(12) << "prediction"
        << std::setw(12) << "token ID"
        << std::setw(23) << "current full input"
        << std::setw(21) << "future phase"
        << std::setw(14) << "future input"
        << std::setw(14) << "start pos"
        << "score keys\n";

    std::cout
        << std::setw(12) << "----------"
        << std::setw(12) << "--------"
        << std::setw(23) << "------------------"
        << std::setw(21) << "------------"
        << std::setw(14) << "------------"
        << std::setw(14) << "---------"
        << "------------\n";

    for (
        std::size_t index = 0;
        index < plan.steps.size();
        ++index
    ) {
        const minitensor::inference::PredictionStep& step =
            plan.steps[index];

        std::string future_phase =
            minitensor::inference::phase_name(
                step.cache_aware_phase
            );

        if (
            step.cache_aware_phase ==
            minitensor::inference::InferencePhase::Decode
        ) {
            future_phase += " (planned)";
        }

        std::cout
            << std::left
            << std::setw(12)
            << (step.prediction_index + 1)
            << std::setw(12)
            << result.generated_token_ids[index]
            << std::setw(23)
            << step.full_recomputation_input_tokens
            << std::setw(21)
            << future_phase
            << std::setw(14)
            << step.cache_aware_input_tokens
            << std::setw(14)
            << step.start_position
            << step.score_key_tokens
            << '\n';
    }

    if (plan.steps.empty()) {
        std::cout
            << "(no prediction requested, so neither prefill nor "
               "decode executed)\n";
    }
}

void print_totals(
    const minitensor::inference::GenerationPhasePlan& plan
) {
    const std::size_t avoided_token_positions =
        plan.full_recomputation
            .model_input_token_positions -
        plan.cache_aware
            .model_input_token_positions;
    const std::size_t avoided_attention_cells =
        plan.full_recomputation
            .attention_score_cells_per_head -
        plan.cache_aware
            .attention_score_cells_per_head;

    std::cout
        << "\nTotals\n"
        << "Current execution (full-prefix recomputation):\n"
        << "  model-input token positions: "
        << plan.full_recomputation
               .model_input_token_positions
        << '\n'
        << "  attention score cells per head per layer: "
        << plan.full_recomputation
               .attention_score_cells_per_head
        << '\n'
        << "Future cache-aware plan (not executed in Lesson 37):\n"
        << "  model-input token positions: "
        << plan.cache_aware
               .model_input_token_positions
        << '\n'
        << "  attention score cells per head per layer: "
        << plan.cache_aware
               .attention_score_cells_per_head
        << '\n'
        << "Work a future KV cache can avoid:\n"
        << "  repeated model-input token positions: "
        << avoided_token_positions
        << '\n'
        << "  repeated attention score cells per head per layer: "
        << avoided_attention_cells
        << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 6) {
            std::cerr
                << "usage: lesson37_prefill_decode "
                   "<model-directory> <greedy|sample> "
                   "<max-new-tokens> <output-ids-file> "
                   "<prompt-token-id> [prompt-token-id ...]\n";
            return 2;
        }

        const std::filesystem::path model_directory =
            argv[1];
        const std::string_view mode = argv[2];
        const std::size_t max_new_tokens =
            parse_size(argv[3], "max-new-tokens");
        const std::filesystem::path output_path =
            argv[4];

        minitensor::hf::GenerationConfig config;
        config.max_new_tokens = max_new_tokens;

        if (mode == "greedy") {
            config.do_sample = false;
        } else if (mode == "sample") {
            config.do_sample = true;
            config.temperature = 0.8f;
            config.top_k = 20;
            config.seed = 1234;
        } else {
            throw std::invalid_argument(
                "mode must be either 'greedy' or 'sample'"
            );
        }

        std::vector<std::uint32_t> prompt_token_ids;
        prompt_token_ids.reserve(
            static_cast<std::size_t>(argc - 5)
        );

        for (int index = 5; index < argc; ++index) {
            prompt_token_ids.push_back(
                parse_token_id(argv[index])
            );
        }

        minigpu::Device device;
        auto queue = device.create_queue();
        minitensor::hf::HuggingFaceLlama model =
            minitensor::hf::HuggingFaceLlama::load(
                device,
                model_directory
            );
        const minitensor::hf::AutoregressiveGenerator
            generator(model);
        const minitensor::hf::GenerationResult result =
            generator.generate(
                *queue,
                prompt_token_ids,
                config
            );

        const minitensor::inference::GenerationPhasePlan plan =
            minitensor::inference::plan_generation_phases(
                prompt_token_ids.size(),
                result.generated_token_ids.size()
            );
        validate_plan_matches_execution(result, plan);

        write_token_ids(output_path, result.token_ids);

        std::cout
            << "mode: " << mode << '\n'
            << "generated token IDs: ";
        print_token_ids(result.generated_token_ids);
        std::cout
            << '\n'
            << "stopped on EOS: "
            << (result.stopped_on_eos ? "yes" : "no")
            << '\n'
            << "actual forward passes: "
            << result.stats.forward_pass_count
            << '\n'
            << "generation time: "
            << result.stats.elapsed_milliseconds
            << " ms\n";

        print_phase_table(result, plan);
        print_totals(plan);

        std::cout
            << "\nLesson boundary\n"
            << "Lesson 37 has no KV cache. It only makes the "
               "prefill/decode schedule explicit.\n"
            << "A real cache will be designed in Lesson 38 and "
               "executed in Lesson 39.\n"
            << "wrote full token sequence: "
            << output_path << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
