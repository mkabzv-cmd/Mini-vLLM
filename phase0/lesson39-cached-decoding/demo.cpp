#include "cached_generation.hpp"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
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

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 6) {
            std::cerr
                << "usage: lesson39_cached_generate "
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
        const minitensor::inference::CachedLlamaExecutor
            executor(model);
        const minitensor::hf::CachedAutoregressiveGenerator
            generator(executor);
        const minitensor::hf::CachedGenerationResult result =
            generator.generate(
                *queue,
                prompt_token_ids,
                config
            );

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
            << "prefill passes: "
            << result.stats.prefill_pass_count << '\n'
            << "one-token decode passes: "
            << result.stats.decode_pass_count << '\n'
            << "total model passes: "
            << result.stats.model_pass_count << '\n'
            << "model input token positions: "
            << result.stats.model_input_token_positions
            << '\n'
            << "final KV-cache length: "
            << result.stats.final_cache_length << '\n'
            << "KV-cache capacity: "
            << result.stats.cache_capacity << '\n'
            << "cache execution: real cached K/V attention "
               "(not a projected work estimate)\n"
            << "generation time: "
            << result.stats.elapsed_milliseconds
            << " ms\n"
            << "wrote full token sequence: "
            << output_path << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
