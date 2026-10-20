#include "generation.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using minitensor::hf::AutoregressiveGenerator;
using minitensor::hf::GenerationConfig;
using minitensor::hf::GenerationResult;
using minitensor::hf::HuggingFaceLlama;

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

void test_zero_new_tokens(
    minigpu::Queue& queue,
    const AutoregressiveGenerator& generator
) {
    const std::vector<std::uint32_t> prompt{
        1,
        15043,
    };
    GenerationConfig config;
    config.max_new_tokens = 0;

    const GenerationResult result =
        generator.generate(queue, prompt, config);

    require(
        result.token_ids == prompt,
        "zero-token generation changed the prompt"
    );
    require(
        result.generated_token_ids.empty(),
        "zero-token generation produced a token"
    );
    require(
        result.stats.forward_pass_count == 0 &&
            result.stats.model_input_token_count == 0,
        "zero-token generation ran the model"
    );
}

void test_greedy_reference(
    minigpu::Queue& queue,
    const AutoregressiveGenerator& generator
) {
    const std::vector<std::uint32_t> prompt{
        1,
        15043,
    };
    const std::vector<std::uint32_t> expected_generated{
        1789,
        19053,
        25470,
        2323,
        21742,
        13486,
    };
    std::vector<std::uint32_t> expected_full = prompt;
    expected_full.insert(
        expected_full.end(),
        expected_generated.begin(),
        expected_generated.end()
    );

    GenerationConfig config;
    config.max_new_tokens = expected_generated.size();
    const GenerationResult result =
        generator.generate(queue, prompt, config);

    require(
        result.generated_token_ids == expected_generated,
        "greedy IDs differ from Hugging Face"
    );
    require(
        result.token_ids == expected_full,
        "full greedy sequence is wrong"
    );
    require(
        !result.stopped_on_eos,
        "greedy generation unexpectedly stopped on EOS"
    );
    require(
        result.stats.prompt_token_count == 2 &&
            result.stats.generated_token_count == 6 &&
            result.stats.forward_pass_count == 6,
        "greedy generation counters are wrong"
    );
    require(
        result.stats.model_input_token_count == 27,
        "expected recomputation lengths 2+3+4+5+6+7"
    );
}

void test_eos_override(
    minigpu::Queue& queue,
    const AutoregressiveGenerator& generator
) {
    GenerationConfig config;
    config.max_new_tokens = 5;
    config.eos_token_id = 1789;

    const GenerationResult result =
        generator.generate(
            queue,
            {1, 15043},
            config
        );

    require(
        result.generated_token_ids ==
            std::vector<std::uint32_t>{1789},
        "EOS override did not stop at the first token"
    );
    require(
        result.stopped_on_eos,
        "EOS stop was not reported"
    );
    require(
        result.stats.forward_pass_count == 1 &&
            result.stats.model_input_token_count == 2,
        "EOS stop counters are wrong"
    );
}

void test_seeded_sampling_is_repeatable(
    minigpu::Queue& queue,
    const AutoregressiveGenerator& generator
) {
    GenerationConfig config;
    config.max_new_tokens = 4;
    config.do_sample = true;
    config.temperature = 0.8f;
    config.top_k = 20;
    config.seed = 1234;
    config.stop_on_eos = false;

    const GenerationResult first =
        generator.generate(
            queue,
            {1, 15043},
            config
        );
    const GenerationResult second =
        generator.generate(
            queue,
            {1, 15043},
            config
        );

    require(
        first.generated_token_ids ==
            second.generated_token_ids,
        "same sampling seed produced different IDs"
    );
    require(
        first.generated_token_ids.size() == 4,
        "sampling produced the wrong token count"
    );
    require(
        first.stats.model_input_token_count == 14 &&
            second.stats.model_input_token_count == 14,
        "sampling recomputation counters are wrong"
    );
}

void test_top_one_sampling_equals_greedy(
    minigpu::Queue& queue,
    const AutoregressiveGenerator& generator
) {
    GenerationConfig config;
    config.max_new_tokens = 4;
    config.do_sample = true;
    config.temperature = 5.0f;
    config.top_k = 1;
    config.seed = 999999;
    config.stop_on_eos = false;

    const GenerationResult result =
        generator.generate(
            queue,
            {1, 15043},
            config
        );

    require(
        result.generated_token_ids ==
            std::vector<std::uint32_t>{
                1789,
                19053,
                25470,
                2323,
            },
        "top_k=1 did not keep only the greedy token"
    );
}

void test_tiny_temperature_is_stable(
    minigpu::Queue& queue,
    const AutoregressiveGenerator& generator
) {
    GenerationConfig config;
    config.max_new_tokens = 1;
    config.do_sample = true;
    config.temperature =
        std::numeric_limits<float>::min();
    config.top_k = 20;
    config.seed = 7;
    config.stop_on_eos = false;

    const GenerationResult result =
        generator.generate(
            queue,
            {1, 15043},
            config
        );

    require(
        result.generated_token_ids ==
            std::vector<std::uint32_t>{1789},
        "tiny-temperature sampling was not stable"
    );
}

void test_request_validation(
    minigpu::Queue& queue,
    const AutoregressiveGenerator& generator,
    std::size_t vocabulary_size
) {
    require_throws<std::invalid_argument>(
        [&] {
            (void)generator.generate(queue, {});
        },
        "at least one prompt token"
    );

    require_throws<std::out_of_range>(
        [&] {
            (void)generator.generate(
                queue,
                {
                    static_cast<std::uint32_t>(
                        vocabulary_size
                    ),
                }
            );
        },
        "outside the model vocabulary"
    );

    GenerationConfig bad_temperature;
    bad_temperature.do_sample = true;
    bad_temperature.temperature = 0.0f;
    require_throws<std::invalid_argument>(
        [&] {
            (void)generator.generate(
                queue,
                {1},
                bad_temperature
            );
        },
        "temperature"
    );

    GenerationConfig bad_top_k;
    bad_top_k.do_sample = true;
    bad_top_k.top_k = vocabulary_size + 1;
    require_throws<std::invalid_argument>(
        [&] {
            (void)generator.generate(
                queue,
                {1},
                bad_top_k
            );
        },
        "top_k"
    );
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            std::cerr
                << "usage: lesson36_tests "
                   "<pinned-tiny-model-directory>\n";
            return 2;
        }

        minigpu::Device device;
        auto queue = device.create_queue();
        const HuggingFaceLlama model =
            HuggingFaceLlama::load(
                device,
                std::filesystem::path(argv[1])
            );
        const AutoregressiveGenerator generator(model);

        test_zero_new_tokens(*queue, generator);
        test_greedy_reference(*queue, generator);
        test_eos_override(*queue, generator);
        test_seeded_sampling_is_repeatable(
            *queue,
            generator
        );
        test_top_one_sampling_equals_greedy(
            *queue,
            generator
        );
        test_tiny_temperature_is_stable(
            *queue,
            generator
        );
        test_request_validation(
            *queue,
            generator,
            model.config().vocabulary_size
        );

        std::cout
            << "Lesson 36 generation tests passed.\n";
    } catch (const std::exception& error) {
        std::cerr << "test failure: "
                  << error.what() << '\n';
        return 1;
    }

    return 0;
}
