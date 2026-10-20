#include "cached_generation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using minitensor::RuntimeTensor;
using minitensor::Shape;
using minitensor::hf::AutoregressiveGenerator;
using minitensor::hf::CachedAutoregressiveGenerator;
using minitensor::hf::CachedGenerationResult;
using minitensor::hf::GenerationConfig;
using minitensor::hf::GenerationResult;
using minitensor::hf::HuggingFaceLlama;
using minitensor::inference::CachedAttention;
using minitensor::inference::CachedLlamaExecutor;
using minitensor::inference::KVCache;
using Values = std::vector<float>;

constexpr float synthetic_absolute_tolerance = 0.0001f;
constexpr float model_absolute_tolerance = 0.001f;
constexpr float model_relative_tolerance = 0.001f;

static_assert(
    !std::is_constructible_v<
        CachedLlamaExecutor,
        HuggingFaceLlama&&
    >
);
static_assert(
    !std::is_constructible_v<
        CachedLlamaExecutor,
        const HuggingFaceLlama&&
    >
);
static_assert(
    !std::is_constructible_v<
        CachedAutoregressiveGenerator,
        CachedLlamaExecutor&&
    >
);
static_assert(
    !std::is_constructible_v<
        CachedAutoregressiveGenerator,
        const CachedLlamaExecutor&&
    >
);

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

template <typename Function>
void run_named_test(
    std::string_view name,
    Function&& function
) {
    try {
        std::forward<Function>(function)();
    } catch (const std::exception& error) {
        throw std::runtime_error(
            std::string(name) + ": " + error.what()
        );
    }
}

[[nodiscard]] RuntimeTensor make_tensor(
    const minigpu::Device& device,
    Shape shape,
    Values values
) {
    return RuntimeTensor::from_host(
        device,
        miniir::Type::tensor(
            miniir::DType::F32,
            std::move(shape)
        ),
        std::move(values)
    );
}

void require_all_close(
    const Values& actual,
    const Values& expected,
    float absolute_tolerance,
    float relative_tolerance,
    std::string_view description
) {
    require(
        actual.size() == expected.size(),
        std::string(description) +
            " has the wrong element count"
    );

    for (
        std::size_t index = 0;
        index < actual.size();
        ++index
    ) {
        const float scale = std::max(
            std::fabs(actual[index]),
            std::fabs(expected[index])
        );
        const float allowed =
            absolute_tolerance +
            relative_tolerance * scale;
        const float difference =
            std::fabs(actual[index] - expected[index]);

        if (
            !std::isfinite(actual[index]) ||
            difference > allowed
        ) {
            throw std::runtime_error(
                std::string(description) +
                " differs at element " +
                std::to_string(index) +
                ": actual=" +
                std::to_string(actual[index]) +
                ", expected=" +
                std::to_string(expected[index]) +
                ", difference=" +
                std::to_string(difference) +
                ", allowed=" +
                std::to_string(allowed)
            );
        }
    }
}

[[nodiscard]] Values cpu_cached_attention(
    const Values& queries,
    const Values& keys,
    const Values& values,
    std::size_t heads,
    std::size_t query_tokens,
    std::size_t capacity,
    std::size_t head_dimension,
    std::size_t valid_length,
    std::size_t query_start
) {
    require(
        queries.size() ==
            heads * query_tokens * head_dimension,
        "CPU attention received the wrong query size"
    );
    require(
        keys.size() ==
                heads * capacity * head_dimension &&
            values.size() == keys.size(),
        "CPU attention received the wrong cache size"
    );
    require(
        query_start + query_tokens == valid_length,
        "CPU attention requires the current chunk at the cache end"
    );

    Values output(
        heads * query_tokens * head_dimension,
        0.0f
    );
    const double scale =
        1.0 /
        std::sqrt(static_cast<double>(head_dimension));

    for (
        std::size_t head = 0;
        head < heads;
        ++head
    ) {
        for (
            std::size_t query = 0;
            query < query_tokens;
            ++query
        ) {
            const std::size_t absolute_query =
                query_start + query;
            std::vector<double> scores(
                absolute_query + 1,
                0.0
            );
            double maximum =
                -std::numeric_limits<double>::infinity();

            for (
                std::size_t key = 0;
                key <= absolute_query;
                ++key
            ) {
                double dot = 0.0;

                for (
                    std::size_t component = 0;
                    component < head_dimension;
                    ++component
                ) {
                    const std::size_t query_index =
                        (
                            head * query_tokens + query
                        ) * head_dimension +
                        component;
                    const std::size_t key_index =
                        (
                            head * capacity + key
                        ) * head_dimension +
                        component;
                    dot +=
                        static_cast<double>(
                            queries[query_index]
                        ) *
                        static_cast<double>(keys[key_index]);
                }

                scores[key] = dot * scale;
                maximum = std::max(maximum, scores[key]);
            }

            double denominator = 0.0;

            for (
                std::size_t key = 0;
                key <= absolute_query;
                ++key
            ) {
                scores[key] =
                    std::exp(scores[key] - maximum);
                denominator += scores[key];
            }

            for (
                std::size_t component = 0;
                component < head_dimension;
                ++component
            ) {
                double context = 0.0;

                for (
                    std::size_t key = 0;
                    key <= absolute_query;
                    ++key
                ) {
                    const std::size_t value_index =
                        (
                            head * capacity + key
                        ) * head_dimension +
                        component;
                    context +=
                        (scores[key] / denominator) *
                        static_cast<double>(
                            values[value_index]
                        );
                }

                const std::size_t output_index =
                    (
                        head * query_tokens + query
                    ) * head_dimension +
                    component;
                output[output_index] =
                    static_cast<float>(context);
            }
        }
    }

    return output;
}

[[nodiscard]] std::size_t final_row_offset(
    const RuntimeTensor& logits
) {
    require(
        logits.rank() == 2 && logits.shape()[0] > 0,
        "logits must have shape [tokens, vocabulary]"
    );

    return
        (logits.shape()[0] - 1) *
        logits.shape()[1];
}

[[nodiscard]] Values final_row(
    const RuntimeTensor& logits
) {
    const Values all = logits.to_host();
    const std::size_t offset = final_row_offset(logits);

    return Values(
        all.begin() +
            static_cast<std::ptrdiff_t>(offset),
        all.end()
    );
}

[[nodiscard]] std::uint32_t greedy_token(
    const RuntimeTensor& logits
) {
    const Values row = final_row(logits);
    const auto best = std::max_element(
        row.begin(),
        row.end()
    );
    const std::size_t token_id =
        static_cast<std::size_t>(
            std::distance(row.begin(), best)
        );

    require(
        token_id <=
            static_cast<std::size_t>(
                std::numeric_limits<std::uint32_t>::max()
            ),
        "greedy token ID does not fit uint32_t"
    );

    return static_cast<std::uint32_t>(token_id);
}

void test_synthetic_prefill_attention(
    const minigpu::Device& device,
    minigpu::Queue& queue
) {
    constexpr std::size_t heads = 2;
    constexpr std::size_t query_tokens = 3;
    constexpr std::size_t capacity = 5;
    constexpr std::size_t head_dimension = 2;
    constexpr std::size_t valid_length = 3;
    constexpr std::size_t query_start = 0;

    const Values queries{
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,

        1.0f, -1.0f,
        2.0f, 0.0f,
        0.0f, 2.0f,
    };
    const Values keys{
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
        1000.0f, 1000.0f,
        2000.0f, 2000.0f,

        1.0f, 1.0f,
        1.0f, -1.0f,
        0.0f, 1.0f,
        3000.0f, 3000.0f,
        4000.0f, 4000.0f,
    };
    const Values values{
        1.0f, 10.0f,
        2.0f, 20.0f,
        4.0f, 40.0f,
        10000.0f, 10000.0f,
        20000.0f, 20000.0f,

        3.0f, 30.0f,
        6.0f, 60.0f,
        9.0f, 90.0f,
        30000.0f, 30000.0f,
        40000.0f, 40000.0f,
    };

    const RuntimeTensor query_tensor = make_tensor(
        device,
        {heads, query_tokens, head_dimension},
        queries
    );
    const RuntimeTensor key_tensor = make_tensor(
        device,
        {heads, capacity, head_dimension},
        keys
    );
    const RuntimeTensor value_tensor = make_tensor(
        device,
        {heads, capacity, head_dimension},
        values
    );

    const CachedAttention attention(device);
    const RuntimeTensor actual = attention.run(
        queue,
        query_tensor,
        key_tensor,
        value_tensor,
        valid_length,
        query_start
    );
    const Values expected = cpu_cached_attention(
        queries,
        keys,
        values,
        heads,
        query_tokens,
        capacity,
        head_dimension,
        valid_length,
        query_start
    );

    require(
        actual.shape() ==
            Shape{heads, query_tokens, head_dimension},
        "prefill attention returned the wrong shape"
    );
    require_all_close(
        actual.to_host(),
        expected,
        synthetic_absolute_tolerance,
        synthetic_absolute_tolerance,
        "synthetic prefill attention"
    );
}

void test_synthetic_one_token_decode_attention(
    const minigpu::Device& device,
    minigpu::Queue& queue
) {
    constexpr std::size_t heads = 2;
    constexpr std::size_t query_tokens = 1;
    constexpr std::size_t capacity = 5;
    constexpr std::size_t head_dimension = 2;
    constexpr std::size_t valid_length = 4;
    constexpr std::size_t query_start = 3;

    const Values queries{
        0.5f, 1.0f,
        -1.0f, 0.25f,
    };
    const Values keys{
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
        -1.0f, 0.5f,
        9000.0f, 9000.0f,

        1.0f, 1.0f,
        1.0f, -1.0f,
        0.0f, 1.0f,
        -0.5f, 2.0f,
        8000.0f, 8000.0f,
    };
    const Values values{
        1.0f, 10.0f,
        2.0f, 20.0f,
        4.0f, 40.0f,
        8.0f, 80.0f,
        90000.0f, 90000.0f,

        3.0f, 30.0f,
        6.0f, 60.0f,
        9.0f, 90.0f,
        12.0f, 120.0f,
        80000.0f, 80000.0f,
    };

    const RuntimeTensor query_tensor = make_tensor(
        device,
        {heads, query_tokens, head_dimension},
        queries
    );
    const RuntimeTensor key_tensor = make_tensor(
        device,
        {heads, capacity, head_dimension},
        keys
    );
    const RuntimeTensor value_tensor = make_tensor(
        device,
        {heads, capacity, head_dimension},
        values
    );

    const CachedAttention attention(device);
    const RuntimeTensor actual = attention.run(
        queue,
        query_tensor,
        key_tensor,
        value_tensor,
        valid_length,
        query_start
    );
    const Values expected = cpu_cached_attention(
        queries,
        keys,
        values,
        heads,
        query_tokens,
        capacity,
        head_dimension,
        valid_length,
        query_start
    );

    require(
        actual.shape() ==
            Shape{heads, 1, head_dimension},
        "decode attention returned the wrong shape"
    );
    require_all_close(
        actual.to_host(),
        expected,
        synthetic_absolute_tolerance,
        synthetic_absolute_tolerance,
        "synthetic one-token decode attention"
    );
}

void test_cached_attention_validation(
    const minigpu::Device& device,
    minigpu::Queue& queue
) {
    const CachedAttention attention(device);
    const RuntimeTensor query = make_tensor(
        device,
        {1, 1, 2},
        {1.0f, 2.0f}
    );
    const RuntimeTensor keys = make_tensor(
        device,
        {1, 3, 2},
        {
            1.0f, 0.0f,
            0.0f, 1.0f,
            1.0f, 1.0f,
        }
    );
    const RuntimeTensor values = make_tensor(
        device,
        {1, 3, 2},
        {
            10.0f, 11.0f,
            20.0f, 21.0f,
            30.0f, 31.0f,
        }
    );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                attention.run(
                    queue,
                    query,
                    keys,
                    values,
                    2,
                    0
                )
            );
        },
        "query_start"
    );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                attention.run(
                    queue,
                    query,
                    keys,
                    values,
                    1,
                    0,
                    0
                )
            );
        },
        "threads_per_group"
    );

    const RuntimeTensor wrong_values = make_tensor(
        device,
        {1, 2, 2},
        {
            1.0f, 2.0f,
            3.0f, 4.0f,
        }
    );
    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                attention.run(
                    queue,
                    query,
                    keys,
                    wrong_values,
                    1,
                    0
                )
            );
        },
        "same shape"
    );
}

void test_prefill_matches_full_forward(
    minigpu::Queue& queue,
    const HuggingFaceLlama& model,
    const CachedLlamaExecutor& executor
) {
    const std::vector<std::uint32_t> prompt{
        1,
        15043,
    };
    KVCache cache = executor.create_cache(10);

    const RuntimeTensor baseline =
        model.forward(queue, prompt);
    const RuntimeTensor cached =
        executor.prefill(queue, prompt, cache);

    require(
        cached.shape() == baseline.shape(),
        "cached prefill logits have the wrong shape"
    );
    require_all_close(
        cached.to_host(),
        baseline.to_host(),
        model_absolute_tolerance,
        model_relative_tolerance,
        "cached prefill logits"
    );
    require(
        cache.has_uniform_length() &&
            cache.sequence_length() == prompt.size(),
        "prefill did not append every prompt token"
    );
}

void test_repeated_decode_matches_full_prefix(
    minigpu::Queue& queue,
    const HuggingFaceLlama& model,
    const CachedLlamaExecutor& executor
) {
    const std::vector<std::uint32_t> expected{
        1789,
        19053,
        25470,
        2323,
        21742,
        13486,
        13934,
        26715,
    };
    std::vector<std::uint32_t> sequence{
        1,
        15043,
    };
    std::vector<std::uint32_t> generated;
    generated.reserve(expected.size());
    KVCache cache = executor.create_cache(
        sequence.size() + expected.size()
    );

    RuntimeTensor cached_logits =
        executor.prefill(queue, sequence, cache);

    for (
        std::size_t step = 0;
        step < expected.size();
        ++step
    ) {
        const RuntimeTensor full_logits =
            model.forward(queue, sequence);

        require_all_close(
            final_row(cached_logits),
            final_row(full_logits),
            model_absolute_tolerance,
            model_relative_tolerance,
            "cached decode final-row logits at step " +
                std::to_string(step)
        );

        const std::uint32_t token =
            greedy_token(cached_logits);
        generated.push_back(token);
        sequence.push_back(token);

        if (step + 1 < expected.size()) {
            cached_logits = executor.decode_one(
                queue,
                token,
                cache
            );
        }
    }

    require(
        generated == expected,
        "cached greedy IDs differ from the exact reference"
    );
    require(
        cache.sequence_length() == 9,
        "eight predictions should leave nine consumed "
        "tokens in the cache"
    );
}

void test_cached_generation_stats_and_exact_ids(
    minigpu::Queue& queue,
    const HuggingFaceLlama& model,
    const CachedAutoregressiveGenerator& generator
) {
    GenerationConfig config;
    config.max_new_tokens = 8;

    const CachedGenerationResult result =
        generator.generate(
            queue,
            {1, 15043},
            config
        );
    const std::vector<std::uint32_t> expected{
        1789,
        19053,
        25470,
        2323,
        21742,
        13486,
        13934,
        26715,
    };

    require(
        result.generated_token_ids == expected,
        "cached generator produced the wrong greedy IDs"
    );
    require(
        result.stats.prompt_token_count == 2 &&
            result.stats.generated_token_count == 8,
        "cached generation token counters are wrong"
    );
    require(
        result.stats.prefill_pass_count == 1 &&
            result.stats.decode_pass_count == 7 &&
            result.stats.model_pass_count == 8,
        "expected one prefill plus seven decode passes"
    );
    require(
        result.stats.model_input_token_positions == 9,
        "cached generation should process 2+7 token positions"
    );
    require(
        result.stats.final_cache_length == 9 &&
            result.stats.cache_capacity == 10,
        "cached generation cache counters are wrong"
    );
    require(
        !result.stopped_on_eos,
        "reference generation unexpectedly stopped on EOS"
    );

    const AutoregressiveGenerator baseline_generator(model);
    const GenerationResult baseline =
        baseline_generator.generate(
            queue,
            {1, 15043},
            config
        );
    require(
        baseline.generated_token_ids == expected,
        "full-recomputation generator produced different IDs"
    );
    require(
        baseline.stats.model_input_token_count == 44,
        "full recomputation should process "
        "2+3+4+5+6+7+8+9 = 44 token positions"
    );
}

void test_zero_tokens_and_eos(
    minigpu::Queue& queue,
    const CachedAutoregressiveGenerator& generator
) {
    GenerationConfig zero_config;
    zero_config.max_new_tokens = 0;
    const CachedGenerationResult zero =
        generator.generate(
            queue,
            {1, 15043},
            zero_config
        );

    require(
        zero.token_ids ==
                std::vector<std::uint32_t>{1, 15043} &&
            zero.generated_token_ids.empty(),
        "zero-token cached generation changed the prompt"
    );
    require(
        zero.stats.prefill_pass_count == 0 &&
            zero.stats.decode_pass_count == 0 &&
            zero.stats.model_pass_count == 0 &&
            zero.stats.model_input_token_positions == 0 &&
            zero.stats.final_cache_length == 0 &&
            zero.stats.cache_capacity == 0,
        "zero-token cached generation performed work"
    );

    GenerationConfig eos_config;
    eos_config.max_new_tokens = 5;
    eos_config.eos_token_id = 1789;
    const CachedGenerationResult eos =
        generator.generate(
            queue,
            {1, 15043},
            eos_config
        );

    require(
        eos.generated_token_ids ==
                std::vector<std::uint32_t>{1789} &&
            eos.stopped_on_eos,
        "cached generation did not stop on EOS"
    );
    require(
        eos.stats.prefill_pass_count == 1 &&
            eos.stats.decode_pass_count == 0 &&
            eos.stats.model_pass_count == 1 &&
            eos.stats.model_input_token_positions == 2 &&
            eos.stats.final_cache_length == 2,
        "EOS cached generation counters are wrong"
    );
}

void test_executor_state_and_capacity_errors(
    minigpu::Queue& queue,
    const CachedLlamaExecutor& executor
) {
    require_throws<std::invalid_argument>(
        [&] {
            KVCache cache = executor.create_cache(4);
            static_cast<void>(
                executor.prefill(queue, {}, cache)
            );
        },
        "at least one"
    );

    require_throws<std::length_error>(
        [&] {
            KVCache cache = executor.create_cache(1);
            static_cast<void>(
                executor.prefill(
                    queue,
                    {1, 15043},
                    cache
                )
            );
        },
        "capacity"
    );

    require_throws<std::invalid_argument>(
        [&] {
            KVCache cache = executor.create_cache(4);
            static_cast<void>(
                executor.decode_one(queue, 1, cache)
            );
        },
        "empty"
    );

    require_throws<std::invalid_argument>(
        [&] {
            KVCache cache = executor.create_cache(4);
            static_cast<void>(
                executor.prefill(queue, {1}, cache)
            );
            static_cast<void>(
                executor.prefill(queue, {1}, cache)
            );
        },
        "empty"
    );

    require_throws<std::length_error>(
        [&] {
            KVCache cache = executor.create_cache(1);
            static_cast<void>(
                executor.prefill(queue, {1}, cache)
            );
            static_cast<void>(
                executor.decode_one(queue, 2, cache)
            );
        },
        "capacity"
    );

    require_throws<std::out_of_range>(
        [&] {
            KVCache cache = executor.create_cache(2);
            static_cast<void>(
                executor.prefill(
                    queue,
                    {
                        static_cast<std::uint32_t>(
                            executor.config().
                                vocabulary_size
                        ),
                    },
                    cache
                )
            );
        },
        "vocabulary"
    );
}

void test_generation_request_validation(
    minigpu::Queue& queue,
    const CachedAutoregressiveGenerator& generator,
    const minitensor::hf::LlamaConfig& model_config
) {
    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                generator.generate(queue, {})
            );
        },
        "at least one"
    );

    require_throws<std::out_of_range>(
        [&] {
            static_cast<void>(
                generator.generate(
                    queue,
                    {
                        static_cast<std::uint32_t>(
                            model_config.vocabulary_size
                        ),
                    }
                )
            );
        },
        "vocabulary"
    );

    GenerationConfig bad_temperature;
    bad_temperature.do_sample = true;
    bad_temperature.temperature = 0.0f;
    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                generator.generate(
                    queue,
                    {1},
                    bad_temperature
                )
            );
        },
        "temperature"
    );

    GenerationConfig bad_top_k;
    bad_top_k.do_sample = true;
    bad_top_k.top_k =
        model_config.vocabulary_size + 1;
    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                generator.generate(
                    queue,
                    {1},
                    bad_top_k
                )
            );
        },
        "top_k"
    );

    GenerationConfig too_long;
    too_long.max_new_tokens =
        model_config.maximum_position_count;
    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                generator.generate(
                    queue,
                    {1},
                    too_long
                )
            );
        },
        "max_position_embeddings"
    );
}

void test_seeded_sampling_matches_full_recomputation(
    minigpu::Queue& queue,
    const HuggingFaceLlama& model,
    const CachedAutoregressiveGenerator& cached_generator
) {
    GenerationConfig config;
    config.max_new_tokens = 4;
    config.do_sample = true;
    config.temperature = 0.8f;
    config.top_k = 20;
    config.seed = 1234;
    config.stop_on_eos = false;

    const AutoregressiveGenerator baseline_generator(model);
    const GenerationResult baseline =
        baseline_generator.generate(
            queue,
            {1, 15043},
            config
        );
    const CachedGenerationResult cached =
        cached_generator.generate(
            queue,
            {1, 15043},
            config
        );

    require(
        cached.generated_token_ids ==
            baseline.generated_token_ids,
        "seeded sampled generation differs between "
        "cached and full recomputation"
    );
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            std::cerr
                << "usage: lesson39_tests "
                   "<pinned-tiny-model-directory>\n";
            return 2;
        }

        minigpu::Device device;
        auto queue = device.create_queue();

        run_named_test(
            "synthetic prefill attention",
            [&] {
                test_synthetic_prefill_attention(
                    device,
                    *queue
                );
            }
        );
        run_named_test(
            "synthetic decode attention",
            [&] {
                test_synthetic_one_token_decode_attention(
                    device,
                    *queue
                );
            }
        );
        run_named_test(
            "cached attention validation",
            [&] {
                test_cached_attention_validation(
                    device,
                    *queue
                );
            }
        );

        const HuggingFaceLlama model =
            HuggingFaceLlama::load(
                device,
                std::filesystem::path(argv[1])
            );
        const CachedLlamaExecutor executor(model);
        const CachedAutoregressiveGenerator generator(
            executor
        );

        run_named_test(
            "prefill versus full forward",
            [&] {
                test_prefill_matches_full_forward(
                    *queue,
                    model,
                    executor
                );
            }
        );
        run_named_test(
            "repeated decode versus full prefix",
            [&] {
                test_repeated_decode_matches_full_prefix(
                    *queue,
                    model,
                    executor
                );
            }
        );
        run_named_test(
            "cached generation statistics",
            [&] {
                test_cached_generation_stats_and_exact_ids(
                    *queue,
                    model,
                    generator
                );
            }
        );
        run_named_test(
            "zero-token and EOS generation",
            [&] {
                test_zero_tokens_and_eos(
                    *queue,
                    generator
                );
            }
        );
        run_named_test(
            "executor state validation",
            [&] {
                test_executor_state_and_capacity_errors(
                    *queue,
                    executor
                );
            }
        );
        run_named_test(
            "generation request validation",
            [&] {
                test_generation_request_validation(
                    *queue,
                    generator,
                    model.config()
                );
            }
        );
        run_named_test(
            "seeded sampling parity",
            [&] {
                test_seeded_sampling_matches_full_recomputation(
                    *queue,
                    model,
                    generator
                );
            }
        );

        std::cout
            << "Lesson 39 cached-decoding tests passed.\n";
    } catch (const std::exception& error) {
        std::cerr << "test failure: "
                  << error.what() << '\n';
        return 1;
    }

    return 0;
}
