#include "cached_generation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace minitensor::hf {

namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] std::uint32_t checked_token_id(
    std::size_t token_id
) {
    if (
        token_id >
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()
        )
    ) {
        throw std::overflow_error(
            "selected token ID does not fit in uint32_t"
        );
    }

    return static_cast<std::uint32_t>(token_id);
}

void require_finite_logits(
    std::span<const float> logits
) {
    for (std::size_t index = 0; index < logits.size(); ++index) {
        if (!std::isfinite(logits[index])) {
            throw std::runtime_error(
                "cached generation received a non-finite logit at "
                "vocabulary index " +
                std::to_string(index)
            );
        }
    }
}

[[nodiscard]] std::uint32_t select_greedy(
    std::span<const float> logits
) {
    std::size_t best_index = 0;

    for (
        std::size_t index = 1;
        index < logits.size();
        ++index
    ) {
        // On an exact tie the lower token ID remains selected.
        if (logits[index] > logits[best_index]) {
            best_index = index;
        }
    }

    return checked_token_id(best_index);
}

struct Candidate {
    std::uint32_t token_id = 0;
    float logit = 0.0f;
};

[[nodiscard]] double next_unit_interval(
    std::mt19937_64& random
) {
    // Generate the same implementation-independent [0, 1) value used by
    // Lesson 36 instead of relying on uniform_real_distribution.
    constexpr double two_to_negative_53 =
        1.0 / 9007199254740992.0;

    return static_cast<double>(random() >> 11U) *
        two_to_negative_53;
}

[[nodiscard]] std::uint32_t select_sampled(
    std::span<const float> logits,
    const GenerationConfig& config,
    std::mt19937_64& random
) {
    std::vector<Candidate> candidates;
    candidates.reserve(logits.size());

    for (
        std::size_t index = 0;
        index < logits.size();
        ++index
    ) {
        candidates.push_back(
            Candidate{
                checked_token_id(index),
                logits[index],
            }
        );
    }

    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const Candidate& left, const Candidate& right) {
            if (left.logit != right.logit) {
                return left.logit > right.logit;
            }

            return left.token_id < right.token_id;
        }
    );

    if (
        config.top_k != 0 &&
        config.top_k < candidates.size()
    ) {
        candidates.resize(config.top_k);
    }

    const double temperature =
        static_cast<double>(config.temperature);
    const double maximum_logit =
        static_cast<double>(candidates.front().logit);

    std::vector<double> weights;
    weights.reserve(candidates.size());

    double weight_sum = 0.0;

    for (const Candidate& candidate : candidates) {
        const double weight = std::exp(
            (
                static_cast<double>(candidate.logit) -
                maximum_logit
            ) /
            temperature
        );

        weights.push_back(weight);
        weight_sum += weight;
    }

    if (
        !std::isfinite(weight_sum) ||
        weight_sum <= 0.0
    ) {
        throw std::runtime_error(
            "temperature sampling produced invalid "
            "probability weights"
        );
    }

    const double target =
        next_unit_interval(random) * weight_sum;
    double cumulative = 0.0;

    for (
        std::size_t index = 0;
        index < candidates.size();
        ++index
    ) {
        cumulative += weights[index];

        if (target < cumulative) {
            return candidates[index].token_id;
        }
    }

    return candidates.back().token_id;
}

void checked_add(
    std::size_t value,
    std::size_t& total,
    const char* description
) {
    if (
        value >
        std::numeric_limits<std::size_t>::max() - total
    ) {
        throw std::overflow_error(
            std::string(description) + " overflowed size_t"
        );
    }

    total += value;
}

[[nodiscard]] std::optional<std::uint32_t>
resolve_eos_token_id(
    const LlamaConfig& model_config,
    const GenerationConfig& generation_config
) {
    if (!generation_config.stop_on_eos) {
        return std::nullopt;
    }

    if (generation_config.eos_token_id.has_value()) {
        return generation_config.eos_token_id;
    }

    if (
        model_config.eos_token_id < 0 ||
        static_cast<std::uint64_t>(
            model_config.eos_token_id
        ) >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::uint32_t>::max()
            )
    ) {
        throw std::invalid_argument(
            "model EOS token ID cannot be represented as "
            "uint32_t"
        );
    }

    return static_cast<std::uint32_t>(
        model_config.eos_token_id
    );
}

void validate_request(
    const LlamaConfig& model_config,
    const std::vector<std::uint32_t>& prompt_token_ids,
    const GenerationConfig& generation_config
) {
    if (prompt_token_ids.empty()) {
        throw std::invalid_argument(
            "cached generation requires at least one prompt token"
        );
    }

    if (
        model_config.vocabulary_size == 0 ||
        model_config.vocabulary_size >
            static_cast<std::size_t>(
                std::numeric_limits<std::uint32_t>::max()
            ) + std::size_t{1}
    ) {
        throw std::invalid_argument(
            "model vocabulary cannot be addressed by "
            "uint32_t token IDs"
        );
    }

    if (
        prompt_token_ids.size() >
        model_config.maximum_position_count
    ) {
        throw std::invalid_argument(
            "prompt length exceeds max_position_embeddings"
        );
    }

    const std::size_t remaining_positions =
        model_config.maximum_position_count -
        prompt_token_ids.size();

    if (
        generation_config.max_new_tokens >
        remaining_positions
    ) {
        throw std::invalid_argument(
            "prompt length plus max_new_tokens exceeds "
            "max_position_embeddings"
        );
    }

    for (std::uint32_t token_id : prompt_token_ids) {
        if (
            static_cast<std::size_t>(token_id) >=
            model_config.vocabulary_size
        ) {
            throw std::out_of_range(
                "prompt token ID " +
                std::to_string(token_id) +
                " is outside the model vocabulary"
            );
        }
    }

    if (
        generation_config.top_k >
        model_config.vocabulary_size
    ) {
        throw std::invalid_argument(
            "top_k cannot exceed the vocabulary size"
        );
    }

    if (
        generation_config.do_sample &&
        (
            !std::isfinite(generation_config.temperature) ||
            generation_config.temperature <= 0.0f
        )
    ) {
        throw std::invalid_argument(
            "sampling temperature must be finite and "
            "greater than zero"
        );
    }

    if (
        generation_config.eos_token_id.has_value() &&
        static_cast<std::size_t>(
            *generation_config.eos_token_id
        ) >= model_config.vocabulary_size
    ) {
        throw std::invalid_argument(
            "EOS token ID is outside the model vocabulary"
        );
    }

    const std::optional<std::uint32_t> eos_token_id =
        resolve_eos_token_id(
            model_config,
            generation_config
        );

    if (
        eos_token_id.has_value() &&
        static_cast<std::size_t>(*eos_token_id) >=
            model_config.vocabulary_size
    ) {
        throw std::invalid_argument(
            "model EOS token ID is outside the vocabulary"
        );
    }
}

[[nodiscard]] std::span<const float> final_logits(
    const RuntimeTensor& logits,
    const std::vector<float>& host_logits,
    std::size_t sequence_length,
    std::size_t vocabulary_size
) {
    if (
        logits.shape() !=
        Shape{
            sequence_length,
            vocabulary_size,
        }
    ) {
        throw std::runtime_error(
            "cached model returned an unexpected logits shape"
        );
    }

    if (host_logits.size() != logits.numel()) {
        throw std::runtime_error(
            "cached model logits host copy has an unexpected size"
        );
    }

    const std::size_t last_row_offset =
        (sequence_length - 1) * vocabulary_size;

    return std::span<const float>(
        host_logits.data() + last_row_offset,
        vocabulary_size
    );
}

[[nodiscard]] std::uint32_t select_next_token(
    const RuntimeTensor& logits,
    std::size_t sequence_length,
    std::size_t vocabulary_size,
    const GenerationConfig& config,
    std::mt19937_64& random
) {
    const std::vector<float> host_logits =
        logits.to_host();
    const std::span<const float> last_logits =
        final_logits(
            logits,
            host_logits,
            sequence_length,
            vocabulary_size
        );

    require_finite_logits(last_logits);

    return config.do_sample
        ? select_sampled(last_logits, config, random)
        : select_greedy(last_logits);
}

} // namespace

CachedAutoregressiveGenerator::CachedAutoregressiveGenerator(
    const inference::CachedLlamaExecutor& executor
)
    : executor_(executor) {}

CachedGenerationResult CachedAutoregressiveGenerator::generate(
    minigpu::Queue& queue,
    const std::vector<std::uint32_t>& prompt_token_ids,
    const GenerationConfig& config
) const {
    const LlamaConfig& model_config = executor_.config();
    validate_request(
        model_config,
        prompt_token_ids,
        config
    );

    CachedGenerationResult result;
    result.token_ids = prompt_token_ids;
    result.token_ids.reserve(
        prompt_token_ids.size() + config.max_new_tokens
    );
    result.generated_token_ids.reserve(
        config.max_new_tokens
    );
    result.stats.prompt_token_count =
        prompt_token_ids.size();

    const Clock::time_point start = Clock::now();

    // There is nothing to predict, so do not allocate a cache and do not call
    // either model entry point.
    if (config.max_new_tokens == 0) {
        const Clock::time_point end = Clock::now();
        result.stats.elapsed_milliseconds =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();
        return result;
    }

    const std::size_t cache_capacity =
        prompt_token_ids.size() + config.max_new_tokens;
    inference::KVCache cache =
        executor_.create_cache(cache_capacity);
    result.stats.cache_capacity =
        cache.config().capacity;

    const std::optional<std::uint32_t> eos_token_id =
        resolve_eos_token_id(model_config, config);
    std::mt19937_64 random(config.seed);

    RuntimeTensor logits = executor_.prefill(
        queue,
        prompt_token_ids,
        cache
    );
    std::uint32_t next_token_id = select_next_token(
        logits,
        prompt_token_ids.size(),
        model_config.vocabulary_size,
        config,
        random
    );

    result.token_ids.push_back(next_token_id);
    result.generated_token_ids.push_back(next_token_id);
    result.stats.prefill_pass_count = 1;
    result.stats.model_pass_count = 1;
    checked_add(
        prompt_token_ids.size(),
        result.stats.model_input_token_positions,
        "model input token positions"
    );

    if (
        eos_token_id.has_value() &&
        next_token_id == *eos_token_id
    ) {
        result.stopped_on_eos = true;
    }

    for (
        std::size_t step = 1;
        step < config.max_new_tokens &&
            !result.stopped_on_eos;
        ++step
    ) {
        logits = executor_.decode_one(
            queue,
            next_token_id,
            cache
        );
        next_token_id = select_next_token(
            logits,
            1,
            model_config.vocabulary_size,
            config,
            random
        );

        result.token_ids.push_back(next_token_id);
        result.generated_token_ids.push_back(
            next_token_id
        );
        ++result.stats.decode_pass_count;
        ++result.stats.model_pass_count;
        checked_add(
            1,
            result.stats.model_input_token_positions,
            "model input token positions"
        );

        if (
            eos_token_id.has_value() &&
            next_token_id == *eos_token_id
        ) {
            result.stopped_on_eos = true;
        }
    }

    result.stats.generated_token_count =
        result.generated_token_ids.size();
    result.stats.final_cache_length =
        cache.sequence_length();

    if (
        result.stats.final_cache_length !=
        result.stats.model_input_token_positions
    ) {
        throw std::logic_error(
            "cached generation cache length does not match "
            "the number of model input token positions"
        );
    }

    const Clock::time_point end = Clock::now();
    result.stats.elapsed_milliseconds =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();

    return result;
}

} // namespace minitensor::hf
