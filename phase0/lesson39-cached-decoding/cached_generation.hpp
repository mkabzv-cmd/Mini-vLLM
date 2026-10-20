#pragma once

#include "../lesson36-autoregressive-generation/generation.hpp"
#include "cached_llama.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace minitensor::hf {

// These counters describe work that was actually executed through
// CachedLlamaExecutor. They are not an estimate of a future cached path.
struct CachedGenerationStats {
    std::size_t prompt_token_count = 0;
    std::size_t generated_token_count = 0;

    // A nonempty generation performs exactly one prefill. Every later model
    // pass consumes one previously selected token through decode_one().
    std::size_t prefill_pass_count = 0;
    std::size_t decode_pass_count = 0;
    std::size_t model_pass_count = 0;

    // Sum of token positions actually sent into the model. Without early EOS:
    //
    //   prompt_token_count + generated_token_count - 1
    //
    // because the final selected token does not need to be consumed unless
    // another token is requested.
    std::size_t model_input_token_positions = 0;

    // The cache contains only tokens that the model has consumed. Therefore,
    // after generation it normally equals model_input_token_positions.
    std::size_t final_cache_length = 0;
    std::size_t cache_capacity = 0;

    double elapsed_milliseconds = 0.0;
};

struct CachedGenerationResult {
    // Original prompt followed by tokens selected by this call.
    std::vector<std::uint32_t> token_ids;
    std::vector<std::uint32_t> generated_token_ids;

    bool stopped_on_eos = false;
    CachedGenerationStats stats;
};

class CachedAutoregressiveGenerator {
public:
    // The executor must outlive this generator and must not be moved while
    // this generator exists.
    explicit CachedAutoregressiveGenerator(
        const inference::CachedLlamaExecutor& executor
    );

    // The generator borrows its executor, so an executor temporary would
    // leave a dangling reference.
    CachedAutoregressiveGenerator(
        inference::CachedLlamaExecutor&& executor
    ) = delete;

    CachedAutoregressiveGenerator(
        const inference::CachedLlamaExecutor&& executor
    ) = delete;

    // max_new_tokens == 0 performs validation but no model pass and creates
    // no KV cache. Otherwise:
    //
    //   prompt -> one prefill -> first selected token
    //   previous selected token -> decode_one -> next selected token
    //
    // The GenerationConfig and token-selection semantics intentionally match
    // Lesson 36 so the two execution strategies can be compared directly.
    [[nodiscard]] CachedGenerationResult generate(
        minigpu::Queue& queue,
        const std::vector<std::uint32_t>& prompt_token_ids,
        const GenerationConfig& config = {}
    ) const;

private:
    const inference::CachedLlamaExecutor& executor_;
};

} // namespace minitensor::hf
