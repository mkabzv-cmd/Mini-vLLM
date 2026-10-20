#pragma once

#include "../lesson35-huggingface-llama/hf_llama.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace minitensor::hf {

// Lesson 36 deliberately keeps decoding policy separate from the model.
// The model produces logits; this configuration decides how one next token
// is selected from the final row of those logits.
struct GenerationConfig {
    std::size_t max_new_tokens = 1;

    // false: choose the largest logit (greedy decoding).
    // true: sample after temperature scaling and optional top-k filtering.
    bool do_sample = false;
    float temperature = 1.0f;

    // 0 keeps the complete vocabulary. A positive value keeps only the
    // tokens with the k largest logits before sampling.
    std::size_t top_k = 0;
    std::uint64_t seed = 0;

    // By default the model's config.json EOS ID is used. An override is
    // useful for small tests or tokenizers with an explicitly chosen stop
    // token.
    bool stop_on_eos = true;
    std::optional<std::uint32_t> eos_token_id;
};

struct GenerationStats {
    std::size_t prompt_token_count = 0;
    std::size_t generated_token_count = 0;
    std::size_t forward_pass_count = 0;

    // Sum of the sequence lengths sent through the model. For a prompt of
    // length 3 and three generated tokens this is 3 + 4 + 5 = 12. This makes
    // Lesson 36's deliberate full-sequence recomputation visible.
    std::size_t model_input_token_count = 0;

    double elapsed_milliseconds = 0.0;
};

struct GenerationResult {
    // The complete sequence: original prompt followed by generated tokens.
    std::vector<std::uint32_t> token_ids;

    // Only the tokens selected by this generation call.
    std::vector<std::uint32_t> generated_token_ids;

    bool stopped_on_eos = false;
    GenerationStats stats;
};

class AutoregressiveGenerator {
public:
    explicit AutoregressiveGenerator(
        const HuggingFaceLlama& model
    );

    // Lesson 36 has no KV cache. Before selecting each token it executes:
    //
    //   complete sequence -> full model forward -> final-row logits
    //
    // It then appends one token and repeats.
    [[nodiscard]] GenerationResult generate(
        minigpu::Queue& queue,
        const std::vector<std::uint32_t>& prompt_token_ids,
        const GenerationConfig& config = {}
    ) const;

private:
    const HuggingFaceLlama& model_;
};

} // namespace minitensor::hf
