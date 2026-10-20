#pragma once

#include "../lesson31-linear-operator/linear.hpp"
#include "../lesson33-transformer-ops/transformer_ops.hpp"
#include "../lesson34-attention-transformer-block/transformer_block.hpp"
#include "hf_loader.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace minitensor::inference {

class CachedLlamaExecutor;

} // namespace minitensor::inference

namespace minitensor::hf {

// A correctness-first, batch-size-one Hugging Face LlamaForCausalLM runtime.
//
// The first version intentionally accepts only the architecture already
// implemented by Lessons 33 and 34:
//
//   * ordinary multi-head attention (Q heads == KV heads),
//   * default, unscaled RoPE,
//   * no attention or MLP projection biases,
//   * f32 RuntimeTensors after checkpoint loading,
//   * no padding mask and no KV cache.
class HuggingFaceLlama {
public:
    [[nodiscard]] static HuggingFaceLlama load(
        const minigpu::Device& device,
        const std::filesystem::path& model_directory
    );

    // Runs the complete causal language model:
    //
    //   token IDs -> embeddings -> decoder blocks
    //             -> final RMSNorm -> LM head -> logits
    //
    // The result has shape
    // [token_ids.size(), config().vocabulary_size].
    [[nodiscard]] RuntimeTensor forward(
        minigpu::Queue& queue,
        const std::vector<std::uint32_t>& token_ids
    ) const;

    [[nodiscard]] const LlamaConfig&
    config() const noexcept;

private:
    friend class minitensor::inference::CachedLlamaExecutor;

    HuggingFaceLlama(
        const minigpu::Device& device,
        LlamaConfig config,
        RuntimeTensor embedding_weight,
        std::vector<TransformerBlockWeights> layers,
        RuntimeTensor final_norm_weight,
        RuntimeTensor lm_head_weight
    );

    minigpu::Device device_;
    LlamaConfig config_;
    RuntimeTensor embedding_weight_;
    std::vector<TransformerBlockWeights> layers_;
    RuntimeTensor final_norm_weight_;
    RuntimeTensor lm_head_weight_;

    TransformerOps transformer_ops_;
    MiniTransformerBlock transformer_block_;
    LinearOperator linear_;
};

} // namespace minitensor::hf
