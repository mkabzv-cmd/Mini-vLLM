#pragma once

#include "../lesson25-tensor-storage-layout/runtime_tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace minitensor {

// A small collection of real Metal operations used around transformer
// attention and feed-forward layers. Every public method waits for its GPU
// command before returning.
class TransformerOps {
public:
    explicit TransformerOps(const minigpu::Device& device);

    // embedding_table: [vocabulary_size, hidden_size]
    // result:          [token_ids.size(), hidden_size]
    //
    // MiniGPU currently exposes only float buffers, so token IDs are
    // validated, encoded exactly into a temporary f32 RuntimeTensor, and
    // converted back to uint inside the Metal kernel.
    [[nodiscard]] RuntimeTensor embedding(
        minigpu::Queue& queue,
        const std::vector<std::uint32_t>& token_ids,
        const RuntimeTensor& embedding_table,
        std::size_t threads_per_group = 64
    ) const;

    // Normalizes each row over the final dimension, then multiplies by the
    // one-dimensional weight. The Metal kernel always uses 64 threads for
    // each row.
    [[nodiscard]] RuntimeTensor rms_norm(
        minigpu::Queue& queue,
        const RuntimeTensor& input,
        const RuntimeTensor& weight,
        float epsilon = 0.00001f
    ) const;

    // Llama-style split-half rotary embedding.
    // input/result shape: [tokens, heads, head_dimension]
    [[nodiscard]] RuntimeTensor rope(
        minigpu::Queue& queue,
        const RuntimeTensor& input,
        std::size_t start_position = 0,
        float theta = 10000.0f,
        std::size_t threads_per_group = 64
    ) const;

    // SiLU(x) = x / (1 + exp(-x)).
    [[nodiscard]] RuntimeTensor silu(
        minigpu::Queue& queue,
        const RuntimeTensor& input,
        std::size_t threads_per_group = 64
    ) const;

private:
    minigpu::Device device_;
    minigpu::Kernel embedding_kernel_;
    minigpu::Kernel rms_norm_kernel_;
    minigpu::Kernel rope_kernel_;
    minigpu::Kernel silu_kernel_;
};

} // namespace minitensor
