#pragma once

#include "../lesson28-contiguous-buffer-reuse/contiguous.hpp"
#include "../lesson31-linear-operator/linear.hpp"
#include "../lesson32-batched-matmul/batched_matmul.hpp"
#include "../lesson33-transformer-ops/transformer_ops.hpp"

#include <cstddef>

namespace minitensor {

// Llama-style decoder-block weights. Every linear weight uses the
// Hugging Face / PyTorch layout [output_features, input_features].
struct TransformerBlockWeights {
    RuntimeTensor attention_norm_weight;
    RuntimeTensor query_weight;
    RuntimeTensor key_weight;
    RuntimeTensor value_weight;
    RuntimeTensor attention_output_weight;

    RuntimeTensor ffn_norm_weight;
    RuntimeTensor gate_weight;
    RuntimeTensor up_weight;
    RuntimeTensor down_weight;
};

// A correctness-first, batch-size-one transformer block.
class MiniTransformerBlock {
public:
    explicit MiniTransformerBlock(
        const minigpu::Device& device
    );

    // The input is already normalized and has shape [tokens, hidden].
    // start_position affects RoPE only. Without a KV cache, this call
    // cannot attend to tokens that precede the supplied tensor.
    [[nodiscard]] RuntimeTensor causal_self_attention(
        minigpu::Queue& queue,
        const RuntimeTensor& normalized_hidden,
        const TransformerBlockWeights& weights,
        std::size_t heads,
        std::size_t start_position = 0,
        float rope_theta = 10000.0f
    ) const;

    // Full Llama-style decoder block:
    //
    //     RMSNorm -> causal attention -> residual
    //     RMSNorm -> SwiGLU feed-forward -> residual
    //
    // start_position has the same no-KV-cache limitation described above.
    [[nodiscard]] RuntimeTensor run(
        minigpu::Queue& queue,
        const RuntimeTensor& hidden,
        const TransformerBlockWeights& weights,
        std::size_t heads,
        std::size_t start_position = 0,
        float epsilon = 0.00001f,
        float rope_theta = 10000.0f
    ) const;

private:
    minigpu::Device device_;
    LinearOperator linear_;
    BatchedMatmul batched_matmul_;
    TransformerOps transformer_ops_;
    ContiguousMaterializer materializer_;
    minigpu::Kernel binary_kernel_;
    minigpu::Kernel causal_softmax_kernel_;
};

} // namespace minitensor
