#pragma once

#include "../lesson28-contiguous-buffer-reuse/contiguous.hpp"
#include "../lesson35-huggingface-llama/hf_llama.hpp"
#include "../lesson38-kv-cache-layout/kv_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace minitensor::inference {

// Computes causal attention for one newly supplied query chunk while reading
// every key and value from preallocated KV-cache-shaped storage.
//
// compact_queries:
//     [heads, query_tokens, head_dimension]
//
// key_storage and value_storage:
//     [heads, capacity, head_dimension]
//
// Only rows [0, valid_length) of K/V are read. The supplied queries occupy
// absolute positions:
//
//     [query_start, query_start + query_tokens)
//
// Therefore query_start + query_tokens must equal valid_length. This matches
// the normal call order: append the current chunk's K/V, then attend over the
// complete cache including the current chunk.
class CachedAttention {
public:
    explicit CachedAttention(
        const minigpu::Device& device
    );

    [[nodiscard]] RuntimeTensor run(
        minigpu::Queue& queue,
        const RuntimeTensor& compact_queries,
        const RuntimeTensor& key_storage,
        const RuntimeTensor& value_storage,
        std::size_t valid_length,
        std::size_t query_start,
        std::size_t threads_per_group = 64
    ) const;

private:
    minigpu::Device device_;
    minigpu::Kernel scores_kernel_;
    minigpu::Kernel softmax_kernel_;
    minigpu::Kernel context_kernel_;
};

// Runs the already-loaded Hugging Face Llama with a real per-layer KV cache.
//
// This object borrows the model and its immutable weights. The model must
// outlive the executor and must not be moved while the executor exists.
class CachedLlamaExecutor {
public:
    explicit CachedLlamaExecutor(
        const hf::HuggingFaceLlama& model
    );

    // Prevent a temporary model from being destroyed immediately while this
    // executor still stores a reference to it.
    CachedLlamaExecutor(
        hf::HuggingFaceLlama&& model
    ) = delete;

    CachedLlamaExecutor(
        const hf::HuggingFaceLlama&& model
    ) = delete;

    [[nodiscard]] KVCache create_cache(
        std::size_t capacity
    ) const;

    // A populated cache belongs to this executor's model. Lesson 39 validates
    // its geometry but does not yet store a model-identity tag inside the
    // generic KVCache object.
    //
    // Processes a non-empty prompt into an empty cache. Returns logits for
    // every prompt position: [prompt_tokens, vocabulary_size].
    [[nodiscard]] RuntimeTensor prefill(
        minigpu::Queue& queue,
        const std::vector<std::uint32_t>& prompt_token_ids,
        KVCache& cache
    ) const;

    // Processes exactly one new token after a non-empty cached prefix.
    // Returns [1, vocabulary_size].
    [[nodiscard]] RuntimeTensor decode_one(
        minigpu::Queue& queue,
        std::uint32_t token_id,
        KVCache& cache
    ) const;

    [[nodiscard]] const hf::LlamaConfig&
    config() const noexcept;

private:
    [[nodiscard]] RuntimeTensor run_chunk(
        minigpu::Queue& queue,
        const std::vector<std::uint32_t>& token_ids,
        KVCache& cache,
        std::size_t query_start
    ) const;

    const hf::HuggingFaceLlama& model_;
    CachedAttention cached_attention_;
    ContiguousMaterializer materializer_;
    minigpu::Kernel binary_kernel_;
};

} // namespace minitensor::inference
