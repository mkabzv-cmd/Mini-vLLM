#pragma once

#include "../lesson25-tensor-storage-layout/runtime_tensor.hpp"

#include <cstddef>
#include <vector>

namespace minitensor::inference {

// One cache contains keys and values for every transformer layer.
//
// The first implementation stores f32 values and assumes batch size one.
// key_value_head_count may differ from the number of query heads in a future
// grouped-query-attention extension. The current tiny model uses equal
// query-head and key/value-head counts.
struct KVCacheConfig {
    std::size_t layer_count = 0;
    std::size_t key_value_head_count = 0;
    std::size_t capacity = 0;
    std::size_t head_dimension = 0;
};

// Memory is reported explicitly because a production-sized KV cache can be
// one of the largest allocations owned by an inference engine.
//
// "tensor" means one K tensor or one V tensor. Each layer owns both.
struct KVCacheMemoryUsage {
    std::size_t elements_per_tensor = 0;
    std::size_t bytes_per_tensor = 0;
    std::size_t elements_per_layer = 0;
    std::size_t bytes_per_layer = 0;
    std::size_t total_elements = 0;
    std::size_t total_bytes = 0;
};

// Performs the same checked shape and arithmetic validation used by the
// constructor, without allocating Metal buffers.
[[nodiscard]] KVCacheMemoryUsage
calculate_kv_cache_memory_usage(
    const KVCacheConfig& config
);

class KVCache {
public:
    KVCache(
        const minigpu::Device& device,
        KVCacheConfig config
    );

    // RuntimeTensor normally behaves like a shared buffer handle. Copying a
    // complete KVCache would be dangerous, however: two cache objects could
    // then share GPU storage while tracking different valid lengths.
    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;
    KVCache(KVCache&&) noexcept = default;
    KVCache& operator=(KVCache&&) noexcept = default;

    [[nodiscard]] const KVCacheConfig&
    config() const noexcept;

    [[nodiscard]] const KVCacheMemoryUsage&
    memory_usage() const noexcept;

    // Number of valid token rows currently stored in one layer.
    [[nodiscard]] std::size_t valid_length(
        std::size_t layer_index
    ) const;

    // During a model pass, layers are appended one after another. Their
    // lengths can temporarily differ until every layer has processed the
    // same input chunk.
    [[nodiscard]] bool has_uniform_length() const noexcept;

    // Returns the common length after a complete model pass. Calling this
    // while layer lengths differ is a logic error.
    [[nodiscard]] std::size_t sequence_length() const;

    // Full preallocated storage shape:
    //
    //   [key_value_head_count, capacity, head_dimension]
    [[nodiscard]] const RuntimeTensor& key_storage(
        std::size_t layer_index
    ) const;

    [[nodiscard]] const RuntimeTensor& value_storage(
        std::size_t layer_index
    ) const;

    // Row-major address inside either one layer's K or V storage:
    //
    //   ((head * capacity) + token) * head_dimension + component
    //
    // token may address any reserved row below capacity. load_key() and
    // load_value() apply the stronger valid-length rule.
    [[nodiscard]] std::size_t storage_index(
        std::size_t head,
        std::size_t token,
        std::size_t component
    ) const;

    [[nodiscard]] float load_key(
        std::size_t layer_index,
        std::size_t head,
        std::size_t token,
        std::size_t component
    ) const;

    [[nodiscard]] float load_value(
        std::size_t layer_index,
        std::size_t head,
        std::size_t token,
        std::size_t component
    ) const;

    // Appends one compact chunk to one layer using the GPU.
    //
    // keys and values must both have shape:
    //
    //   [key_value_head_count, new_tokens, head_dimension]
    //
    // The method waits for the Metal command. Only after that wait succeeds
    // does it publish the new valid length on the CPU.
    //
    // GPU-produced inputs must have completed already or have been submitted
    // earlier to this same FIFO queue. Different queues require explicit
    // synchronization by the caller.
    void append(
        minigpu::Queue& queue,
        std::size_t layer_index,
        const RuntimeTensor& keys,
        const RuntimeTensor& values,
        std::size_t threads_per_group = 64
    );

    // Makes every layer logically empty. Old bytes are intentionally left in
    // the preallocated buffers because future appends overwrite them before
    // those rows become valid again.
    void reset() noexcept;

    // Reduces every layer to an earlier common valid length without clearing
    // physical bytes. Used to roll back an interrupted model pass.
    void truncate(std::size_t new_length);

private:
    struct LayerStorage {
        RuntimeTensor keys;
        RuntimeTensor values;
        std::size_t valid_length = 0;
    };

    [[nodiscard]] LayerStorage& checked_layer(
        std::size_t layer_index
    );

    [[nodiscard]] const LayerStorage& checked_layer(
        std::size_t layer_index
    ) const;

    minigpu::Device device_;
    KVCacheConfig config_;
    KVCacheMemoryUsage memory_usage_;
    minigpu::Kernel append_kernel_;
    std::vector<LayerStorage> layers_;
};

} // namespace minitensor::inference
