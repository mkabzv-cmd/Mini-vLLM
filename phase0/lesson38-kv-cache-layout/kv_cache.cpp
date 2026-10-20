#include "kv_cache.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace minitensor::inference {

namespace {

constexpr std::string_view append_kernel_name =
    "append_kv_cache";

constexpr std::string_view append_kernel_source = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void append_kv_cache(
    device const float* new_keys [[buffer(0)]],
    device const float* new_values [[buffer(1)]],
    device float* key_cache [[buffer(2)]],
    device float* value_cache [[buffer(3)]],
    constant uint& new_token_count [[buffer(4)]],
    constant uint& capacity [[buffer(5)]],
    constant uint& head_dimension [[buffer(6)]],
    constant uint& destination_start [[buffer(7)]],
    constant uint& element_count [[buffer(8)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id >= element_count) {
        return;
    }

    // The compact input is laid out as [head, new_token, component].
    const uint elements_per_input_head =
        new_token_count * head_dimension;
    const uint head =
        global_id / elements_per_input_head;
    const uint within_head =
        global_id % elements_per_input_head;
    const uint token =
        within_head / head_dimension;
    const uint component =
        within_head % head_dimension;

    // The preallocated destination is laid out as
    // [head, capacity, component]. Its token stride therefore uses the
    // complete capacity, not new_token_count.
    const uint destination_index =
        ((head * capacity) + destination_start + token) *
            head_dimension +
        component;

    // new_keys and new_values are compact and start at offset zero, so their
    // row-major source index is exactly global_id.
    key_cache[destination_index] = new_keys[global_id];
    value_cache[destination_index] = new_values[global_id];
}
)metal";

constexpr std::size_t new_keys_binding = 0;
constexpr std::size_t new_values_binding = 1;
constexpr std::size_t key_cache_binding = 2;
constexpr std::size_t value_cache_binding = 3;
constexpr std::size_t new_token_count_binding = 4;
constexpr std::size_t capacity_binding = 5;
constexpr std::size_t head_dimension_binding = 6;
constexpr std::size_t destination_start_binding = 7;
constexpr std::size_t element_count_binding = 8;

[[nodiscard]] std::size_t checked_add(
    std::size_t left,
    std::size_t right,
    const char* description
) {
    if (
        left >
        std::numeric_limits<std::size_t>::max() - right
    ) {
        throw std::overflow_error(
            std::string(description) +
            " overflows size_t"
        );
    }

    return left + right;
}

[[nodiscard]] std::size_t checked_multiply(
    std::size_t left,
    std::size_t right,
    const char* description
) {
    if (
        left != 0 &&
        right >
            std::numeric_limits<std::size_t>::max() / left
    ) {
        throw std::overflow_error(
            std::string(description) +
            " overflows size_t"
        );
    }

    return left * right;
}

[[nodiscard]] std::uint32_t require_u32(
    std::size_t value,
    const char* description
) {
    if (
        value >
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()
        )
    ) {
        throw std::overflow_error(
            std::string(description) +
            " does not fit the Lesson 38 uint32 "
            "Metal indexing model"
        );
    }

    return static_cast<std::uint32_t>(value);
}

void require_positive(
    std::size_t value,
    const char* description
) {
    if (value == 0) {
        throw std::invalid_argument(
            std::string(description) +
            " must be greater than zero"
        );
    }
}

void validate_config_dimensions(
    const KVCacheConfig& config
) {
    require_positive(
        config.layer_count,
        "KV-cache layer_count"
    );
    require_positive(
        config.key_value_head_count,
        "KV-cache key_value_head_count"
    );
    require_positive(
        config.capacity,
        "KV-cache capacity"
    );
    require_positive(
        config.head_dimension,
        "KV-cache head_dimension"
    );

    // The Metal kernel uses uint for every shape and address calculation.
    static_cast<void>(
        require_u32(
            config.layer_count,
            "KV-cache layer_count"
        )
    );
    static_cast<void>(
        require_u32(
            config.key_value_head_count,
            "KV-cache key_value_head_count"
        )
    );
    static_cast<void>(
        require_u32(
            config.capacity,
            "KV-cache capacity"
        )
    );
    static_cast<void>(
        require_u32(
            config.head_dimension,
            "KV-cache head_dimension"
        )
    );
}

[[nodiscard]] miniir::Type cache_tensor_type(
    const KVCacheConfig& config
) {
    return miniir::Type::tensor(
        miniir::DType::F32,
        {
            config.key_value_head_count,
            config.capacity,
            config.head_dimension,
        }
    );
}

void require_compact_chunk(
    const RuntimeTensor& tensor,
    const char* name
) {
    if (
        !tensor.is_contiguous() ||
        tensor.offset() != 0
    ) {
        throw std::invalid_argument(
            std::string("KV-cache ") + name +
            " must use compact row-major storage "
            "(canonical strides and offset zero)"
        );
    }
}

void require_chunk_shape(
    const RuntimeTensor& tensor,
    const KVCacheConfig& config,
    const char* name
) {
    if (tensor.rank() != 3) {
        throw std::invalid_argument(
            std::string("KV-cache ") + name +
            " must have rank 3 and shape "
            "[key_value_heads, new_tokens, "
            "head_dimension]"
        );
    }

    if (
        tensor.shape()[0] !=
            config.key_value_head_count ||
        tensor.shape()[2] !=
            config.head_dimension
    ) {
        throw std::invalid_argument(
            std::string("KV-cache ") + name +
            " shape must be [" +
            std::to_string(
                config.key_value_head_count
            ) +
            ", new_tokens, " +
            std::to_string(config.head_dimension) +
            "]; received [" +
            std::to_string(tensor.shape()[0]) +
            ", " +
            std::to_string(tensor.shape()[1]) +
            ", " +
            std::to_string(tensor.shape()[2]) +
            "]"
        );
    }

    require_compact_chunk(tensor, name);
}

} // namespace

KVCacheMemoryUsage calculate_kv_cache_memory_usage(
    const KVCacheConfig& config
) {
    validate_config_dimensions(config);

    KVCacheMemoryUsage usage;
    usage.elements_per_tensor =
        checked_multiply(
            checked_multiply(
                config.key_value_head_count,
                config.capacity,
                "KV-cache elements per tensor"
            ),
            config.head_dimension,
            "KV-cache elements per tensor"
        );

    // Every address expression in the Metal kernel must fit uint.
    static_cast<void>(
        require_u32(
            usage.elements_per_tensor,
            "KV-cache elements per tensor"
        )
    );

    usage.bytes_per_tensor =
        checked_multiply(
            usage.elements_per_tensor,
            sizeof(float),
            "KV-cache bytes per tensor"
        );
    usage.elements_per_layer =
        checked_multiply(
            usage.elements_per_tensor,
            2,
            "KV-cache elements per layer"
        );
    usage.bytes_per_layer =
        checked_multiply(
            usage.bytes_per_tensor,
            2,
            "KV-cache bytes per layer"
        );
    usage.total_elements =
        checked_multiply(
            usage.elements_per_layer,
            config.layer_count,
            "KV-cache total elements"
        );
    usage.total_bytes =
        checked_multiply(
            usage.bytes_per_layer,
            config.layer_count,
            "KV-cache total bytes"
        );

    return usage;
}

KVCache::KVCache(
    const minigpu::Device& device,
    KVCacheConfig config
)
    : device_(device),
      config_(std::move(config)),
      memory_usage_(
          calculate_kv_cache_memory_usage(config_)
      ),
      append_kernel_(
          device_.compile_kernel(
              std::string(append_kernel_name),
              append_kernel_source
          )
      ) {
    layers_.reserve(config_.layer_count);

    for (
        std::size_t layer = 0;
        layer < config_.layer_count;
        ++layer
    ) {
        layers_.push_back(
            LayerStorage{
                RuntimeTensor::empty(
                    device_,
                    cache_tensor_type(config_)
                ),
                RuntimeTensor::empty(
                    device_,
                    cache_tensor_type(config_)
                ),
                0,
            }
        );
    }
}

const KVCacheConfig& KVCache::config() const noexcept {
    return config_;
}

const KVCacheMemoryUsage&
KVCache::memory_usage() const noexcept {
    return memory_usage_;
}

KVCache::LayerStorage& KVCache::checked_layer(
    std::size_t layer_index
) {
    if (layer_index >= layers_.size()) {
        throw std::out_of_range(
            "KV-cache layer index " +
            std::to_string(layer_index) +
            " is outside layer count " +
            std::to_string(layers_.size())
        );
    }

    return layers_[layer_index];
}

const KVCache::LayerStorage& KVCache::checked_layer(
    std::size_t layer_index
) const {
    if (layer_index >= layers_.size()) {
        throw std::out_of_range(
            "KV-cache layer index " +
            std::to_string(layer_index) +
            " is outside layer count " +
            std::to_string(layers_.size())
        );
    }

    return layers_[layer_index];
}

std::size_t KVCache::valid_length(
    std::size_t layer_index
) const {
    return checked_layer(layer_index).valid_length;
}

bool KVCache::has_uniform_length() const noexcept {
    // A moved-from cache may have no layer objects. Keep observers safe even
    // though only destruction or move-assignment is normally useful then.
    if (layers_.empty()) {
        return true;
    }

    const std::size_t first_length =
        layers_.front().valid_length;

    return std::all_of(
        layers_.begin() + 1,
        layers_.end(),
        [first_length](const LayerStorage& layer) {
            return layer.valid_length == first_length;
        }
    );
}

std::size_t KVCache::sequence_length() const {
    if (!has_uniform_length()) {
        throw std::logic_error(
            "KV-cache layers do not have one common "
            "sequence length; finish appending the current "
            "model chunk to every layer first"
        );
    }

    return layers_.empty()
        ? 0
        : layers_.front().valid_length;
}

const RuntimeTensor& KVCache::key_storage(
    std::size_t layer_index
) const {
    return checked_layer(layer_index).keys;
}

const RuntimeTensor& KVCache::value_storage(
    std::size_t layer_index
) const {
    return checked_layer(layer_index).values;
}

std::size_t KVCache::storage_index(
    std::size_t head,
    std::size_t token,
    std::size_t component
) const {
    if (head >= config_.key_value_head_count) {
        throw std::out_of_range(
            "KV-cache head index is outside "
            "key_value_head_count"
        );
    }

    if (token >= config_.capacity) {
        throw std::out_of_range(
            "KV-cache token index is outside capacity"
        );
    }

    if (component >= config_.head_dimension) {
        throw std::out_of_range(
            "KV-cache component index is outside "
            "head_dimension"
        );
    }

    const std::size_t head_and_token =
        checked_add(
            checked_multiply(
                head,
                config_.capacity,
                "KV-cache storage index"
            ),
            token,
            "KV-cache storage index"
        );

    return checked_add(
        checked_multiply(
            head_and_token,
            config_.head_dimension,
            "KV-cache storage index"
        ),
        component,
        "KV-cache storage index"
    );
}

float KVCache::load_key(
    std::size_t layer_index,
    std::size_t head,
    std::size_t token,
    std::size_t component
) const {
    const LayerStorage& layer =
        checked_layer(layer_index);

    if (token >= layer.valid_length) {
        throw std::out_of_range(
            "KV-cache key token index is outside the "
            "layer's valid length"
        );
    }

    return layer.keys.buffer().load(
        storage_index(head, token, component)
    );
}

float KVCache::load_value(
    std::size_t layer_index,
    std::size_t head,
    std::size_t token,
    std::size_t component
) const {
    const LayerStorage& layer =
        checked_layer(layer_index);

    if (token >= layer.valid_length) {
        throw std::out_of_range(
            "KV-cache value token index is outside the "
            "layer's valid length"
        );
    }

    return layer.values.buffer().load(
        storage_index(head, token, component)
    );
}

void KVCache::append(
    minigpu::Queue& queue,
    std::size_t layer_index,
    const RuntimeTensor& keys,
    const RuntimeTensor& values,
    std::size_t threads_per_group
) {
    if (threads_per_group == 0) {
        throw std::invalid_argument(
            "KV-cache threads_per_group must be greater "
            "than zero"
        );
    }

    LayerStorage& layer = checked_layer(layer_index);
    require_chunk_shape(keys, config_, "keys");
    require_chunk_shape(values, config_, "values");

    if (keys.shape() != values.shape()) {
        throw std::invalid_argument(
            "KV-cache key and value chunks must have "
            "identical shapes"
        );
    }

    if (
        keys.buffer().shares_storage_with(
            layer.keys.buffer()
        ) ||
        keys.buffer().shares_storage_with(
            layer.values.buffer()
        ) ||
        values.buffer().shares_storage_with(
            layer.keys.buffer()
        ) ||
        values.buffer().shares_storage_with(
            layer.values.buffer()
        )
    ) {
        throw std::invalid_argument(
            "KV-cache source chunks must not alias the "
            "destination layer's K or V storage"
        );
    }

    const std::size_t new_token_count =
        keys.shape()[1];
    const std::size_t next_length =
        checked_add(
            layer.valid_length,
            new_token_count,
            "KV-cache appended sequence length"
        );

    if (next_length > config_.capacity) {
        throw std::length_error(
            "KV-cache append would exceed capacity: "
            "current length " +
            std::to_string(layer.valid_length) +
            " + new tokens " +
            std::to_string(new_token_count) +
            " > capacity " +
            std::to_string(config_.capacity)
        );
    }

    const std::size_t element_count =
        keys.numel();
    const std::uint32_t element_count_u32 =
        require_u32(
            element_count,
            "KV-cache append element count"
        );
    const std::uint32_t new_token_count_u32 =
        require_u32(
            new_token_count,
            "KV-cache appended token count"
        );
    const std::uint32_t capacity_u32 =
        require_u32(
            config_.capacity,
            "KV-cache capacity"
        );
    const std::uint32_t head_dimension_u32 =
        require_u32(
            config_.head_dimension,
            "KV-cache head_dimension"
        );
    const std::uint32_t destination_start_u32 =
        require_u32(
            layer.valid_length,
            "KV-cache append destination start"
        );

    const minigpu::LaunchConfig launch =
        minigpu::LaunchConfig::cover(
            element_count,
            threads_per_group
        );
    static_cast<void>(
        require_u32(
            launch.total_threads(),
            "KV-cache launched thread count"
        )
    );

    minigpu::Dispatch dispatch(append_kernel_);
    dispatch
        .bind_buffer(
            new_keys_binding,
            keys.buffer()
        )
        .bind_buffer(
            new_values_binding,
            values.buffer()
        )
        .bind_buffer(
            key_cache_binding,
            layer.keys.buffer()
        )
        .bind_buffer(
            value_cache_binding,
            layer.values.buffer()
        )
        .bind_u32(
            new_token_count_binding,
            new_token_count_u32
        )
        .bind_u32(capacity_binding, capacity_u32)
        .bind_u32(
            head_dimension_binding,
            head_dimension_u32
        )
        .bind_u32(
            destination_start_binding,
            destination_start_u32
        )
        .bind_u32(
            element_count_binding,
            element_count_u32
        );

    minigpu::Event finished = queue.submit(
        std::move(dispatch),
        launch
    );
    finished.wait();

    // Publish the new logical length only after the GPU copy succeeded.
    layer.valid_length = next_length;
}

void KVCache::reset() noexcept {
    for (LayerStorage& layer : layers_) {
        layer.valid_length = 0;
    }
}

void KVCache::truncate(
    std::size_t new_length
) {
    for (const LayerStorage& layer : layers_) {
        if (new_length > layer.valid_length) {
            throw std::invalid_argument(
                "KV-cache truncate cannot increase a "
                "layer's valid length"
            );
        }
    }

    for (LayerStorage& layer : layers_) {
        layer.valid_length = new_length;
    }
}

} // namespace minitensor::inference
