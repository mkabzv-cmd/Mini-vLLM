#include "cached_llama.hpp"

#include "../lesson26-tensor-views/tensor_views.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace minitensor::inference {

namespace {

constexpr std::string_view scores_kernel_name =
    "cached_attention_scores";

constexpr std::string_view scores_kernel_source = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void cached_attention_scores(
    device const float* queries [[buffer(0)]],
    device const float* key_cache [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& query_tokens [[buffer(3)]],
    constant uint& valid_length [[buffer(4)]],
    constant uint& capacity [[buffer(5)]],
    constant uint& head_dimension [[buffer(6)]],
    constant uint& output_count [[buffer(7)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id >= output_count) {
        return;
    }

    const uint key = global_id % valid_length;
    const uint head_and_query =
        global_id / valid_length;
    const uint query =
        head_and_query % query_tokens;
    const uint head =
        head_and_query / query_tokens;

    const uint query_start =
        (head * query_tokens + query) *
        head_dimension;
    const uint key_start =
        (head * capacity + key) *
        head_dimension;

    float score = 0.0f;

    for (
        uint component = 0;
        component < head_dimension;
        ++component
    ) {
        score +=
            queries[query_start + component] *
            key_cache[key_start + component];
    }

    scores[global_id] = score;
}
)metal";

constexpr std::string_view softmax_kernel_name =
    "cached_causal_softmax";

constexpr std::string_view softmax_kernel_source = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void cached_causal_softmax(
    device const float* scores [[buffer(0)]],
    device float* probabilities [[buffer(1)]],
    constant uint& query_tokens [[buffer(2)]],
    constant uint& valid_length [[buffer(3)]],
    constant uint& query_start [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    uint lane [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[64];

    const uint local_query = row % query_tokens;
    const uint absolute_query =
        query_start + local_query;
    const uint row_start = row * valid_length;

    float lane_maximum = -3.402823466e+38f;

    for (
        uint key = lane;
        key < valid_length;
        key += 64
    ) {
        if (key <= absolute_query) {
            lane_maximum = max(
                lane_maximum,
                scores[row_start + key] * scale
            );
        }
    }

    scratch[lane] = lane_maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) {
            scratch[lane] = max(
                scratch[lane],
                scratch[lane + stride]
            );
        }

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );
    }

    const float row_maximum = scratch[0];
    float lane_sum = 0.0f;

    for (
        uint key = lane;
        key < valid_length;
        key += 64
    ) {
        if (key <= absolute_query) {
            lane_sum += exp(
                scores[row_start + key] * scale -
                row_maximum
            );
        }
    }

    scratch[lane] = lane_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) {
            scratch[lane] += scratch[lane + stride];
        }

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );
    }

    const float inverse_sum = 1.0f / scratch[0];

    for (
        uint key = lane;
        key < valid_length;
        key += 64
    ) {
        probabilities[row_start + key] =
            key <= absolute_query
                ? exp(
                      scores[row_start + key] * scale -
                      row_maximum
                  ) * inverse_sum
                : 0.0f;
    }
}
)metal";

constexpr std::string_view context_kernel_name =
    "cached_attention_context";

constexpr std::string_view context_kernel_source = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void cached_attention_context(
    device const float* probabilities [[buffer(0)]],
    device const float* value_cache [[buffer(1)]],
    device float* context [[buffer(2)]],
    constant uint& query_tokens [[buffer(3)]],
    constant uint& valid_length [[buffer(4)]],
    constant uint& capacity [[buffer(5)]],
    constant uint& head_dimension [[buffer(6)]],
    constant uint& output_count [[buffer(7)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id >= output_count) {
        return;
    }

    const uint component =
        global_id % head_dimension;
    const uint head_and_query =
        global_id / head_dimension;
    const uint query =
        head_and_query % query_tokens;
    const uint head =
        head_and_query / query_tokens;

    const uint probability_start =
        (head * query_tokens + query) *
        valid_length;

    float sum = 0.0f;

    for (uint key = 0; key < valid_length; ++key) {
        const uint value_index =
            (head * capacity + key) *
                head_dimension +
            component;

        sum +=
            probabilities[probability_start + key] *
            value_cache[value_index];
    }

    context[global_id] = sum;
}
)metal";

constexpr std::string_view binary_kernel_name =
    "cached_llama_binary";

constexpr std::string_view binary_kernel_source = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void cached_llama_binary(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& element_count [[buffer(3)]],
    constant uint& operation [[buffer(4)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id >= element_count) {
        return;
    }

    output[global_id] =
        operation == 0
            ? left[global_id] + right[global_id]
            : left[global_id] * right[global_id];
}
)metal";

constexpr std::size_t softmax_threads = 64;

enum class BinaryOperation : std::uint32_t {
    Add = 0,
    Multiply = 1,
};

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
            " does not fit the Lesson 39 uint32 "
            "Metal indexing model"
        );
    }

    return static_cast<std::uint32_t>(value);
}

[[nodiscard]] miniir::Type tensor_type(
    Shape shape
) {
    return miniir::Type::tensor(
        miniir::DType::F32,
        std::move(shape)
    );
}

void require_compact(
    const RuntimeTensor& tensor,
    std::string_view name
) {
    if (
        !tensor.is_contiguous() ||
        tensor.offset() != 0
    ) {
        throw std::invalid_argument(
            std::string(name) +
            " must use compact row-major storage "
            "(canonical strides and offset zero)"
        );
    }
}

[[nodiscard]] std::size_t shape_elements(
    const Shape& shape
) {
    std::size_t count = 1;

    for (std::size_t dimension : shape) {
        count = checked_multiply(
            count,
            dimension,
            "cached Llama reshape element count"
        );
    }

    return count;
}

[[nodiscard]] RuntimeTensor reinterpret_compact(
    const RuntimeTensor& source,
    Shape shape
) {
    require_compact(source, "reshape source");

    if (shape_elements(shape) != source.numel()) {
        throw std::invalid_argument(
            "cached Llama reshape changes element count"
        );
    }

    miniir::Type type =
        shape.empty()
            ? miniir::Type::scalar(
                  source.type().dtype()
              )
            : miniir::Type::tensor(
                  source.type().dtype(),
                  shape
              );

    return RuntimeTensor(
        std::move(type),
        TensorLayout::contiguous(std::move(shape)),
        source.buffer()
    );
}

void wait_for(
    minigpu::Queue& queue,
    minigpu::Dispatch dispatch,
    minigpu::LaunchConfig launch
) {
    minigpu::Event finished = queue.submit(
        std::move(dispatch),
        std::move(launch)
    );
    finished.wait();
}

[[nodiscard]] minigpu::LaunchConfig
elementwise_launch(
    std::size_t element_count,
    std::size_t threads_per_group
) {
    if (threads_per_group == 0) {
        throw std::invalid_argument(
            "threads_per_group must be greater than zero"
        );
    }

    static_cast<void>(
        require_u32(
            element_count,
            "logical element count"
        )
    );

    minigpu::LaunchConfig launch =
        minigpu::LaunchConfig::cover(
            element_count,
            threads_per_group
        );

    static_cast<void>(
        require_u32(
            launch.total_threads(),
            "launched thread count"
        )
    );

    return launch;
}

[[nodiscard]] RuntimeTensor run_binary(
    const minigpu::Device& device,
    const minigpu::Kernel& kernel,
    minigpu::Queue& queue,
    const RuntimeTensor& left,
    const RuntimeTensor& right,
    BinaryOperation operation
) {
    if (left.type() != right.type()) {
        throw std::invalid_argument(
            "cached Llama binary operands must have "
            "the same type"
        );
    }

    require_compact(left, "left binary operand");
    require_compact(right, "right binary operand");

    RuntimeTensor output =
        RuntimeTensor::empty(device, left.type());
    const minigpu::LaunchConfig launch =
        elementwise_launch(left.numel(), 64);

    minigpu::Dispatch dispatch(kernel);
    dispatch
        .bind_buffer(0, left.buffer())
        .bind_buffer(1, right.buffer())
        .bind_buffer(2, output.buffer())
        .bind_u32(
            3,
            require_u32(
                left.numel(),
                "binary element count"
            )
        )
        .bind_u32(
            4,
            static_cast<std::uint32_t>(operation)
        );

    wait_for(
        queue,
        std::move(dispatch),
        launch
    );
    return output;
}

struct AttentionDimensions {
    std::size_t heads;
    std::size_t query_tokens;
    std::size_t valid_length;
    std::size_t capacity;
    std::size_t head_dimension;
    std::size_t score_count;
    std::size_t context_count;
    std::size_t row_count;
};

[[nodiscard]] AttentionDimensions
validate_cached_attention(
    const RuntimeTensor& queries,
    const RuntimeTensor& keys,
    const RuntimeTensor& values,
    std::size_t valid_length,
    std::size_t query_start,
    std::size_t threads_per_group
) {
    require_compact(queries, "cached attention queries");
    require_compact(keys, "cached attention keys");
    require_compact(values, "cached attention values");

    if (queries.rank() != 3) {
        throw std::invalid_argument(
            "cached attention queries must have shape "
            "[heads, query_tokens, head_dimension]"
        );
    }

    if (keys.rank() != 3 || values.rank() != 3) {
        throw std::invalid_argument(
            "cached attention K/V storage must have shape "
            "[heads, capacity, head_dimension]"
        );
    }

    if (keys.shape() != values.shape()) {
        throw std::invalid_argument(
            "cached attention key and value storage must "
            "have the same shape"
        );
    }

    const std::size_t heads = queries.shape()[0];
    const std::size_t query_tokens =
        queries.shape()[1];
    const std::size_t head_dimension =
        queries.shape()[2];
    const std::size_t capacity = keys.shape()[1];

    if (
        heads == 0 ||
        query_tokens == 0 ||
        head_dimension == 0 ||
        capacity == 0
    ) {
        throw std::invalid_argument(
            "cached attention dimensions must all be "
            "greater than zero"
        );
    }

    if (
        keys.shape()[0] != heads ||
        keys.shape()[2] != head_dimension
    ) {
        throw std::invalid_argument(
            "cached attention Q/K/V head count and "
            "head dimension must match"
        );
    }

    if (valid_length > capacity) {
        throw std::length_error(
            "cached attention valid_length exceeds "
            "K/V capacity"
        );
    }

    const std::size_t query_end =
        checked_add(
            query_start,
            query_tokens,
            "cached attention query end"
        );

    if (query_end != valid_length) {
        throw std::invalid_argument(
            "cached attention requires query_start + "
            "query_tokens == valid_length"
        );
    }

    const std::size_t row_count =
        checked_multiply(
            heads,
            query_tokens,
            "cached attention row count"
        );
    const std::size_t score_count =
        checked_multiply(
            row_count,
            valid_length,
            "cached attention score count"
        );
    const std::size_t context_count =
        checked_multiply(
            row_count,
            head_dimension,
            "cached attention context count"
        );

    static_cast<void>(
        require_u32(heads, "attention head count")
    );
    static_cast<void>(
        require_u32(
            query_tokens,
            "attention query token count"
        )
    );
    static_cast<void>(
        require_u32(
            valid_length,
            "attention valid length"
        )
    );
    static_cast<void>(
        require_u32(capacity, "attention capacity")
    );
    static_cast<void>(
        require_u32(
            head_dimension,
            "attention head dimension"
        )
    );
    static_cast<void>(
        require_u32(
            query_start,
            "attention query start"
        )
    );
    static_cast<void>(
        require_u32(row_count, "attention row count")
    );
    static_cast<void>(
        require_u32(
            score_count,
            "attention score count"
        )
    );
    static_cast<void>(
        require_u32(
            context_count,
            "attention context count"
        )
    );
    static_cast<void>(
        require_u32(
            queries.numel(),
            "attention query storage size"
        )
    );
    static_cast<void>(
        require_u32(
            keys.numel(),
            "attention key storage size"
        )
    );
    static_cast<void>(
        require_u32(
            values.numel(),
            "attention value storage size"
        )
    );

    // Validate both padded elementwise launches before allocating outputs.
    static_cast<void>(
        elementwise_launch(
            score_count,
            threads_per_group
        )
    );
    static_cast<void>(
        elementwise_launch(
            context_count,
            threads_per_group
        )
    );

    return AttentionDimensions{
        heads,
        query_tokens,
        valid_length,
        capacity,
        head_dimension,
        score_count,
        context_count,
        row_count,
    };
}

void validate_cache_geometry(
    const hf::LlamaConfig& model_config,
    const KVCache& cache
) {
    const KVCacheConfig& cache_config =
        cache.config();

    if (
        cache_config.layer_count !=
            model_config.layer_count ||
        cache_config.key_value_head_count !=
            model_config.key_value_head_count ||
        cache_config.head_dimension !=
            model_config.head_dimension()
    ) {
        throw std::invalid_argument(
            "KV-cache geometry does not match the "
            "loaded Llama model"
        );
    }

    if (
        cache_config.capacity >
        model_config.maximum_position_count
    ) {
        throw std::invalid_argument(
            "KV-cache capacity exceeds the model's "
            "maximum_position_count"
        );
    }
}

void validate_token_ids(
    const hf::LlamaConfig& config,
    const std::vector<std::uint32_t>& token_ids
) {
    for (std::uint32_t token_id : token_ids) {
        if (
            static_cast<std::size_t>(token_id) >=
            config.vocabulary_size
        ) {
            throw std::out_of_range(
                "token ID " +
                std::to_string(token_id) +
                " is outside vocabulary size " +
                std::to_string(config.vocabulary_size)
            );
        }
    }
}

// A model chunk updates the cache one layer at a time. If any later
// operation fails, restore every layer's logical length so the caller can
// safely retry or abandon the request. The old bytes do not need clearing:
// rows beyond valid_length are invisible and a retry overwrites them.
class CacheRollbackGuard {
public:
    CacheRollbackGuard(
        KVCache& cache,
        std::size_t original_length
    ) noexcept
        : cache_(cache),
          original_length_(original_length) {}

    CacheRollbackGuard(
        const CacheRollbackGuard&
    ) = delete;

    CacheRollbackGuard& operator=(
        const CacheRollbackGuard&
    ) = delete;

    ~CacheRollbackGuard() noexcept {
        if (committed_) {
            return;
        }

        try {
            cache_.truncate(original_length_);
        } catch (...) {
            // This should be unreachable because run_chunk validates every
            // layer before mutation and append only increases lengths.
            // Resetting is the safest fallback if an invariant is broken.
            cache_.reset();
        }
    }

    void commit() noexcept {
        committed_ = true;
    }

private:
    KVCache& cache_;
    std::size_t original_length_;
    bool committed_ = false;
};

} // namespace

CachedAttention::CachedAttention(
    const minigpu::Device& device
)
    : device_(device),
      scores_kernel_(
          device_.compile_kernel(
              std::string(scores_kernel_name),
              scores_kernel_source
          )
      ),
      softmax_kernel_(
          device_.compile_kernel(
              std::string(softmax_kernel_name),
              softmax_kernel_source
          )
      ),
      context_kernel_(
          device_.compile_kernel(
              std::string(context_kernel_name),
              context_kernel_source
          )
      ) {}

RuntimeTensor CachedAttention::run(
    minigpu::Queue& queue,
    const RuntimeTensor& compact_queries,
    const RuntimeTensor& key_storage,
    const RuntimeTensor& value_storage,
    std::size_t valid_length,
    std::size_t query_start,
    std::size_t threads_per_group
) const {
    const AttentionDimensions dimensions =
        validate_cached_attention(
            compact_queries,
            key_storage,
            value_storage,
            valid_length,
            query_start,
            threads_per_group
        );

    const std::uint32_t query_tokens_u32 =
        require_u32(
            dimensions.query_tokens,
            "attention query token count"
        );
    const std::uint32_t valid_length_u32 =
        require_u32(
            dimensions.valid_length,
            "attention valid length"
        );
    const std::uint32_t capacity_u32 =
        require_u32(
            dimensions.capacity,
            "attention capacity"
        );
    const std::uint32_t head_dimension_u32 =
        require_u32(
            dimensions.head_dimension,
            "attention head dimension"
        );

    RuntimeTensor scores =
        RuntimeTensor::empty(
            device_,
            tensor_type(
                {
                    dimensions.heads,
                    dimensions.query_tokens,
                    dimensions.valid_length,
                }
            )
        );

    minigpu::Dispatch scores_dispatch(
        scores_kernel_
    );
    scores_dispatch
        .bind_buffer(0, compact_queries.buffer())
        .bind_buffer(1, key_storage.buffer())
        .bind_buffer(2, scores.buffer())
        .bind_u32(3, query_tokens_u32)
        .bind_u32(4, valid_length_u32)
        .bind_u32(5, capacity_u32)
        .bind_u32(6, head_dimension_u32)
        .bind_u32(
            7,
            require_u32(
                dimensions.score_count,
                "attention score count"
            )
        );

    wait_for(
        queue,
        std::move(scores_dispatch),
        elementwise_launch(
            dimensions.score_count,
            threads_per_group
        )
    );

    RuntimeTensor probabilities =
        RuntimeTensor::empty(
            device_,
            scores.type()
        );
    const float scale =
        1.0f /
        std::sqrt(
            static_cast<float>(
                dimensions.head_dimension
            )
        );

    minigpu::Dispatch softmax_dispatch(
        softmax_kernel_
    );
    softmax_dispatch
        .bind_buffer(0, scores.buffer())
        .bind_buffer(1, probabilities.buffer())
        .bind_u32(2, query_tokens_u32)
        .bind_u32(3, valid_length_u32)
        .bind_u32(
            4,
            require_u32(
                query_start,
                "attention query start"
            )
        )
        .bind_f32(5, scale);

    wait_for(
        queue,
        std::move(softmax_dispatch),
        minigpu::LaunchConfig(
            dimensions.row_count,
            softmax_threads
        )
    );

    RuntimeTensor context =
        RuntimeTensor::empty(
            device_,
            tensor_type(
                {
                    dimensions.heads,
                    dimensions.query_tokens,
                    dimensions.head_dimension,
                }
            )
        );

    minigpu::Dispatch context_dispatch(
        context_kernel_
    );
    context_dispatch
        .bind_buffer(0, probabilities.buffer())
        .bind_buffer(1, value_storage.buffer())
        .bind_buffer(2, context.buffer())
        .bind_u32(3, query_tokens_u32)
        .bind_u32(4, valid_length_u32)
        .bind_u32(5, capacity_u32)
        .bind_u32(6, head_dimension_u32)
        .bind_u32(
            7,
            require_u32(
                dimensions.context_count,
                "attention context count"
            )
        );

    wait_for(
        queue,
        std::move(context_dispatch),
        elementwise_launch(
            dimensions.context_count,
            threads_per_group
        )
    );

    return context;
}

CachedLlamaExecutor::CachedLlamaExecutor(
    const hf::HuggingFaceLlama& model
)
    : model_(model),
      cached_attention_(model_.device_),
      materializer_(model_.device_),
      binary_kernel_(
          model_.device_.compile_kernel(
              std::string(binary_kernel_name),
              binary_kernel_source
          )
      ) {
    if (
        model_.config_.attention_head_count !=
        model_.config_.key_value_head_count
    ) {
        throw std::invalid_argument(
            "Lesson 39 cached decoding does not yet "
            "support grouped-query attention"
        );
    }
}

KVCache CachedLlamaExecutor::create_cache(
    std::size_t capacity
) const {
    if (capacity == 0) {
        throw std::invalid_argument(
            "KV-cache capacity must be greater than zero"
        );
    }

    if (
        capacity >
        model_.config_.maximum_position_count
    ) {
        throw std::invalid_argument(
            "KV-cache capacity " +
            std::to_string(capacity) +
            " exceeds max_position_embeddings " +
            std::to_string(
                model_.config_.maximum_position_count
            )
        );
    }

    return KVCache(
        model_.device_,
        KVCacheConfig{
            model_.config_.layer_count,
            model_.config_.key_value_head_count,
            capacity,
            model_.config_.head_dimension(),
        }
    );
}

RuntimeTensor CachedLlamaExecutor::prefill(
    minigpu::Queue& queue,
    const std::vector<std::uint32_t>& prompt_token_ids,
    KVCache& cache
) const {
    validate_cache_geometry(model_.config_, cache);

    if (!cache.has_uniform_length()) {
        throw std::logic_error(
            "prefill requires uniform KV-cache layer "
            "lengths"
        );
    }

    if (cache.sequence_length() != 0) {
        throw std::invalid_argument(
            "prefill requires an empty KV cache; call "
            "reset() before processing a new prompt"
        );
    }

    if (prompt_token_ids.empty()) {
        throw std::invalid_argument(
            "prefill requires at least one prompt token"
        );
    }

    if (
        prompt_token_ids.size() >
        cache.config().capacity
    ) {
        throw std::length_error(
            "prompt token count exceeds KV-cache capacity"
        );
    }

    if (
        prompt_token_ids.size() >
        model_.config_.maximum_position_count
    ) {
        throw std::length_error(
            "prompt token count exceeds the model's "
            "maximum_position_count"
        );
    }

    validate_token_ids(
        model_.config_,
        prompt_token_ids
    );

    return run_chunk(
        queue,
        prompt_token_ids,
        cache,
        0
    );
}

RuntimeTensor CachedLlamaExecutor::decode_one(
    minigpu::Queue& queue,
    std::uint32_t token_id,
    KVCache& cache
) const {
    validate_cache_geometry(model_.config_, cache);

    if (!cache.has_uniform_length()) {
        throw std::logic_error(
            "decode requires uniform KV-cache layer "
            "lengths"
        );
    }

    const std::size_t query_start =
        cache.sequence_length();

    if (query_start == 0) {
        throw std::invalid_argument(
            "decode_one requires a non-empty cached "
            "prefix; call prefill first"
        );
    }

    const std::size_t next_length =
        checked_add(
            query_start,
            1,
            "decoded sequence length"
        );

    if (next_length > cache.config().capacity) {
        throw std::length_error(
            "decode_one would exceed KV-cache capacity"
        );
    }

    if (
        next_length >
        model_.config_.maximum_position_count
    ) {
        throw std::length_error(
            "decode_one would exceed the model's "
            "maximum_position_count"
        );
    }

    const std::vector<std::uint32_t> token_ids{
        token_id,
    };
    validate_token_ids(model_.config_, token_ids);

    return run_chunk(
        queue,
        token_ids,
        cache,
        query_start
    );
}

const hf::LlamaConfig&
CachedLlamaExecutor::config() const noexcept {
    return model_.config_;
}

RuntimeTensor CachedLlamaExecutor::run_chunk(
    minigpu::Queue& queue,
    const std::vector<std::uint32_t>& token_ids,
    KVCache& cache,
    std::size_t query_start
) const {
    const std::size_t query_tokens =
        token_ids.size();
    const std::size_t expected_length =
        checked_add(
            query_start,
            query_tokens,
            "cached Llama sequence length"
        );
    const std::size_t heads =
        model_.config_.attention_head_count;
    const std::size_t head_dimension =
        model_.config_.head_dimension();
    const std::size_t hidden =
        model_.config_.hidden_size;

    // Validate the complete starting state before the first layer mutates
    // the cache. After this point, the guard makes the model pass
    // transactional with respect to logical cache lengths.
    for (
        std::size_t layer_index = 0;
        layer_index < model_.layers_.size();
        ++layer_index
    ) {
        if (
            cache.valid_length(layer_index) !=
            query_start
        ) {
            throw std::logic_error(
                "KV-cache layer " +
                std::to_string(layer_index) +
                " does not begin this model chunk at the "
                "expected sequence length"
            );
        }
    }

    CacheRollbackGuard rollback_guard(
        cache,
        query_start
    );

    RuntimeTensor hidden_states =
        model_.transformer_ops_.embedding(
            queue,
            token_ids,
            model_.embedding_weight_
        );

    for (
        std::size_t layer_index = 0;
        layer_index < model_.layers_.size();
        ++layer_index
    ) {
        const TransformerBlockWeights& weights =
            model_.layers_[layer_index];

        RuntimeTensor attention_input =
            model_.transformer_ops_.rms_norm(
                queue,
                hidden_states,
                weights.attention_norm_weight,
                model_.config_.rms_norm_epsilon
            );
        RuntimeTensor query = model_.linear_.run(
            queue,
            attention_input,
            weights.query_weight
        );
        RuntimeTensor key = model_.linear_.run(
            queue,
            attention_input,
            weights.key_weight
        );
        RuntimeTensor value = model_.linear_.run(
            queue,
            attention_input,
            weights.value_weight
        );

        const Shape token_head_shape{
            query_tokens,
            heads,
            head_dimension,
        };
        RuntimeTensor query_by_token =
            reinterpret_compact(
                query,
                token_head_shape
            );
        RuntimeTensor key_by_token =
            reinterpret_compact(
                key,
                token_head_shape
            );
        RuntimeTensor value_by_token =
            reinterpret_compact(
                value,
                token_head_shape
            );

        RuntimeTensor rotated_query =
            model_.transformer_ops_.rope(
                queue,
                query_by_token,
                query_start,
                model_.config_.rope_theta
            );
        RuntimeTensor rotated_key =
            model_.transformer_ops_.rope(
                queue,
                key_by_token,
                query_start,
                model_.config_.rope_theta
            );

        RuntimeTensor query_heads =
            materializer_.run(
                queue,
                transpose(rotated_query, 0, 1)
            );
        RuntimeTensor key_heads =
            materializer_.run(
                queue,
                transpose(rotated_key, 0, 1)
            );
        RuntimeTensor value_heads =
            materializer_.run(
                queue,
                transpose(value_by_token, 0, 1)
            );

        cache.append(
            queue,
            layer_index,
            key_heads,
            value_heads
        );

        RuntimeTensor context_heads =
            cached_attention_.run(
                queue,
                query_heads,
                cache.key_storage(layer_index),
                cache.value_storage(layer_index),
                expected_length,
                query_start
            );
        RuntimeTensor context_by_token =
            materializer_.run(
                queue,
                transpose(context_heads, 0, 1)
            );
        RuntimeTensor context =
            reinterpret_compact(
                context_by_token,
                {query_tokens, hidden}
            );
        RuntimeTensor attention_output =
            model_.linear_.run(
                queue,
                context,
                weights.attention_output_weight
            );
        RuntimeTensor after_attention =
            run_binary(
                model_.device_,
                binary_kernel_,
                queue,
                hidden_states,
                attention_output,
                BinaryOperation::Add
            );

        RuntimeTensor feed_forward_input =
            model_.transformer_ops_.rms_norm(
                queue,
                after_attention,
                weights.ffn_norm_weight,
                model_.config_.rms_norm_epsilon
            );
        RuntimeTensor gate = model_.linear_.run(
            queue,
            feed_forward_input,
            weights.gate_weight
        );
        RuntimeTensor activated_gate =
            model_.transformer_ops_.silu(
                queue,
                gate
            );
        RuntimeTensor up = model_.linear_.run(
            queue,
            feed_forward_input,
            weights.up_weight
        );
        RuntimeTensor gated = run_binary(
            model_.device_,
            binary_kernel_,
            queue,
            activated_gate,
            up,
            BinaryOperation::Multiply
        );
        RuntimeTensor down = model_.linear_.run(
            queue,
            gated,
            weights.down_weight
        );

        hidden_states = run_binary(
            model_.device_,
            binary_kernel_,
            queue,
            after_attention,
            down,
            BinaryOperation::Add
        );
    }

    if (
        !cache.has_uniform_length() ||
        cache.sequence_length() != expected_length
    ) {
        throw std::logic_error(
            "cached Llama pass did not advance every "
            "layer to the expected sequence length"
        );
    }

    RuntimeTensor normalized =
        model_.transformer_ops_.rms_norm(
            queue,
            hidden_states,
            model_.final_norm_weight_,
            model_.config_.rms_norm_epsilon
        );

    RuntimeTensor logits = model_.linear_.run(
        queue,
        normalized,
        model_.lm_head_weight_
    );

    rollback_guard.commit();
    return logits;
}

} // namespace minitensor::inference
