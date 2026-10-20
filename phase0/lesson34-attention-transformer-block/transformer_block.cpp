#include "transformer_block.hpp"

#include "../lesson26-tensor-views/tensor_views.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace minitensor {

namespace {

constexpr std::string_view binary_kernel_name =
    "transformer_binary";

constexpr std::string_view binary_kernel_source = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void transformer_binary(
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

constexpr std::string_view softmax_kernel_name =
    "causal_attention_softmax";

constexpr std::string_view softmax_kernel_source = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void causal_attention_softmax(
    device const float* scores [[buffer(0)]],
    device float* probabilities [[buffer(1)]],
    constant uint& tokens [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint lane [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[64];

    const uint head = group_id / tokens;
    const uint query = group_id % tokens;
    const uint row_start =
        (head * tokens + query) * tokens;

    float lane_maximum = -3.402823466e+38f;

    for (
        uint key = lane;
        key < tokens;
        key += 64
    ) {
        if (key <= query) {
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
        key < tokens;
        key += 64
    ) {
        if (key <= query) {
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
        key < tokens;
        key += 64
    ) {
        probabilities[row_start + key] =
            key <= query
                ? exp(
                      scores[row_start + key] * scale -
                      row_maximum
                  ) * inverse_sum
                : 0.0f;
    }
}
)metal";

constexpr std::size_t softmax_threads = 64;

enum class BinaryOperation : std::uint32_t {
    Add = 0,
    Multiply = 1,
};

[[nodiscard]] std::size_t checked_multiply(
    std::size_t left,
    std::size_t right,
    const char* message
) {
    if (
        left != 0 &&
        right >
            std::numeric_limits<std::size_t>::max() / left
    ) {
        throw std::overflow_error(message);
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
            " does not fit the Lesson 34 uint32 "
            "Metal indexing model"
        );
    }

    return static_cast<std::uint32_t>(value);
}

void require_compact(
    const RuntimeTensor& tensor,
    std::string_view name
) {
    if (!tensor.is_contiguous() || tensor.offset() != 0) {
        throw std::invalid_argument(
            std::string(name) +
            " must be contiguous with offset zero"
        );
    }
}

[[nodiscard]] std::string format_shape(
    const Shape& shape
) {
    std::string result = "[";

    for (
        std::size_t index = 0;
        index < shape.size();
        ++index
    ) {
        if (index != 0) {
            result += ", ";
        }

        result += std::to_string(shape[index]);
    }

    result += "]";
    return result;
}

void require_shape(
    const RuntimeTensor& tensor,
    const Shape& expected,
    std::string_view name
) {
    if (tensor.shape() != expected) {
        throw std::invalid_argument(
            std::string(name) + " has type " +
            miniir::format_type(tensor.type()) +
            ", expected shape " +
            format_shape(expected)
        );
    }

    require_compact(tensor, name);
}

[[nodiscard]] std::size_t shape_elements(
    const Shape& shape
) {
    std::size_t count = 1;

    for (std::size_t dimension : shape) {
        count = checked_multiply(
            count,
            dimension,
            "transformer reshape element count "
            "overflows size_t"
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
            "transformer reshape changes element count"
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
            "transformer binary operands must have "
            "the same type"
        );
    }

    require_compact(left, "left binary operand");
    require_compact(right, "right binary operand");

    RuntimeTensor output =
        RuntimeTensor::empty(device, left.type());
    const minigpu::LaunchConfig launch =
        minigpu::LaunchConfig::cover(
            left.numel(),
            64
        );

    static_cast<void>(
        require_u32(
            launch.total_threads(),
            "binary launched thread count"
        )
    );

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

    minigpu::Event finished = queue.submit(
        std::move(dispatch),
        launch
    );
    finished.wait();
    return output;
}

[[nodiscard]] RuntimeTensor run_causal_softmax(
    const minigpu::Device& device,
    const minigpu::Kernel& kernel,
    minigpu::Queue& queue,
    const RuntimeTensor& scores,
    float scale
) {
    require_compact(scores, "attention scores");

    if (
        scores.rank() != 3 ||
        scores.shape()[1] != scores.shape()[2]
    ) {
        throw std::invalid_argument(
            "attention scores must have shape "
            "[heads, tokens, tokens]"
        );
    }

    if (!std::isfinite(scale) || scale <= 0.0f) {
        throw std::invalid_argument(
            "attention scale must be finite and positive"
        );
    }

    const std::size_t heads = scores.shape()[0];
    const std::size_t tokens = scores.shape()[1];
    const std::size_t row_count =
        checked_multiply(
            heads,
            tokens,
            "attention row count overflows size_t"
        );
    static_cast<void>(
        require_u32(
            row_count,
            "attention row count"
        )
    );

    RuntimeTensor output =
        RuntimeTensor::empty(device, scores.type());

    minigpu::Dispatch dispatch(kernel);
    dispatch
        .bind_buffer(0, scores.buffer())
        .bind_buffer(1, output.buffer())
        .bind_u32(
            2,
            require_u32(tokens, "attention token count")
        )
        .bind_f32(3, scale);

    minigpu::Event finished = queue.submit(
        std::move(dispatch),
        minigpu::LaunchConfig(
            row_count,
            softmax_threads
        )
    );
    finished.wait();
    return output;
}

struct AttentionDimensions {
    std::size_t tokens;
    std::size_t hidden;
    std::size_t heads;
    std::size_t head_dimension;
};

[[nodiscard]] AttentionDimensions
validate_attention_inputs(
    const RuntimeTensor& hidden,
    const TransformerBlockWeights& weights,
    std::size_t heads
) {
    require_compact(hidden, "attention input");

    if (hidden.rank() != 2) {
        throw std::invalid_argument(
            "attention input must have shape "
            "[tokens, hidden]"
        );
    }

    if (heads == 0) {
        throw std::invalid_argument(
            "attention head count must be positive"
        );
    }

    const std::size_t tokens = hidden.shape()[0];
    const std::size_t hidden_size = hidden.shape()[1];

    if (hidden_size % heads != 0) {
        throw std::invalid_argument(
            "hidden size must be divisible by heads"
        );
    }

    const std::size_t head_dimension =
        hidden_size / heads;

    if (head_dimension % 2 != 0) {
        throw std::invalid_argument(
            "RoPE requires an even head dimension"
        );
    }

    const Shape projection_shape{
        hidden_size,
        hidden_size,
    };

    require_shape(
        weights.query_weight,
        projection_shape,
        "query weight"
    );
    require_shape(
        weights.key_weight,
        projection_shape,
        "key weight"
    );
    require_shape(
        weights.value_weight,
        projection_shape,
        "value weight"
    );
    require_shape(
        weights.attention_output_weight,
        projection_shape,
        "attention output weight"
    );

    return AttentionDimensions{
        tokens,
        hidden_size,
        heads,
        head_dimension,
    };
}

void validate_full_block_weights(
    const RuntimeTensor& hidden,
    const TransformerBlockWeights& weights
) {
    const std::size_t hidden_size = hidden.shape()[1];

    require_shape(
        weights.attention_norm_weight,
        {hidden_size},
        "attention norm weight"
    );
    require_shape(
        weights.ffn_norm_weight,
        {hidden_size},
        "feed-forward norm weight"
    );

    if (
        weights.gate_weight.rank() != 2 ||
        weights.gate_weight.shape()[1] != hidden_size
    ) {
        throw std::invalid_argument(
            "gate weight must have shape "
            "[intermediate, hidden]"
        );
    }

    require_compact(weights.gate_weight, "gate weight");

    const std::size_t intermediate =
        weights.gate_weight.shape()[0];

    require_shape(
        weights.up_weight,
        {intermediate, hidden_size},
        "up weight"
    );
    require_shape(
        weights.down_weight,
        {hidden_size, intermediate},
        "down weight"
    );
}

} // namespace

MiniTransformerBlock::MiniTransformerBlock(
    const minigpu::Device& device
)
    : device_(device),
      linear_(device_),
      batched_matmul_(device_),
      transformer_ops_(device_),
      materializer_(device_),
      binary_kernel_(
          device_.compile_kernel(
              std::string(binary_kernel_name),
              binary_kernel_source
          )
      ),
      causal_softmax_kernel_(
          device_.compile_kernel(
              std::string(softmax_kernel_name),
              softmax_kernel_source
          )
      ) {}

RuntimeTensor MiniTransformerBlock::causal_self_attention(
    minigpu::Queue& queue,
    const RuntimeTensor& normalized_hidden,
    const TransformerBlockWeights& weights,
    std::size_t heads,
    std::size_t start_position,
    float rope_theta
) const {
    const AttentionDimensions dimensions =
        validate_attention_inputs(
            normalized_hidden,
            weights,
            heads
        );

    RuntimeTensor query = linear_.run(
        queue,
        normalized_hidden,
        weights.query_weight
    );
    RuntimeTensor key = linear_.run(
        queue,
        normalized_hidden,
        weights.key_weight
    );
    RuntimeTensor value = linear_.run(
        queue,
        normalized_hidden,
        weights.value_weight
    );

    const Shape token_head_shape{
        dimensions.tokens,
        dimensions.heads,
        dimensions.head_dimension,
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
        transformer_ops_.rope(
            queue,
            query_by_token,
            start_position,
            rope_theta
        );
    RuntimeTensor rotated_key =
        transformer_ops_.rope(
            queue,
            key_by_token,
            start_position,
            rope_theta
        );

    RuntimeTensor query_heads_view =
        transpose(rotated_query, 0, 1);
    RuntimeTensor key_heads_view =
        transpose(rotated_key, 0, 1);
    RuntimeTensor value_heads_view =
        transpose(value_by_token, 0, 1);

    RuntimeTensor query_heads =
        materializer_.run(queue, query_heads_view);
    RuntimeTensor key_heads =
        materializer_.run(queue, key_heads_view);
    RuntimeTensor value_heads =
        materializer_.run(queue, value_heads_view);

    RuntimeTensor scores = batched_matmul_.run(
        queue,
        query_heads,
        key_heads,
        true
    );

    const float scale =
        1.0f /
        std::sqrt(
            static_cast<float>(
                dimensions.head_dimension
            )
        );
    RuntimeTensor probabilities =
        run_causal_softmax(
            device_,
            causal_softmax_kernel_,
            queue,
            scores,
            scale
        );

    RuntimeTensor context_heads =
        batched_matmul_.run(
            queue,
            probabilities,
            value_heads,
            false
        );
    RuntimeTensor context_by_token_view =
        transpose(context_heads, 0, 1);
    RuntimeTensor context_by_token =
        materializer_.run(
            queue,
            context_by_token_view
        );
    RuntimeTensor context = reinterpret_compact(
        context_by_token,
        {
            dimensions.tokens,
            dimensions.hidden,
        }
    );

    return linear_.run(
        queue,
        context,
        weights.attention_output_weight
    );
}

RuntimeTensor MiniTransformerBlock::run(
    minigpu::Queue& queue,
    const RuntimeTensor& hidden,
    const TransformerBlockWeights& weights,
    std::size_t heads,
    std::size_t start_position,
    float epsilon,
    float rope_theta
) const {
    // This also validates the attention projections and head geometry.
    static_cast<void>(
        validate_attention_inputs(
            hidden,
            weights,
            heads
        )
    );
    validate_full_block_weights(hidden, weights);

    RuntimeTensor attention_input =
        transformer_ops_.rms_norm(
            queue,
            hidden,
            weights.attention_norm_weight,
            epsilon
        );
    RuntimeTensor attention_output =
        causal_self_attention(
            queue,
            attention_input,
            weights,
            heads,
            start_position,
            rope_theta
        );
    RuntimeTensor after_attention = run_binary(
        device_,
        binary_kernel_,
        queue,
        hidden,
        attention_output,
        BinaryOperation::Add
    );

    RuntimeTensor feed_forward_input =
        transformer_ops_.rms_norm(
            queue,
            after_attention,
            weights.ffn_norm_weight,
            epsilon
        );
    RuntimeTensor gate = linear_.run(
        queue,
        feed_forward_input,
        weights.gate_weight
    );
    RuntimeTensor activated_gate =
        transformer_ops_.silu(queue, gate);
    RuntimeTensor up = linear_.run(
        queue,
        feed_forward_input,
        weights.up_weight
    );
    RuntimeTensor gated = run_binary(
        device_,
        binary_kernel_,
        queue,
        activated_gate,
        up,
        BinaryOperation::Multiply
    );
    RuntimeTensor down = linear_.run(
        queue,
        gated,
        weights.down_weight
    );

    return run_binary(
        device_,
        binary_kernel_,
        queue,
        after_attention,
        down,
        BinaryOperation::Add
    );
}

} // namespace minitensor
