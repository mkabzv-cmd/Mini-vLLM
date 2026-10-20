#include "transformer_ops.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace minitensor {

namespace {

constexpr std::size_t default_rms_threads = 64;

constexpr std::string_view embedding_source = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void embedding_lookup(
    device const float* table [[buffer(0)]],
    device const float* token_ids [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& element_count [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    constant uint& vocabulary_size [[buffer(5)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id >= element_count) {
        return;
    }

    const uint token_index = global_id / hidden_size;
    const uint hidden_index = global_id % hidden_size;
    const uint token_id = uint(token_ids[token_index]);

    // Host validation makes this branch unreachable. Keeping the guard in
    // the kernel prevents an accidental out-of-bounds table read if that
    // contract is changed later.
    if (token_id >= vocabulary_size) {
        output[global_id] = 0.0f;
        return;
    }

    output[global_id] =
        table[token_id * hidden_size + hidden_index];
}
)metal";

constexpr std::string_view rms_norm_source = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_rows(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& hidden_size [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]]
) {
    threadgroup float partial_sums[64];

    const uint row_start = row * hidden_size;
    float local_sum = 0.0f;

    for (
        uint column = local_id;
        column < hidden_size;
        column += 64
    ) {
        const float value = input[row_start + column];
        local_sum += value * value;
    }

    partial_sums[local_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All 64 lanes reach every barrier. At each step, the lower half adds
    // values from the upper half until lane 0 owns the complete row sum.
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            partial_sums[local_id] +=
                partial_sums[local_id + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inverse_rms =
        rsqrt(partial_sums[0] / float(hidden_size) + epsilon);

    for (
        uint column = local_id;
        column < hidden_size;
        column += 64
    ) {
        output[row_start + column] =
            input[row_start + column] *
            inverse_rms *
            weight[column];
    }
}
)metal";

constexpr std::string_view rope_source = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void llama_split_half_rope(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& pair_count [[buffer(2)]],
    constant uint& head_count [[buffer(3)]],
    constant uint& head_dimension [[buffer(4)]],
    constant uint& start_position [[buffer(5)]],
    constant float& theta [[buffer(6)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id >= pair_count) {
        return;
    }

    const uint half_dimension = head_dimension / 2;
    const uint pairs_per_token =
        head_count * half_dimension;
    const uint token = global_id / pairs_per_token;
    const uint within_token = global_id % pairs_per_token;
    const uint head = within_token / half_dimension;
    const uint pair = within_token % half_dimension;

    const uint vector_start =
        (token * head_count + head) * head_dimension;
    const uint first_index = vector_start + pair;
    const uint second_index =
        first_index + half_dimension;

    const uint position = start_position + token;
    const float exponent =
        (2.0f * float(pair)) / float(head_dimension);
    const float inverse_frequency =
        1.0f / pow(theta, exponent);
    const float angle =
        float(position) * inverse_frequency;
    const float cosine = cos(angle);
    const float sine = sin(angle);

    const float first = input[first_index];
    const float second = input[second_index];

    output[first_index] =
        first * cosine - second * sine;
    output[second_index] =
        second * cosine + first * sine;
}
)metal";

constexpr std::string_view silu_source = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void silu_elementwise(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id >= element_count) {
        return;
    }

    const float value = input[global_id];
    output[global_id] =
        value / (1.0f + exp(-value));
}
)metal";

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
            " does not fit the Lesson 33 uint32 "
            "Metal indexing model"
        );
    }

    return static_cast<std::uint32_t>(value);
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
        throw std::overflow_error(description);
    }

    return left * right;
}

void require_threads_per_group(
    std::size_t threads_per_group
) {
    if (threads_per_group == 0) {
        throw std::invalid_argument(
            "threads_per_group must be greater than zero"
        );
    }
}

void require_compact(
    const RuntimeTensor& tensor,
    const char* description
) {
    if (
        !tensor.is_contiguous() ||
        tensor.offset() != 0
    ) {
        throw std::invalid_argument(
            std::string(description) +
            " must use compact row-major storage "
            "(canonical strides and offset zero)"
        );
    }
}

[[nodiscard]] minigpu::LaunchConfig elementwise_launch(
    std::size_t element_count,
    std::size_t threads_per_group
) {
    require_threads_per_group(threads_per_group);
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

[[nodiscard]] miniir::Type tensor_type(
    Shape shape
) {
    return miniir::Type::tensor(
        miniir::DType::F32,
        std::move(shape)
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

} // namespace

TransformerOps::TransformerOps(
    const minigpu::Device& device
)
    : device_(device),
      embedding_kernel_(
          device_.compile_kernel(
              "embedding_lookup",
              embedding_source
          )
      ),
      rms_norm_kernel_(
          device_.compile_kernel(
              "rms_norm_rows",
              rms_norm_source
          )
      ),
      rope_kernel_(
          device_.compile_kernel(
              "llama_split_half_rope",
              rope_source
          )
      ),
      silu_kernel_(
          device_.compile_kernel(
              "silu_elementwise",
              silu_source
          )
      ) {}

RuntimeTensor TransformerOps::embedding(
    minigpu::Queue& queue,
    const std::vector<std::uint32_t>& token_ids,
    const RuntimeTensor& embedding_table,
    std::size_t threads_per_group
) const {
    require_compact(
        embedding_table,
        "embedding table"
    );

    if (embedding_table.rank() != 2) {
        throw std::invalid_argument(
            "embedding table must have shape "
            "[vocabulary_size, hidden_size]"
        );
    }

    if (token_ids.empty()) {
        throw std::invalid_argument(
            "embedding lookup requires at least one token ID"
        );
    }

    const std::size_t vocabulary_size =
        embedding_table.shape()[0];
    const std::size_t hidden_size =
        embedding_table.shape()[1];

    const std::uint32_t vocabulary_u32 =
        require_u32(
            vocabulary_size,
            "embedding vocabulary size"
        );
    const std::uint32_t hidden_u32 =
        require_u32(
            hidden_size,
            "embedding hidden size"
        );
    static_cast<void>(
        require_u32(
            embedding_table.numel(),
            "embedding table element count"
        )
    );
    static_cast<void>(
        require_u32(
            token_ids.size(),
            "embedding token count"
        )
    );

    std::vector<float> encoded_ids;
    encoded_ids.reserve(token_ids.size());

    for (std::uint32_t token_id : token_ids) {
        const float encoded =
            static_cast<float>(token_id);

        if (
            static_cast<double>(encoded) !=
            static_cast<double>(token_id)
        ) {
            throw std::invalid_argument(
                "token ID " +
                std::to_string(token_id) +
                " cannot be represented exactly by the "
                "temporary f32 ID tensor"
            );
        }

        if (
            static_cast<std::size_t>(token_id) >=
            vocabulary_size
        ) {
            throw std::out_of_range(
                "token ID " +
                std::to_string(token_id) +
                " is outside embedding vocabulary size " +
                std::to_string(vocabulary_size)
            );
        }

        encoded_ids.push_back(encoded);
    }

    RuntimeTensor encoded_token_ids =
        RuntimeTensor::from_host(
            device_,
            tensor_type({token_ids.size()}),
            std::move(encoded_ids)
        );

    RuntimeTensor output =
        RuntimeTensor::empty(
            device_,
            tensor_type(
                {token_ids.size(), hidden_size}
            )
        );

    const minigpu::LaunchConfig launch =
        elementwise_launch(
            output.numel(),
            threads_per_group
        );

    minigpu::Dispatch dispatch(embedding_kernel_);
    dispatch
        .bind_buffer(0, embedding_table.buffer())
        .bind_buffer(1, encoded_token_ids.buffer())
        .bind_buffer(2, output.buffer())
        .bind_u32(
            3,
            require_u32(
                output.numel(),
                "embedding output element count"
            )
        )
        .bind_u32(4, hidden_u32)
        .bind_u32(5, vocabulary_u32);

    wait_for(
        queue,
        std::move(dispatch),
        launch
    );
    return output;
}

RuntimeTensor TransformerOps::rms_norm(
    minigpu::Queue& queue,
    const RuntimeTensor& input,
    const RuntimeTensor& weight,
    float epsilon
) const {
    require_compact(input, "RMSNorm input");
    require_compact(weight, "RMSNorm weight");

    if (input.rank() == 0) {
        throw std::invalid_argument(
            "RMSNorm input must have at least one dimension"
        );
    }

    const std::size_t hidden_size =
        input.shape().back();

    if (
        weight.rank() != 1 ||
        weight.shape()[0] != hidden_size
    ) {
        throw std::invalid_argument(
            "RMSNorm weight must have shape [hidden_size]"
        );
    }

    if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
        throw std::invalid_argument(
            "RMSNorm epsilon must be finite and greater "
            "than zero"
        );
    }

    static_cast<void>(
        require_u32(
            input.numel(),
            "RMSNorm input element count"
        )
    );
    const std::uint32_t hidden_u32 =
        require_u32(
            hidden_size,
            "RMSNorm hidden size"
        );
    const std::size_t row_count =
        input.numel() / hidden_size;
    static_cast<void>(
        require_u32(
            row_count,
            "RMSNorm row count"
        )
    );

    RuntimeTensor output =
        RuntimeTensor::empty(device_, input.type());

    minigpu::Dispatch dispatch(rms_norm_kernel_);
    dispatch
        .bind_buffer(0, input.buffer())
        .bind_buffer(1, weight.buffer())
        .bind_buffer(2, output.buffer())
        .bind_u32(3, hidden_u32)
        .bind_f32(4, epsilon);

    wait_for(
        queue,
        std::move(dispatch),
        minigpu::LaunchConfig(
            row_count,
            default_rms_threads
        )
    );
    return output;
}

RuntimeTensor TransformerOps::rope(
    minigpu::Queue& queue,
    const RuntimeTensor& input,
    std::size_t start_position,
    float theta,
    std::size_t threads_per_group
) const {
    require_compact(input, "RoPE input");

    if (input.rank() != 3) {
        throw std::invalid_argument(
            "RoPE input must have shape "
            "[tokens, heads, head_dimension]"
        );
    }

    const std::size_t token_count = input.shape()[0];
    const std::size_t head_count = input.shape()[1];
    const std::size_t head_dimension = input.shape()[2];

    if (head_dimension % 2 != 0) {
        throw std::invalid_argument(
            "RoPE head_dimension must be even"
        );
    }

    if (!std::isfinite(theta) || theta <= 0.0f) {
        throw std::invalid_argument(
            "RoPE theta must be finite and greater than zero"
        );
    }

    if (
        start_position >
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()
        ) ||
        token_count - 1 >
            static_cast<std::size_t>(
                std::numeric_limits<std::uint32_t>::max()
            ) - start_position
    ) {
        throw std::overflow_error(
            "RoPE final token position does not fit uint32"
        );
    }

    static_cast<void>(
        require_u32(
            input.numel(),
            "RoPE input element count"
        )
    );
    const std::uint32_t heads_u32 =
        require_u32(head_count, "RoPE head count");
    const std::uint32_t head_dimension_u32 =
        require_u32(
            head_dimension,
            "RoPE head dimension"
        );
    const std::uint32_t start_position_u32 =
        require_u32(
            start_position,
            "RoPE start position"
        );

    const std::size_t half_dimension =
        head_dimension / 2;
    const std::size_t pairs_per_token =
        checked_multiply(
            head_count,
            half_dimension,
            "RoPE pair count overflows size_t"
        );
    const std::size_t pair_count =
        checked_multiply(
            token_count,
            pairs_per_token,
            "RoPE pair count overflows size_t"
        );
    const std::uint32_t pair_count_u32 =
        require_u32(
            pair_count,
            "RoPE pair count"
        );

    RuntimeTensor output =
        RuntimeTensor::empty(device_, input.type());
    const minigpu::LaunchConfig launch =
        elementwise_launch(
            pair_count,
            threads_per_group
        );

    minigpu::Dispatch dispatch(rope_kernel_);
    dispatch
        .bind_buffer(0, input.buffer())
        .bind_buffer(1, output.buffer())
        .bind_u32(2, pair_count_u32)
        .bind_u32(3, heads_u32)
        .bind_u32(4, head_dimension_u32)
        .bind_u32(5, start_position_u32)
        .bind_f32(6, theta);

    wait_for(
        queue,
        std::move(dispatch),
        launch
    );
    return output;
}

RuntimeTensor TransformerOps::silu(
    minigpu::Queue& queue,
    const RuntimeTensor& input,
    std::size_t threads_per_group
) const {
    require_compact(input, "SiLU input");

    RuntimeTensor output =
        RuntimeTensor::empty(device_, input.type());
    const minigpu::LaunchConfig launch =
        elementwise_launch(
            input.numel(),
            threads_per_group
        );

    minigpu::Dispatch dispatch(silu_kernel_);
    dispatch
        .bind_buffer(0, input.buffer())
        .bind_buffer(1, output.buffer())
        .bind_u32(
            2,
            require_u32(
                input.numel(),
                "SiLU input element count"
            )
        );

    wait_for(
        queue,
        std::move(dispatch),
        launch
    );
    return output;
}

} // namespace minitensor
