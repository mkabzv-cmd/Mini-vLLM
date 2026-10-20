#include "batched_matmul.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace minitensor {

namespace {

constexpr std::string_view kernel_name =
    "batched_matmul";

constexpr std::string_view kernel_source = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void batched_matmul(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& m_size [[buffer(4)]],
    constant uint& n_size [[buffer(5)]],
    constant uint& k_size [[buffer(6)]],
    constant uint& left_offset [[buffer(7)]],
    constant uint& right_offset [[buffer(8)]],
    constant uint& transpose_right [[buffer(9)]],
    uint global_id [[thread_position_in_grid]]
) {
    const uint output_count =
        batch_size * m_size * n_size;

    if (global_id >= output_count) {
        return;
    }

    const uint column = global_id % n_size;
    const uint matrix_index = global_id / n_size;
    const uint row = matrix_index % m_size;
    const uint batch = matrix_index / m_size;

    const uint left_base =
        left_offset +
        (batch * m_size + row) * k_size;

    float sum = 0.0f;

    for (uint inner = 0; inner < k_size; ++inner) {
        uint right_index;

        if (transpose_right == 0) {
            right_index =
                right_offset +
                (batch * k_size + inner) * n_size +
                column;
        } else {
            right_index =
                right_offset +
                (batch * n_size + column) * k_size +
                inner;
        }

        sum +=
            left[left_base + inner] *
            right[right_index];
    }

    output[global_id] = sum;
}
)metal";

constexpr std::size_t left_binding = 0;
constexpr std::size_t right_binding = 1;
constexpr std::size_t output_binding = 2;
constexpr std::size_t batch_binding = 3;
constexpr std::size_t m_binding = 4;
constexpr std::size_t n_binding = 5;
constexpr std::size_t k_binding = 6;
constexpr std::size_t left_offset_binding = 7;
constexpr std::size_t right_offset_binding = 8;
constexpr std::size_t transpose_right_binding = 9;

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
            " does not fit the Lesson 32 uint32 "
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
            std::numeric_limits<std::size_t>::max() /
                left
    ) {
        throw std::overflow_error(
            std::string(description) +
            " overflows size_t"
        );
    }

    return left * right;
}

void require_rank_three(
    const RuntimeTensor& tensor,
    const char* name
) {
    if (tensor.rank() != 3) {
        throw std::invalid_argument(
            std::string("batched matmul ") + name +
            " must have rank 3; received rank " +
            std::to_string(tensor.rank())
        );
    }
}

void require_contiguous(
    const RuntimeTensor& tensor,
    const char* name
) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(
            std::string("batched matmul ") + name +
            " must use contiguous row-major storage; "
            "materialize the view first"
        );
    }
}

void require_addressable_u32(
    const RuntimeTensor& tensor,
    const char* description
) {
    static_cast<void>(
        require_u32(
            tensor.layout().minimum_storage_size() - 1,
            description
        )
    );
}

} // namespace

BatchedMatmul::BatchedMatmul(
    const minigpu::Device& device
)
    : device_(device),
      kernel_(
          device_.compile_kernel(
              std::string(kernel_name),
              kernel_source
          )
      ) {}

RuntimeTensor BatchedMatmul::run(
    minigpu::Queue& queue,
    const RuntimeTensor& left,
    const RuntimeTensor& right,
    bool transpose_right,
    std::size_t threads_per_group
) const {
    if (threads_per_group == 0) {
        throw std::invalid_argument(
            "threads_per_group must be greater than zero"
        );
    }

    require_rank_three(left, "left input");
    require_rank_three(right, "right input");
    require_contiguous(left, "left input");
    require_contiguous(right, "right input");

    const std::size_t batch = left.shape()[0];
    const std::size_t m = left.shape()[1];
    const std::size_t k = left.shape()[2];

    if (right.shape()[0] != batch) {
        throw std::invalid_argument(
            "batched matmul batch dimensions do not "
            "match: left B=" +
            std::to_string(batch) +
            ", right B=" +
            std::to_string(right.shape()[0])
        );
    }

    const std::size_t right_k =
        transpose_right
            ? right.shape()[2]
            : right.shape()[1];
    const std::size_t n =
        transpose_right
            ? right.shape()[1]
            : right.shape()[2];

    if (right_k != k) {
        throw std::invalid_argument(
            "batched matmul inner dimensions do not "
            "match: left K=" +
            std::to_string(k) +
            ", right K=" +
            std::to_string(right_k)
        );
    }

    const std::size_t matrices =
        checked_multiply(
            batch,
            m,
            "batched matmul B*M"
        );
    const std::size_t output_count =
        checked_multiply(
            matrices,
            n,
            "batched matmul output element count"
        );

    const std::uint32_t batch_u32 =
        require_u32(batch, "batched matmul B");
    const std::uint32_t m_u32 =
        require_u32(m, "batched matmul M");
    const std::uint32_t n_u32 =
        require_u32(n, "batched matmul N");
    const std::uint32_t k_u32 =
        require_u32(k, "batched matmul K");
    static_cast<void>(
        require_u32(
            output_count,
            "batched matmul output element count"
        )
    );

    require_addressable_u32(
        left,
        "highest batched-matmul left storage index"
    );
    require_addressable_u32(
        right,
        "highest batched-matmul right storage index"
    );

    const minigpu::LaunchConfig launch =
        minigpu::LaunchConfig::cover(
            output_count,
            threads_per_group
        );
    static_cast<void>(
        require_u32(
            launch.total_threads(),
            "batched matmul launched thread count"
        )
    );

    const std::uint32_t left_offset =
        require_u32(
            left.offset(),
            "batched matmul left offset"
        );
    const std::uint32_t right_offset =
        require_u32(
            right.offset(),
            "batched matmul right offset"
        );
    const std::uint32_t transpose_value =
        transpose_right ? 1 : 0;

    miniir::Type output_type =
        miniir::Type::tensor(
            miniir::DType::F32,
            {batch, m, n}
        );
    RuntimeTensor output =
        RuntimeTensor::empty(
            device_,
            std::move(output_type)
        );

    minigpu::Dispatch dispatch(kernel_);
    dispatch
        .bind_buffer(left_binding, left.buffer())
        .bind_buffer(right_binding, right.buffer())
        .bind_buffer(output_binding, output.buffer())
        .bind_u32(batch_binding, batch_u32)
        .bind_u32(m_binding, m_u32)
        .bind_u32(n_binding, n_u32)
        .bind_u32(k_binding, k_u32)
        .bind_u32(
            left_offset_binding,
            left_offset
        )
        .bind_u32(
            right_offset_binding,
            right_offset
        )
        .bind_u32(
            transpose_right_binding,
            transpose_value
        );

    minigpu::Event finished = queue.submit(
        std::move(dispatch),
        launch
    );
    finished.wait();

    return output;
}

} // namespace minitensor
