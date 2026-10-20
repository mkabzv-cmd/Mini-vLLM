#include "naive_matmul.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace minitensor {

namespace {

constexpr std::string_view kernel_name = "naive_matmul";

constexpr std::string_view kernel_source = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void naive_matmul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    constant uint& columns [[buffer(5)]],
    uint global_id [[thread_position_in_grid]]
) {
    const uint output_count = rows * columns;

    if (global_id >= output_count) {
        return;
    }

    const uint row = global_id / columns;
    const uint column = global_id % columns;

    float sum = 0.0f;

    for (uint k = 0; k < inner; ++k) {
        sum +=
            a[row * inner + k] *
            b[k * columns + column];
    }

    output[global_id] = sum;
}
)metal";

constexpr std::size_t a_binding = 0;
constexpr std::size_t b_binding = 1;
constexpr std::size_t output_binding = 2;
constexpr std::size_t rows_binding = 3;
constexpr std::size_t inner_binding = 4;
constexpr std::size_t columns_binding = 5;

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
            " does not fit the Lesson 29 uint32 "
            "Metal indexing model"
        );
    }

    return static_cast<std::uint32_t>(value);
}

void require_compact_matrix(
    const RuntimeTensor& tensor,
    const char* name
) {
    if (tensor.type().dtype() != miniir::DType::F32) {
        throw std::invalid_argument(
            std::string(name) + " must use f32"
        );
    }

    if (tensor.rank() != 2) {
        throw std::invalid_argument(
            std::string(name) +
            " must have rank 2; received rank " +
            std::to_string(tensor.rank())
        );
    }

    if (
        !tensor.is_contiguous() ||
        tensor.offset() != 0 ||
        tensor.storage_size() != tensor.numel()
    ) {
        throw std::invalid_argument(
            std::string(name) +
            " must use compact row-major storage"
        );
    }
}

struct MatmulShape {
    std::uint32_t rows;
    std::uint32_t inner;
    std::uint32_t columns;
    std::size_t output_count;
};

[[nodiscard]] MatmulShape validate_inputs(
    const RuntimeTensor& a,
    const RuntimeTensor& b
) {
    require_compact_matrix(a, "A");
    require_compact_matrix(b, "B");

    const std::size_t rows = a.shape()[0];
    const std::size_t inner = a.shape()[1];
    const std::size_t b_inner = b.shape()[0];
    const std::size_t columns = b.shape()[1];

    if (inner != b_inner) {
        throw std::invalid_argument(
            "matmul inner dimensions must match: "
            "A is [" +
            std::to_string(rows) + ", " +
            std::to_string(inner) +
            "] and B is [" +
            std::to_string(b_inner) + ", " +
            std::to_string(columns) + "]"
        );
    }

    // If each flat buffer length fits uint32, every index formed by the
    // Metal kernel also fits uint32.
    static_cast<void>(
        require_u32(a.numel(), "A element count")
    );
    static_cast<void>(
        require_u32(b.numel(), "B element count")
    );

    const std::size_t output_count =
        checked_multiply(
            rows,
            columns,
            "output element count"
        );

    // RuntimeTensor shapes have already checked size_t multiplication. The
    // explicit output check protects the kernel's uint arithmetic.
    static_cast<void>(
        require_u32(output_count, "output element count")
    );

    return MatmulShape{
        require_u32(rows, "M dimension"),
        require_u32(inner, "K dimension"),
        require_u32(columns, "N dimension"),
        output_count,
    };
}

} // namespace

NaiveMatmul::NaiveMatmul(
    const minigpu::Device& device
)
    : device_(device),
      kernel_(
          device_.compile_kernel(
              std::string(kernel_name),
              kernel_source
          )
      ) {}

RuntimeTensor NaiveMatmul::run(
    minigpu::Queue& queue,
    const RuntimeTensor& a,
    const RuntimeTensor& b,
    std::size_t threads_per_group
) const {
    if (threads_per_group == 0) {
        throw std::invalid_argument(
            "threads_per_group must be greater than zero"
        );
    }

    const MatmulShape shape = validate_inputs(a, b);
    const minigpu::LaunchConfig launch =
        minigpu::LaunchConfig::cover(
            shape.output_count,
            threads_per_group
        );

    static_cast<void>(
        require_u32(
            launch.total_threads(),
            "launched thread count"
        )
    );

    RuntimeTensor output = RuntimeTensor::empty(
        device_,
        miniir::Type::tensor(
            miniir::DType::F32,
            {
                static_cast<std::size_t>(shape.rows),
                static_cast<std::size_t>(shape.columns),
            }
        )
    );

    minigpu::Dispatch dispatch(kernel_);
    dispatch
        .bind_buffer(a_binding, a.buffer())
        .bind_buffer(b_binding, b.buffer())
        .bind_buffer(output_binding, output.buffer())
        .bind_u32(rows_binding, shape.rows)
        .bind_u32(inner_binding, shape.inner)
        .bind_u32(columns_binding, shape.columns);

    minigpu::Event finished = queue.submit(
        std::move(dispatch),
        launch
    );

    finished.wait();
    return output;
}

} // namespace minitensor
