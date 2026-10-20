#include "tiled_matmul.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace minitensor {

namespace {

constexpr std::string_view kernel_name = "tiled_matmul_8x8";

constexpr std::string_view kernel_source = R"metal(
#include <metal_stdlib>
using namespace metal;

constant uint tile_size = 8;

kernel void tiled_matmul_8x8(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    constant uint& columns [[buffer(5)]],
    constant uint& inner_tile_count [[buffer(6)]],
    constant uint& output_tile_columns [[buffer(7)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint local_id [[thread_index_in_threadgroup]]
) {
    threadgroup float a_tile[64];
    threadgroup float b_tile[64];

    const uint local_row = local_id / tile_size;
    const uint local_column = local_id % tile_size;
    const uint output_tile_row =
        group_id / output_tile_columns;
    const uint output_tile_column =
        group_id % output_tile_columns;
    const uint row =
        output_tile_row * tile_size + local_row;
    const uint column =
        output_tile_column * tile_size + local_column;

    float sum = 0.0f;

    for (
        uint tile = 0;
        tile < inner_tile_count;
        ++tile
    ) {
        const uint tile_start = tile * tile_size;
        const uint a_column = tile_start + local_column;
        const uint b_row = tile_start + local_row;

        a_tile[local_id] =
            row < rows && a_column < inner
                ? a[row * inner + a_column]
                : 0.0f;

        b_tile[local_id] =
            b_row < inner && column < columns
                ? b[b_row * columns + column]
                : 0.0f;

        // Every lane must finish producing its two tile values before any
        // lane starts consuming values written by neighboring lanes.
        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );

        for (uint k = 0; k < tile_size; ++k) {
            sum +=
                a_tile[local_row * tile_size + k] *
                b_tile[k * tile_size + local_column];
        }

        // Every lane must finish consuming this tile before lanes overwrite
        // threadgroup memory with the next K tile.
        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );
    }

    if (row < rows && column < columns) {
        output[row * columns + column] = sum;
    }
}
)metal";

constexpr std::size_t a_binding = 0;
constexpr std::size_t b_binding = 1;
constexpr std::size_t output_binding = 2;
constexpr std::size_t rows_binding = 3;
constexpr std::size_t inner_binding = 4;
constexpr std::size_t columns_binding = 5;
constexpr std::size_t inner_tile_count_binding = 6;
constexpr std::size_t output_tile_columns_binding = 7;

static_assert(tiled_matmul_tile_size == 8);
static_assert(tiled_matmul_threads_per_group == 64);

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
            " does not fit the Lesson 30 uint32 "
            "Metal indexing model"
        );
    }

    return static_cast<std::uint32_t>(value);
}

[[nodiscard]] std::size_t ceil_divide(
    std::size_t value,
    std::size_t divisor
) {
    return
        value / divisor +
        static_cast<std::size_t>(
            value % divisor != 0
        );
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

struct TiledShape {
    std::uint32_t rows;
    std::uint32_t inner;
    std::uint32_t columns;
    std::uint32_t inner_tile_count;
    std::uint32_t output_tile_columns;
    std::size_t group_count;
};

[[nodiscard]] TiledShape validate_inputs(
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
    static_cast<void>(
        require_u32(
            output_count,
            "output element count"
        )
    );

    const std::size_t row_tiles =
        ceil_divide(rows, tiled_matmul_tile_size);
    const std::size_t column_tiles =
        ceil_divide(
            columns,
            tiled_matmul_tile_size
        );
    const std::size_t inner_tiles =
        ceil_divide(
            inner,
            tiled_matmul_tile_size
        );
    const std::size_t group_count =
        checked_multiply(
            row_tiles,
            column_tiles,
            "tiled matmul threadgroup count"
        );

    static_cast<void>(
        require_u32(
            group_count,
            "tiled matmul threadgroup count"
        )
    );

    return TiledShape{
        require_u32(rows, "M dimension"),
        require_u32(inner, "K dimension"),
        require_u32(columns, "N dimension"),
        require_u32(
            inner_tiles,
            "K tile count"
        ),
        require_u32(
            column_tiles,
            "output tile-column count"
        ),
        group_count,
    };
}

} // namespace

TiledMatmul::TiledMatmul(
    const minigpu::Device& device
)
    : device_(device),
      kernel_(
          device_.compile_kernel(
              std::string(kernel_name),
              kernel_source
          )
      ) {}

RuntimeTensor TiledMatmul::run(
    minigpu::Queue& queue,
    const RuntimeTensor& a,
    const RuntimeTensor& b
) const {
    const TiledShape shape = validate_inputs(a, b);

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
        .bind_u32(columns_binding, shape.columns)
        .bind_u32(
            inner_tile_count_binding,
            shape.inner_tile_count
        )
        .bind_u32(
            output_tile_columns_binding,
            shape.output_tile_columns
        );

    minigpu::Event finished = queue.submit(
        std::move(dispatch),
        minigpu::LaunchConfig(
            shape.group_count,
            tiled_matmul_threads_per_group
        )
    );

    finished.wait();
    return output;
}

} // namespace minitensor
