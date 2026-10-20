#include "layout_relu.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace minitensor {

namespace {

constexpr std::string_view kernel_name =
    "layout_aware_relu";

constexpr std::string_view kernel_source = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void layout_aware_relu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    constant uint& rank [[buffer(3)]],
    constant uint& offset [[buffer(4)]],
    constant uint& shape_0 [[buffer(5)]],
    constant uint& shape_1 [[buffer(6)]],
    constant uint& shape_2 [[buffer(7)]],
    constant uint& shape_3 [[buffer(8)]],
    constant uint& stride_0 [[buffer(9)]],
    constant uint& stride_1 [[buffer(10)]],
    constant uint& stride_2 [[buffer(11)]],
    constant uint& stride_3 [[buffer(12)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id >= element_count) {
        return;
    }

    const uint shapes[4] = {
        shape_0,
        shape_1,
        shape_2,
        shape_3
    };

    const uint strides[4] = {
        stride_0,
        stride_1,
        stride_2,
        stride_3
    };

    uint remaining = global_id;
    uint storage_index = offset;

    // Convert the flat logical index into coordinates, starting with the
    // rightmost axis, then use strides to find the physical input slot.
    for (uint axis = rank; axis > 0; --axis) {
        const uint current = axis - 1;
        const uint coordinate =
            remaining % shapes[current];
        remaining /= shapes[current];
        storage_index +=
            coordinate * strides[current];
    }

    output[global_id] =
        max(input[storage_index], 0.0f);
}
)metal";

constexpr std::size_t input_binding = 0;
constexpr std::size_t output_binding = 1;
constexpr std::size_t element_count_binding = 2;
constexpr std::size_t rank_binding = 3;
constexpr std::size_t offset_binding = 4;
constexpr std::size_t first_shape_binding = 5;
constexpr std::size_t first_stride_binding = 9;

// The MSL function signature above explicitly contains four shape and four
// stride arguments. These assertions make the public constant a fixed ABI
// description rather than a knob that can be changed on only the C++ side.
static_assert(layout_kernel_max_rank == 4);
static_assert(
    first_shape_binding + layout_kernel_max_rank ==
    first_stride_binding
);
static_assert(
    first_stride_binding + layout_kernel_max_rank == 13
);

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
            " does not fit the Lesson 27 uint32 "
            "Metal indexing model"
        );
    }

    return static_cast<std::uint32_t>(value);
}

struct PackedLayout {
    std::uint32_t element_count;
    std::uint32_t rank;
    std::uint32_t offset;
    std::array<std::uint32_t, layout_kernel_max_rank>
        shapes;
    std::array<std::uint32_t, layout_kernel_max_rank>
        strides;
};

[[nodiscard]] PackedLayout pack_layout(
    const RuntimeTensor& input
) {
    if (input.rank() > layout_kernel_max_rank) {
        throw std::invalid_argument(
            "Lesson 27 layout-aware kernel supports "
            "rank at most " +
            std::to_string(layout_kernel_max_rank) +
            "; received rank " +
            std::to_string(input.rank())
        );
    }

    // Every physical index calculated by the GPU must fit its uint.
    const std::size_t highest_storage_index =
        input.layout().minimum_storage_size() - 1;
    static_cast<void>(
        require_u32(
            highest_storage_index,
            "highest input storage index"
        )
    );

    PackedLayout packed{
        require_u32(
            input.numel(),
            "logical element count"
        ),
        require_u32(input.rank(), "tensor rank"),
        require_u32(input.offset(), "tensor offset"),
        {1, 1, 1, 1},
        {0, 0, 0, 0},
    };

    for (
        std::size_t axis = 0;
        axis < input.rank();
        ++axis
    ) {
        packed.shapes[axis] =
            require_u32(
                input.shape()[axis],
                "tensor dimension"
            );

        // A size-one axis never advances, so its stride cannot affect an
        // address. Zero is a safe GPU representation for that unused value.
        if (input.shape()[axis] > 1) {
            packed.strides[axis] =
                require_u32(
                    input.strides()[axis],
                    "tensor stride"
                );
        }
    }

    return packed;
}

} // namespace

LayoutAwareRelu::LayoutAwareRelu(
    const minigpu::Device& device
)
    : device_(device),
      kernel_(
          device_.compile_kernel(
              std::string(kernel_name),
              kernel_source
          )
      ) {}

RuntimeTensor LayoutAwareRelu::run(
    minigpu::Queue& queue,
    const RuntimeTensor& input,
    std::size_t threads_per_group
) const {
    if (threads_per_group == 0) {
        throw std::invalid_argument(
            "threads_per_group must be greater than zero"
        );
    }

    const PackedLayout packed = pack_layout(input);
    const minigpu::LaunchConfig launch =
        minigpu::LaunchConfig::cover(
            input.numel(),
            threads_per_group
        );

    static_cast<void>(
        require_u32(
            launch.total_threads(),
            "launched thread count"
        )
    );

    // A distinct contiguous output avoids read/write aliasing and gives the
    // next operator a simple row-major tensor.
    RuntimeTensor output =
        RuntimeTensor::empty(device_, input.type());

    minigpu::Dispatch dispatch(kernel_);
    dispatch
        .bind_buffer(input_binding, input.buffer())
        .bind_buffer(output_binding, output.buffer())
        .bind_u32(
            element_count_binding,
            packed.element_count
        )
        .bind_u32(rank_binding, packed.rank)
        .bind_u32(offset_binding, packed.offset);

    for (
        std::size_t axis = 0;
        axis < layout_kernel_max_rank;
        ++axis
    ) {
        dispatch.bind_u32(
            first_shape_binding + axis,
            packed.shapes[axis]
        );
        dispatch.bind_u32(
            first_stride_binding + axis,
            packed.strides[axis]
        );
    }

    minigpu::Event finished = queue.submit(
        std::move(dispatch),
        launch
    );

    // run() is intentionally synchronous: do not return a tensor while the
    // GPU may still be writing its shared output buffer.
    finished.wait();
    return output;
}

} // namespace minitensor
