#include "contiguous.hpp"

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
    "materialize_contiguous";

constexpr std::string_view kernel_source = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void materialize_contiguous(
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
    uint input_index = offset;

    for (uint axis = rank; axis > 0; --axis) {
        const uint current = axis - 1;
        const uint coordinate =
            remaining % shapes[current];
        remaining /= shapes[current];
        input_index +=
            coordinate * strides[current];
    }

    output[global_id] = input[input_index];
}
)metal";

constexpr std::size_t input_binding = 0;
constexpr std::size_t output_binding = 1;
constexpr std::size_t element_count_binding = 2;
constexpr std::size_t rank_binding = 3;
constexpr std::size_t offset_binding = 4;
constexpr std::size_t first_shape_binding = 5;
constexpr std::size_t first_stride_binding = 9;

static_assert(contiguous_copy_max_rank == 4);
static_assert(
    first_shape_binding + contiguous_copy_max_rank ==
    first_stride_binding
);
static_assert(
    first_stride_binding + contiguous_copy_max_rank == 13
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
            " does not fit the Lesson 28 uint32 "
            "Metal indexing model"
        );
    }

    return static_cast<std::uint32_t>(value);
}

struct PackedLayout {
    std::uint32_t element_count;
    std::uint32_t rank;
    std::uint32_t offset;
    std::array<std::uint32_t, contiguous_copy_max_rank>
        shapes;
    std::array<std::uint32_t, contiguous_copy_max_rank>
        strides;
};

struct PreparedCopy {
    PackedLayout layout;
    minigpu::LaunchConfig launch;
};

[[nodiscard]] PackedLayout pack_layout(
    const RuntimeTensor& input
) {
    if (input.rank() > contiguous_copy_max_rank) {
        throw std::invalid_argument(
            "Lesson 28 contiguous copy supports rank "
            "at most " +
            std::to_string(contiguous_copy_max_rank) +
            "; received rank " +
            std::to_string(input.rank())
        );
    }

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

void validate_output(
    const RuntimeTensor& input,
    const RuntimeTensor& output
) {
    if (output.type() != input.type()) {
        throw std::invalid_argument(
            "contiguous-copy output type " +
            miniir::format_type(output.type()) +
            " does not match input type " +
            miniir::format_type(input.type())
        );
    }

    if (!output.is_contiguous()) {
        throw std::invalid_argument(
            "contiguous-copy output must use canonical "
            "row-major strides"
        );
    }

    if (output.offset() != 0) {
        throw std::invalid_argument(
            "contiguous-copy output offset must be zero"
        );
    }

    if (
        input.buffer().shares_storage_with(
            output.buffer()
        )
    ) {
        throw std::invalid_argument(
            "contiguous copy requires distinct input and "
            "output storage"
        );
    }
}

[[nodiscard]] PreparedCopy prepare_copy(
    const RuntimeTensor& input,
    std::size_t threads_per_group
) {
    if (threads_per_group == 0) {
        throw std::invalid_argument(
            "threads_per_group must be greater than zero"
        );
    }

    PackedLayout layout = pack_layout(input);
    minigpu::LaunchConfig launch =
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

    return PreparedCopy{
        std::move(layout),
        std::move(launch),
    };
}

[[nodiscard]] minigpu::Event submit_copy(
    const minigpu::Kernel& kernel,
    minigpu::Queue& queue,
    const RuntimeTensor& input,
    RuntimeTensor& output,
    const PreparedCopy& prepared
) {
    minigpu::Dispatch dispatch(kernel);
    dispatch
        .bind_buffer(input_binding, input.buffer())
        .bind_buffer(output_binding, output.buffer())
        .bind_u32(
            element_count_binding,
            prepared.layout.element_count
        )
        .bind_u32(
            rank_binding,
            prepared.layout.rank
        )
        .bind_u32(
            offset_binding,
            prepared.layout.offset
        );

    for (
        std::size_t axis = 0;
        axis < contiguous_copy_max_rank;
        ++axis
    ) {
        dispatch.bind_u32(
            first_shape_binding + axis,
            prepared.layout.shapes[axis]
        );
        dispatch.bind_u32(
            first_stride_binding + axis,
            prepared.layout.strides[axis]
        );
    }

    return queue.submit(
        std::move(dispatch),
        prepared.launch
    );
}

} // namespace

ContiguousMaterializer::ContiguousMaterializer(
    const minigpu::Device& device
)
    : device_(device),
      kernel_(
          device_.compile_kernel(
              std::string(kernel_name),
              kernel_source
          )
      ) {}

RuntimeTensor ContiguousMaterializer::run(
    minigpu::Queue& queue,
    const RuntimeTensor& input,
    std::size_t threads_per_group
) const {
    const PreparedCopy prepared =
        prepare_copy(input, threads_per_group);
    RuntimeTensor output =
        RuntimeTensor::empty(device_, input.type());

    validate_output(input, output);
    minigpu::Event finished = submit_copy(
        kernel_,
        queue,
        input,
        output,
        prepared
    );

    finished.wait();
    return output;
}

minigpu::Event ContiguousMaterializer::enqueue_into(
    minigpu::Queue& queue,
    const RuntimeTensor& input,
    RuntimeTensor& output,
    std::size_t threads_per_group
) const {
    const PreparedCopy prepared =
        prepare_copy(input, threads_per_group);
    validate_output(input, output);

    return submit_copy(
        kernel_,
        queue,
        input,
        output,
        prepared
    );
}

} // namespace minitensor
