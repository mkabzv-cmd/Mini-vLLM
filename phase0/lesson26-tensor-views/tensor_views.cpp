#include "tensor_views.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace minitensor {

namespace {

[[nodiscard]] std::size_t checked_add(
    std::size_t left,
    std::size_t right,
    const char* message
) {
    if (
        left >
        std::numeric_limits<std::size_t>::max() - right
    ) {
        throw std::overflow_error(message);
    }

    return left + right;
}

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

[[nodiscard]] miniir::Type type_for_view(
    miniir::DType dtype,
    const Shape& shape
) {
    if (shape.empty()) {
        return miniir::Type::scalar(dtype);
    }

    return miniir::Type::tensor(dtype, shape);
}

[[nodiscard]] RuntimeTensor make_view(
    const RuntimeTensor& source,
    Shape shape,
    Strides strides,
    std::size_t offset
) {
    miniir::Type type =
        type_for_view(source.type().dtype(), shape);
    TensorLayout layout(
        std::move(shape),
        std::move(strides),
        offset
    );

    // Buffer is a shared handle. This copies the handle, not the values.
    return RuntimeTensor(
        std::move(type),
        std::move(layout),
        source.buffer()
    );
}

void require_axis(
    const RuntimeTensor& source,
    std::size_t axis,
    const char* operation
) {
    if (axis >= source.rank()) {
        throw std::out_of_range(
            std::string(operation) + " axis " +
            std::to_string(axis) +
            " is outside tensor rank " +
            std::to_string(source.rank())
        );
    }
}

} // namespace

RuntimeTensor transpose(
    const RuntimeTensor& source,
    std::size_t first_axis,
    std::size_t second_axis
) {
    require_axis(source, first_axis, "transpose");
    require_axis(source, second_axis, "transpose");

    Shape shape = source.shape();
    Strides strides = source.strides();

    std::swap(shape[first_axis], shape[second_axis]);
    std::swap(
        strides[first_axis],
        strides[second_axis]
    );

    return make_view(
        source,
        std::move(shape),
        std::move(strides),
        source.offset()
    );
}

RuntimeTensor slice(
    const RuntimeTensor& source,
    std::size_t axis,
    std::size_t start,
    std::size_t length,
    std::size_t step
) {
    require_axis(source, axis, "slice");

    if (length == 0) {
        throw std::invalid_argument(
            "Lesson 26 does not support empty slices"
        );
    }

    if (step == 0) {
        throw std::invalid_argument(
            "slice step must be greater than zero"
        );
    }

    const std::size_t dimension = source.shape()[axis];

    if (start >= dimension) {
        throw std::out_of_range(
            "slice start " + std::to_string(start) +
            " is outside axis " + std::to_string(axis) +
            " with size " + std::to_string(dimension)
        );
    }

    // The final selected coordinate is:
    //
    //     start + (length - 1) * step
    //
    // Compare using division so the check itself cannot overflow.
    const std::size_t available_steps =
        (dimension - 1 - start) / step;

    if (length - 1 > available_steps) {
        throw std::out_of_range(
            "slice extends past axis " +
            std::to_string(axis) +
            " with size " + std::to_string(dimension)
        );
    }

    Shape shape = source.shape();
    Strides strides = source.strides();

    const std::size_t offset_delta =
        checked_multiply(
            start,
            strides[axis],
            "slice offset calculation overflows size_t"
        );

    const std::size_t view_offset =
        checked_add(
            source.offset(),
            offset_delta,
            "slice offset calculation overflows size_t"
        );

    shape[axis] = length;

    // A one-element axis never advances, so its step cannot affect any
    // address. Keep the existing representable stride instead of rejecting
    // an otherwise valid slice because an unused product would overflow.
    if (length > 1) {
        strides[axis] =
            checked_multiply(
                strides[axis],
                step,
                "slice stride calculation overflows size_t"
            );
    }

    return make_view(
        source,
        std::move(shape),
        std::move(strides),
        view_offset
    );
}

} // namespace minitensor
