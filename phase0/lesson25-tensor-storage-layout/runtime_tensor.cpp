#include "runtime_tensor.hpp"

#include <algorithm>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
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

void validate_shape(const Shape& shape) {
    for (std::size_t dimension : shape) {
        if (dimension == 0) {
            throw std::invalid_argument(
                "tensor dimensions must be greater than zero"
            );
        }
    }
}

[[nodiscard]] std::size_t compute_element_count(
    const Shape& shape
) {
    validate_shape(shape);

    std::size_t count = 1;

    for (std::size_t dimension : shape) {
        count = checked_multiply(
            count,
            dimension,
            "tensor element count overflows size_t"
        );
    }

    return count;
}

[[nodiscard]] Strides canonical_strides(
    const Shape& shape
) {
    validate_shape(shape);

    Strides strides(shape.size());
    std::size_t running = 1;

    for (
        std::size_t axis = shape.size();
        axis > 0;
        --axis
    ) {
        const std::size_t current = axis - 1;
        strides[current] = running;
        running = checked_multiply(
            running,
            shape[current],
            "contiguous stride calculation overflows size_t"
        );
    }

    return strides;
}

[[nodiscard]] std::size_t compute_minimum_storage_size(
    const Shape& shape,
    const Strides& strides,
    std::size_t offset
) {
    std::size_t maximum_index = offset;

    for (
        std::size_t axis = 0;
        axis < shape.size();
        ++axis
    ) {
        const std::size_t axis_span = checked_multiply(
            shape[axis] - 1,
            strides[axis],
            "tensor layout address calculation "
            "overflows size_t"
        );

        maximum_index = checked_add(
            maximum_index,
            axis_span,
            "tensor layout address calculation "
            "overflows size_t"
        );
    }

    return checked_add(
        maximum_index,
        1,
        "tensor layout storage size overflows size_t"
    );
}

void require_provably_non_overlapping(
    const Shape& shape,
    const Strides& strides
) {
    struct Axis {
        std::size_t stride;
        std::size_t size;
    };

    std::vector<Axis> axes;
    axes.reserve(shape.size());

    for (
        std::size_t axis = 0;
        axis < shape.size();
        ++axis
    ) {
        if (shape[axis] > 1) {
            axes.push_back(Axis{
                strides[axis],
                shape[axis],
            });
        }
    }

    std::sort(
        axes.begin(),
        axes.end(),
        [](const Axis& left, const Axis& right) {
            return left.stride < right.stride;
        }
    );

    std::size_t occupied_span = 1;

    for (const Axis& axis : axes) {
        if (axis.stride < occupied_span) {
            throw std::invalid_argument(
                "Lesson 25 does not support overlapping "
                "tensor layouts"
            );
        }

        occupied_span = checked_add(
            occupied_span,
            checked_multiply(
                axis.size - 1,
                axis.stride,
                "tensor overlap check overflows size_t"
            ),
            "tensor overlap check overflows size_t"
        );
    }
}

void validate_runtime_type(const miniir::Type& type) {
    if (type.dtype() != miniir::DType::F32) {
        throw std::invalid_argument(
            "Lesson 25 RuntimeTensor supports only f32"
        );
    }
}

[[nodiscard]] std::string format_numbers(
    const std::vector<std::size_t>& numbers
) {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << '[';

    for (
        std::size_t index = 0;
        index < numbers.size();
        ++index
    ) {
        if (index != 0) {
            output << ", ";
        }

        output << numbers[index];
    }

    output << ']';
    return output.str();
}

} // namespace

TensorLayout TensorLayout::contiguous(
    Shape shape,
    std::size_t offset
) {
    Strides strides = canonical_strides(shape);

    return TensorLayout(
        std::move(shape),
        std::move(strides),
        offset
    );
}

TensorLayout::TensorLayout(
    Shape shape,
    Strides strides,
    std::size_t offset
)
    : shape_(std::move(shape)),
      strides_(std::move(strides)),
      offset_(offset),
      element_count_(compute_element_count(shape_)),
      minimum_storage_size_(0),
      contiguous_(false) {
    if (shape_.size() != strides_.size()) {
        throw std::invalid_argument(
            "tensor shape rank must equal stride rank"
        );
    }

    if (
        std::any_of(
            strides_.begin(),
            strides_.end(),
            [](std::size_t stride) {
                return stride == 0;
            }
        )
    ) {
        throw std::invalid_argument(
            "Lesson 25 requires positive strides"
        );
    }

    require_provably_non_overlapping(
        shape_,
        strides_
    );

    minimum_storage_size_ =
        compute_minimum_storage_size(
            shape_,
            strides_,
            offset_
        );

    contiguous_ =
        strides_ == canonical_strides(shape_);
}

const Shape& TensorLayout::shape() const noexcept {
    return shape_;
}

const Strides& TensorLayout::strides() const noexcept {
    return strides_;
}

std::size_t TensorLayout::offset() const noexcept {
    return offset_;
}

std::size_t TensorLayout::rank() const noexcept {
    return shape_.size();
}

std::size_t TensorLayout::element_count() const noexcept {
    return element_count_;
}

std::size_t
TensorLayout::minimum_storage_size() const noexcept {
    return minimum_storage_size_;
}

bool TensorLayout::is_contiguous() const noexcept {
    return contiguous_;
}

std::size_t TensorLayout::storage_index(
    const std::vector<std::size_t>& coordinates
) const {
    if (coordinates.size() != rank()) {
        throw std::invalid_argument(
            "coordinate rank " +
            std::to_string(coordinates.size()) +
            " does not match tensor rank " +
            std::to_string(rank())
        );
    }

    std::size_t result = offset_;

    for (
        std::size_t axis = 0;
        axis < rank();
        ++axis
    ) {
        if (coordinates[axis] >= shape_[axis]) {
            throw std::out_of_range(
                "coordinate " +
                std::to_string(coordinates[axis]) +
                " is outside axis " +
                std::to_string(axis) +
                " with size " +
                std::to_string(shape_[axis])
            );
        }

        const std::size_t contribution =
            checked_multiply(
                coordinates[axis],
                strides_[axis],
                "tensor coordinate calculation "
                "overflows size_t"
            );

        result = checked_add(
            result,
            contribution,
            "tensor coordinate calculation "
            "overflows size_t"
        );
    }

    return result;
}

std::size_t TensorLayout::storage_index_from_linear(
    std::size_t linear_index
) const {
    if (linear_index >= element_count_) {
        throw std::out_of_range(
            "logical linear index is outside the tensor"
        );
    }

    std::size_t result = offset_;
    std::size_t remaining = linear_index;

    for (
        std::size_t axis = rank();
        axis > 0;
        --axis
    ) {
        const std::size_t current = axis - 1;
        const std::size_t coordinate =
            remaining % shape_[current];
        remaining /= shape_[current];

        result = checked_add(
            result,
            checked_multiply(
                coordinate,
                strides_[current],
                "tensor linear-index calculation "
                "overflows size_t"
            ),
            "tensor linear-index calculation "
            "overflows size_t"
        );
    }

    return result;
}

std::string format_layout(const TensorLayout& layout) {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output
        << "shape=" << format_numbers(layout.shape())
        << ", strides=" << format_numbers(layout.strides())
        << ", offset=" << layout.offset()
        << ", elements=" << layout.element_count()
        << ", minimum_storage="
        << layout.minimum_storage_size()
        << ", contiguous="
        << (layout.is_contiguous() ? "yes" : "no");

    return output.str();
}

RuntimeTensor RuntimeTensor::empty(
    const minigpu::Device& device,
    miniir::Type type
) {
    validate_runtime_type(type);

    TensorLayout layout =
        TensorLayout::contiguous(type.shape());
    minigpu::Buffer storage =
        device.allocate(layout.minimum_storage_size());

    return RuntimeTensor(
        std::move(type),
        std::move(layout),
        std::move(storage)
    );
}

RuntimeTensor RuntimeTensor::from_host(
    const minigpu::Device& device,
    miniir::Type type,
    std::vector<float> values
) {
    validate_runtime_type(type);

    TensorLayout layout =
        TensorLayout::contiguous(type.shape());

    if (values.size() != layout.element_count()) {
        throw std::invalid_argument(
            "host data for " + miniir::format_type(type) +
            " requires " +
            std::to_string(layout.element_count()) +
            " value(s), received " +
            std::to_string(values.size())
        );
    }

    minigpu::Buffer storage =
        device.allocate(layout.minimum_storage_size());
    RuntimeTensor tensor(
        std::move(type),
        std::move(layout),
        std::move(storage)
    );

    for (
        std::size_t index = 0;
        index < values.size();
        ++index
    ) {
        tensor.store_flat(index, values[index]);
    }

    return tensor;
}

RuntimeTensor::RuntimeTensor(
    miniir::Type type,
    TensorLayout layout,
    minigpu::Buffer storage
)
    : type_(std::move(type)),
      layout_(std::move(layout)),
      storage_(std::move(storage)) {
    validate_runtime_type(type_);

    if (type_.shape() != layout_.shape()) {
        throw std::invalid_argument(
            "RuntimeTensor type shape " +
            format_numbers(type_.shape()) +
            " does not match layout shape " +
            format_numbers(layout_.shape())
        );
    }

    if (
        storage_.size() <
        layout_.minimum_storage_size()
    ) {
        throw std::invalid_argument(
            "RuntimeTensor layout requires " +
            std::to_string(
                layout_.minimum_storage_size()
            ) +
            " storage element(s), received " +
            std::to_string(storage_.size())
        );
    }
}

const miniir::Type& RuntimeTensor::type() const noexcept {
    return type_;
}

const TensorLayout&
RuntimeTensor::layout() const noexcept {
    return layout_;
}

const Shape& RuntimeTensor::shape() const noexcept {
    return layout_.shape();
}

const Strides& RuntimeTensor::strides() const noexcept {
    return layout_.strides();
}

std::size_t RuntimeTensor::offset() const noexcept {
    return layout_.offset();
}

std::size_t RuntimeTensor::rank() const noexcept {
    return layout_.rank();
}

std::size_t RuntimeTensor::numel() const noexcept {
    return layout_.element_count();
}

std::size_t RuntimeTensor::storage_size() const noexcept {
    return storage_.size();
}

bool RuntimeTensor::is_contiguous() const noexcept {
    return layout_.is_contiguous();
}

std::size_t RuntimeTensor::storage_index(
    const std::vector<std::size_t>& coordinates
) const {
    return layout_.storage_index(coordinates);
}

float RuntimeTensor::load(
    const std::vector<std::size_t>& coordinates
) const {
    return storage_.load(
        layout_.storage_index(coordinates)
    );
}

void RuntimeTensor::store(
    const std::vector<std::size_t>& coordinates,
    float value
) const {
    storage_.store(
        layout_.storage_index(coordinates),
        value
    );
}

float RuntimeTensor::load_flat(
    std::size_t logical_index
) const {
    return storage_.load(
        layout_.storage_index_from_linear(
            logical_index
        )
    );
}

void RuntimeTensor::store_flat(
    std::size_t logical_index,
    float value
) const {
    storage_.store(
        layout_.storage_index_from_linear(
            logical_index
        ),
        value
    );
}

std::vector<float> RuntimeTensor::to_host() const {
    std::vector<float> values;
    values.reserve(numel());

    for (
        std::size_t index = 0;
        index < numel();
        ++index
    ) {
        values.push_back(load_flat(index));
    }

    return values;
}

const minigpu::Buffer&
RuntimeTensor::buffer() const noexcept {
    return storage_;
}

} // namespace minitensor
