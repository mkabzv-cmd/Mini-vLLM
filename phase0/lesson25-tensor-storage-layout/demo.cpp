#include "runtime_tensor.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Exception, typename Function>
void require_throws(
    Function&& function,
    std::string_view expected_message_part
) {
    try {
        std::forward<Function>(function)();
    } catch (const Exception& error) {
        if (
            std::string_view(error.what()).find(
                expected_message_part
            ) == std::string_view::npos
        ) {
            throw std::runtime_error(
                "exception message did not contain '" +
                std::string(expected_message_part) +
                "': " + error.what()
            );
        }

        return;
    } catch (const std::exception& error) {
        throw std::runtime_error(
            "unexpected exception type: " +
            std::string(error.what())
        );
    }

    throw std::runtime_error(
        "expected an exception containing '" +
        std::string(expected_message_part) + "'"
    );
}

[[nodiscard]] bool equal_values(
    const std::vector<float>& actual,
    const std::vector<float>& expected
) {
    if (actual.size() != expected.size()) {
        return false;
    }

    for (
        std::size_t index = 0;
        index < actual.size();
        ++index
    ) {
        if (
            !std::isfinite(actual[index]) ||
            std::fabs(actual[index] - expected[index]) >
                0.00001f
        ) {
            return false;
        }
    }

    return true;
}

void print_values(const std::vector<float>& values) {
    std::cout << '[';

    for (
        std::size_t index = 0;
        index < values.size();
        ++index
    ) {
        if (index != 0) {
            std::cout << ", ";
        }

        std::cout << values[index];
    }

    std::cout << ']';
}

[[nodiscard]] std::vector<float> read_storage(
    const minigpu::Buffer& storage
) {
    std::vector<float> values;
    values.reserve(storage.size());

    for (
        std::size_t index = 0;
        index < storage.size();
        ++index
    ) {
        values.push_back(storage.load(index));
    }

    return values;
}

void check_layout_metadata() {
    const minitensor::TensorLayout vector =
        minitensor::TensorLayout::contiguous({5});
    const minitensor::TensorLayout matrix =
        minitensor::TensorLayout::contiguous({2, 3});
    const minitensor::TensorLayout cube =
        minitensor::TensorLayout::contiguous(
            {2, 3, 4}
        );
    const minitensor::TensorLayout scalar =
        minitensor::TensorLayout::contiguous({});

    require(
        vector.strides() == minitensor::Strides{1} &&
            matrix.strides() ==
                minitensor::Strides{3, 1} &&
            cube.strides() ==
                minitensor::Strides{12, 4, 1},
        "canonical row-major strides are incorrect"
    );
    require(
        scalar.rank() == 0 &&
            scalar.element_count() == 1 &&
            scalar.minimum_storage_size() == 1 &&
            scalar.storage_index({}) == 0 &&
            scalar.is_contiguous(),
        "rank-zero scalar layout is incorrect"
    );

    const minitensor::TensorLayout offset_contiguous =
        minitensor::TensorLayout::contiguous(
            {2, 3},
            2
        );

    require(
        offset_contiguous.is_contiguous() &&
            offset_contiguous.minimum_storage_size() == 8 &&
            offset_contiguous.storage_index({1, 2}) == 7,
        "a nonzero offset broke contiguous layout semantics"
    );

    const minitensor::TensorLayout transpose_like(
        {3, 2},
        {1, 3}
    );

    require(
        !transpose_like.is_contiguous() &&
            transpose_like.storage_index({0, 0}) == 0 &&
            transpose_like.storage_index({0, 1}) == 3 &&
            transpose_like.storage_index({2, 0}) == 2 &&
            transpose_like.storage_index({2, 1}) == 5,
        "transpose-like layout mapping is incorrect"
    );
}

void check_errors(const minigpu::Device& device) {
    require_throws<std::invalid_argument>(
        [] {
            static_cast<void>(
                minitensor::TensorLayout(
                    {2, 3},
                    {1}
                )
            );
        },
        "shape rank must equal stride rank"
    );

    require_throws<std::invalid_argument>(
        [] {
            static_cast<void>(
                minitensor::TensorLayout::contiguous(
                    {2, 0, 3}
                )
            );
        },
        "greater than zero"
    );

    require_throws<std::invalid_argument>(
        [] {
            static_cast<void>(
                minitensor::TensorLayout(
                    {2, 3},
                    {0, 1}
                )
            );
        },
        "positive strides"
    );

    require_throws<std::invalid_argument>(
        [] {
            static_cast<void>(
                minitensor::TensorLayout(
                    {2, 2},
                    {1, 1}
                )
            );
        },
        "overlapping tensor layouts"
    );

    const minitensor::TensorLayout matrix =
        minitensor::TensorLayout::contiguous({2, 3});

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                matrix.storage_index({1})
            );
        },
        "coordinate rank"
    );

    require_throws<std::out_of_range>(
        [&] {
            static_cast<void>(
                matrix.storage_index({2, 0})
            );
        },
        "outside axis 0"
    );

    require_throws<std::out_of_range>(
        [&] {
            static_cast<void>(
                matrix.storage_index_from_linear(6)
            );
        },
        "linear index"
    );

    require_throws<std::overflow_error>(
        [] {
            static_cast<void>(
                minitensor::TensorLayout::contiguous(
                    {
                        std::numeric_limits<
                            std::size_t
                        >::max(),
                        2,
                    }
                )
            );
        },
        "overflows"
    );

    require_throws<std::overflow_error>(
        [] {
            static_cast<void>(
                minitensor::TensorLayout(
                    {},
                    {},
                    std::numeric_limits<
                        std::size_t
                    >::max()
                )
            );
        },
        "storage size overflows"
    );

    const miniir::Type matrix_type =
        miniir::Type::tensor(
            miniir::DType::F32,
            {2, 3}
        );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                minitensor::RuntimeTensor::from_host(
                    device,
                    matrix_type,
                    {1.0f, 2.0f}
                )
            );
        },
        "requires 6 value(s)"
    );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                minitensor::RuntimeTensor::empty(
                    device,
                    miniir::Type::tensor(
                        miniir::DType::F16,
                        {2, 3}
                    )
                )
            );
        },
        "supports only f32"
    );

    const minitensor::TensorLayout padded(
        {2, 3},
        {4, 1},
        2
    );
    minigpu::Buffer too_small = device.allocate(8);

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                minitensor::RuntimeTensor(
                    matrix_type,
                    padded,
                    too_small
                )
            );
        },
        "requires 9 storage element(s), received 8"
    );

    minigpu::Buffer six_elements = device.allocate(6);

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                minitensor::RuntimeTensor(
                    matrix_type,
                    minitensor::TensorLayout::
                        contiguous({6}),
                    six_elements
                )
            );
        },
        "does not match layout shape"
    );
}

} // namespace

int main() {
    try {
        std::cout
            << "Lesson 25: tensor storage and layout\n\n";

        check_layout_metadata();

        minigpu::Device device;
        check_errors(device);

        const miniir::Type matrix_type =
            miniir::Type::tensor(
                miniir::DType::F32,
                {2, 3}
            );

        minitensor::RuntimeTensor matrix =
            minitensor::RuntimeTensor::from_host(
                device,
                matrix_type,
                {
                    10.0f,
                    11.0f,
                    12.0f,
                    20.0f,
                    21.0f,
                    22.0f,
                }
            );

        require(
            matrix.shape() == minitensor::Shape{2, 3} &&
                matrix.strides() ==
                    minitensor::Strides{3, 1} &&
                matrix.offset() == 0 &&
                matrix.numel() == 6 &&
                matrix.storage_size() == 6 &&
                matrix.is_contiguous(),
            "contiguous RuntimeTensor metadata is wrong"
        );
        require(
            matrix.storage_index({1, 2}) == 5 &&
                matrix.load({1, 2}) == 22.0f,
            "logical coordinate did not reach storage[5]"
        );

        matrix.store({0, 1}, 99.0f);
        require(
            equal_values(
                matrix.to_host(),
                {
                    10.0f,
                    99.0f,
                    12.0f,
                    20.0f,
                    21.0f,
                    22.0f,
                }
            ),
            "contiguous tensor store or download failed"
        );

        // RuntimeTensor copies copy a Buffer handle. The underlying Metal
        // allocation remains shared rather than duplicating six floats.
        minitensor::RuntimeTensor copied = matrix;
        copied.store({1, 2}, 77.0f);

        require(
            matrix.load({1, 2}) == 77.0f,
            "RuntimeTensor copy did not share storage"
        );

        const minitensor::TensorLayout padded_layout(
            {2, 3},
            {4, 1},
            2
        );
        minigpu::Buffer padded_storage =
            device.allocate(
                padded_layout.minimum_storage_size()
            );

        for (
            std::size_t index = 0;
            index < padded_storage.size();
            ++index
        ) {
            padded_storage.store(index, -1.0f);
        }

        minitensor::RuntimeTensor padded(
            matrix_type,
            padded_layout,
            padded_storage
        );

        const std::vector<float> logical_values{
            10.0f,
            11.0f,
            12.0f,
            20.0f,
            21.0f,
            22.0f,
        };

        for (
            std::size_t index = 0;
            index < logical_values.size();
            ++index
        ) {
            padded.store_flat(
                index,
                logical_values[index]
            );
        }

        const std::vector<float> expected_storage{
            -1.0f,
            -1.0f,
            10.0f,
            11.0f,
            12.0f,
            -1.0f,
            20.0f,
            21.0f,
            22.0f,
        };

        require(
            padded.storage_index({0, 0}) == 2 &&
                padded.storage_index({0, 1}) == 3 &&
                padded.storage_index({0, 2}) == 4 &&
                padded.storage_index({1, 0}) == 6 &&
                padded.storage_index({1, 1}) == 7 &&
                padded.storage_index({1, 2}) == 8 &&
                padded.numel() == 6 &&
                padded.storage_size() == 9 &&
                !padded.is_contiguous() &&
                equal_values(
                    padded.to_host(),
                    logical_values
                ) &&
                equal_values(
                    read_storage(padded.buffer()),
                    expected_storage
                ),
            "padded layout mapping is incorrect"
        );

        minitensor::RuntimeTensor scalar =
            minitensor::RuntimeTensor::from_host(
                device,
                miniir::Type::scalar(
                    miniir::DType::F32
                ),
                {7.0f}
            );

        require(
            scalar.rank() == 0 &&
                scalar.numel() == 1 &&
                scalar.load({}) == 7.0f,
            "RuntimeTensor scalar is incorrect"
        );

        std::cout
            << "Device: " << device.name() << "\n\n"
            << "Contiguous matrix\n"
            << minitensor::format_layout(matrix.layout())
            << "\n"
            << "[1, 2] -> storage["
            << matrix.storage_index({1, 2})
            << "] -> " << matrix.load({1, 2})
            << "\nlogical values: ";
        print_values(matrix.to_host());

        std::cout
            << "\n\nPadded matrix\n"
            << minitensor::format_layout(padded.layout())
            << "\n"
            << "[0,0] [0,1] [0,2] [1,0] [1,1] [1,2]\n"
            << "   2     3     4     6     7     8"
            << "   <- storage indexes\n"
            << "raw storage: ";
        print_values(read_storage(padded.buffer()));

        std::cout
            << "\n\nShared-copy check: copied[1,2] = 77"
            << ", original[1,2] = "
            << matrix.load({1, 2})
            << "\nScalar check: shape=[], value="
            << scalar.load({})
            << "\n\nPASS: shape, strides, offset, and "
               "storage map logical tensors onto one "
               "flat Metal buffer.\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
