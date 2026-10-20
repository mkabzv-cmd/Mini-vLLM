#include "tensor_views.hpp"

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

[[nodiscard]] minitensor::RuntimeTensor
make_view_that_outlives_source(
    const minigpu::Device& device
) {
    minitensor::RuntimeTensor temporary =
        minitensor::RuntimeTensor::from_host(
            device,
            miniir::Type::tensor(
                miniir::DType::F32,
                {2, 3}
            ),
            {
                1.0f,
                2.0f,
                3.0f,
                4.0f,
                5.0f,
                6.0f,
            }
        );

    return minitensor::slice(
        temporary,
        1,
        1,
        2
    );
}

void check_general_view_metadata(
    const minigpu::Device& device
) {
    minitensor::RuntimeTensor cube =
        minitensor::RuntimeTensor::empty(
            device,
            miniir::Type::tensor(
                miniir::DType::F32,
                {2, 3, 4}
            )
        );

    minitensor::RuntimeTensor transposed =
        minitensor::transpose(cube, 0, 2);

    require(
        transposed.shape() ==
            minitensor::Shape{4, 3, 2} &&
            transposed.strides() ==
                minitensor::Strides{1, 4, 12} &&
            transposed.type().shape() ==
                minitensor::Shape{4, 3, 2} &&
            transposed.type().dtype() ==
                miniir::DType::F32 &&
            transposed.storage_index({3, 2, 1}) ==
                cube.storage_index({1, 2, 3}),
        "higher-rank transpose metadata is incorrect"
    );

    minitensor::RuntimeTensor no_op =
        minitensor::transpose(cube, 1, 1);
    minitensor::RuntimeTensor full_slice =
        minitensor::slice(cube, 1, 0, 3);
    minitensor::RuntimeTensor singleton =
        minitensor::slice(
            cube,
            0,
            1,
            1,
            std::numeric_limits<std::size_t>::max()
        );

    require(
        no_op.shape() == cube.shape() &&
            no_op.strides() == cube.strides() &&
            no_op.offset() == cube.offset() &&
            full_slice.shape() == cube.shape() &&
            full_slice.strides() == cube.strides() &&
            full_slice.offset() == cube.offset() &&
            singleton.shape() ==
                minitensor::Shape{1, 3, 4} &&
            singleton.strides() == cube.strides() &&
            singleton.offset() == 12,
        "no-op views changed the source layout"
    );

    no_op.store({1, 2, 3}, 42.0f);
    require(
        cube.load({1, 2, 3}) == 42.0f,
        "no-op transpose did not share storage"
    );
}

void check_errors(
    const minigpu::Device& device,
    const minitensor::RuntimeTensor& matrix
) {
    require_throws<std::out_of_range>(
        [&] {
            static_cast<void>(
                minitensor::transpose(matrix, 0, 2)
            );
        },
        "outside tensor rank 2"
    );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                minitensor::slice(matrix, 1, 0, 0)
            );
        },
        "empty slices"
    );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                minitensor::slice(matrix, 1, 0, 2, 0)
            );
        },
        "step must be greater than zero"
    );

    require_throws<std::out_of_range>(
        [&] {
            static_cast<void>(
                minitensor::slice(matrix, 1, 4, 1)
            );
        },
        "slice start 4"
    );

    require_throws<std::out_of_range>(
        [&] {
            static_cast<void>(
                minitensor::slice(matrix, 1, 1, 3, 2)
            );
        },
        "extends past axis 1"
    );

    const minitensor::RuntimeTensor scalar =
        minitensor::RuntimeTensor::from_host(
            device,
            miniir::Type::scalar(
                miniir::DType::F32
            ),
            {7.0f}
        );

    require_throws<std::out_of_range>(
        [&] {
            static_cast<void>(
                minitensor::slice(scalar, 0, 0, 1)
            );
        },
        "outside tensor rank 0"
    );
}

} // namespace

int main() {
    try {
        std::cout << "Lesson 26: zero-copy tensor views\n\n";

        minigpu::Device device;

        minitensor::RuntimeTensor matrix =
            minitensor::RuntimeTensor::from_host(
                device,
                miniir::Type::tensor(
                    miniir::DType::F32,
                    {3, 4}
                ),
                {
                    0.0f,
                    1.0f,
                    2.0f,
                    3.0f,
                    4.0f,
                    5.0f,
                    6.0f,
                    7.0f,
                    8.0f,
                    9.0f,
                    10.0f,
                    11.0f,
                }
            );

        minitensor::RuntimeTensor transposed =
            minitensor::transpose(matrix, 0, 1);

        require(
            transposed.shape() ==
                minitensor::Shape{4, 3} &&
                transposed.strides() ==
                    minitensor::Strides{1, 4} &&
                transposed.offset() == 0 &&
                transposed.storage_size() == 12 &&
                !transposed.is_contiguous(),
            "transpose metadata is incorrect"
        );
        require(
            equal_values(
                transposed.to_host(),
                {
                    0.0f,
                    4.0f,
                    8.0f,
                    1.0f,
                    5.0f,
                    9.0f,
                    2.0f,
                    6.0f,
                    10.0f,
                    3.0f,
                    7.0f,
                    11.0f,
                }
            ),
            "transpose logical values are incorrect"
        );
        require(
            matrix.shape() == minitensor::Shape{3, 4} &&
                matrix.strides() ==
                    minitensor::Strides{4, 1} &&
                matrix.offset() == 0,
            "creating a view changed the source metadata"
        );

        // transposed[2,1] and matrix[1,2] are the same storage slot.
        transposed.store({2, 1}, 60.0f);
        require(
            matrix.load({1, 2}) == 60.0f &&
                transposed.storage_index({2, 1}) == 6 &&
                matrix.storage_index({1, 2}) == 6,
            "transpose did not share source storage"
        );

        minitensor::RuntimeTensor rows =
            minitensor::slice(matrix, 0, 1, 2);

        require(
            rows.shape() == minitensor::Shape{2, 4} &&
                rows.strides() ==
                    minitensor::Strides{4, 1} &&
                rows.offset() == 4 &&
                rows.is_contiguous() &&
                equal_values(
                    rows.to_host(),
                    {
                        4.0f,
                        5.0f,
                        60.0f,
                        7.0f,
                        8.0f,
                        9.0f,
                        10.0f,
                        11.0f,
                    }
                ),
            "contiguous row slice is incorrect"
        );

        minitensor::RuntimeTensor columns =
            minitensor::slice(matrix, 1, 1, 2);

        require(
            columns.shape() ==
                minitensor::Shape{3, 2} &&
                columns.strides() ==
                    minitensor::Strides{4, 1} &&
                columns.offset() == 1 &&
                !columns.is_contiguous() &&
                equal_values(
                    columns.to_host(),
                    {
                        1.0f,
                        2.0f,
                        5.0f,
                        60.0f,
                        9.0f,
                        10.0f,
                    }
                ),
            "non-contiguous column slice is incorrect"
        );

        // These start from a view whose offset is already 1. The next
        // transformation must preserve or add to that existing offset.
        minitensor::RuntimeTensor columns_transposed =
            minitensor::transpose(columns, 0, 1);
        minitensor::RuntimeTensor nested_slice =
            minitensor::slice(columns, 0, 1, 2);

        require(
            columns_transposed.shape() ==
                minitensor::Shape{2, 3} &&
                columns_transposed.strides() ==
                    minitensor::Strides{1, 4} &&
                columns_transposed.offset() == 1 &&
                equal_values(
                    columns_transposed.to_host(),
                    {
                        1.0f,
                        5.0f,
                        9.0f,
                        2.0f,
                        60.0f,
                        10.0f,
                    }
                ) &&
                nested_slice.shape() ==
                    minitensor::Shape{2, 2} &&
                nested_slice.offset() == 5 &&
                equal_values(
                    nested_slice.to_host(),
                    {
                        5.0f,
                        60.0f,
                        9.0f,
                        10.0f,
                    }
                ),
            "composition lost an existing view offset"
        );

        minitensor::RuntimeTensor every_other_column =
            minitensor::slice(matrix, 1, 0, 2, 2);

        require(
            every_other_column.shape() ==
                minitensor::Shape{3, 2} &&
                every_other_column.strides() ==
                    minitensor::Strides{4, 2} &&
                every_other_column.offset() == 0 &&
                equal_values(
                    every_other_column.to_host(),
                    {
                        0.0f,
                        2.0f,
                        4.0f,
                        60.0f,
                        8.0f,
                        10.0f,
                    }
                ),
            "stepped slice is incorrect"
        );

        // Views compose: slice the already-transposed view.
        minitensor::RuntimeTensor composed =
            minitensor::slice(
                transposed,
                0,
                1,
                2
            );

        require(
            composed.shape() ==
                minitensor::Shape{2, 3} &&
                composed.strides() ==
                    minitensor::Strides{1, 4} &&
                composed.offset() == 1 &&
                equal_values(
                    composed.to_host(),
                    {
                        1.0f,
                        5.0f,
                        9.0f,
                        2.0f,
                        60.0f,
                        10.0f,
                    }
                ),
            "composed view is incorrect"
        );

        nested_slice.store({1, 1}, 100.0f);
        require(
            matrix.load({2, 2}) == 100.0f,
            "write through composed view was not shared"
        );

        minitensor::RuntimeTensor surviving =
            make_view_that_outlives_source(device);

        require(
            equal_values(
                surviving.to_host(),
                {
                    2.0f,
                    3.0f,
                    5.0f,
                    6.0f,
                }
            ),
            "view did not keep shared storage alive"
        );

        check_general_view_metadata(device);
        check_errors(device, matrix);

        std::cout
            << "Device: " << device.name() << "\n\n"
            << "Base tensor after writes through views\n"
            << minitensor::format_layout(matrix.layout())
            << "\nlogical values: ";
        print_values(matrix.to_host());

        std::cout
            << "\n\nTranspose axes 0 and 1\n"
            << minitensor::format_layout(
                   transposed.layout()
               )
            << "\nlogical values: ";
        print_values(transposed.to_host());

        std::cout
            << "\n\nSlice columns [1, 3)\n"
            << minitensor::format_layout(columns.layout())
            << "\nlogical values: ";
        print_values(columns.to_host());

        std::cout
            << "\n\nStepped slice columns 0, 2\n"
            << minitensor::format_layout(
                   every_other_column.layout()
               )
            << "\nlogical values: ";
        print_values(every_other_column.to_host());

        std::cout
            << "\n\nShared-storage proof\n"
            << "transposed[2,1] and matrix[1,2]"
               " both use storage[6]\n"
            << "a view remains valid after its source"
               " handle is destroyed\n\n"
            << "PASS: transpose and slice changed only "
               "layout metadata while sharing one Metal "
               "buffer.\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
