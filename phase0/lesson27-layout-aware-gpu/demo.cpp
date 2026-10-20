#include "layout_relu.hpp"

#include "../lesson26-tensor-views/tensor_views.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using minitensor::RuntimeTensor;

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

[[nodiscard]] std::vector<float> cpu_relu(
    const RuntimeTensor& input
) {
    std::vector<float> expected = input.to_host();

    for (float& value : expected) {
        value = std::max(value, 0.0f);
    }

    return expected;
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

void require_contiguous_output(
    const RuntimeTensor& output,
    const RuntimeTensor& input,
    const std::vector<float>& expected,
    std::string_view message
) {
    require(
        output.type() == input.type() &&
            output.shape() == input.shape() &&
            output.offset() == 0 &&
            output.is_contiguous() &&
            output.storage_size() == output.numel() &&
            equal_values(output.to_host(), expected),
        message
    );
}

void check_errors(
    const minigpu::Device& device,
    minigpu::Queue& queue,
    const minitensor::LayoutAwareRelu& relu,
    const RuntimeTensor& matrix
) {
    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                relu.run(queue, matrix, 0)
            );
        },
        "threads_per_group must be greater than zero"
    );

    RuntimeTensor rank_five =
        RuntimeTensor::from_host(
            device,
            miniir::Type::tensor(
                miniir::DType::F32,
                {1, 1, 1, 1, 1}
            ),
            {-1.0f}
        );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                relu.run(queue, rank_five)
            );
        },
        "rank at most 4"
    );

    const std::size_t oversized_threads_per_group =
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()
        ) + 1;

    require_throws<std::overflow_error>(
        [&] {
            static_cast<void>(
                relu.run(
                    queue,
                    matrix,
                    oversized_threads_per_group
                )
            );
        },
        "launched thread count"
    );
}

} // namespace

int main() {
    try {
        std::cout
            << "Lesson 27: layout-aware Metal indexing\n\n";

        minigpu::Device device;
        std::unique_ptr<minigpu::Queue> queue =
            device.create_queue();
        minitensor::LayoutAwareRelu relu(device);

        RuntimeTensor matrix =
            RuntimeTensor::from_host(
                device,
                miniir::Type::tensor(
                    miniir::DType::F32,
                    {3, 4}
                ),
                {
                    -10.0f,
                    -1.0f,
                    2.0f,
                    -30.0f,
                    4.0f,
                    -5.0f,
                    -6.0f,
                    7.0f,
                    -8.0f,
                    9.0f,
                    -10.0f,
                    11.0f,
                }
            );

        RuntimeTensor contiguous_output =
            relu.run(*queue, matrix, 5);

        require_contiguous_output(
            contiguous_output,
            matrix,
            {
                0.0f,
                0.0f,
                2.0f,
                0.0f,
                4.0f,
                0.0f,
                0.0f,
                7.0f,
                0.0f,
                9.0f,
                0.0f,
                11.0f,
            },
            "contiguous GPU ReLU is incorrect"
        );

        RuntimeTensor columns =
            minitensor::slice(matrix, 1, 1, 2);
        RuntimeTensor columns_output =
            relu.run(*queue, columns, 4);

        require_contiguous_output(
            columns_output,
            columns,
            {
                0.0f,
                2.0f,
                0.0f,
                0.0f,
                9.0f,
                0.0f,
            },
            "sliced GPU ReLU is incorrect"
        );

        RuntimeTensor transposed =
            minitensor::transpose(matrix, 0, 1);
        RuntimeTensor transposed_output =
            relu.run(*queue, transposed, 7);

        require_contiguous_output(
            transposed_output,
            transposed,
            {
                0.0f,
                4.0f,
                0.0f,
                0.0f,
                0.0f,
                9.0f,
                2.0f,
                0.0f,
                0.0f,
                0.0f,
                7.0f,
                11.0f,
            },
            "transposed GPU ReLU is incorrect"
        );

        RuntimeTensor stepped =
            minitensor::slice(matrix, 1, 0, 2, 2);
        RuntimeTensor stepped_output =
            relu.run(*queue, stepped, 4);

        require_contiguous_output(
            stepped_output,
            stepped,
            {
                0.0f,
                2.0f,
                4.0f,
                0.0f,
                0.0f,
                0.0f,
            },
            "stepped GPU ReLU is incorrect"
        );

        RuntimeTensor vector =
            RuntimeTensor::from_host(
                device,
                miniir::Type::tensor(
                    miniir::DType::F32,
                    {8}
                ),
                {
                    99.0f,
                    -1.0f,
                    99.0f,
                    2.0f,
                    99.0f,
                    -3.0f,
                    99.0f,
                    4.0f,
                }
            );
        RuntimeTensor vector_view =
            minitensor::slice(vector, 0, 1, 4, 2);
        RuntimeTensor vector_output =
            relu.run(*queue, vector_view, 3);

        require_contiguous_output(
            vector_output,
            vector_view,
            {
                0.0f,
                2.0f,
                0.0f,
                4.0f,
            },
            "rank-one stepped GPU indexing is incorrect"
        );

        RuntimeTensor composed =
            minitensor::slice(
                transposed,
                0,
                1,
                2
            );
        RuntimeTensor composed_output =
            relu.run(*queue, composed, 4);

        require_contiguous_output(
            composed_output,
            composed,
            {
                0.0f,
                0.0f,
                9.0f,
                2.0f,
                0.0f,
                0.0f,
            },
            "composed-view GPU ReLU is incorrect"
        );

        RuntimeTensor cube =
            RuntimeTensor::from_host(
                device,
                miniir::Type::tensor(
                    miniir::DType::F32,
                    {2, 2, 3}
                ),
                {
                    -1.0f,
                    2.0f,
                    -3.0f,
                    4.0f,
                    -5.0f,
                    6.0f,
                    -7.0f,
                    8.0f,
                    -9.0f,
                    10.0f,
                    -11.0f,
                    12.0f,
                }
            );
        RuntimeTensor cube_view =
            minitensor::transpose(cube, 0, 2);
        RuntimeTensor cube_output =
            relu.run(*queue, cube_view, 5);

        require_contiguous_output(
            cube_output,
            cube_view,
            cpu_relu(cube_view),
            "rank-three GPU indexing is incorrect"
        );

        RuntimeTensor rank_four =
            RuntimeTensor::from_host(
                device,
                miniir::Type::tensor(
                    miniir::DType::F32,
                    {2, 1, 2, 2}
                ),
                {
                    -1.0f,
                    2.0f,
                    -3.0f,
                    4.0f,
                    -5.0f,
                    6.0f,
                    -7.0f,
                    8.0f,
                }
            );
        RuntimeTensor rank_four_view =
            minitensor::transpose(rank_four, 0, 3);
        RuntimeTensor rank_four_output =
            relu.run(*queue, rank_four_view, 3);

        require_contiguous_output(
            rank_four_output,
            rank_four_view,
            {
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                2.0f,
                6.0f,
                4.0f,
                8.0f,
            },
            "rank-four GPU indexing is incorrect"
        );

        minigpu::Buffer scalar_storage =
            device.allocate(3);
        scalar_storage.store(0, 7.0f);
        scalar_storage.store(1, 8.0f);
        scalar_storage.store(2, -3.0f);

        RuntimeTensor scalar =
            RuntimeTensor(
                miniir::Type::scalar(
                    miniir::DType::F32
                ),
                minitensor::TensorLayout::contiguous(
                    {},
                    2
                ),
                scalar_storage
            );
        RuntimeTensor scalar_output =
            relu.run(*queue, scalar, 4);

        require_contiguous_output(
            scalar_output,
            scalar,
            {0.0f},
            "rank-zero offset GPU indexing is incorrect"
        );

        require(
            equal_values(
                matrix.to_host(),
                {
                    -10.0f,
                    -1.0f,
                    2.0f,
                    -30.0f,
                    4.0f,
                    -5.0f,
                    -6.0f,
                    7.0f,
                    -8.0f,
                    9.0f,
                    -10.0f,
                    11.0f,
                }
            ),
            "out-of-place kernel modified its input"
        );

        check_errors(device, *queue, relu, matrix);

        std::cout
            << "Device: " << device.name() << "\n\n"
            << "Input column view\n"
            << minitensor::format_layout(columns.layout())
            << "\nlogical values: ";
        print_values(columns.to_host());

        std::cout
            << "\nGPU ReLU output\n"
            << minitensor::format_layout(
                   columns_output.layout()
               )
            << "\nlogical values: ";
        print_values(columns_output.to_host());

        std::cout
            << "\n\nInput transpose view\n"
            << minitensor::format_layout(
                   transposed.layout()
               )
            << "\nGPU output values: ";
        print_values(transposed_output.to_host());

        std::cout
            << "\n\nPadded launch example: 6 values with "
               "4 threads per group launches 8 lanes; "
               "2 return immediately.\n\n"
            << "PASS: Metal converted logical lane indexes "
               "through shape, strides, and offset before "
               "reading every view.\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
