#include "naive_matmul.hpp"

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

[[nodiscard]] std::vector<float> cpu_matmul(
    const RuntimeTensor& a,
    const RuntimeTensor& b
) {
    const std::size_t rows = a.shape()[0];
    const std::size_t inner = a.shape()[1];
    const std::size_t columns = b.shape()[1];
    const std::vector<float> a_values = a.to_host();
    const std::vector<float> b_values = b.to_host();
    std::vector<float> output(rows * columns, 0.0f);

    for (
        std::size_t row = 0;
        row < rows;
        ++row
    ) {
        for (
            std::size_t column = 0;
            column < columns;
            ++column
        ) {
            float sum = 0.0f;

            for (
                std::size_t k = 0;
                k < inner;
                ++k
            ) {
                sum +=
                    a_values[row * inner + k] *
                    b_values[k * columns + column];
            }

            output[row * columns + column] = sum;
        }
    }

    return output;
}

[[nodiscard]] bool nearly_equal(
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
            !std::isfinite(expected[index])
        ) {
            return false;
        }

        const float scale = std::max(
            1.0f,
            std::fabs(expected[index])
        );

        if (
            std::fabs(
                actual[index] - expected[index]
            ) >
            0.0001f * scale
        ) {
            return false;
        }
    }

    return true;
}

[[nodiscard]] std::vector<float> patterned_values(
    std::size_t count,
    int seed
) {
    std::vector<float> values;
    values.reserve(count);

    for (
        std::size_t index = 0;
        index < count;
        ++index
    ) {
        const int folded =
            static_cast<int>(
                (index * 17U +
                 static_cast<std::size_t>(seed * seed + 3)) %
                19U
            ) -
            9;
        values.push_back(
            static_cast<float>(folded) * 0.125f
        );
    }

    return values;
}

void require_correct_output(
    const RuntimeTensor& output,
    const RuntimeTensor& a,
    const RuntimeTensor& b,
    std::string_view message
) {
    require(
        output.rank() == 2 &&
            output.shape()[0] == a.shape()[0] &&
            output.shape()[1] == b.shape()[1] &&
            output.is_contiguous() &&
            output.offset() == 0 &&
            output.storage_size() == output.numel() &&
            nearly_equal(
                output.to_host(),
                cpu_matmul(a, b)
            ),
        message
    );
}

void print_matrix(const RuntimeTensor& tensor) {
    const std::vector<float> values = tensor.to_host();
    const std::size_t rows = tensor.shape()[0];
    const std::size_t columns = tensor.shape()[1];

    for (
        std::size_t row = 0;
        row < rows;
        ++row
    ) {
        std::cout << "  [";

        for (
            std::size_t column = 0;
            column < columns;
            ++column
        ) {
            if (column != 0) {
                std::cout << ", ";
            }

            std::cout
                << values[row * columns + column];
        }

        std::cout << "]\n";
    }
}

void check_errors(
    const minigpu::Device& device,
    minigpu::Queue& queue,
    const minitensor::NaiveMatmul& matmul,
    const RuntimeTensor& valid_a,
    const RuntimeTensor& valid_b
) {
    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                matmul.run(
                    queue,
                    valid_a,
                    valid_b,
                    0
                )
            );
        },
        "threads_per_group must be greater than zero"
    );

    RuntimeTensor vector = RuntimeTensor::from_host(
        device,
        miniir::Type::tensor(
            miniir::DType::F32,
            {3}
        ),
        {1.0f, 2.0f, 3.0f}
    );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                matmul.run(queue, vector, valid_b)
            );
        },
        "A must have rank 2"
    );

    RuntimeTensor wrong_inner =
        RuntimeTensor::from_host(
            device,
            miniir::Type::tensor(
                miniir::DType::F32,
                {4, 2}
            ),
            std::vector<float>(8, 1.0f)
        );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                matmul.run(
                    queue,
                    valid_a,
                    wrong_inner
                )
            );
        },
        "inner dimensions must match"
    );

    minigpu::Buffer transpose_storage =
        device.allocate(6);

    for (
        std::size_t index = 0;
        index < transpose_storage.size();
        ++index
    ) {
        transpose_storage.store(
            index,
            static_cast<float>(index)
        );
    }

    RuntimeTensor transposed_a(
        miniir::Type::tensor(
            miniir::DType::F32,
            {2, 3}
        ),
        minitensor::TensorLayout(
            {2, 3},
            {1, 2}
        ),
        transpose_storage
    );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                matmul.run(
                    queue,
                    transposed_a,
                    valid_b
                )
            );
        },
        "A must use compact row-major storage"
    );

    minigpu::Buffer offset_storage =
        device.allocate(9);
    RuntimeTensor offset_a(
        valid_a.type(),
        minitensor::TensorLayout::contiguous(
            valid_a.shape(),
            3
        ),
        offset_storage
    );

    require(
        offset_a.is_contiguous(),
        "offset test tensor should have canonical strides"
    );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                matmul.run(
                    queue,
                    offset_a,
                    valid_b
                )
            );
        },
        "A must use compact row-major storage"
    );

    const std::size_t oversized_threads_per_group =
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()
        ) +
        1;

    require_throws<std::overflow_error>(
        [&] {
            static_cast<void>(
                matmul.run(
                    queue,
                    valid_a,
                    valid_b,
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
            << "Lesson 29: naive matrix multiplication\n\n";

        minigpu::Device device;
        std::unique_ptr<minigpu::Queue> queue =
            device.create_queue();
        minitensor::NaiveMatmul matmul(device);

        RuntimeTensor a = RuntimeTensor::from_host(
            device,
            miniir::Type::tensor(
                miniir::DType::F32,
                {2, 3}
            ),
            {
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f,
            }
        );
        RuntimeTensor b = RuntimeTensor::from_host(
            device,
            miniir::Type::tensor(
                miniir::DType::F32,
                {3, 2}
            ),
            {
                7.0f, 8.0f,
                9.0f, 10.0f,
                11.0f, 12.0f,
            }
        );

        RuntimeTensor output =
            matmul.run(*queue, a, b, 7);

        require(
            nearly_equal(
                output.to_host(),
                {
                    58.0f, 64.0f,
                    139.0f, 154.0f,
                }
            ),
            "known 2x3 by 3x2 result is incorrect"
        );
        require_correct_output(
            output,
            a,
            b,
            "known matrix did not match CPU reference"
        );

        RuntimeTensor rectangular_a =
            RuntimeTensor::from_host(
                device,
                miniir::Type::tensor(
                    miniir::DType::F32,
                    {5, 7}
                ),
                patterned_values(35, 2)
            );
        RuntimeTensor rectangular_b =
            RuntimeTensor::from_host(
                device,
                miniir::Type::tensor(
                    miniir::DType::F32,
                    {7, 4}
                ),
                patterned_values(28, 5)
            );

        RuntimeTensor rectangular_output =
            matmul.run(
                *queue,
                rectangular_a,
                rectangular_b,
                6
            );

        require_correct_output(
            rectangular_output,
            rectangular_a,
            rectangular_b,
            "rectangular matrix result is incorrect"
        );

        RuntimeTensor second_a =
            RuntimeTensor::from_host(
                device,
                miniir::Type::tensor(
                    miniir::DType::F32,
                    {1, 4}
                ),
                {-1.0f, 2.0f, -3.0f, 4.0f}
            );
        RuntimeTensor second_b =
            RuntimeTensor::from_host(
                device,
                miniir::Type::tensor(
                    miniir::DType::F32,
                    {4, 3}
                ),
                {
                    1.0f, 2.0f, 3.0f,
                    4.0f, 5.0f, 6.0f,
                    7.0f, 8.0f, 9.0f,
                    10.0f, 11.0f, 12.0f,
                }
            );
        RuntimeTensor second_output =
            matmul.run(*queue, second_a, second_b);

        require_correct_output(
            second_output,
            second_a,
            second_b,
            "reused NaiveMatmul pipeline is incorrect"
        );

        check_errors(
            device,
            *queue,
            matmul,
            a,
            b
        );

        std::cout
            << "Device: " << device.name() << "\n\n"
            << "A [2,3]\n";
        print_matrix(a);
        std::cout << "\nB [3,2]\n";
        print_matrix(b);
        std::cout << "\nC = A x B [2,2]\n";
        print_matrix(output);
        std::cout
            << "\nPASS: one Metal lane computed each "
               "output cell and matched the CPU.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
