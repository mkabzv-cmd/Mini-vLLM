#include "tiled_matmul.hpp"

#include "../lesson29-naive-matmul/naive_matmul.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
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

[[nodiscard]] std::vector<float> patterned_values(
    std::size_t count,
    std::size_t seed
) {
    std::vector<float> values;
    values.reserve(count);

    for (
        std::size_t index = 0;
        index < count;
        ++index
    ) {
        const std::size_t folded =
            (index * 29U + seed * 11U + 5U) % 23U;
        const int centered =
            static_cast<int>(folded) - 11;
        values.push_back(
            static_cast<float>(centered) * 0.0625f
        );
    }

    return values;
}

[[nodiscard]] RuntimeTensor make_matrix(
    const minigpu::Device& device,
    std::size_t rows,
    std::size_t columns,
    std::size_t seed
) {
    return RuntimeTensor::from_host(
        device,
        miniir::Type::tensor(
            miniir::DType::F32,
            {rows, columns}
        ),
        patterned_values(rows * columns, seed)
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
            0.0002f * scale
        ) {
            return false;
        }
    }

    return true;
}

void compare_all_three(
    minigpu::Queue& queue,
    const minitensor::NaiveMatmul& naive,
    const minitensor::TiledMatmul& tiled,
    const RuntimeTensor& a,
    const RuntimeTensor& b,
    std::string_view message
) {
    RuntimeTensor naive_output =
        naive.run(queue, a, b);
    RuntimeTensor tiled_output =
        tiled.run(queue, a, b);
    const std::vector<float> cpu_output =
        cpu_matmul(a, b);

    require(
        tiled_output.rank() == 2 &&
            tiled_output.shape()[0] == a.shape()[0] &&
            tiled_output.shape()[1] == b.shape()[1] &&
            tiled_output.is_contiguous() &&
            tiled_output.offset() == 0 &&
            tiled_output.storage_size() ==
                tiled_output.numel() &&
            nearly_equal(
                tiled_output.to_host(),
                naive_output.to_host()
            ) &&
            nearly_equal(
                tiled_output.to_host(),
                cpu_output
            ),
        message
    );
}

template <typename Function>
[[nodiscard]] double median_microseconds(
    Function&& function,
    std::size_t iterations,
    float& checksum
) {
    std::vector<double> samples;
    samples.reserve(iterations);

    for (
        std::size_t iteration = 0;
        iteration < iterations;
        ++iteration
    ) {
        const auto start =
            std::chrono::steady_clock::now();
        RuntimeTensor output =
            std::forward<Function>(function)();
        const auto finish =
            std::chrono::steady_clock::now();

        checksum += output.load_flat(0);
        samples.push_back(
            std::chrono::duration<
                double,
                std::micro
            >(finish - start)
                .count()
        );
    }

    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void check_errors(
    const minigpu::Device& device,
    minigpu::Queue& queue,
    const minitensor::TiledMatmul& tiled,
    const RuntimeTensor& valid_a,
    const RuntimeTensor& valid_b
) {
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
                tiled.run(queue, vector, valid_b)
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
                tiled.run(
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
                tiled.run(
                    queue,
                    transposed_a,
                    valid_b
                )
            );
        },
        "A must use compact row-major storage"
    );
}

} // namespace

int main() {
    try {
        std::cout
            << "Lesson 30: tiled matrix multiplication\n\n";

        minigpu::Device device;
        std::unique_ptr<minigpu::Queue> queue =
            device.create_queue();
        minitensor::NaiveMatmul naive(device);
        minitensor::TiledMatmul tiled(device);

        RuntimeTensor known_a = RuntimeTensor::from_host(
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
        RuntimeTensor known_b = RuntimeTensor::from_host(
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
        RuntimeTensor known_output =
            tiled.run(*queue, known_a, known_b);

        require(
            nearly_equal(
                known_output.to_host(),
                {
                    58.0f, 64.0f,
                    139.0f, 154.0f,
                }
            ),
            "known tiled result is incorrect"
        );
        compare_all_three(
            *queue,
            naive,
            tiled,
            known_a,
            known_b,
            "small edge-padded matrices disagree"
        );

        RuntimeTensor edge_a =
            make_matrix(device, 9, 10, 3);
        RuntimeTensor edge_b =
            make_matrix(device, 10, 11, 7);

        compare_all_three(
            *queue,
            naive,
            tiled,
            edge_a,
            edge_b,
            "multi-tile edge-padded matrices disagree"
        );

        RuntimeTensor exact_a =
            make_matrix(device, 8, 8, 13);
        RuntimeTensor exact_b =
            make_matrix(device, 8, 8, 17);

        compare_all_three(
            *queue,
            naive,
            tiled,
            exact_a,
            exact_b,
            "exact 8x8 tile matrices disagree"
        );

        RuntimeTensor narrow_a =
            make_matrix(device, 1, 13, 19);
        RuntimeTensor narrow_b =
            make_matrix(device, 13, 1, 23);

        compare_all_three(
            *queue,
            naive,
            tiled,
            narrow_a,
            narrow_b,
            "single-cell output matrices disagree"
        );

        check_errors(
            device,
            *queue,
            tiled,
            known_a,
            known_b
        );

        RuntimeTensor timing_a =
            make_matrix(device, 73, 69, 29);
        RuntimeTensor timing_b =
            make_matrix(device, 69, 77, 31);

        // Warm-up prevents first-use effects from dominating the samples.
        static_cast<void>(
            naive.run(*queue, timing_a, timing_b)
        );
        static_cast<void>(
            tiled.run(*queue, timing_a, timing_b)
        );

        constexpr std::size_t timing_iterations = 7;
        float checksum = 0.0f;

        const double naive_median =
            median_microseconds(
                [&] {
                    return naive.run(
                        *queue,
                        timing_a,
                        timing_b
                    );
                },
                timing_iterations,
                checksum
            );
        const double tiled_median =
            median_microseconds(
                [&] {
                    return tiled.run(
                        *queue,
                        timing_a,
                        timing_b
                    );
                },
                timing_iterations,
                checksum
            );

        std::cout
            << "Device: " << device.name() << "\n"
            << "Tile: 8x8, 64 lanes per group\n"
            << "Edge test: [9,10] x [10,11]\n"
            << "Timing shape: [73,69] x [69,77]\n"
            << "Naive median: " << naive_median
            << " us\n"
            << "Tiled median: " << tiled_median
            << " us\n"
            << "Timing checksum: " << checksum << "\n\n"
            << "PASS: tiled, naive, and CPU matmul "
               "agree, including padded edges.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
