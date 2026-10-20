#include "batched_matmul.hpp"

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

[[nodiscard]] bool close_values(
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
                0.0001f
        ) {
            return false;
        }
    }

    return true;
}

[[nodiscard]] std::vector<float> cpu_batched_matmul(
    const RuntimeTensor& left,
    const RuntimeTensor& right,
    bool transpose_right
) {
    const std::size_t batch = left.shape()[0];
    const std::size_t m = left.shape()[1];
    const std::size_t k = left.shape()[2];
    const std::size_t n =
        transpose_right
            ? right.shape()[1]
            : right.shape()[2];
    std::vector<float> output(
        batch * m * n,
        0.0f
    );

    for (
        std::size_t batch_index = 0;
        batch_index < batch;
        ++batch_index
    ) {
        for (std::size_t row = 0; row < m; ++row) {
            for (
                std::size_t column = 0;
                column < n;
                ++column
            ) {
                float sum = 0.0f;

                for (
                    std::size_t inner = 0;
                    inner < k;
                    ++inner
                ) {
                    const float right_value =
                        transpose_right
                            ? right.load({
                                  batch_index,
                                  column,
                                  inner,
                              })
                            : right.load({
                                  batch_index,
                                  inner,
                                  column,
                              });

                    sum +=
                        left.load({
                            batch_index,
                            row,
                            inner,
                        }) *
                        right_value;
                }

                output[
                    (batch_index * m + row) * n +
                    column
                ] = sum;
            }
        }
    }

    return output;
}

void require_output(
    const RuntimeTensor& output,
    const minitensor::Shape& expected_shape,
    const std::vector<float>& expected,
    std::string_view message
) {
    require(
        output.shape() == expected_shape &&
            output.offset() == 0 &&
            output.is_contiguous() &&
            output.storage_size() == output.numel() &&
            close_values(output.to_host(), expected),
        message
    );
}

void print_batch(
    const RuntimeTensor& tensor,
    std::size_t batch_index
) {
    const std::size_t rows = tensor.shape()[1];
    const std::size_t columns = tensor.shape()[2];

    for (std::size_t row = 0; row < rows; ++row) {
        std::cout << "  [";

        for (
            std::size_t column = 0;
            column < columns;
            ++column
        ) {
            if (column != 0) {
                std::cout << ", ";
            }

            std::cout << tensor.load({
                batch_index,
                row,
                column,
            });
        }

        std::cout << "]\n";
    }
}

[[nodiscard]] RuntimeTensor offset_tensor(
    const minigpu::Device& device,
    miniir::Type type,
    std::size_t offset,
    const std::vector<float>& values
) {
    const minitensor::TensorLayout layout =
        minitensor::TensorLayout::contiguous(
            type.shape(),
            offset
        );
    minigpu::Buffer storage =
        device.allocate(layout.minimum_storage_size());
    RuntimeTensor tensor(
        std::move(type),
        layout,
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

void check_nonzero_offsets(
    const minigpu::Device& device,
    minigpu::Queue& queue,
    const minitensor::BatchedMatmul& batched
) {
    RuntimeTensor left = offset_tensor(
        device,
        miniir::Type::tensor(
            miniir::DType::F32,
            {1, 1, 2}
        ),
        2,
        {2.0f, 3.0f}
    );
    RuntimeTensor right = offset_tensor(
        device,
        miniir::Type::tensor(
            miniir::DType::F32,
            {1, 2, 2}
        ),
        3,
        {
            4.0f,
            5.0f,
            -1.0f,
            2.0f,
        }
    );
    RuntimeTensor output =
        batched.run(queue, left, right, false, 3);

    require_output(
        output,
        {1, 1, 2},
        {5.0f, 16.0f},
        "batched matmul ignored a storage offset"
    );
}

void check_errors(
    const minigpu::Device& device,
    minigpu::Queue& queue,
    const minitensor::BatchedMatmul& batched,
    const RuntimeTensor& left,
    const RuntimeTensor& right
) {
    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                batched.run(
                    queue,
                    left,
                    right,
                    false,
                    0
                )
            );
        },
        "threads_per_group must be greater than zero"
    );

    RuntimeTensor matrix =
        RuntimeTensor::from_host(
            device,
            miniir::Type::tensor(
                miniir::DType::F32,
                {2, 3}
            ),
            std::vector<float>(6, 1.0f)
        );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                batched.run(queue, matrix, right)
            );
        },
        "left input must have rank 3"
    );

    RuntimeTensor wrong_batch =
        RuntimeTensor::from_host(
            device,
            miniir::Type::tensor(
                miniir::DType::F32,
                {1, 3, 2}
            ),
            std::vector<float>(6, 1.0f)
        );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                batched.run(queue, left, wrong_batch)
            );
        },
        "batch dimensions do not match"
    );

    RuntimeTensor wrong_k =
        RuntimeTensor::from_host(
            device,
            miniir::Type::tensor(
                miniir::DType::F32,
                {2, 2, 2}
            ),
            std::vector<float>(8, 1.0f)
        );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                batched.run(queue, left, wrong_k)
            );
        },
        "inner dimensions do not match"
    );

    RuntimeTensor wrong_transposed_k =
        RuntimeTensor::from_host(
            device,
            miniir::Type::tensor(
                miniir::DType::F32,
                {2, 2, 4}
            ),
            std::vector<float>(16, 1.0f)
        );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                batched.run(
                    queue,
                    left,
                    wrong_transposed_k,
                    true
                )
            );
        },
        "inner dimensions do not match"
    );

    const minitensor::TensorLayout padded_layout(
        {2, 2, 3},
        {8, 4, 1}
    );
    RuntimeTensor padded_left(
        left.type(),
        padded_layout,
        device.allocate(
            padded_layout.minimum_storage_size()
        )
    );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                batched.run(
                    queue,
                    padded_left,
                    right
                )
            );
        },
        "must use contiguous row-major storage"
    );
}

} // namespace

int main() {
    try {
        std::cout
            << "Lesson 32: batched matrix multiplication\n\n";

        minigpu::Device device;
        std::unique_ptr<minigpu::Queue> queue =
            device.create_queue();
        minitensor::BatchedMatmul batched(device);

        RuntimeTensor left =
            RuntimeTensor::from_host(
                device,
                miniir::Type::tensor(
                    miniir::DType::F32,
                    {2, 2, 3}
                ),
                {
                    1.0f,
                    2.0f,
                    3.0f,
                    4.0f,
                    5.0f,
                    6.0f,
                    -1.0f,
                    0.0f,
                    1.0f,
                    2.0f,
                    -2.0f,
                    1.0f,
                }
            );
        RuntimeTensor right =
            RuntimeTensor::from_host(
                device,
                miniir::Type::tensor(
                    miniir::DType::F32,
                    {2, 3, 2}
                ),
                {
                    1.0f,
                    0.0f,
                    0.0f,
                    1.0f,
                    1.0f,
                    1.0f,
                    2.0f,
                    1.0f,
                    1.0f,
                    0.0f,
                    0.0f,
                    -1.0f,
                }
            );

        // Eight outputs with three threads per group launches nine lanes.
        RuntimeTensor normal =
            batched.run(
                *queue,
                left,
                right,
                false,
                3
            );

        const std::vector<float> expected{
            4.0f,
            5.0f,
            10.0f,
            11.0f,
            -2.0f,
            -2.0f,
            2.0f,
            1.0f,
        };
        require_output(
            normal,
            {2, 2, 2},
            expected,
            "ordinary batched matmul is incorrect"
        );
        require(
            close_values(
                normal.to_host(),
                cpu_batched_matmul(
                    left,
                    right,
                    false
                )
            ),
            "ordinary Metal result disagrees with CPU"
        );

        // The same matrices, physically stored as [B,N,K].
        RuntimeTensor right_rows =
            RuntimeTensor::from_host(
                device,
                miniir::Type::tensor(
                    miniir::DType::F32,
                    {2, 2, 3}
                ),
                {
                    1.0f,
                    0.0f,
                    1.0f,
                    0.0f,
                    1.0f,
                    1.0f,
                    2.0f,
                    1.0f,
                    0.0f,
                    1.0f,
                    0.0f,
                    -1.0f,
                }
            );
        RuntimeTensor transposed_mode =
            batched.run(
                *queue,
                left,
                right_rows,
                true,
                5
            );

        require_output(
            transposed_mode,
            {2, 2, 2},
            expected,
            "transpose-right mode is incorrect"
        );

        // Attention treats every (batch, head) pair as one independent
        // matrix batch. Here batch_heads = 2.
        RuntimeTensor queries =
            RuntimeTensor::from_host(
                device,
                miniir::Type::tensor(
                    miniir::DType::F32,
                    {2, 3, 2}
                ),
                {
                    1.0f,
                    0.0f,
                    0.0f,
                    1.0f,
                    1.0f,
                    1.0f,
                    1.0f,
                    -1.0f,
                    2.0f,
                    0.0f,
                    0.0f,
                    2.0f,
                }
            );
        RuntimeTensor keys =
            RuntimeTensor::from_host(
                device,
                miniir::Type::tensor(
                    miniir::DType::F32,
                    {2, 2, 2}
                ),
                {
                    1.0f,
                    0.0f,
                    1.0f,
                    1.0f,
                    1.0f,
                    1.0f,
                    -1.0f,
                    1.0f,
                }
            );
        RuntimeTensor scores =
            batched.run(
                *queue,
                queries,
                keys,
                true,
                4
            );

        require_output(
            scores,
            {2, 3, 2},
            {
                1.0f,
                1.0f,
                0.0f,
                1.0f,
                1.0f,
                2.0f,
                0.0f,
                -2.0f,
                2.0f,
                -2.0f,
                2.0f,
                2.0f,
            },
            "attention QK^T scores are incorrect"
        );
        require(
            close_values(
                scores.to_host(),
                cpu_batched_matmul(
                    queries,
                    keys,
                    true
                )
            ),
            "attention scores disagree with CPU"
        );

        check_nonzero_offsets(device, *queue, batched);
        check_errors(
            device,
            *queue,
            batched,
            left,
            right
        );

        std::cout
            << "Device: " << device.name() << "\n\n"
            << "Generic result, batch 0\n";
        print_batch(normal, 0);

        std::cout
            << "\nGeneric result, batch 1\n";
        print_batch(normal, 1);

        std::cout
            << "\nAttention scores, head-batch 0\n";
        print_batch(scores, 0);

        std::cout
            << "\nAttention scores, head-batch 1\n";
        print_batch(scores, 1);

        std::cout
            << "\nPASS: one Metal dispatch computed "
               "independent matrix products for every "
               "batch/head.\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
