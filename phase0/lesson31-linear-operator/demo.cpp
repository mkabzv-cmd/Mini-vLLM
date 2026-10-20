#include "linear.hpp"

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

[[nodiscard]] std::vector<float> cpu_linear(
    const RuntimeTensor& input,
    const RuntimeTensor& weight,
    const RuntimeTensor* bias,
    minitensor::LinearActivation activation
) {
    const std::size_t m = input.shape()[0];
    const std::size_t k = input.shape()[1];
    const std::size_t n = weight.shape()[0];
    std::vector<float> output(m * n, 0.0f);

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
                sum +=
                    input.load({row, inner}) *
                    weight.load({column, inner});
            }

            if (bias != nullptr) {
                sum += bias->load({column});
            }

            if (
                activation ==
                    minitensor::LinearActivation::Relu &&
                sum < 0.0f
            ) {
                sum = 0.0f;
            }

            output[row * n + column] = sum;
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

void print_matrix(const RuntimeTensor& tensor) {
    const std::size_t rows = tensor.shape()[0];
    const std::size_t columns = tensor.shape()[1];

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

            std::cout << tensor.load({row, column});
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

    for (
        std::size_t index = 0;
        index < storage.size();
        ++index
    ) {
        storage.store(index, -999.0f);
    }

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
    const minitensor::LinearOperator& linear
) {
    RuntimeTensor input = offset_tensor(
        device,
        miniir::Type::tensor(
            miniir::DType::F32,
            {1, 2}
        ),
        2,
        {2.0f, 3.0f}
    );
    RuntimeTensor weight = offset_tensor(
        device,
        miniir::Type::tensor(
            miniir::DType::F32,
            {2, 2}
        ),
        1,
        {
            4.0f,
            5.0f,
            -1.0f,
            2.0f,
        }
    );
    RuntimeTensor bias = offset_tensor(
        device,
        miniir::Type::tensor(
            miniir::DType::F32,
            {2}
        ),
        3,
        {1.0f, -2.0f}
    );

    RuntimeTensor output = linear.run(
        queue,
        input,
        weight,
        &bias,
        minitensor::LinearActivation::None,
        3
    );

    require_output(
        output,
        {1, 2},
        {24.0f, 2.0f},
        "linear ignored a contiguous tensor offset"
    );
}

void check_errors(
    const minigpu::Device& device,
    minigpu::Queue& queue,
    const minitensor::LinearOperator& linear,
    const RuntimeTensor& input,
    const RuntimeTensor& weight
) {
    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                linear.run(
                    queue,
                    input,
                    weight,
                    nullptr,
                    minitensor::LinearActivation::None,
                    0
                )
            );
        },
        "threads_per_group must be greater than zero"
    );

    RuntimeTensor vector =
        RuntimeTensor::from_host(
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
                linear.run(queue, vector, weight)
            );
        },
        "input must have rank 2"
    );

    RuntimeTensor wrong_k =
        RuntimeTensor::from_host(
            device,
            miniir::Type::tensor(
                miniir::DType::F32,
                {2, 2}
            ),
            {
                1.0f,
                2.0f,
                3.0f,
                4.0f,
            }
        );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                linear.run(queue, input, wrong_k)
            );
        },
        "inner dimensions do not match"
    );

    RuntimeTensor wrong_bias =
        RuntimeTensor::from_host(
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
                linear.run(
                    queue,
                    input,
                    weight,
                    &wrong_bias
                )
            );
        },
        "does not match output width"
    );

    const minitensor::TensorLayout padded_layout(
        {2, 3},
        {4, 1}
    );
    RuntimeTensor padded_input(
        input.type(),
        padded_layout,
        device.allocate(
            padded_layout.minimum_storage_size()
        )
    );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                linear.run(
                    queue,
                    padded_input,
                    weight
                )
            );
        },
        "must use contiguous row-major storage"
    );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                linear.run(
                    queue,
                    input,
                    weight,
                    nullptr,
                    static_cast<
                        minitensor::LinearActivation
                    >(99)
                )
            );
        },
        "activation is not supported"
    );
}

} // namespace

int main() {
    try {
        std::cout
            << "Lesson 31: fused linear operator\n\n";

        minigpu::Device device;
        std::unique_ptr<minigpu::Queue> queue =
            device.create_queue();
        minitensor::LinearOperator linear(device);

        RuntimeTensor input =
            RuntimeTensor::from_host(
                device,
                miniir::Type::tensor(
                    miniir::DType::F32,
                    {2, 3}
                ),
                {
                    1.0f,
                    2.0f,
                    3.0f,
                    -1.0f,
                    0.0f,
                    2.0f,
                }
            );
        RuntimeTensor weight =
            RuntimeTensor::from_host(
                device,
                miniir::Type::tensor(
                    miniir::DType::F32,
                    {4, 3}
                ),
                {
                    1.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    1.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    1.0f,
                    1.0f,
                    1.0f,
                    1.0f,
                }
            );
        RuntimeTensor bias =
            RuntimeTensor::from_host(
                device,
                miniir::Type::tensor(
                    miniir::DType::F32,
                    {4}
                ),
                {
                    -2.0f,
                    0.5f,
                    -4.0f,
                    -3.0f,
                }
            );

        // Eight outputs with three threads per group launches nine lanes.
        // The final padded lane exits before reading or writing memory.
        RuntimeTensor projection =
            linear.run(
                *queue,
                input,
                weight,
                nullptr,
                minitensor::LinearActivation::None,
                3
            );

        require_output(
            projection,
            {2, 4},
            cpu_linear(
                input,
                weight,
                nullptr,
                minitensor::LinearActivation::None
            ),
            "linear projection without bias is incorrect"
        );

        RuntimeTensor biased =
            linear.run(
                *queue,
                input,
                weight,
                &bias,
                minitensor::LinearActivation::None,
                5
            );
        require_output(
            biased,
            {2, 4},
            {
                -1.0f,
                2.5f,
                -1.0f,
                3.0f,
                -3.0f,
                0.5f,
                -2.0f,
                -2.0f,
            },
            "optional bias without activation is incorrect"
        );

        RuntimeTensor relu_only =
            linear.run(
                *queue,
                input,
                weight,
                nullptr,
                minitensor::LinearActivation::Relu,
                5
            );
        require_output(
            relu_only,
            {2, 4},
            {
                1.0f,
                2.0f,
                3.0f,
                6.0f,
                0.0f,
                0.0f,
                2.0f,
                1.0f,
            },
            "optional ReLU without bias is incorrect"
        );

        RuntimeTensor activated =
            linear.run(
                *queue,
                input,
                weight,
                &bias,
                minitensor::LinearActivation::Relu,
                5
            );

        require_output(
            activated,
            {2, 4},
            {
                0.0f,
                2.5f,
                0.0f,
                3.0f,
                0.0f,
                0.5f,
                0.0f,
                0.0f,
            },
            "fused bias and ReLU are incorrect"
        );
        require(
            close_values(
                activated.to_host(),
                cpu_linear(
                    input,
                    weight,
                    &bias,
                    minitensor::LinearActivation::Relu
                )
            ),
            "Metal linear disagrees with CPU reference"
        );

        require(
            close_values(
                input.to_host(),
                {
                    1.0f,
                    2.0f,
                    3.0f,
                    -1.0f,
                    0.0f,
                    2.0f,
                }
            ),
            "linear modified its input"
        );

        check_nonzero_offsets(device, *queue, linear);
        check_errors(
            device,
            *queue,
            linear,
            input,
            weight
        );

        std::cout
            << "Device: " << device.name() << "\n\n"
            << "Input [M=2, K=3]\n";
        print_matrix(input);

        std::cout
            << "\nWeight [N=4, K=3]\n";
        print_matrix(weight);

        std::cout
            << "\ninput @ weight^T [M=2, N=4]\n";
        print_matrix(projection);

        std::cout
            << "\nReLU(input @ weight^T + bias)\n";
        print_matrix(activated);

        std::cout
            << "\nPASS: one reusable Metal kernel performed "
               "projection, optional bias, and optional "
               "ReLU.\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
