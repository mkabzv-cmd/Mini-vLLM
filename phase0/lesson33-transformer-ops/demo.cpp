#include "transformer_ops.hpp"

#include "../lesson26-tensor-views/tensor_views.hpp"

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

[[nodiscard]] miniir::Type tensor_type(
    std::vector<std::size_t> shape
) {
    return miniir::Type::tensor(
        miniir::DType::F32,
        std::move(shape)
    );
}

[[nodiscard]] bool all_close(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float absolute_tolerance = 0.0001f,
    float relative_tolerance = 0.0001f
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

        const float difference =
            std::fabs(actual[index] - expected[index]);
        const float scale =
            std::fmax(
                std::fabs(actual[index]),
                std::fabs(expected[index])
            );

        if (
            difference >
            absolute_tolerance +
                relative_tolerance * scale
        ) {
            return false;
        }
    }

    return true;
}

void require_fresh_output(
    const RuntimeTensor& output,
    const RuntimeTensor& input,
    std::string_view operation
) {
    require(
        output.type() == input.type() &&
            output.is_contiguous() &&
            output.offset() == 0 &&
            !output.buffer().shares_storage_with(
                input.buffer()
            ),
        std::string(operation) +
            " did not return fresh compact output"
    );
}

[[nodiscard]] std::vector<float> reference_embedding(
    const std::vector<float>& table,
    std::size_t hidden_size,
    const std::vector<std::uint32_t>& token_ids
) {
    std::vector<float> output;
    output.reserve(token_ids.size() * hidden_size);

    for (std::uint32_t token_id : token_ids) {
        const std::size_t row =
            static_cast<std::size_t>(token_id);

        for (
            std::size_t column = 0;
            column < hidden_size;
            ++column
        ) {
            output.push_back(
                table[row * hidden_size + column]
            );
        }
    }

    return output;
}

[[nodiscard]] std::vector<float> reference_rms_norm(
    const std::vector<float>& input,
    std::size_t hidden_size,
    const std::vector<float>& weight,
    float epsilon
) {
    std::vector<float> output(input.size());
    const std::size_t row_count =
        input.size() / hidden_size;

    for (
        std::size_t row = 0;
        row < row_count;
        ++row
    ) {
        double square_sum = 0.0;

        for (
            std::size_t column = 0;
            column < hidden_size;
            ++column
        ) {
            const double value = static_cast<double>(
                input[row * hidden_size + column]
            );
            square_sum += value * value;
        }

        const double mean_square =
            square_sum /
            static_cast<double>(hidden_size);
        const float inverse_rms =
            static_cast<float>(
                1.0 /
                std::sqrt(
                    mean_square +
                    static_cast<double>(epsilon)
                )
            );

        for (
            std::size_t column = 0;
            column < hidden_size;
            ++column
        ) {
            const std::size_t index =
                row * hidden_size + column;
            output[index] =
                input[index] *
                inverse_rms *
                weight[column];
        }
    }

    return output;
}

[[nodiscard]] std::vector<float> reference_rope(
    const std::vector<float>& input,
    std::size_t token_count,
    std::size_t head_count,
    std::size_t head_dimension,
    std::size_t start_position,
    float theta
) {
    std::vector<float> output(input.size());
    const std::size_t half_dimension =
        head_dimension / 2;

    for (
        std::size_t token = 0;
        token < token_count;
        ++token
    ) {
        for (
            std::size_t head = 0;
            head < head_count;
            ++head
        ) {
            const std::size_t vector_start =
                (token * head_count + head) *
                head_dimension;

            for (
                std::size_t pair = 0;
                pair < half_dimension;
                ++pair
            ) {
                const double exponent =
                    2.0 *
                    static_cast<double>(pair) /
                    static_cast<double>(head_dimension);
                const double inverse_frequency =
                    1.0 /
                    std::pow(
                        static_cast<double>(theta),
                        exponent
                    );
                const double angle =
                    static_cast<double>(
                        start_position + token
                    ) *
                    inverse_frequency;
                const float cosine =
                    static_cast<float>(std::cos(angle));
                const float sine =
                    static_cast<float>(std::sin(angle));
                const std::size_t first_index =
                    vector_start + pair;
                const std::size_t second_index =
                    first_index + half_dimension;
                const float first = input[first_index];
                const float second = input[second_index];

                output[first_index] =
                    first * cosine - second * sine;
                output[second_index] =
                    second * cosine + first * sine;
            }
        }
    }

    return output;
}

[[nodiscard]] std::vector<float> reference_silu(
    const std::vector<float>& input
) {
    std::vector<float> output;
    output.reserve(input.size());

    for (float value : input) {
        output.push_back(
            value /
            (1.0f + std::exp(-value))
        );
    }

    return output;
}

void check_embedding(
    const minigpu::Device& device,
    minigpu::Queue& queue,
    const minitensor::TransformerOps& ops
) {
    const std::vector<float> table_values{
        0.0f, 1.0f, 2.0f, 3.0f,
        10.0f, 11.0f, 12.0f, 13.0f,
        20.0f, 21.0f, 22.0f, 23.0f,
        30.0f, 31.0f, 32.0f, 33.0f,
        40.0f, 41.0f, 42.0f, 43.0f,
        50.0f, 51.0f, 52.0f, 53.0f,
    };
    const std::vector<std::uint32_t> token_ids{
        4,
        1,
        4,
        0,
    };
    RuntimeTensor table = RuntimeTensor::from_host(
        device,
        tensor_type({6, 4}),
        table_values
    );
    RuntimeTensor output = ops.embedding(
        queue,
        token_ids,
        table,
        5
    );

    require(
        output.shape() ==
            std::vector<std::size_t>({4, 4}) &&
            output.is_contiguous() &&
            output.offset() == 0 &&
            !output.buffer().shares_storage_with(
                table.buffer()
            ) &&
            all_close(
                output.to_host(),
                reference_embedding(
                    table_values,
                    4,
                    token_ids
                )
            ),
        "embedding lookup is incorrect"
    );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                ops.embedding(queue, {}, table)
            );
        },
        "at least one token ID"
    );
    require_throws<std::out_of_range>(
        [&] {
            static_cast<void>(
                ops.embedding(queue, {6}, table)
            );
        },
        "outside embedding vocabulary"
    );
    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                ops.embedding(
                    queue,
                    {16777217U},
                    table
                )
            );
        },
        "cannot be represented exactly"
    );

    RuntimeTensor transposed =
        minitensor::transpose(table, 0, 1);
    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                ops.embedding(
                    queue,
                    {0},
                    transposed
                )
            );
        },
        "compact row-major storage"
    );
}

void check_rms_norm(
    const minigpu::Device& device,
    minigpu::Queue& queue,
    const minitensor::TransformerOps& ops
) {
    const std::vector<float> input_values{
        1.0f,
        2.0f,
        3.0f,
        4.0f,
        -1.0f,
        0.5f,
        2.0f,
        -2.0f,
    };
    const std::vector<float> weight_values{
        1.0f,
        0.5f,
        2.0f,
        -1.0f,
    };
    constexpr float epsilon = 0.00001f;

    RuntimeTensor input = RuntimeTensor::from_host(
        device,
        tensor_type({2, 4}),
        input_values
    );
    RuntimeTensor weight = RuntimeTensor::from_host(
        device,
        tensor_type({4}),
        weight_values
    );
    RuntimeTensor output =
        ops.rms_norm(queue, input, weight, epsilon);

    require_fresh_output(output, input, "RMSNorm");
    require(
        all_close(
            output.to_host(),
            reference_rms_norm(
                input_values,
                4,
                weight_values,
                epsilon
            ),
            0.0002f,
            0.0002f
        ),
        "RMSNorm values are incorrect"
    );

    // More than 64 columns proves each lane can process a strided sequence
    // rather than assuming one lane per hidden value.
    std::vector<float> wide_values;
    std::vector<float> wide_weight;
    wide_values.reserve(70);
    wide_weight.reserve(70);

    for (
        std::size_t index = 0;
        index < 70;
        ++index
    ) {
        wide_values.push_back(
            static_cast<float>(index % 11) - 5.0f
        );
        wide_weight.push_back(
            0.5f +
            0.01f * static_cast<float>(index % 7)
        );
    }

    RuntimeTensor wide_input = RuntimeTensor::from_host(
        device,
        tensor_type({1, 70}),
        wide_values
    );
    RuntimeTensor wide_scale = RuntimeTensor::from_host(
        device,
        tensor_type({70}),
        wide_weight
    );
    RuntimeTensor wide_output = ops.rms_norm(
        queue,
        wide_input,
        wide_scale,
        epsilon
    );

    require(
        all_close(
            wide_output.to_host(),
            reference_rms_norm(
                wide_values,
                70,
                wide_weight,
                epsilon
            ),
            0.0003f,
            0.0003f
        ),
        "RMSNorm lane-strided row is incorrect"
    );

    RuntimeTensor wrong_weight =
        RuntimeTensor::from_host(
            device,
            tensor_type({3}),
            {1.0f, 1.0f, 1.0f}
        );
    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                ops.rms_norm(
                    queue,
                    input,
                    wrong_weight
                )
            );
        },
        "weight must have shape"
    );
    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                ops.rms_norm(
                    queue,
                    input,
                    weight,
                    0.0f
                )
            );
        },
        "epsilon must be finite"
    );

    RuntimeTensor scalar = RuntimeTensor::from_host(
        device,
        miniir::Type::scalar(miniir::DType::F32),
        {2.0f}
    );
    RuntimeTensor scalar_weight =
        RuntimeTensor::from_host(
            device,
            tensor_type({1}),
            {1.0f}
        );
    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                ops.rms_norm(
                    queue,
                    scalar,
                    scalar_weight
                )
            );
        },
        "at least one dimension"
    );
}

void check_rope(
    const minigpu::Device& device,
    minigpu::Queue& queue,
    const minitensor::TransformerOps& ops
) {
    const std::vector<float> input_values{
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f,
    };
    constexpr std::size_t token_count = 2;
    constexpr std::size_t head_count = 2;
    constexpr std::size_t head_dimension = 4;
    constexpr std::size_t start_position = 3;
    constexpr float theta = 10000.0f;

    RuntimeTensor input = RuntimeTensor::from_host(
        device,
        tensor_type(
            {
                token_count,
                head_count,
                head_dimension,
            }
        ),
        input_values
    );
    RuntimeTensor output = ops.rope(
        queue,
        input,
        start_position,
        theta,
        5
    );

    require_fresh_output(output, input, "RoPE");
    require(
        all_close(
            output.to_host(),
            reference_rope(
                input_values,
                token_count,
                head_count,
                head_dimension,
                start_position,
                theta
            ),
            0.0003f,
            0.0003f
        ),
        "Llama split-half RoPE values are incorrect"
    );

    RuntimeTensor odd_dimension =
        RuntimeTensor::from_host(
            device,
            tensor_type({1, 1, 3}),
            {1.0f, 2.0f, 3.0f}
        );
    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                ops.rope(queue, odd_dimension)
            );
        },
        "head_dimension must be even"
    );
    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                ops.rope(
                    queue,
                    input,
                    0,
                    -1.0f
                )
            );
        },
        "theta must be finite"
    );
    require_throws<std::overflow_error>(
        [&] {
            static_cast<void>(
                ops.rope(
                    queue,
                    input,
                    static_cast<std::size_t>(
                        std::numeric_limits<
                            std::uint32_t
                        >::max()
                    )
                )
            );
        },
        "final token position"
    );
}

void check_silu(
    const minigpu::Device& device,
    minigpu::Queue& queue,
    const minitensor::TransformerOps& ops
) {
    const std::vector<float> input_values{
        -8.0f,
        -2.0f,
        -1.0f,
        0.0f,
        1.0f,
        2.0f,
        4.0f,
        8.0f,
    };
    RuntimeTensor input = RuntimeTensor::from_host(
        device,
        tensor_type({2, 4}),
        input_values
    );
    RuntimeTensor output = ops.silu(queue, input, 3);

    require_fresh_output(output, input, "SiLU");
    require(
        all_close(
            output.to_host(),
            reference_silu(input_values)
        ),
        "SiLU values are incorrect"
    );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                ops.silu(queue, input, 0)
            );
        },
        "threads_per_group must be greater than zero"
    );

    RuntimeTensor transposed =
        minitensor::transpose(input, 0, 1);
    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                ops.silu(queue, transposed)
            );
        },
        "compact row-major storage"
    );
}

} // namespace

int main() {
    try {
        std::cout
            << "Lesson 33: transformer operations\n\n";

        minigpu::Device device;
        std::unique_ptr<minigpu::Queue> queue =
            device.create_queue();
        minitensor::TransformerOps ops(device);

        check_embedding(device, *queue, ops);
        check_rms_norm(device, *queue, ops);
        check_rope(device, *queue, ops);
        check_silu(device, *queue, ops);

        std::cout
            << "Device: " << device.name() << '\n'
            << "PASS: embedding lookup, RMSNorm, "
               "Llama split-half RoPE, and SiLU matched "
               "CPU references.\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
