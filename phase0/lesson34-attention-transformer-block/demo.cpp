#include "transformer_block.hpp"

#include <algorithm>
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

using Values = std::vector<float>;
using minitensor::RuntimeTensor;
using minitensor::TransformerBlockWeights;

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
    const Values& actual,
    const Values& expected,
    float tolerance = 0.0005f
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
                tolerance
        ) {
            return false;
        }
    }

    return true;
}

void print_values(const Values& values) {
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

[[nodiscard]] RuntimeTensor make_tensor(
    const minigpu::Device& device,
    minitensor::Shape shape,
    Values values
) {
    return RuntimeTensor::from_host(
        device,
        miniir::Type::tensor(
            miniir::DType::F32,
            std::move(shape)
        ),
        std::move(values)
    );
}

[[nodiscard]] TransformerBlockWeights make_weights(
    const minigpu::Device& device
) {
    return TransformerBlockWeights{
        make_tensor(
            device,
            {4},
            {1.0f, 0.9f, 1.1f, 1.0f}
        ),
        make_tensor(
            device,
            {4, 4},
            {
                0.5f, 0.1f, -0.2f, 0.0f,
                -0.1f, 0.4f, 0.0f, 0.2f,
                0.3f, -0.2f, 0.5f, 0.1f,
                0.0f, 0.2f, -0.1f, 0.6f,
            }
        ),
        make_tensor(
            device,
            {4, 4},
            {
                0.4f, -0.1f, 0.2f, 0.0f,
                0.0f, 0.5f, -0.2f, 0.1f,
                -0.3f, 0.1f, 0.4f, 0.2f,
                0.2f, 0.0f, 0.1f, 0.5f,
            }
        ),
        make_tensor(
            device,
            {4, 4},
            {
                0.6f, 0.0f, 0.1f, -0.1f,
                0.2f, 0.5f, 0.0f, 0.1f,
                -0.1f, 0.2f, 0.7f, 0.0f,
                0.0f, -0.2f, 0.1f, 0.6f,
            }
        ),
        make_tensor(
            device,
            {4, 4},
            {
                0.5f, 0.1f, 0.0f, 0.2f,
                -0.1f, 0.6f, 0.2f, 0.0f,
                0.1f, 0.0f, 0.5f, -0.2f,
                0.0f, 0.2f, 0.1f, 0.7f,
            }
        ),
        make_tensor(
            device,
            {4},
            {1.0f, 1.05f, 0.95f, 1.0f}
        ),
        make_tensor(
            device,
            {6, 4},
            {
                0.2f, -0.1f, 0.3f, 0.0f,
                -0.2f, 0.4f, 0.1f, 0.2f,
                0.1f, 0.2f, -0.3f, 0.4f,
                0.3f, 0.0f, 0.2f, -0.1f,
                -0.1f, 0.3f, 0.4f, 0.1f,
                0.2f, -0.2f, 0.0f, 0.5f,
            }
        ),
        make_tensor(
            device,
            {6, 4},
            {
                0.4f, 0.1f, -0.1f, 0.2f,
                0.0f, 0.3f, 0.2f, -0.2f,
                -0.3f, 0.2f, 0.5f, 0.1f,
                0.1f, -0.1f, 0.3f, 0.4f,
                0.2f, 0.4f, -0.2f, 0.0f,
                -0.1f, 0.0f, 0.2f, 0.5f,
            }
        ),
        make_tensor(
            device,
            {4, 6},
            {
                0.3f, -0.1f, 0.2f,
                0.0f, 0.1f, 0.2f,
                -0.2f, 0.4f, 0.1f,
                0.3f, 0.0f, -0.1f,
                0.1f, 0.2f, 0.3f,
                -0.2f, 0.4f, 0.0f,
                0.0f, -0.1f, 0.2f,
                0.3f, 0.2f, 0.1f,
            }
        ),
    };
}

[[nodiscard]] Values cpu_linear(
    const Values& input,
    std::size_t rows,
    std::size_t input_size,
    const Values& weight,
    std::size_t output_size
) {
    Values output(rows * output_size, 0.0f);

    for (std::size_t row = 0; row < rows; ++row) {
        for (
            std::size_t column = 0;
            column < output_size;
            ++column
        ) {
            float sum = 0.0f;

            for (
                std::size_t inner = 0;
                inner < input_size;
                ++inner
            ) {
                sum +=
                    input[row * input_size + inner] *
                    weight[
                        column * input_size + inner
                    ];
            }

            output[row * output_size + column] = sum;
        }
    }

    return output;
}

[[nodiscard]] Values cpu_rms_norm(
    const Values& input,
    std::size_t rows,
    std::size_t hidden,
    const Values& weight,
    float epsilon
) {
    Values output(input.size(), 0.0f);

    for (std::size_t row = 0; row < rows; ++row) {
        float sum_squares = 0.0f;

        for (
            std::size_t column = 0;
            column < hidden;
            ++column
        ) {
            const float value =
                input[row * hidden + column];
            sum_squares += value * value;
        }

        const float inverse_rms =
            1.0f /
            std::sqrt(
                sum_squares /
                    static_cast<float>(hidden) +
                epsilon
            );

        for (
            std::size_t column = 0;
            column < hidden;
            ++column
        ) {
            output[row * hidden + column] =
                input[row * hidden + column] *
                inverse_rms *
                weight[column];
        }
    }

    return output;
}

[[nodiscard]] Values cpu_rope(
    const Values& input,
    std::size_t tokens,
    std::size_t heads,
    std::size_t head_dimension,
    std::size_t start_position,
    float theta
) {
    Values output(input.size(), 0.0f);
    const std::size_t half = head_dimension / 2;

    for (
        std::size_t token = 0;
        token < tokens;
        ++token
    ) {
        const float position = static_cast<float>(
            start_position + token
        );

        for (
            std::size_t head = 0;
            head < heads;
            ++head
        ) {
            const std::size_t base =
                (token * heads + head) *
                head_dimension;

            for (
                std::size_t pair = 0;
                pair < half;
                ++pair
            ) {
                const float exponent =
                    2.0f *
                    static_cast<float>(pair) /
                    static_cast<float>(head_dimension);
                const float angle =
                    position /
                    std::pow(theta, exponent);
                const float cosine = std::cos(angle);
                const float sine = std::sin(angle);
                const float first = input[base + pair];
                const float second =
                    input[base + half + pair];

                output[base + pair] =
                    first * cosine - second * sine;
                output[base + half + pair] =
                    second * cosine + first * sine;
            }
        }
    }

    return output;
}

[[nodiscard]] Values tokens_to_heads(
    const Values& input,
    std::size_t tokens,
    std::size_t heads,
    std::size_t head_dimension
) {
    Values output(input.size(), 0.0f);

    for (
        std::size_t head = 0;
        head < heads;
        ++head
    ) {
        for (
            std::size_t token = 0;
            token < tokens;
            ++token
        ) {
            for (
                std::size_t dimension = 0;
                dimension < head_dimension;
                ++dimension
            ) {
                output[
                    (head * tokens + token) *
                        head_dimension +
                    dimension
                ] =
                    input[
                        (token * heads + head) *
                            head_dimension +
                        dimension
                    ];
            }
        }
    }

    return output;
}

[[nodiscard]] Values heads_to_tokens(
    const Values& input,
    std::size_t tokens,
    std::size_t heads,
    std::size_t head_dimension
) {
    Values output(input.size(), 0.0f);

    for (
        std::size_t token = 0;
        token < tokens;
        ++token
    ) {
        for (
            std::size_t head = 0;
            head < heads;
            ++head
        ) {
            for (
                std::size_t dimension = 0;
                dimension < head_dimension;
                ++dimension
            ) {
                output[
                    (token * heads + head) *
                        head_dimension +
                    dimension
                ] =
                    input[
                        (head * tokens + token) *
                            head_dimension +
                        dimension
                    ];
            }
        }
    }

    return output;
}

[[nodiscard]] Values cpu_attention(
    const Values& hidden,
    std::size_t tokens,
    std::size_t hidden_size,
    std::size_t heads,
    const TransformerBlockWeights& weights,
    std::size_t start_position,
    float theta
) {
    const std::size_t head_dimension =
        hidden_size / heads;

    Values query = cpu_linear(
        hidden,
        tokens,
        hidden_size,
        weights.query_weight.to_host(),
        hidden_size
    );
    Values key = cpu_linear(
        hidden,
        tokens,
        hidden_size,
        weights.key_weight.to_host(),
        hidden_size
    );
    Values value = cpu_linear(
        hidden,
        tokens,
        hidden_size,
        weights.value_weight.to_host(),
        hidden_size
    );

    query = tokens_to_heads(
        cpu_rope(
            query,
            tokens,
            heads,
            head_dimension,
            start_position,
            theta
        ),
        tokens,
        heads,
        head_dimension
    );
    key = tokens_to_heads(
        cpu_rope(
            key,
            tokens,
            heads,
            head_dimension,
            start_position,
            theta
        ),
        tokens,
        heads,
        head_dimension
    );
    value = tokens_to_heads(
        value,
        tokens,
        heads,
        head_dimension
    );

    const float scale =
        1.0f /
        std::sqrt(
            static_cast<float>(head_dimension)
        );
    Values context(
        heads * tokens * head_dimension,
        0.0f
    );

    for (
        std::size_t head = 0;
        head < heads;
        ++head
    ) {
        for (
            std::size_t query_token = 0;
            query_token < tokens;
            ++query_token
        ) {
            Values row_scores(query_token + 1, 0.0f);

            for (
                std::size_t key_token = 0;
                key_token <= query_token;
                ++key_token
            ) {
                float dot = 0.0f;

                for (
                    std::size_t dimension = 0;
                    dimension < head_dimension;
                    ++dimension
                ) {
                    dot +=
                        query[
                            (head * tokens +
                             query_token) *
                                head_dimension +
                            dimension
                        ] *
                        key[
                            (head * tokens +
                             key_token) *
                                head_dimension +
                            dimension
                        ];
                }

                row_scores[key_token] = dot * scale;
            }

            const float maximum =
                *std::max_element(
                    row_scores.begin(),
                    row_scores.end()
                );
            float sum = 0.0f;

            for (float& score : row_scores) {
                score = std::exp(score - maximum);
                sum += score;
            }

            for (
                std::size_t key_token = 0;
                key_token <= query_token;
                ++key_token
            ) {
                const float probability =
                    row_scores[key_token] / sum;

                for (
                    std::size_t dimension = 0;
                    dimension < head_dimension;
                    ++dimension
                ) {
                    context[
                        (head * tokens +
                         query_token) *
                            head_dimension +
                        dimension
                    ] +=
                        probability *
                        value[
                            (head * tokens +
                             key_token) *
                                head_dimension +
                            dimension
                        ];
                }
            }
        }
    }

    const Values context_by_token =
        heads_to_tokens(
            context,
            tokens,
            heads,
            head_dimension
        );

    return cpu_linear(
        context_by_token,
        tokens,
        hidden_size,
        weights.attention_output_weight.to_host(),
        hidden_size
    );
}

[[nodiscard]] Values add_values(
    const Values& left,
    const Values& right
) {
    Values output(left.size(), 0.0f);

    for (
        std::size_t index = 0;
        index < output.size();
        ++index
    ) {
        output[index] = left[index] + right[index];
    }

    return output;
}

[[nodiscard]] Values cpu_block(
    const Values& hidden,
    std::size_t tokens,
    std::size_t hidden_size,
    std::size_t heads,
    const TransformerBlockWeights& weights,
    std::size_t start_position,
    float epsilon,
    float theta
) {
    const Values attention_input = cpu_rms_norm(
        hidden,
        tokens,
        hidden_size,
        weights.attention_norm_weight.to_host(),
        epsilon
    );
    const Values attention_output = cpu_attention(
        attention_input,
        tokens,
        hidden_size,
        heads,
        weights,
        start_position,
        theta
    );
    const Values after_attention =
        add_values(hidden, attention_output);
    const Values feed_forward_input = cpu_rms_norm(
        after_attention,
        tokens,
        hidden_size,
        weights.ffn_norm_weight.to_host(),
        epsilon
    );

    const std::size_t intermediate =
        weights.gate_weight.shape()[0];
    Values gate = cpu_linear(
        feed_forward_input,
        tokens,
        hidden_size,
        weights.gate_weight.to_host(),
        intermediate
    );
    const Values up = cpu_linear(
        feed_forward_input,
        tokens,
        hidden_size,
        weights.up_weight.to_host(),
        intermediate
    );

    for (
        std::size_t index = 0;
        index < gate.size();
        ++index
    ) {
        const float activated =
            gate[index] /
            (1.0f + std::exp(-gate[index]));
        gate[index] = activated * up[index];
    }

    const Values down = cpu_linear(
        gate,
        tokens,
        intermediate,
        weights.down_weight.to_host(),
        hidden_size
    );

    return add_values(after_attention, down);
}

} // namespace

int main() {
    try {
        std::cout
            << "Lesson 34: causal attention and a "
               "complete transformer block\n\n";

        minigpu::Device device;
        std::unique_ptr<minigpu::Queue> queue =
            device.create_queue();
        minitensor::MiniTransformerBlock block(device);
        const TransformerBlockWeights weights =
            make_weights(device);

        constexpr std::size_t tokens = 3;
        constexpr std::size_t hidden_size = 4;
        constexpr std::size_t heads = 2;
        constexpr float epsilon = 0.00001f;
        constexpr float theta = 10000.0f;

        const Values hidden_values{
            0.2f, -0.1f, 0.3f, 0.4f,
            -0.2f, 0.5f, 0.1f, -0.3f,
            0.6f, -0.4f, 0.2f, 0.1f,
        };
        RuntimeTensor hidden = make_tensor(
            device,
            {tokens, hidden_size},
            hidden_values
        );

        RuntimeTensor attention =
            block.causal_self_attention(
                *queue,
                hidden,
                weights,
                heads
            );
        const Values expected_attention =
            cpu_attention(
                hidden_values,
                tokens,
                hidden_size,
                heads,
                weights,
                0,
                theta
            );

        require(
            attention.shape() ==
                minitensor::Shape{
                    tokens,
                    hidden_size,
                } &&
                attention.is_contiguous() &&
                attention.offset() == 0 &&
                close_values(
                    attention.to_host(),
                    expected_attention
                ),
            "causal self-attention disagrees with CPU"
        );

        Values changed_future = hidden_values;
        changed_future[4] = 20.0f;
        changed_future[5] = -30.0f;
        changed_future[6] = 15.0f;
        changed_future[7] = -25.0f;
        changed_future[8] = -40.0f;
        changed_future[9] = 50.0f;
        changed_future[10] = -35.0f;
        changed_future[11] = 45.0f;

        RuntimeTensor future_changed_tensor =
            make_tensor(
                device,
                {tokens, hidden_size},
                changed_future
            );
        RuntimeTensor future_changed_attention =
            block.causal_self_attention(
                *queue,
                future_changed_tensor,
                weights,
                heads
            );

        const Values first_attention =
            attention.to_host();
        const Values changed_attention =
            future_changed_attention.to_host();

        require(
            close_values(
                Values(
                    first_attention.begin(),
                    first_attention.begin() +
                        hidden_size
                ),
                Values(
                    changed_attention.begin(),
                    changed_attention.begin() +
                        hidden_size
                )
            ),
            "token zero attended to future tokens"
        );

        RuntimeTensor output = block.run(
            *queue,
            hidden,
            weights,
            heads,
            0,
            epsilon,
            theta
        );
        const Values expected_output = cpu_block(
            hidden_values,
            tokens,
            hidden_size,
            heads,
            weights,
            0,
            epsilon,
            theta
        );

        require(
            output.shape() ==
                minitensor::Shape{
                    tokens,
                    hidden_size,
                } &&
                output.is_contiguous() &&
                output.offset() == 0 &&
                close_values(
                    output.to_host(),
                    expected_output
                ),
            "complete transformer block disagrees with CPU"
        );

        constexpr std::size_t shifted_start_position = 7;
        RuntimeTensor shifted_output = block.run(
            *queue,
            hidden,
            weights,
            heads,
            shifted_start_position,
            epsilon,
            theta
        );
        const Values expected_shifted_output = cpu_block(
            hidden_values,
            tokens,
            hidden_size,
            heads,
            weights,
            shifted_start_position,
            epsilon,
            theta
        );

        require(
            close_values(
                shifted_output.to_host(),
                expected_shifted_output
            ),
            "nonzero start position disagrees with CPU"
        );

        require_throws<std::invalid_argument>(
            [&] {
                static_cast<void>(
                    block.run(
                        *queue,
                        hidden,
                        weights,
                        3
                    )
                );
            },
            "divisible by heads"
        );

        RuntimeTensor odd_head_dimension =
            make_tensor(
                device,
                {1, 6},
                Values(6, 0.0f)
            );

        require_throws<std::invalid_argument>(
            [&] {
                static_cast<void>(
                    block.causal_self_attention(
                        *queue,
                        odd_head_dimension,
                        weights,
                        2
                    )
                );
            },
            "even head dimension"
        );

        TransformerBlockWeights bad_weights = weights;
        bad_weights.up_weight = make_tensor(
            device,
            {5, 4},
            Values(20, 0.0f)
        );

        require_throws<std::invalid_argument>(
            [&] {
                static_cast<void>(
                    block.run(
                        *queue,
                        hidden,
                        bad_weights,
                        heads
                    )
                );
            },
            "up weight"
        );

        std::cout
            << "Device: " << device.name() << "\n\n"
            << "Input hidden states [tokens=3, hidden=4]\n";
        print_values(hidden_values);

        std::cout
            << "\n\nCausal attention output\n";
        print_values(attention.to_host());

        std::cout
            << "\n\nComplete block output\n";
        print_values(output.to_host());

        std::cout
            << "\n\nCausal-mask proof: changing tokens 1 "
               "and 2 did not change token 0's output "
               "within floating-point tolerance.\n\n"
            << "PASS: Q/K/V attention, RoPE, causal "
               "softmax, residuals, RMSNorm, and SwiGLU "
               "executed as one miniature transformer "
               "block.\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
