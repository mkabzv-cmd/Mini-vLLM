#import <Foundation/Foundation.h>

#include "minitensor.hpp"

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

[[nodiscard]] miniir::Type vector_type(
    std::size_t elements,
    miniir::DType dtype = miniir::DType::F32
) {
    return miniir::Type::tensor(dtype, {elements});
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

[[nodiscard]] miniir::Program
make_explicit_builder_reference() {
    const miniir::Type vector4 = vector_type(4);

    miniir::ProgramBuilder builder;
    const miniir::Value a =
        builder.input("a", vector4);
    const miniir::Value b =
        builder.input("b", vector4);
    const miniir::Value two =
        builder.constant_f32(2.0f);
    const miniir::Value scaled =
        builder.multiply(b, two);
    const miniir::Value added =
        builder.add(a, scaled);
    const miniir::Value output =
        builder.relu(added);

    builder.return_value(output);
    return builder.build();
}

[[nodiscard]] minitensor::CapturedExpression
make_main_expression() {
    const miniir::Type vector4 = vector_type(4);

    minitensor::Graph graph;
    const minitensor::Tensor a =
        graph.input("a", vector4);
    const minitensor::Tensor b =
        graph.input("b", vector4);

    // These operators only append MiniIR operations. There is no Device,
    // Buffer, Queue, or GPU dispatch yet.
    const minitensor::Tensor output =
        minitensor::relu(a + b * 2.0f);

    require(
        output.type() == vector4,
        "MiniTensor did not preserve the inferred type"
    );

    minitensor::CapturedExpression captured =
        minitensor::capture(output);

    require_throws<minitensor::FrontendError>(
        [&] {
            static_cast<void>(output + 1.0f);
        },
        "sealed"
    );
    require_throws<minitensor::FrontendError>(
        [&] {
            static_cast<void>(graph.scalar(1.0f));
        },
        "sealed"
    );
    require_throws<minitensor::FrontendError>(
        [&] {
            static_cast<void>(
                minitensor::capture(output)
            );
        },
        "sealed"
    );

    return captured;
}

void check_frontend_errors() {
    {
        minitensor::Graph first;
        minitensor::Graph second;
        const minitensor::Tensor left =
            first.input("left", {4});
        const minitensor::Tensor right =
            second.input("right", {4});

        require_throws<minitensor::FrontendError>(
            [&] {
                static_cast<void>(left + right);
            },
            "different MiniTensor graphs"
        );
    }

    {
        minitensor::Graph graph;
        const minitensor::Tensor vector4 =
            graph.input("vector4", {4});
        const minitensor::Tensor vector3 =
            graph.input("vector3", {3});

        require_throws<miniir::TypeError>(
            [&] {
                static_cast<void>(vector4 + vector3);
            },
            "shape mismatch"
        );
    }

    {
        minitensor::Graph graph;
        const minitensor::Tensor f32 =
            graph.input(
                "f32",
                {4},
                miniir::DType::F32
            );
        const minitensor::Tensor f16 =
            graph.input(
                "f16",
                {4},
                miniir::DType::F16
            );

        require_throws<miniir::TypeError>(
            [&] {
                static_cast<void>(f32 * f16);
            },
            "dtype mismatch"
        );
    }

    {
        minitensor::Graph graph;
        static_cast<void>(graph.input("x", {4}));

        require_throws<std::invalid_argument>(
            [&] {
                static_cast<void>(
                    graph.input("x", {4})
                );
            },
            "duplicate MiniTensor input"
        );
    }

    {
        minitensor::Graph graph;
        const minitensor::Tensor x =
            graph.input("x", {4});

        require_throws<std::invalid_argument>(
            [&] {
                static_cast<void>(
                    x *
                    std::numeric_limits<float>::
                        infinity()
                );
            },
            "finite"
        );
    }
}

void check_shared_symbolic_handle() {
    minitensor::Graph graph;
    const minitensor::Tensor x =
        graph.input("x", {4});
    const minitensor::Tensor scaled = x * 2.0f;

    // Copying a Tensor copies one symbolic handle. It does not append a
    // second multiply and does not copy runtime tensor data.
    const minitensor::Tensor same_scaled = scaled;
    const minitensor::Tensor output =
        scaled + same_scaled;
    const minitensor::CapturedExpression captured =
        minitensor::capture(output);

    std::size_t multiply_count = 0;
    const miniir::Operation* final_add = nullptr;

    for (const miniir::Operation& operation :
         captured.program().operations()) {
        if (operation.opcode == miniir::Opcode::Multiply) {
            ++multiply_count;
        }

        if (operation.opcode == miniir::Opcode::Add) {
            final_add = &operation;
        }
    }

    require(
        multiply_count == 1,
        "copying a Tensor duplicated its defining operation"
    );
    require(
        final_add != nullptr &&
            final_add->operands[0] ==
                final_add->operands[1],
        "reused Tensor did not reuse one SSA value"
    );
}

[[nodiscard]] minitensor::Tensor
make_tensor_that_outlives_graph_handle() {
    minitensor::Graph graph;
    const minitensor::Tensor x =
        graph.input("x", {3});

    return minitensor::relu(x + 1.0f);
}

void check_graph_state_lifetime() {
    const minitensor::Tensor output =
        make_tensor_that_outlives_graph_handle();
    const minitensor::CapturedExpression captured =
        minitensor::capture(output);

    require(
        captured.program().output_type() ==
            vector_type(3),
        "Tensor did not retain its shared graph state"
    );
}

void check_optimizer_integration(
    const minigpu::Device& device,
    minigpu::Queue& queue
) {
    minitensor::Graph graph;
    const minitensor::Tensor x =
        graph.input("x", {4});
    const minitensor::Tensor unused =
        graph.input("unused", {4});

    // This branch is captured but cannot affect the selected output.
    static_cast<void>(unused + 100.0f);

    const minitensor::Tensor one = graph.scalar(1.0f);
    const minitensor::Tensor two = one + one;
    const minitensor::Tensor output = x * two;

    const minitensor::CapturedExpression captured =
        minitensor::capture(output);
    const minitensor::CompiledExpression compiled =
        minitensor::compile(device, captured);

    const std::string expected =
        "%0 = input @x : tensor<4xf32>\n"
        "%1 = constant_f32 2.0 : f32\n"
        "%2 = multiply %0, %1 : tensor<4xf32>\n"
        "return %2\n";

    require(
        compiled.optimization().folds.size() == 1 &&
            compiled.optimization().removed_values == 4 &&
            miniir::format_program(
                compiled.optimized_program()
            ) ==
                expected,
        "MiniTensor compile did not apply Lesson 23 passes"
    );
    require(
        compiled.plan().inputs().size() == 1 &&
            compiled.plan().inputs()[0].name == "x",
        "dead frontend input survived optimization"
    );

    const minitensor::TensorData with_dead_feed =
        minitensor::evaluate(
            device,
            queue,
            compiled,
            {
                {
                    "x",
                    minitensor::TensorData(
                        vector_type(4),
                        {1.0f, 2.0f, 3.0f, 4.0f}
                    ),
                },
                {
                    "unused",
                    minitensor::TensorData(
                        vector_type(4),
                        {
                            100.0f,
                            200.0f,
                            300.0f,
                            400.0f,
                        }
                    ),
                },
            }
        );

    const minitensor::TensorData without_dead_feed =
        minitensor::evaluate(
            device,
            queue,
            compiled,
            {
                {
                    "x",
                    minitensor::TensorData(
                        vector_type(4),
                        {1.0f, 2.0f, 3.0f, 4.0f}
                    ),
                },
            }
        );

    require(
        equal_values(
            with_dead_feed.values(),
            {2.0f, 4.0f, 6.0f, 8.0f}
        ) &&
            equal_values(
                without_dead_feed.values(),
                with_dead_feed.values()
            ),
        "optimized-away input feed handling is incorrect"
    );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                minitensor::evaluate(
                    device,
                    queue,
                    compiled,
                    {
                        {
                            "x",
                            minitensor::TensorData(
                                vector_type(4),
                                {
                                    1.0f,
                                    2.0f,
                                    3.0f,
                                    4.0f,
                                }
                            ),
                        },
                        {
                            "unused",
                            minitensor::TensorData(
                                miniir::Type::tensor(
                                    miniir::DType::F32,
                                    {2, 2}
                                ),
                                {
                                    100.0f,
                                    200.0f,
                                    300.0f,
                                    400.0f,
                                }
                            ),
                        },
                    }
                )
            );
        },
        "input @unused requires tensor<4xf32>"
    );
}

void check_scalar_operator_forms() {
    minitensor::Graph graph;
    const minitensor::Tensor x =
        graph.input("x", {4});

    const minitensor::Tensor right_multiply = x * 2.0f;
    const minitensor::Tensor left_multiply = 2.0f * x;
    const minitensor::Tensor right_add = x + 1.0f;
    const minitensor::Tensor left_add = 1.0f + x;

    require(
        right_multiply.type() == x.type() &&
            left_multiply.type() == x.type() &&
            right_add.type() == x.type() &&
            left_add.type() == x.type(),
        "a scalar operator form inferred the wrong type"
    );

    static_cast<void>(
        minitensor::capture(
            (right_multiply + left_multiply) *
            (right_add + left_add)
        )
    );
}

void check_tensor_data_errors() {
    require_throws<std::invalid_argument>(
        [] {
            static_cast<void>(
                minitensor::TensorData(
                    vector_type(4),
                    {1.0f, 2.0f, 3.0f}
                )
            );
        },
        "requires 4 value(s)"
    );

    require_throws<std::invalid_argument>(
        [] {
            static_cast<void>(
                minitensor::TensorData(
                    vector_type(
                        4,
                        miniir::DType::F16
                    ),
                    {
                        1.0f,
                        2.0f,
                        3.0f,
                        4.0f,
                    }
                )
            );
        },
        "supports only f32"
    );
}

void check_evaluation_errors(
    const minigpu::Device& device,
    minigpu::Queue& queue,
    const minitensor::CompiledExpression& expression
) {
    const miniir::Type vector4 = vector_type(4);

    const minitensor::TensorData a(
        vector4,
        {-3.0f, 1.0f, 5.0f, -2.0f}
    );
    const minitensor::TensorData b(
        vector4,
        {1.0f, 2.0f, -4.0f, 10.0f}
    );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                minitensor::evaluate(
                    device,
                    queue,
                    expression,
                    {{"a", a}}
                )
            );
        },
        "missing MiniTensor input @b"
    );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                minitensor::evaluate(
                    device,
                    queue,
                    expression,
                    {
                        {"a", a},
                        {"b", b},
                        {
                            "extra",
                            minitensor::TensorData(
                                miniir::Type::scalar(
                                    miniir::DType::F32
                                ),
                                {1.0f}
                            ),
                        },
                    }
                )
            );
        },
        "unknown MiniTensor input @extra"
    );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                minitensor::evaluate(
                    device,
                    queue,
                    expression,
                    {
                        {"a", a},
                        {"b", b},
                        {
                            "zeta",
                            minitensor::TensorData(
                                miniir::Type::scalar(
                                    miniir::DType::F32
                                ),
                                {1.0f}
                            ),
                        },
                        {
                            "alpha",
                            minitensor::TensorData(
                                miniir::Type::scalar(
                                    miniir::DType::F32
                                ),
                                {1.0f}
                            ),
                        },
                    }
                )
            );
        },
        "unknown MiniTensor input @alpha"
    );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                minitensor::evaluate(
                    device,
                    queue,
                    expression,
                    {
                        {
                            "a",
                            minitensor::TensorData(
                                miniir::Type::tensor(
                                    miniir::DType::F32,
                                    {2, 2}
                                ),
                                {
                                    -3.0f,
                                    1.0f,
                                    5.0f,
                                    -2.0f,
                                }
                            ),
                        },
                        {"b", b},
                    }
                )
            );
        },
        "requires tensor<4xf32>, received "
        "tensor<2x2xf32>"
    );
}

[[nodiscard]] std::vector<float>
run_scalar_input_expression(
    const minigpu::Device& device,
    minigpu::Queue& queue
) {
    const miniir::Type vector5 = vector_type(5);

    minitensor::Graph graph;
    const minitensor::Tensor x =
        graph.input("x", vector5);
    const minitensor::Tensor scale =
        graph.scalar_input("scale");
    const minitensor::Tensor bias =
        graph.scalar_input("bias");
    const minitensor::Tensor output =
        minitensor::relu(x * scale + bias);

    const minitensor::CapturedExpression captured =
        minitensor::capture(output);
    const minitensor::CompiledExpression compiled =
        minitensor::compile(device, captured);

    const minitensor::TensorData result =
        minitensor::evaluate(
            device,
            queue,
            compiled,
            {
                {
                    "x",
                    minitensor::TensorData(
                        vector5,
                        {
                            -2.0f,
                            -1.0f,
                            0.0f,
                            1.0f,
                            2.0f,
                        }
                    ),
                },
                {
                    "scale",
                    minitensor::TensorData(
                        miniir::Type::scalar(
                            miniir::DType::F32
                        ),
                        {2.0f}
                    ),
                },
                {
                    "bias",
                    minitensor::TensorData(
                        miniir::Type::scalar(
                            miniir::DType::F32
                        ),
                        {-1.0f}
                    ),
                },
            }
        );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                minitensor::evaluate(
                    device,
                    queue,
                    compiled,
                    {
                        {
                            "x",
                            minitensor::TensorData(
                                vector5,
                                {
                                    -2.0f,
                                    -1.0f,
                                    0.0f,
                                    1.0f,
                                    2.0f,
                                }
                            ),
                        },
                        {
                            "scale",
                            minitensor::TensorData(
                                miniir::Type::tensor(
                                    miniir::DType::F32,
                                    {1}
                                ),
                                {2.0f}
                            ),
                        },
                        {
                            "bias",
                            minitensor::TensorData(
                                miniir::Type::scalar(
                                    miniir::DType::F32
                                ),
                                {-1.0f}
                            ),
                        },
                    }
                )
            );
        },
        "requires f32, received tensor<1xf32>"
    );

    return result.values();
}

void check_backend_boundary(
    const minigpu::Device& device
) {
    minitensor::Graph graph;
    const minitensor::Tensor x =
        graph.input(
            "x",
            {4},
            miniir::DType::F16
        );

    require_throws<miniir::TypeError>(
        [&] {
            static_cast<void>(x + 1.0f);
        },
        "dtype mismatch"
    );

    const minitensor::CapturedExpression captured =
        minitensor::capture(minitensor::relu(x));

    for (const miniir::Operation& operation :
         captured.program().operations()) {
        require(
            operation.opcode !=
                miniir::Opcode::ConstantF32,
            "failed scalar operation mutated the graph"
        );
    }

    require_throws<miniir::metal::CodegenError>(
        [&] {
            static_cast<void>(
                minitensor::compile(device, captured)
            );
        },
        "supports only f32"
    );
}

} // namespace

int main() {
    @autoreleasepool {
        try {
            std::cout
                << "Lesson 24: MiniTensor frontend\n\n";

            check_frontend_errors();
            check_shared_symbolic_handle();
            check_graph_state_lifetime();
            check_scalar_operator_forms();
            check_tensor_data_errors();

            const minitensor::CapturedExpression captured =
                make_main_expression();
            const miniir::Program explicit_reference =
                make_explicit_builder_reference();

            const std::string captured_ir =
                miniir::format_program(
                    captured.program()
                );

            require(
                captured_ir ==
                    miniir::format_program(
                        explicit_reference
                    ),
                "MiniTensor syntax did not create the same "
                "MiniIR as the explicit builder"
            );

            const std::string expected_ir =
                "%0 = input @a : tensor<4xf32>\n"
                "%1 = input @b : tensor<4xf32>\n"
                "%2 = constant_f32 2.0 : f32\n"
                "%3 = multiply %1, %2 : tensor<4xf32>\n"
                "%4 = add %0, %3 : tensor<4xf32>\n"
                "%5 = relu %4 : tensor<4xf32>\n"
                "return %5\n";

            require(
                captured_ir == expected_ir,
                "captured MiniTensor IR changed unexpectedly"
            );

            minigpu::Device device;
            auto queue = device.create_queue();

            check_optimizer_integration(
                device,
                *queue
            );

            const minitensor::CompiledExpression compiled =
                minitensor::compile(device, captured);

            require(
                miniir::format_program(
                    compiled.captured_program()
                ) == captured_ir,
                "compilation mutated the captured expression"
            );

            const miniir::metal::MetalKernelPlan& plan =
                compiled.plan();

            require(
                plan.inputs().size() == 2 &&
                    plan.inputs()[0].name == "a" &&
                    plan.inputs()[0].buffer_index == 0 &&
                    plan.inputs()[1].name == "b" &&
                    plan.inputs()[1].buffer_index == 1 &&
                    plan.output_buffer_index() == 2 &&
                    plan.element_count_index() == 3 &&
                    plan.launch().logical_work_items == 4 &&
                    plan.launch().threads_per_group == 64,
                "MiniTensor produced incorrect Metal metadata"
            );
            require(
                plan.source().find(
                    "if (global_id >= element_count)"
                ) != std::string::npos &&
                    plan.source().find(" * ") !=
                        std::string::npos &&
                    plan.source().find(" + ") !=
                        std::string::npos &&
                    plan.source().find("max(") !=
                        std::string::npos,
                "MiniTensor-generated MSL is missing "
                "expected operations or its bounds guard"
            );

            const miniir::Type vector4 = vector_type(4);

            const minitensor::TensorData first =
                minitensor::evaluate(
                    device,
                    *queue,
                    compiled,
                    {
                        {
                            "a",
                            minitensor::TensorData(
                                vector4,
                                {
                                    -3.0f,
                                    1.0f,
                                    5.0f,
                                    -2.0f,
                                }
                            ),
                        },
                        {
                            "b",
                            minitensor::TensorData(
                                vector4,
                                {
                                    1.0f,
                                    2.0f,
                                    -4.0f,
                                    10.0f,
                                }
                            ),
                        },
                    }
                );

            const minitensor::TensorData second =
                minitensor::evaluate(
                    device,
                    *queue,
                    compiled,
                    {
                        {
                            "a",
                            minitensor::TensorData(
                                vector4,
                                {
                                    0.0f,
                                    0.0f,
                                    0.0f,
                                    0.0f,
                                }
                            ),
                        },
                        {
                            "b",
                            minitensor::TensorData(
                                vector4,
                                {
                                    1.0f,
                                    -1.0f,
                                    3.0f,
                                    -3.0f,
                                }
                            ),
                        },
                    }
                );

            require(
                first.type() == vector4 &&
                    equal_values(
                        first.values(),
                        {0.0f, 5.0f, 0.0f, 18.0f}
                    ),
                "first MiniTensor GPU evaluation is wrong"
            );
            require(
                equal_values(
                    second.values(),
                    {2.0f, 0.0f, 6.0f, 0.0f}
                ),
                "compiled expression was not safely reusable"
            );

            check_evaluation_errors(
                device,
                *queue,
                compiled
            );

            const std::vector<float> scalar_result =
                run_scalar_input_expression(
                    device,
                    *queue
                );

            require(
                equal_values(
                    scalar_result,
                    {0.0f, 0.0f, 0.0f, 1.0f, 3.0f}
                ),
                "runtime scalar broadcasting is wrong"
            );

            check_backend_boundary(device);

            std::cout
                << "Captured MiniIR:\n"
                << captured_ir
                << "\nOptimized MiniIR:\n"
                << miniir::format_program(
                       compiled.optimized_program()
                   )
                << "\nFirst GPU evaluation:  ";
            print_values(first.values());
            std::cout
                << "\nReused compiled graph:  ";
            print_values(second.values());
            std::cout
                << "\nRuntime scalar inputs: ";
            print_values(scalar_result);
            std::cout
                << "\n\nPASS: MiniTensor syntax captured, "
                   "compiled, and reused one real Metal "
                   "kernel.\n";
        } catch (const std::exception& error) {
            std::cerr << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }

    return 0;
}
