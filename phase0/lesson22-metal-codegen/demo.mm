#import <Foundation/Foundation.h>

#include "metal_backend.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using HostInputs =
    std::unordered_map<std::string, std::vector<float>>;

using MetalKernelPlan =
    miniir::metal::MetalKernelPlan;
using MetalExecutable =
    miniir::metal::MetalExecutable;

static_assert(
    !std::is_default_constructible_v<MetalKernelPlan>
);
static_assert(!std::is_aggregate_v<MetalKernelPlan>);
static_assert(
    std::is_same_v<
        decltype(
            std::declval<MetalKernelPlan&>().inputs()
        ),
        const std::vector<
            miniir::metal::InputBinding
        >&
    >
);
static_assert(
    !std::is_default_constructible_v<MetalExecutable>
);
static_assert(
    std::is_same_v<
        decltype(
            std::declval<
                const MetalExecutable&
            >().plan()
        ),
        const MetalKernelPlan&
    >
);

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
        std::string(expected_message_part) +
        "'"
    );
}

[[nodiscard]] bool equal_vectors(
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

void print_vector(const std::vector<float>& values) {
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

[[nodiscard]] miniir::Program make_main_program() {
    const miniir::Type vector4 =
        miniir::Type::tensor(
            miniir::DType::F32,
            {4}
        );

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
    const miniir::Value result =
        builder.relu(added);

    builder.return_value(result);
    return builder.build();
}

[[nodiscard]] miniir::Program
make_scalar_broadcast_program() {
    const miniir::Type matrix =
        miniir::Type::tensor(
            miniir::DType::F32,
            {2, 3}
        );
    const miniir::Type scalar =
        miniir::Type::scalar(
            miniir::DType::F32
        );

    miniir::ProgramBuilder builder;
    const miniir::Value x =
        builder.input("x", matrix);
    const miniir::Value scale =
        builder.input("scale", scalar);
    const miniir::Value bias =
        builder.input("bias", scalar);
    const miniir::Value scaled =
        builder.multiply(x, scale);
    const miniir::Value shifted =
        builder.add(scaled, bias);
    const miniir::Value result =
        builder.relu(shifted);

    builder.return_value(result);
    return builder.build();
}

[[nodiscard]] std::vector<float> cpu_reference(
    const miniir::Program& program,
    const miniir::metal::MetalKernelPlan& plan,
    const HostInputs& host_inputs
) {
    std::vector<float> output;
    output.reserve(
        plan.launch().logical_work_items
    );

    for (
        std::size_t index = 0;
        index < plan.launch().logical_work_items;
        ++index
    ) {
        miniir::InputBindings element_inputs;

        for (
            const miniir::metal::InputBinding& input :
            plan.inputs()
        ) {
            const auto found =
                host_inputs.find(input.name);

            if (found == host_inputs.end()) {
                throw std::invalid_argument(
                    "missing host input @" + input.name
                );
            }

            if (
                found->second.size() !=
                input.required_elements
            ) {
                throw std::invalid_argument(
                    "host input @" + input.name +
                    " has an incorrect size"
                );
            }

            const std::size_t input_index =
                input.type.is_scalar() ? 0 : index;

            element_inputs.emplace(
                input.name,
                found->second[input_index]
            );
        }

        output.push_back(
            miniir::evaluate_f32_element(
                program,
                element_inputs
            )
        );
    }

    return output;
}

[[nodiscard]] std::vector<float> execute_gpu(
    const miniir::metal::MetalExecutable& executable,
    const HostInputs& host_inputs,
    const minigpu::Device& device,
    minigpu::Queue& queue
) {
    const miniir::metal::MetalKernelPlan& plan =
        executable.plan();

    miniir::metal::RuntimeInputs runtime_inputs;

    for (
        const miniir::metal::InputBinding& input :
        plan.inputs()
    ) {
        const auto found =
            host_inputs.find(input.name);

        if (found == host_inputs.end()) {
            throw std::invalid_argument(
                "missing host input @" + input.name
            );
        }

        if (
            found->second.size() !=
            input.required_elements
        ) {
            throw std::invalid_argument(
                "host input @" + input.name +
                " has an incorrect size"
            );
        }

        minigpu::Buffer buffer =
            device.allocate(input.required_elements);

        for (
            std::size_t index = 0;
            index < input.required_elements;
            ++index
        ) {
            buffer.store(index, found->second[index]);
        }

        runtime_inputs.emplace(
            input.name,
            std::move(buffer)
        );
    }

    minigpu::Buffer output =
        device.allocate(
            plan.launch().logical_work_items
        );

    for (
        std::size_t index = 0;
        index < plan.launch().logical_work_items;
        ++index
    ) {
        output.store(index, -999.0f);
    }

    minigpu::Dispatch dispatch =
        executable.make_dispatch(
            runtime_inputs,
            output
        );

    const minigpu::LaunchConfig launch =
        executable.make_launch();

    minigpu::Event finished = queue.submit(
        std::move(dispatch),
        launch
    );
    finished.wait();

    std::vector<float> result;
    result.reserve(
        plan.launch().logical_work_items
    );

    for (
        std::size_t index = 0;
        index < plan.launch().logical_work_items;
        ++index
    ) {
        result.push_back(output.load(index));
    }

    return result;
}

void check_main_codegen(
    const miniir::Program& program,
    const miniir::metal::MetalKernelPlan& plan
) {
    const std::string expected_source =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "\n"
        "kernel void miniir_elementwise(\n"
        "    device const float* input_0 [[buffer(0)]],\n"
        "    device const float* input_1 [[buffer(1)]],\n"
        "    device float* output [[buffer(2)]],\n"
        "    constant uint& element_count [[buffer(3)]],\n"
        "    uint global_id [[thread_position_in_grid]]\n"
        ") {\n"
        "    if (global_id >= element_count) {\n"
        "        return;\n"
        "    }\n"
        "\n"
        "    const float v0 = input_0[global_id];\n"
        "    const float v1 = input_1[global_id];\n"
        "    const float v2 = 2.0f;\n"
        "    const float v3 = v1 * v2;\n"
        "    const float v4 = v0 + v3;\n"
        "    const float v5 = max(v4, 0.0f);\n"
        "\n"
        "    output[global_id] = v5;\n"
        "}\n";

    const std::string expected_plan =
        "function: miniir_elementwise\n"
        "buffer(0) input @a -> %0 : "
        "tensor<4xf32> [4 element(s)]\n"
        "buffer(1) input @b -> %1 : "
        "tensor<4xf32> [4 element(s)]\n"
        "buffer(2) output <- %5 : tensor<4xf32>\n"
        "buffer(3) element_count = 4\n"
        "launch: 4 work item(s), "
        "64 thread(s) per group\n";

    require(
        plan.function_name() == "miniir_elementwise",
        "generated function name is incorrect"
    );
    require(
        plan.source() == expected_source,
        "generated MSL did not match the golden source"
    );
    require(
        miniir::metal::format_plan(plan) ==
            expected_plan,
        "generated plan did not match the golden metadata"
    );
    require(
        plan.inputs().size() == 2 &&
            plan.inputs()[0].name == "a" &&
            plan.inputs()[0].buffer_index == 0 &&
            plan.inputs()[1].name == "b" &&
            plan.inputs()[1].buffer_index == 1 &&
            plan.output_buffer_index() == 2 &&
            plan.element_count_index() == 3 &&
            plan.output_value() == 5 &&
            plan.launch().logical_work_items == 4 &&
            plan.launch().threads_per_group == 64,
        "generated binding or launch metadata is incorrect"
    );

    const miniir::metal::MetalKernelPlan second =
        miniir::metal::lower_to_metal(program);

    require(
        second.source() == plan.source() &&
            miniir::metal::format_plan(second) ==
                miniir::metal::format_plan(plan),
        "Metal lowering was not deterministic"
    );
}

void check_scalar_codegen(
    const miniir::metal::MetalKernelPlan& plan
) {
    require(
        plan.inputs().size() == 3 &&
            plan.inputs()[0].name == "x" &&
            plan.inputs()[1].name == "scale" &&
            plan.inputs()[2].name == "bias",
        "scalar program input metadata is incorrect"
    );
    require(
        plan.source().find(
            "const float v0 = "
            "input_0[global_id];"
        ) != std::string::npos,
        "tensor input did not use global_id"
    );
    require(
        plan.source().find(
            "const float v1 = input_1[0];"
        ) != std::string::npos,
        "scale scalar did not use index zero"
    );
    require(
        plan.source().find(
            "const float v2 = input_2[0];"
        ) != std::string::npos,
        "bias scalar did not use index zero"
    );
}

void check_codegen_errors() {
    {
        const miniir::Type vector4_f16 =
            miniir::Type::tensor(
                miniir::DType::F16,
                {4}
            );

        miniir::ProgramBuilder builder;
        const miniir::Value input =
            builder.input("input", vector4_f16);
        const miniir::Value output =
            builder.relu(input);
        builder.return_value(output);

        require_throws<miniir::metal::CodegenError>(
            [&] {
                static_cast<void>(
                    miniir::metal::lower_to_metal(
                        builder.build()
                    )
                );
            },
            "supports only f32"
        );
    }

    {
        const miniir::Type vector3 =
            miniir::Type::tensor(
                miniir::DType::F32,
                {3}
            );
        const miniir::Type vector4 =
            miniir::Type::tensor(
                miniir::DType::F32,
                {4}
            );

        miniir::ProgramBuilder builder;
        static_cast<void>(
            builder.input("unused", vector3)
        );
        const miniir::Value input =
            builder.input("input", vector4);
        const miniir::Value output =
            builder.relu(input);
        builder.return_value(output);

        require_throws<miniir::metal::CodegenError>(
            [&] {
                static_cast<void>(
                    miniir::metal::lower_to_metal(
                        builder.build()
                    )
                );
            },
            "every non-scalar value"
        );
    }

    {
        const std::size_t too_many_elements =
            static_cast<std::size_t>(
                std::numeric_limits<std::uint32_t>::max()
            ) + 1;

        miniir::ProgramBuilder builder;
        const miniir::Value input =
            builder.input(
                "input",
                miniir::Type::tensor(
                    miniir::DType::F32,
                    {too_many_elements}
                )
            );
        builder.return_value(input);

        require_throws<miniir::metal::CodegenError>(
            [&] {
                static_cast<void>(
                    miniir::metal::lower_to_metal(
                        builder.build()
                    )
                );
            },
            "does not fit"
        );
    }
}

void check_dispatch_errors(
    const miniir::metal::MetalExecutable& executable,
    const minigpu::Device& device
) {
    minigpu::Buffer a = device.allocate(4);
    minigpu::Buffer b = device.allocate(4);
    minigpu::Buffer output = device.allocate(4);

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                executable.make_dispatch(
                    {{"a", a}},
                    output
                )
            );
        },
        "missing runtime input @b"
    );

    minigpu::Buffer extra = device.allocate(1);

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                executable.make_dispatch(
                    {
                        {"a", a},
                        {"b", b},
                        {"extra", extra},
                    },
                    output
                )
            );
        },
        "unknown runtime input @extra"
    );

    minigpu::Buffer wrong_a = device.allocate(3);

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                executable.make_dispatch(
                    {
                        {"a", wrong_a},
                        {"b", b},
                    },
                    output
                )
            );
        },
        "requires 4 element(s), received 3"
    );

    minigpu::Buffer wrong_output = device.allocate(3);

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                executable.make_dispatch(
                    {
                        {"a", a},
                        {"b", b},
                    },
                    wrong_output
                )
            );
        },
        "runtime output requires 4 element(s)"
    );
}

} // namespace

int main() {
    @autoreleasepool {
        try {
            const miniir::Program main_program =
                make_main_program();
            const miniir::metal::MetalKernelPlan main_plan =
                miniir::metal::lower_to_metal(
                    main_program
                );

            check_main_codegen(main_program, main_plan);
            check_codegen_errors();

            minigpu::Device device;
            auto queue = device.create_queue();

            const miniir::metal::MetalExecutable
                main_executable =
                    miniir::metal::compile_metal(
                        device,
                        main_plan
                    );

            require(
                main_executable.plan().source() ==
                    main_plan.source() &&
                    miniir::metal::format_plan(
                        main_executable.plan()
                    ) ==
                        miniir::metal::format_plan(
                            main_plan
                        ),
                "compiled executable did not retain "
                "its exact plan"
            );

            check_dispatch_errors(
                main_executable,
                device
            );

            const HostInputs main_inputs{
                {
                    "a",
                    {-3.0f, 1.0f, 5.0f, -2.0f},
                },
                {
                    "b",
                    {1.0f, 2.0f, -4.0f, 10.0f},
                },
            };
            const std::vector<float> main_expected{
                0.0f,
                5.0f,
                0.0f,
                18.0f,
            };

            const std::vector<float> main_cpu =
                cpu_reference(
                    main_program,
                    main_plan,
                    main_inputs
                );
            const std::vector<float> main_gpu =
                execute_gpu(
                    main_executable,
                    main_inputs,
                    device,
                    *queue
                );

            require(
                equal_vectors(main_cpu, main_expected),
                "main CPU reference is incorrect"
            );
            require(
                equal_vectors(main_gpu, main_cpu),
                "generated main GPU kernel is incorrect"
            );

            const miniir::Program scalar_program =
                make_scalar_broadcast_program();
            const miniir::metal::MetalKernelPlan
                scalar_plan =
                    miniir::metal::lower_to_metal(
                        scalar_program
                    );

            check_scalar_codegen(scalar_plan);

            const miniir::metal::MetalExecutable
                scalar_executable =
                    miniir::metal::compile_metal(
                        device,
                        scalar_plan
                    );

            const HostInputs scalar_inputs{
                {
                    "x",
                    {
                        -2.0f,
                        -1.0f,
                        0.0f,
                        1.0f,
                        2.0f,
                        3.0f,
                    },
                },
                {"scale", {2.0f}},
                {"bias", {-1.0f}},
            };
            const std::vector<float> scalar_expected{
                0.0f,
                0.0f,
                0.0f,
                1.0f,
                3.0f,
                5.0f,
            };

            const std::vector<float> scalar_cpu =
                cpu_reference(
                    scalar_program,
                    scalar_plan,
                    scalar_inputs
                );
            const std::vector<float> scalar_gpu =
                execute_gpu(
                    scalar_executable,
                    scalar_inputs,
                    device,
                    *queue
                );

            require(
                equal_vectors(
                    scalar_cpu,
                    scalar_expected
                ),
                "scalar CPU reference is incorrect"
            );
            require(
                equal_vectors(scalar_gpu, scalar_cpu),
                "generated scalar GPU kernel is incorrect"
            );

            const minigpu::LaunchConfig launch =
                main_executable.make_launch();

            require(
                launch.groups() == 1 &&
                    launch.threads_per_group() == 64 &&
                    launch.total_threads() == 64,
                "generated launch geometry is incorrect"
            );

            std::cout
                << "Lesson 22: typed IR to generated Metal\n\n"
                << "Typed IR:\n"
                << miniir::format_program(main_program)
                << '\n'
                << "Generated ABI and launch plan:\n"
                << miniir::metal::format_plan(main_plan)
                << '\n'
                << "Generated MSL:\n"
                << main_plan.source()
                << '\n'
                << "Metal device: " << device.name()
                << '\n'
                << "Launch: " << launch.groups()
                << " group(s) x "
                << launch.threads_per_group()
                << " threads = "
                << launch.total_threads()
                << " launched for "
                << main_plan.launch().logical_work_items
                << " logical elements\n"
                << "Main GPU output: ";
            print_vector(main_gpu);
            std::cout
                << "\nScalar-broadcast GPU output: ";
            print_vector(scalar_gpu);
            std::cout
                << "\n\nCodegen, metadata, binding, "
                   "launch and GPU checks: PASS\n"
                << "PASS: generated MSL ran on the "
                   "real Apple GPU.\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr
                << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
}
