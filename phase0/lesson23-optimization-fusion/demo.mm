#import <Foundation/Foundation.h>

#include "optimizer.hpp"
#include "../lesson22-metal-codegen/metal_backend.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using HostInputs =
    std::unordered_map<std::string, std::vector<float>>;

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

[[nodiscard]] miniir::Type vector_type(
    std::size_t elements
) {
    return miniir::Type::tensor(
        miniir::DType::F32,
        {elements}
    );
}

[[nodiscard]] miniir::Program make_scale_program() {
    const miniir::Type vector5 = vector_type(5);

    miniir::ProgramBuilder builder;
    const miniir::Value x =
        builder.input("x", vector5);
    const miniir::Value one_left =
        builder.constant_f32(1.0f);
    const miniir::Value one_right =
        builder.constant_f32(1.0f);
    const miniir::Value two =
        builder.add(one_left, one_right);
    const miniir::Value scaled =
        builder.multiply(x, two);
    const miniir::Value one_hundred =
        builder.constant_f32(100.0f);

    static_cast<void>(
        builder.add(x, one_hundred)
    );

    builder.return_value(scaled);
    return builder.build();
}

[[nodiscard]] miniir::Program make_shift_program() {
    const miniir::Type vector5 = vector_type(5);

    miniir::ProgramBuilder builder;
    const miniir::Value scaled =
        builder.input("scaled", vector5);
    const miniir::Value three =
        builder.constant_f32(3.0f);
    const miniir::Value shifted =
        builder.add(scaled, three);

    builder.return_value(shifted);
    return builder.build();
}

[[nodiscard]] miniir::Program
make_activation_program() {
    const miniir::Type vector5 = vector_type(5);

    miniir::ProgramBuilder builder;
    const miniir::Value shifted =
        builder.input("shifted", vector5);
    const miniir::Value activated =
        builder.relu(shifted);

    builder.return_value(activated);
    return builder.build();
}

[[nodiscard]] miniir::Program
make_nested_constant_program() {
    miniir::ProgramBuilder builder;
    const miniir::Value two =
        builder.constant_f32(2.0f);
    const miniir::Value three =
        builder.constant_f32(3.0f);
    const miniir::Value six =
        builder.multiply(two, three);
    const miniir::Value negative =
        builder.constant_f32(-4.0f);
    const miniir::Value zero =
        builder.relu(negative);
    const miniir::Value result =
        builder.add(six, zero);

    builder.return_value(result);
    return builder.build();
}

[[nodiscard]] miniir::Program
make_nonfinite_fold_program() {
    miniir::ProgramBuilder builder;
    const miniir::Value maximum =
        builder.constant_f32(
            std::numeric_limits<float>::max()
        );
    const miniir::Value two =
        builder.constant_f32(2.0f);
    const miniir::Value overflow =
        builder.multiply(maximum, two);

    builder.return_value(overflow);
    return builder.build();
}

[[nodiscard]] miniir::Program
make_multi_use_program() {
    const miniir::Type vector5 = vector_type(5);

    miniir::ProgramBuilder builder;
    const miniir::Value x =
        builder.input("x", vector5);

    static_cast<void>(
        builder.constant_f32(100.0f)
    );

    const miniir::Value two =
        builder.constant_f32(2.0f);
    const miniir::Value scaled =
        builder.multiply(x, two);
    const miniir::Value doubled =
        builder.add(scaled, scaled);
    const miniir::Value result =
        builder.relu(doubled);

    builder.return_value(result);
    return builder.build();
}

[[nodiscard]] miniir::Program
make_wrong_shape_consumer() {
    const miniir::Type vector4 = vector_type(4);

    miniir::ProgramBuilder builder;
    const miniir::Value scaled =
        builder.input("scaled", vector4);
    const miniir::Value result =
        builder.relu(scaled);

    builder.return_value(result);
    return builder.build();
}

[[nodiscard]] miniir::Program
make_unused_connection_consumer() {
    const miniir::Type vector5 = vector_type(5);

    miniir::ProgramBuilder builder;
    static_cast<void>(
        builder.input("scaled", vector5)
    );
    const miniir::Value fallback =
        builder.input("fallback", vector5);

    builder.return_value(fallback);
    return builder.build();
}

[[nodiscard]] miniir::Program
make_repeated_external_consumer() {
    const miniir::Type vector5 = vector_type(5);

    miniir::ProgramBuilder builder;
    const miniir::Value scaled =
        builder.input("scaled", vector5);
    const miniir::Value repeated_x =
        builder.input("x", vector5);
    const miniir::Value result =
        builder.add(scaled, repeated_x);

    builder.return_value(result);
    return builder.build();
}

[[nodiscard]] const miniir::opt::ValueRemap&
find_remap(
    const std::vector<miniir::opt::ValueRemap>& remap,
    miniir::ValueId old_value
) {
    for (const miniir::opt::ValueRemap& entry : remap) {
        if (entry.old_value == old_value) {
            return entry;
        }
    }

    throw std::runtime_error(
        "optimization remap is missing old value %" +
        std::to_string(old_value)
    );
}

void require_remap(
    const std::vector<miniir::opt::ValueRemap>& remap,
    miniir::ValueId old_value,
    std::optional<miniir::ValueId> expected
) {
    require(
        find_remap(remap, old_value).new_value == expected,
        "optimization value remap is incorrect"
    );
}

void check_main_optimization(
    const miniir::Program& original,
    const miniir::opt::OptimizationResult& optimized
) {
    const std::string expected_original =
        "%0 = input @x : tensor<5xf32>\n"
        "%1 = constant_f32 1.0 : f32\n"
        "%2 = constant_f32 1.0 : f32\n"
        "%3 = add %1, %2 : f32\n"
        "%4 = multiply %0, %3 : tensor<5xf32>\n"
        "%5 = constant_f32 100.0 : f32\n"
        "%6 = add %0, %5 : tensor<5xf32>\n"
        "return %4\n";

    const std::string expected_folded =
        "%0 = input @x : tensor<5xf32>\n"
        "%1 = constant_f32 1.0 : f32\n"
        "%2 = constant_f32 1.0 : f32\n"
        "%3 = constant_f32 2.0 : f32\n"
        "%4 = multiply %0, %3 : tensor<5xf32>\n"
        "%5 = constant_f32 100.0 : f32\n"
        "%6 = add %0, %5 : tensor<5xf32>\n"
        "return %4\n";

    const std::string expected_optimized =
        "%0 = input @x : tensor<5xf32>\n"
        "%1 = constant_f32 2.0 : f32\n"
        "%2 = multiply %0, %1 : tensor<5xf32>\n"
        "return %2\n";

    require(
        miniir::format_program(original) ==
            expected_original,
        "original scale IR is incorrect"
    );
    require(
        miniir::format_program(
            optimized.after_constant_folding
        ) == expected_folded,
        "constant-folded scale IR is incorrect"
    );
    require(
        miniir::format_program(optimized.program) ==
            expected_optimized,
        "optimized scale IR is incorrect"
    );

    require(
        optimized.folds.size() == 1 &&
            optimized.folds[0].value == 3 &&
            optimized.folds[0].original_opcode ==
                miniir::Opcode::Add &&
            optimized.folds[0].folded_value == 2.0f,
        "constant-folding statistics are incorrect"
    );
    require(
        optimized.removed_values == 4 &&
            optimized.remap.size() == 7,
        "dead-code-elimination statistics are incorrect"
    );

    require_remap(optimized.remap, 0, 0);
    require_remap(optimized.remap, 1, std::nullopt);
    require_remap(optimized.remap, 2, std::nullopt);
    require_remap(optimized.remap, 3, 1);
    require_remap(optimized.remap, 4, 2);
    require_remap(optimized.remap, 5, std::nullopt);
    require_remap(optimized.remap, 6, std::nullopt);

    const std::string report =
        miniir::opt::format_optimization_report(
            optimized
        );

    require(
        !report.empty() &&
            report ==
                miniir::opt::format_optimization_report(
                    optimized
                ),
        "optimization report is not deterministic"
    );

    const miniir::opt::OptimizationResult repeated =
        miniir::opt::optimize_elementwise(original);

    require(
        miniir::format_program(
            repeated.after_constant_folding
        ) == expected_folded &&
            miniir::format_program(repeated.program) ==
                expected_optimized &&
            miniir::opt::format_optimization_report(
                repeated
            ) == report,
        "repeated optimization is not deterministic"
    );

    const miniir::opt::OptimizationResult second =
        miniir::opt::optimize_elementwise(
            optimized.program
        );

    require(
        miniir::format_program(second.program) ==
            expected_optimized &&
            second.folds.empty() &&
            second.removed_values == 0,
        "optimization is not idempotent"
    );
}

void check_additional_optimization_cases() {
    {
        const miniir::Program program =
            make_nested_constant_program();
        const miniir::opt::ConstantFoldingResult folded =
            miniir::opt::fold_constants(program);

        const std::string expected_folded =
            "%0 = constant_f32 2.0 : f32\n"
            "%1 = constant_f32 3.0 : f32\n"
            "%2 = constant_f32 6.0 : f32\n"
            "%3 = constant_f32 -4.0 : f32\n"
            "%4 = constant_f32 0.0 : f32\n"
            "%5 = constant_f32 6.0 : f32\n"
            "return %5\n";

        require(
            miniir::format_program(folded.program) ==
                expected_folded,
            "nested or ReLU constant folding is incorrect"
        );
        require(
            folded.folds.size() == 3 &&
                folded.folds[0].value == 2 &&
                folded.folds[0].original_opcode ==
                    miniir::Opcode::Multiply &&
                folded.folds[0].folded_value == 6.0f &&
                folded.folds[1].value == 4 &&
                folded.folds[1].original_opcode ==
                    miniir::Opcode::Relu &&
                folded.folds[1].folded_value == 0.0f &&
                folded.folds[2].value == 5 &&
                folded.folds[2].original_opcode ==
                    miniir::Opcode::Add &&
                folded.folds[2].folded_value == 6.0f,
            "nested constant-fold records are incorrect"
        );

        const miniir::opt::OptimizationResult optimized =
            miniir::opt::optimize_elementwise(program);

        require(
            miniir::format_program(optimized.program) ==
                "%0 = constant_f32 6.0 : f32\n"
                "return %0\n",
            "nested constants were not cleaned up by DCE"
        );
    }

    {
        const miniir::Program program =
            make_nonfinite_fold_program();
        const miniir::opt::ConstantFoldingResult folded =
            miniir::opt::fold_constants(program);

        require(
            folded.folds.empty() &&
                miniir::format_program(folded.program) ==
                    miniir::format_program(program),
            "a non-finite constant result must not be folded"
        );
    }

    {
        const miniir::Program program =
            make_multi_use_program();
        const miniir::opt::DeadCodeEliminationResult dce =
            miniir::opt::eliminate_dead_code(program);

        const std::string expected =
            "%0 = input @x : tensor<5xf32>\n"
            "%1 = constant_f32 2.0 : f32\n"
            "%2 = multiply %0, %1 : tensor<5xf32>\n"
            "%3 = add %2, %2 : tensor<5xf32>\n"
            "%4 = relu %3 : tensor<5xf32>\n"
            "return %4\n";

        require(
            dce.removed_values == 1 &&
                miniir::format_program(dce.program) ==
                    expected,
            "multi-use DCE result is incorrect"
        );

        require_remap(dce.remap, 0, 0);
        require_remap(dce.remap, 1, std::nullopt);
        require_remap(dce.remap, 2, 1);
        require_remap(dce.remap, 3, 2);
        require_remap(dce.remap, 4, 3);
        require_remap(dce.remap, 5, 4);
    }
}

[[nodiscard]] miniir::opt::FusionResult
make_and_check_fusion(
    const miniir::Program& optimized_scale,
    const miniir::Program& shift,
    const miniir::Program& activation
) {
    const std::vector<miniir::opt::PipelineStage> stages{
        {
            "scale",
            optimized_scale,
            std::nullopt,
        },
        {
            "shift",
            shift,
            std::string("scaled"),
        },
        {
            "activation",
            activation,
            std::string("shifted"),
        },
    };

    const miniir::opt::FusionResult fusion =
        miniir::opt::fuse_linear_elementwise_pipeline(
            stages
        );

    const std::string expected_fused =
        "%0 = input @x : tensor<5xf32>\n"
        "%1 = constant_f32 2.0 : f32\n"
        "%2 = multiply %0, %1 : tensor<5xf32>\n"
        "%3 = constant_f32 3.0 : f32\n"
        "%4 = add %2, %3 : tensor<5xf32>\n"
        "%5 = relu %4 : tensor<5xf32>\n"
        "return %5\n";

    require(
        miniir::format_program(fusion.program) ==
            expected_fused,
        "fused pipeline IR is incorrect"
    );
    require(
        fusion.stage_names ==
            std::vector<std::string>{
                "scale",
                "shift",
                "activation",
            },
        "fused stage names are incorrect"
    );
    require(
        fusion.edges.size() == 2 &&
            fusion.edges[0].producer_stage == "scale" &&
            fusion.edges[0].consumer_stage == "shift" &&
            fusion.edges[0].consumer_input == "scaled" &&
            fusion.edges[1].producer_stage == "shift" &&
            fusion.edges[1].consumer_stage ==
                "activation" &&
            fusion.edges[1].consumer_input == "shifted",
        "fusion edge metadata is incorrect"
    );
    require(
        fusion.dispatches_before == 3 &&
            fusion.dispatches_after == 1 &&
            fusion.intermediate_buffers_avoided == 2,
        "fusion statistics are incorrect"
    );

    const miniir::opt::FusionResult second =
        miniir::opt::fuse_linear_elementwise_pipeline(
            stages
        );

    require(
        miniir::format_program(second.program) ==
            expected_fused &&
            miniir::opt::format_fusion_report(second) ==
                miniir::opt::format_fusion_report(fusion),
        "fusion is not deterministic"
    );

    const miniir::opt::FusionResult single =
        miniir::opt::fuse_linear_elementwise_pipeline(
            {
                {
                    "already_fused",
                    fusion.program,
                    std::nullopt,
                },
            }
        );

    require(
        miniir::format_program(single.program) ==
            expected_fused &&
            single.dispatches_before == 1 &&
            single.dispatches_after == 1 &&
            single.intermediate_buffers_avoided == 0,
        "single-stage fusion is not idempotent"
    );

    return fusion;
}

void check_fusion_errors(
    const miniir::Program& optimized_scale,
    const miniir::Program& shift
) {
    require_throws<miniir::opt::FusionError>(
        [] {
            static_cast<void>(
                miniir::opt::
                    fuse_linear_elementwise_pipeline({})
            );
        },
        "cannot fuse an empty pipeline"
    );

    require_throws<miniir::opt::FusionError>(
        [&] {
            static_cast<void>(
                miniir::opt::
                    fuse_linear_elementwise_pipeline(
                        {
                            {
                                "scale",
                                optimized_scale,
                                std::string("x"),
                            },
                        }
                    )
            );
        },
        "must not declare a previous-output input"
    );

    require_throws<miniir::opt::FusionError>(
        [&] {
            static_cast<void>(
                miniir::opt::
                    fuse_linear_elementwise_pipeline(
                        {
                            {
                                "scale",
                                optimized_scale,
                                std::nullopt,
                            },
                            {
                                "shift",
                                shift,
                                std::nullopt,
                            },
                        }
                    )
            );
        },
        "must declare which input receives"
    );

    require_throws<miniir::opt::FusionError>(
        [&] {
            static_cast<void>(
                miniir::opt::
                    fuse_linear_elementwise_pipeline(
                        {
                            {
                                "scale",
                                optimized_scale,
                                std::nullopt,
                            },
                            {
                                "shift",
                                shift,
                                std::string("missing"),
                            },
                        }
                    )
            );
        },
        "has no declared input @missing"
    );

    const miniir::Program wrong_shape =
        make_wrong_shape_consumer();

    require_throws<miniir::opt::FusionError>(
        [&] {
            static_cast<void>(
                miniir::opt::
                    fuse_linear_elementwise_pipeline(
                        {
                            {
                                "scale",
                                optimized_scale,
                                std::nullopt,
                            },
                            {
                                "wrong_shape",
                                wrong_shape,
                                std::string("scaled"),
                            },
                        }
                    )
            );
        },
        "cannot connect stage 'scale' output"
    );

    require_throws<miniir::opt::FusionError>(
        [&] {
            static_cast<void>(
                miniir::opt::
                    fuse_linear_elementwise_pipeline(
                        {
                            {
                                "duplicate",
                                optimized_scale,
                                std::nullopt,
                            },
                            {
                                "duplicate",
                                shift,
                                std::string("scaled"),
                            },
                        }
                    )
            );
        },
        "duplicate pipeline stage name"
    );

    const miniir::Program unused_connection =
        make_unused_connection_consumer();

    require_throws<miniir::opt::FusionError>(
        [&] {
            static_cast<void>(
                miniir::opt::
                    fuse_linear_elementwise_pipeline(
                        {
                            {
                                "scale",
                                optimized_scale,
                                std::nullopt,
                            },
                            {
                                "unused",
                                unused_connection,
                                std::string("scaled"),
                            },
                        }
                    )
            );
        },
        "does not contribute to the stage output"
    );

    const miniir::Program repeated_external =
        make_repeated_external_consumer();

    require_throws<miniir::opt::FusionError>(
        [&] {
            static_cast<void>(
                miniir::opt::
                    fuse_linear_elementwise_pipeline(
                        {
                            {
                                "scale",
                                optimized_scale,
                                std::nullopt,
                            },
                            {
                                "repeat",
                                repeated_external,
                                std::string("scaled"),
                            },
                        }
                    )
            );
        },
        "repeated external names are ambiguous"
    );
}

[[nodiscard]] std::vector<float> evaluate_program(
    const miniir::Program& program,
    const HostInputs& host_inputs
) {
    miniir::verify(program);

    const std::size_t work_items =
        program.output_type().element_count();
    std::vector<float> output;
    output.reserve(work_items);

    for (
        std::size_t index = 0;
        index < work_items;
        ++index
    ) {
        miniir::InputBindings element_inputs;

        for (const miniir::Operation& operation :
             program.operations()) {
            if (operation.opcode != miniir::Opcode::Input) {
                continue;
            }

            const auto found =
                host_inputs.find(operation.input_name);

            if (found == host_inputs.end()) {
                throw std::invalid_argument(
                    "missing host input @" +
                    operation.input_name
                );
            }

            const miniir::Type& input_type =
                *operation.result_type;
            const std::size_t expected_elements =
                input_type.element_count();

            if (
                found->second.size() !=
                expected_elements
            ) {
                throw std::invalid_argument(
                    "host input @" +
                    operation.input_name +
                    " has an incorrect size"
                );
            }

            const std::size_t input_index =
                input_type.is_scalar() ? 0 : index;

            element_inputs.emplace(
                operation.input_name,
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

[[nodiscard]] minigpu::Buffer upload(
    const minigpu::Device& device,
    const std::vector<float>& values
) {
    minigpu::Buffer buffer =
        device.allocate(values.size());

    for (
        std::size_t index = 0;
        index < values.size();
        ++index
    ) {
        buffer.store(index, values[index]);
    }

    return buffer;
}

void fill(
    const minigpu::Buffer& buffer,
    float value
) {
    for (
        std::size_t index = 0;
        index < buffer.size();
        ++index
    ) {
        buffer.store(index, value);
    }
}

[[nodiscard]] std::vector<float> download(
    const minigpu::Buffer& buffer
) {
    std::vector<float> values;
    values.reserve(buffer.size());

    for (
        std::size_t index = 0;
        index < buffer.size();
        ++index
    ) {
        values.push_back(buffer.load(index));
    }

    return values;
}

struct GpuExecutionResult {
    std::vector<float> values;
    std::size_t dispatches;
    std::size_t intermediate_buffers;
};

[[nodiscard]] GpuExecutionResult execute_unfused_gpu(
    const miniir::metal::MetalExecutable& scale,
    const miniir::metal::MetalExecutable& shift,
    const miniir::metal::MetalExecutable& activation,
    const minigpu::Device& device,
    minigpu::Queue& queue,
    const std::vector<float>& input_values
) {
    std::size_t dispatches = 0;
    std::size_t intermediate_buffers = 0;

    minigpu::Buffer input =
        upload(device, input_values);
    minigpu::Buffer scaled =
        device.allocate(input_values.size());
    ++intermediate_buffers;
    minigpu::Buffer shifted =
        device.allocate(input_values.size());
    ++intermediate_buffers;
    minigpu::Buffer output =
        device.allocate(input_values.size());

    fill(scaled, -999.0f);
    fill(shifted, -999.0f);
    fill(output, -999.0f);

    minigpu::Event scale_finished = queue.submit(
        scale.make_dispatch(
            {{"x", input}},
            scaled
        ),
        scale.make_launch()
    );
    ++dispatches;

    minigpu::Event shift_finished = queue.submit(
        shift.make_dispatch(
            {{"scaled", scaled}},
            shifted
        ),
        shift.make_launch()
    );
    ++dispatches;

    minigpu::Event activation_finished = queue.submit(
        activation.make_dispatch(
            {{"shifted", shifted}},
            output
        ),
        activation.make_launch()
    );
    ++dispatches;

    // All three commands have already been submitted to the same FIFO
    // Metal queue. Waiting in submission order propagates the earliest
    // producer error instead of letting a later command's error hide it.
    scale_finished.wait();
    shift_finished.wait();
    activation_finished.wait();

    return GpuExecutionResult{
        download(output),
        dispatches,
        intermediate_buffers,
    };
}

[[nodiscard]] GpuExecutionResult execute_fused_gpu(
    const miniir::metal::MetalExecutable& executable,
    const minigpu::Device& device,
    minigpu::Queue& queue,
    const std::vector<float>& input_values
) {
    std::size_t dispatches = 0;
    constexpr std::size_t intermediate_buffers = 0;

    minigpu::Buffer input =
        upload(device, input_values);
    minigpu::Buffer output =
        device.allocate(input_values.size());

    fill(output, -999.0f);

    minigpu::Event finished = queue.submit(
        executable.make_dispatch(
            {{"x", input}},
            output
        ),
        executable.make_launch()
    );
    ++dispatches;

    finished.wait();

    return GpuExecutionResult{
        download(output),
        dispatches,
        intermediate_buffers,
    };
}

void check_fused_metal_plan(
    const miniir::metal::MetalKernelPlan& plan
) {
    require(
        plan.inputs().size() == 1 &&
            plan.inputs()[0].name == "x" &&
            plan.inputs()[0].buffer_index == 0 &&
            plan.inputs()[0].required_elements == 5,
        "fused Metal input ABI is incorrect"
    );
    require(
        plan.output_buffer_index() == 1 &&
            plan.element_count_index() == 2 &&
            plan.launch().logical_work_items == 5,
        "fused Metal output or launch ABI is incorrect"
    );
    require(
        plan.source().find(
            "const float v2 = v0 * v1;"
        ) != std::string::npos,
        "fused MSL does not contain multiply"
    );
    require(
        plan.source().find(
            "const float v4 = v2 + v3;"
        ) != std::string::npos,
        "fused MSL does not contain add"
    );
    require(
        plan.source().find(
            "const float v5 = max(v4, 0.0f);"
        ) != std::string::npos,
        "fused MSL does not contain ReLU"
    );
    require(
        plan.source().find("input_1") ==
            std::string::npos,
        "fused Metal plan retained an intermediate input"
    );
}

} // namespace

int main() {
    @autoreleasepool {
        try {
            const miniir::Program scale_program =
                make_scale_program();
            const std::string original_scale_ir =
                miniir::format_program(scale_program);

            const miniir::opt::OptimizationResult
                optimized_scale =
                    miniir::opt::optimize_elementwise(
                        scale_program
                    );

            check_main_optimization(
                scale_program,
                optimized_scale
            );
            require(
                miniir::format_program(scale_program) ==
                    original_scale_ir,
                "optimization mutated the original Program"
            );

            check_additional_optimization_cases();

            const miniir::Program shift_program =
                make_shift_program();
            const miniir::Program activation_program =
                make_activation_program();

            const miniir::opt::FusionResult fusion =
                make_and_check_fusion(
                    optimized_scale.program,
                    shift_program,
                    activation_program
                );

            check_fusion_errors(
                optimized_scale.program,
                shift_program
            );

            const std::vector<float> input_values{
                -2.0f,
                -1.0f,
                0.0f,
                1.0f,
                2.0f,
            };
            const std::vector<float> expected{
                0.0f,
                1.0f,
                3.0f,
                5.0f,
                7.0f,
            };

            const std::vector<float> cpu_original_scaled =
                evaluate_program(
                    scale_program,
                    {{"x", input_values}}
                );
            const std::vector<float> cpu_scaled =
                evaluate_program(
                    optimized_scale.program,
                    {{"x", input_values}}
                );
            const std::vector<float> cpu_shifted =
                evaluate_program(
                    shift_program,
                    {{"scaled", cpu_scaled}}
                );
            const std::vector<float> cpu_unfused =
                evaluate_program(
                    activation_program,
                    {{"shifted", cpu_shifted}}
                );
            const std::vector<float> cpu_fused =
                evaluate_program(
                    fusion.program,
                    {{"x", input_values}}
                );

            require(
                equal_vectors(
                    cpu_original_scaled,
                    cpu_scaled
                ) &&
                    equal_vectors(cpu_unfused, expected) &&
                    equal_vectors(cpu_fused, expected),
                "optimization or fusion changed CPU semantics"
            );

            const miniir::metal::MetalKernelPlan scale_plan =
                miniir::metal::lower_to_metal(
                    optimized_scale.program
                );
            const miniir::metal::MetalKernelPlan shift_plan =
                miniir::metal::lower_to_metal(
                    shift_program
                );
            const miniir::metal::MetalKernelPlan
                activation_plan =
                    miniir::metal::lower_to_metal(
                        activation_program
                    );
            const miniir::metal::MetalKernelPlan fused_plan =
                miniir::metal::lower_to_metal(
                    fusion.program
                );

            check_fused_metal_plan(fused_plan);

            minigpu::Device device;
            auto queue = device.create_queue();

            const miniir::metal::MetalExecutable scale =
                miniir::metal::compile_metal(
                    device,
                    scale_plan
                );
            const miniir::metal::MetalExecutable shift =
                miniir::metal::compile_metal(
                    device,
                    shift_plan
                );
            const miniir::metal::MetalExecutable activation =
                miniir::metal::compile_metal(
                    device,
                    activation_plan
                );
            const miniir::metal::MetalExecutable fused =
                miniir::metal::compile_metal(
                    device,
                    fused_plan
                );

            const GpuExecutionResult gpu_unfused =
                execute_unfused_gpu(
                    scale,
                    shift,
                    activation,
                    device,
                    *queue,
                    input_values
                );
            const GpuExecutionResult gpu_fused =
                execute_fused_gpu(
                    fused,
                    device,
                    *queue,
                    input_values
                );

            require(
                gpu_unfused.dispatches == 3 &&
                    gpu_unfused.intermediate_buffers == 2 &&
                    gpu_fused.dispatches == 1 &&
                    gpu_fused.intermediate_buffers == 0 &&
                    fusion.dispatches_before == 3 &&
                    fusion.dispatches_after == 1 &&
                    fusion.intermediate_buffers_avoided == 2,
                "runtime and fusion counts disagree"
            );
            require(
                equal_vectors(gpu_unfused.values, expected) &&
                    equal_vectors(gpu_fused.values, expected) &&
                    equal_vectors(
                        gpu_unfused.values,
                        cpu_unfused
                    ) &&
                    equal_vectors(
                        gpu_fused.values,
                        cpu_fused
                    ),
                "CPU, unfused GPU and fused GPU disagree"
            );

            // Reuse the same compiled fused executable with completely
            // different input/output buffers.
            const std::vector<float> reuse_input{
                0.0f,
                1.0f,
                2.0f,
                3.0f,
                4.0f,
            };
            const std::vector<float> reuse_expected{
                3.0f,
                5.0f,
                7.0f,
                9.0f,
                11.0f,
            };
            const GpuExecutionResult reused_gpu =
                execute_fused_gpu(
                    fused,
                    device,
                    *queue,
                    reuse_input
                );

            require(
                reused_gpu.dispatches == 1 &&
                    reused_gpu.intermediate_buffers == 0 &&
                    equal_vectors(
                        reused_gpu.values,
                        reuse_expected
                    ),
                "compiled fused executable was not reusable"
            );

            std::cout
                << "Lesson 23: optimization and linear "
                   "elementwise fusion\n\n"
                << "Original scale IR:\n"
                << original_scale_ir << '\n'
                << "After constant folding:\n"
                << miniir::format_program(
                       optimized_scale
                           .after_constant_folding
                   )
                << '\n'
                << "After dead-code elimination:\n"
                << miniir::format_program(
                       optimized_scale.program
                   )
                << '\n'
                << "Fused three-stage IR:\n"
                << miniir::format_program(fusion.program)
                << '\n'
                << miniir::opt::
                       format_optimization_report(
                           optimized_scale
                       )
                << '\n'
                << miniir::opt::format_fusion_report(
                       fusion
                   )
                << '\n'
                << "Metal device: " << device.name()
                << '\n'
                << "Unfused GPU: "
                << gpu_unfused.dispatches
                << " dispatches, "
                << gpu_unfused.intermediate_buffers
                << " intermediate buffers, output ";
            print_vector(gpu_unfused.values);
            std::cout
                << "\nFused GPU:   "
                << gpu_fused.dispatches
                << " dispatch, "
                << gpu_fused.intermediate_buffers
                << " intermediate buffers, output ";
            print_vector(gpu_fused.values);
            std::cout
                << "\nReused fused executable output: ";
            print_vector(reused_gpu.values);
            std::cout
                << "\n\nOptimization, fusion, Metal "
                   "equivalence and reuse checks: PASS\n"
                << "PASS: optimized unfused and fused "
                   "programs agree on the real Apple GPU.\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr
                << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
}
