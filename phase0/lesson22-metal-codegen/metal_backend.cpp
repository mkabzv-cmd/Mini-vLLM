#include "metal_backend.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <string_view>
#include <utility>

namespace miniir::metal {

namespace {

constexpr std::string_view function_name =
    "miniir_elementwise";
constexpr std::size_t threads_per_group = 64;

[[nodiscard]] std::string format_msl_float(float value) {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output
        << std::setprecision(
               std::numeric_limits<float>::max_digits10
           )
        << value;

    std::string text = output.str();

    if (
        text.find('.') == std::string::npos &&
        text.find('e') == std::string::npos &&
        text.find('E') == std::string::npos
    ) {
        text += ".0";
    }

    text += 'f';
    return text;
}

[[nodiscard]] std::string value_name(ValueId value) {
    return "v" + std::to_string(value);
}

[[nodiscard]] std::string input_name(
    std::size_t buffer_index
) {
    return "input_" + std::to_string(buffer_index);
}

[[nodiscard]] std::string input_index(
    const Type& type
) {
    return type.is_scalar() ? "0" : "global_id";
}

[[nodiscard]] const InputBinding& find_input_binding(
    const std::vector<InputBinding>& inputs,
    ValueId value
) {
    const auto found = std::find_if(
        inputs.begin(),
        inputs.end(),
        [value](const InputBinding& binding) {
            return binding.value == value;
        }
    );

    if (found == inputs.end()) {
        throw std::logic_error(
            "input value has no generated binding"
        );
    }

    return *found;
}

void validate_plan(const MetalKernelPlan& plan) {
    if (plan.function_name().empty()) {
        throw std::invalid_argument(
            "Metal plan function name must not be empty"
        );
    }

    if (plan.source().empty()) {
        throw std::invalid_argument(
            "Metal plan source must not be empty"
        );
    }

    for (
        std::size_t index = 0;
        index < plan.inputs().size();
        ++index
    ) {
        const InputBinding& input = plan.inputs()[index];

        if (input.buffer_index != index) {
            throw std::invalid_argument(
                "Metal plan input buffer indexes "
                "must be sequential"
            );
        }

        if (
            input.required_elements !=
            input.type.element_count()
        ) {
            throw std::invalid_argument(
                "Metal plan input element count "
                "does not match its type"
            );
        }
    }

    if (
        plan.output_buffer_index() !=
        plan.inputs().size()
    ) {
        throw std::invalid_argument(
            "Metal plan output buffer index is invalid"
        );
    }

    if (
        plan.element_count_index() !=
        plan.output_buffer_index() + 1
    ) {
        throw std::invalid_argument(
            "Metal plan element-count index is invalid"
        );
    }

    if (
        plan.output_type().dtype() != DType::F32 ||
        plan.launch().logical_work_items !=
            plan.output_type().element_count()
    ) {
        throw std::invalid_argument(
            "Metal plan output metadata is inconsistent"
        );
    }

    if (plan.launch().threads_per_group == 0) {
        throw std::invalid_argument(
            "Metal plan threads_per_group must be positive"
        );
    }
}

void require_f32_value(
    ValueId value,
    const Type& type
) {
    if (type.dtype() != DType::F32) {
        throw CodegenError(
            "Lesson 22 Metal backend supports only f32; "
            "value %" + std::to_string(value) +
            " has type " + format_type(type)
        );
    }
}

void require_same_iteration_domain(
    ValueId value,
    const Type& type,
    const Type& output_type
) {
    if (
        !type.is_scalar() &&
        type.shape() != output_type.shape()
    ) {
        throw CodegenError(
            "Lesson 22 requires every non-scalar value "
            "to use the output shape; value %" +
            std::to_string(value) + " has type " +
            format_type(type) + ", while the output is " +
            format_type(output_type)
        );
    }
}

[[nodiscard]] std::string generate_source(
    const Program& program,
    const std::vector<InputBinding>& inputs,
    std::size_t output_buffer_index,
    std::size_t element_count_index
) {
    std::ostringstream source;
    source.imbue(std::locale::classic());

    source
        << "#include <metal_stdlib>\n"
        << "using namespace metal;\n\n"
        << "kernel void " << function_name << "(\n";

    for (const InputBinding& input : inputs) {
        source
            << "    device const float* "
            << input_name(input.buffer_index)
            << " [[buffer(" << input.buffer_index
            << ")]],\n";
    }

    source
        << "    device float* output [[buffer("
        << output_buffer_index << ")]],\n"
        << "    constant uint& element_count [[buffer("
        << element_count_index << ")]],\n"
        << "    uint global_id "
           "[[thread_position_in_grid]]\n"
        << ") {\n"
        << "    if (global_id >= element_count) {\n"
        << "        return;\n"
        << "    }\n\n";

    for (const Operation& operation : program.operations()) {
        switch (operation.opcode) {
            case Opcode::Input: {
                const InputBinding& binding =
                    find_input_binding(
                        inputs,
                        *operation.result
                    );

                source
                    << "    const float "
                    << value_name(*operation.result)
                    << " = "
                    << input_name(binding.buffer_index)
                    << '['
                    << input_index(*operation.result_type)
                    << "];\n";
                break;
            }

            case Opcode::ConstantF32:
                source
                    << "    const float "
                    << value_name(*operation.result)
                    << " = "
                    << format_msl_float(
                           operation.constant_value
                       )
                    << ";\n";
                break;

            case Opcode::Multiply:
                source
                    << "    const float "
                    << value_name(*operation.result)
                    << " = "
                    << value_name(operation.operands[0])
                    << " * "
                    << value_name(operation.operands[1])
                    << ";\n";
                break;

            case Opcode::Add:
                source
                    << "    const float "
                    << value_name(*operation.result)
                    << " = "
                    << value_name(operation.operands[0])
                    << " + "
                    << value_name(operation.operands[1])
                    << ";\n";
                break;

            case Opcode::Relu:
                source
                    << "    const float "
                    << value_name(*operation.result)
                    << " = max("
                    << value_name(operation.operands[0])
                    << ", 0.0f);\n";
                break;

            case Opcode::Return:
                source
                    << "\n    output[global_id] = "
                    << value_name(operation.operands[0])
                    << ";\n";
                break;
        }
    }

    source << "}\n";
    return source.str();
}

} // namespace

MetalKernelPlan::MetalKernelPlan(
    std::string function_name,
    std::string source,
    std::vector<InputBinding> inputs,
    std::size_t output_buffer_index,
    std::size_t element_count_index,
    ValueId output_value,
    Type output_type,
    LaunchMetadata launch
)
    : function_name_(std::move(function_name)),
      source_(std::move(source)),
      inputs_(std::move(inputs)),
      output_buffer_index_(output_buffer_index),
      element_count_index_(element_count_index),
      output_value_(output_value),
      output_type_(std::move(output_type)),
      launch_(launch) {}

const std::string&
MetalKernelPlan::function_name() const noexcept {
    return function_name_;
}

const std::string&
MetalKernelPlan::source() const noexcept {
    return source_;
}

const std::vector<InputBinding>&
MetalKernelPlan::inputs() const noexcept {
    return inputs_;
}

std::size_t
MetalKernelPlan::output_buffer_index() const noexcept {
    return output_buffer_index_;
}

std::size_t
MetalKernelPlan::element_count_index() const noexcept {
    return element_count_index_;
}

ValueId MetalKernelPlan::output_value() const noexcept {
    return output_value_;
}

const Type&
MetalKernelPlan::output_type() const noexcept {
    return output_type_;
}

const LaunchMetadata&
MetalKernelPlan::launch() const noexcept {
    return launch_;
}

MetalKernelPlan lower_to_metal(const Program& program) {
    verify(program);

    const Type output_type = program.output_type();
    const ValueId output_value =
        program.operations().back().operands.front();
    const std::size_t element_count =
        output_type.element_count();

    if (
        element_count >
        std::numeric_limits<std::uint32_t>::max()
    ) {
        throw CodegenError(
            "Lesson 22 output element count does not fit "
            "the MiniGPU u32 binding"
        );
    }

    std::vector<InputBinding> inputs;

    for (const Operation& operation : program.operations()) {
        if (!operation.result.has_value()) {
            continue;
        }

        const ValueId value = *operation.result;
        const Type& type = *operation.result_type;

        require_f32_value(value, type);
        require_same_iteration_domain(
            value,
            type,
            output_type
        );

        if (operation.opcode == Opcode::Input) {
            const std::size_t buffer_index =
                inputs.size();

            inputs.push_back(InputBinding{
                operation.input_name,
                value,
                type,
                buffer_index,
                type.element_count(),
            });
        }
    }

    const std::size_t output_buffer_index =
        inputs.size();
    const std::size_t element_count_index =
        output_buffer_index + 1;

    MetalKernelPlan plan{
        std::string(function_name),
        generate_source(
            program,
            inputs,
            output_buffer_index,
            element_count_index
        ),
        std::move(inputs),
        output_buffer_index,
        element_count_index,
        output_value,
        output_type,
        LaunchMetadata{
            static_cast<std::uint32_t>(element_count),
            threads_per_group,
        },
    };

    validate_plan(plan);
    return plan;
}

std::string format_plan(const MetalKernelPlan& plan) {
    validate_plan(plan);

    std::ostringstream output;
    output.imbue(std::locale::classic());

    output
        << "function: " << plan.function_name() << '\n';

    for (const InputBinding& input : plan.inputs()) {
        output
            << "buffer(" << input.buffer_index
            << ") input @" << input.name
            << " -> %" << input.value
            << " : " << format_type(input.type)
            << " [" << input.required_elements
            << " element(s)]\n";
    }

    output
        << "buffer(" << plan.output_buffer_index()
        << ") output <- %" << plan.output_value()
        << " : " << format_type(plan.output_type())
        << '\n'
        << "buffer(" << plan.element_count_index()
        << ") element_count = "
        << plan.launch().logical_work_items << '\n'
        << "launch: "
        << plan.launch().logical_work_items
        << " work item(s), "
        << plan.launch().threads_per_group
        << " thread(s) per group\n";

    return output.str();
}

MetalExecutable::MetalExecutable(
    MetalKernelPlan plan,
    minigpu::Kernel kernel
)
    : plan_(std::move(plan)),
      kernel_(std::move(kernel)) {}

const MetalKernelPlan&
MetalExecutable::plan() const noexcept {
    return plan_;
}

MetalExecutable compile_metal(
    const minigpu::Device& device,
    const MetalKernelPlan& plan
) {
    validate_plan(plan);

    minigpu::Kernel kernel = device.compile_kernel(
        plan.function_name(),
        plan.source()
    );

    return MetalExecutable(
        plan,
        std::move(kernel)
    );
}

minigpu::Dispatch MetalExecutable::make_dispatch(
    const RuntimeInputs& inputs,
    minigpu::Buffer output
) const {
    validate_plan(plan_);

    for (const auto& [name, buffer] : inputs) {
        static_cast<void>(buffer);

        const auto known = std::find_if(
            plan_.inputs().begin(),
            plan_.inputs().end(),
            [&name](const InputBinding& input) {
                return input.name == name;
            }
        );

        if (known == plan_.inputs().end()) {
            throw std::invalid_argument(
                "unknown runtime input @" + name
            );
        }
    }

    minigpu::Dispatch dispatch(kernel_);

    for (const InputBinding& input : plan_.inputs()) {
        const auto found = inputs.find(input.name);

        if (found == inputs.end()) {
            throw std::invalid_argument(
                "missing runtime input @" + input.name
            );
        }

        if (
            found->second.size() !=
            input.required_elements
        ) {
            throw std::invalid_argument(
                "runtime input @" + input.name +
                " requires " +
                std::to_string(input.required_elements) +
                " element(s), received " +
                std::to_string(found->second.size())
            );
        }

        dispatch.bind_buffer(
            input.buffer_index,
            found->second
        );
    }

    if (
        output.size() !=
        plan_.launch().logical_work_items
    ) {
        throw std::invalid_argument(
            "runtime output requires " +
            std::to_string(
                plan_.launch().logical_work_items
            ) +
            " element(s), received " +
            std::to_string(output.size())
        );
    }

    dispatch
        .bind_buffer(
            plan_.output_buffer_index(),
            std::move(output)
        )
        .bind_u32(
            plan_.element_count_index(),
            plan_.launch().logical_work_items
        );

    return dispatch;
}

minigpu::LaunchConfig
MetalExecutable::make_launch() const {
    validate_plan(plan_);

    return minigpu::LaunchConfig::cover(
        plan_.launch().logical_work_items,
        plan_.launch().threads_per_group
    );
}

} // namespace miniir::metal
