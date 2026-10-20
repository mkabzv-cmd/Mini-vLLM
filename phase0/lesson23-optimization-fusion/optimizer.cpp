#include "optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace miniir::opt {

namespace {

[[nodiscard]] std::string format_float(float value) {
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

    return text;
}

[[nodiscard]] std::size_t value_count(
    const Program& program
) {
    std::size_t count = 0;

    for (const Operation& operation : program.operations()) {
        if (operation.result.has_value()) {
            ++count;
        }
    }

    return count;
}

[[nodiscard]] std::vector<bool> find_live_values(
    const Program& program
) {
    const std::size_t count = value_count(program);
    std::vector<bool> live(count, false);

    const Operation& return_operation =
        program.operations().back();
    live[return_operation.operands.front()] = true;

    for (
        auto operation = program.operations().rbegin();
        operation != program.operations().rend();
        ++operation
    ) {
        if (!operation->result.has_value()) {
            continue;
        }

        const ValueId result = *operation->result;

        if (!live[result]) {
            continue;
        }

        for (ValueId operand : operation->operands) {
            live[operand] = true;
        }
    }

    return live;
}

[[nodiscard]] bool is_scalar_f32(
    const Operation& operation
) {
    return
        operation.result_type.has_value() &&
        operation.result_type->is_scalar() &&
        operation.result_type->dtype() == DType::F32;
}

[[nodiscard]] std::optional<float> try_fold(
    const Operation& operation,
    const std::vector<std::optional<float>>& constants
) {
    if (!is_scalar_f32(operation)) {
        return std::nullopt;
    }

    float result = 0.0f;

    switch (operation.opcode) {
        case Opcode::Add:
        case Opcode::Multiply: {
            const std::optional<float>& left =
                constants[operation.operands[0]];
            const std::optional<float>& right =
                constants[operation.operands[1]];

            if (!left.has_value() || !right.has_value()) {
                return std::nullopt;
            }

            result =
                operation.opcode == Opcode::Add
                    ? *left + *right
                    : *left * *right;
            break;
        }

        case Opcode::Relu: {
            const std::optional<float>& input =
                constants[operation.operands[0]];

            if (!input.has_value()) {
                return std::nullopt;
            }

            result = std::max(*input, 0.0f);
            break;
        }

        case Opcode::Input:
        case Opcode::ConstantF32:
        case Opcode::Return:
            return std::nullopt;
    }

    if (!std::isfinite(result)) {
        return std::nullopt;
    }

    return result;
}

[[nodiscard]] std::vector<ValueId> remap_operands(
    const Operation& operation,
    const std::vector<std::optional<ValueId>>& remap
) {
    std::vector<ValueId> operands;
    operands.reserve(operation.operands.size());

    for (ValueId old_operand : operation.operands) {
        if (!remap[old_operand].has_value()) {
            throw std::logic_error(
                "live operand %" +
                std::to_string(old_operand) +
                " has no rewritten value"
            );
        }

        operands.push_back(*remap[old_operand]);
    }

    return operands;
}

[[nodiscard]] const Operation* find_input(
    const Program& program,
    std::string_view name
) {
    for (const Operation& operation : program.operations()) {
        if (
            operation.opcode == Opcode::Input &&
            operation.input_name == name
        ) {
            return &operation;
        }
    }

    return nullptr;
}

void require_elementwise_region(const Program& program) {
    verify(program);

    const Type output_type = program.output_type();

    for (const Operation& operation : program.operations()) {
        if (!operation.result_type.has_value()) {
            continue;
        }

        const Type& type = *operation.result_type;

        if (
            !type.is_scalar() &&
            type.shape() != output_type.shape()
        ) {
            throw FusionError(
                "fused elementwise pipeline requires every "
                "non-scalar value to use the final output "
                "shape; value %" +
                std::to_string(*operation.result) +
                " has type " + format_type(type) +
                ", while the final output is " +
                format_type(output_type)
            );
        }
    }
}

} // namespace

ConstantFoldingResult fold_constants(
    const Program& program
) {
    verify(program);

    std::vector<Operation> rewritten;
    rewritten.reserve(program.operations().size());

    std::vector<std::optional<float>> constants;
    constants.reserve(value_count(program));

    std::vector<ConstantFold> folds;

    for (const Operation& operation : program.operations()) {
        if (!operation.result.has_value()) {
            rewritten.push_back(operation);
            continue;
        }

        const ValueId result = *operation.result;

        if (result != constants.size()) {
            throw std::logic_error(
                "verified program does not have dense SSA IDs"
            );
        }

        Operation replacement = operation;
        std::optional<float> known_constant;

        if (operation.opcode == Opcode::ConstantF32) {
            known_constant = operation.constant_value;
        } else {
            const std::optional<float> folded =
                try_fold(operation, constants);

            if (folded.has_value()) {
                replacement.opcode = Opcode::ConstantF32;
                replacement.operands.clear();
                replacement.result_type =
                    Type::scalar(DType::F32);
                replacement.input_name.clear();
                replacement.constant_value = *folded;

                folds.push_back(ConstantFold{
                    result,
                    operation.opcode,
                    *folded,
                });
                known_constant = folded;
            }
        }

        rewritten.push_back(std::move(replacement));
        constants.push_back(known_constant);
    }

    Program folded(std::move(rewritten));
    verify(folded);

    return ConstantFoldingResult{
        std::move(folded),
        std::move(folds),
    };
}

DeadCodeEliminationResult eliminate_dead_code(
    const Program& program
) {
    verify(program);

    const std::size_t count = value_count(program);
    const std::vector<bool> live =
        find_live_values(program);

    std::vector<std::optional<ValueId>> old_to_new(
        count,
        std::nullopt
    );
    std::vector<Operation> rewritten;
    rewritten.reserve(program.operations().size());

    ValueId next_value = 0;
    std::size_t removed_values = 0;

    for (const Operation& operation : program.operations()) {
        if (operation.opcode == Opcode::Return) {
            std::vector<ValueId> operands =
                remap_operands(operation, old_to_new);

            Operation rewritten_return = operation;
            rewritten_return.operands = std::move(operands);
            rewritten.push_back(std::move(rewritten_return));
            continue;
        }

        const ValueId old_result = *operation.result;

        if (!live[old_result]) {
            ++removed_values;
            continue;
        }

        Operation kept = operation;
        kept.operands =
            remap_operands(operation, old_to_new);
        kept.result = next_value;

        old_to_new[old_result] = next_value;
        ++next_value;
        rewritten.push_back(std::move(kept));
    }

    std::vector<ValueRemap> remap;
    remap.reserve(count);

    for (ValueId old_value = 0; old_value < count; ++old_value) {
        remap.push_back(ValueRemap{
            old_value,
            old_to_new[old_value],
        });
    }

    Program eliminated(std::move(rewritten));
    verify(eliminated);

    return DeadCodeEliminationResult{
        std::move(eliminated),
        std::move(remap),
        removed_values,
    };
}

OptimizationResult optimize_elementwise(
    const Program& program
) {
    verify(program);

    ConstantFoldingResult folded =
        fold_constants(program);
    DeadCodeEliminationResult eliminated =
        eliminate_dead_code(folded.program);

    verify(folded.program);
    verify(eliminated.program);

    return OptimizationResult{
        std::move(folded.program),
        std::move(eliminated.program),
        std::move(folded.folds),
        std::move(eliminated.remap),
        eliminated.removed_values,
    };
}

std::string format_optimization_report(
    const OptimizationResult& result
) {
    verify(result.after_constant_folding);
    verify(result.program);

    std::ostringstream output;
    output.imbue(std::locale::classic());

    output
        << "constant folding:\n"
        << "  folded values: " << result.folds.size()
        << '\n';

    for (const ConstantFold& fold : result.folds) {
        output
            << "  %" << fold.value << ' '
            << opcode_name(fold.original_opcode)
            << " -> constant_f32 "
            << format_float(fold.folded_value)
            << '\n';
    }

    output
        << "dead-code elimination:\n"
        << "  removed values: " << result.removed_values
        << '\n'
        << "  remap:\n";

    for (const ValueRemap& mapping : result.remap) {
        output << "    %" << mapping.old_value << " -> ";

        if (mapping.new_value.has_value()) {
            output << '%' << *mapping.new_value;
        } else {
            output << "removed";
        }

        output << '\n';
    }

    return output.str();
}

FusionResult fuse_linear_elementwise_pipeline(
    const std::vector<PipelineStage>& stages
) {
    if (stages.empty()) {
        throw FusionError(
            "cannot fuse an empty pipeline"
        );
    }

    std::unordered_set<std::string> seen_stage_names;

    for (
        std::size_t stage_index = 0;
        stage_index < stages.size();
        ++stage_index
    ) {
        const PipelineStage& stage = stages[stage_index];

        if (stage.name.empty()) {
            throw FusionError(
                "pipeline stage name must not be empty"
            );
        }

        if (!seen_stage_names.insert(stage.name).second) {
            throw FusionError(
                "duplicate pipeline stage name '" +
                stage.name + "'"
            );
        }

        verify(stage.program);

        if (
            stage_index == 0 &&
            stage.previous_output_input.has_value()
        ) {
            throw FusionError(
                "first pipeline stage '" + stage.name +
                "' must not declare a previous-output input"
            );
        }

        if (
            stage_index != 0 &&
            !stage.previous_output_input.has_value()
        ) {
            throw FusionError(
                "pipeline stage '" + stage.name +
                "' must declare which input receives the "
                "previous stage output"
            );
        }
    }

    std::vector<Operation> fused_operations;
    std::unordered_map<std::string, std::string>
        external_input_owners;
    std::vector<std::string> stage_names;
    stage_names.reserve(stages.size());
    std::vector<FusionEdge> edges;
    edges.reserve(stages.size() - 1);

    ValueId next_value = 0;
    std::optional<ValueId> previous_output_value;
    std::optional<Type> previous_output_type;

    for (
        std::size_t stage_index = 0;
        stage_index < stages.size();
        ++stage_index
    ) {
        const PipelineStage& stage = stages[stage_index];
        stage_names.push_back(stage.name);

        const std::size_t local_count =
            value_count(stage.program);
        std::vector<std::optional<ValueId>> local_to_fused(
            local_count,
            std::nullopt
        );

        std::optional<ValueId> connected_input;

        if (stage_index != 0) {
            const std::string& consumer_input =
                *stage.previous_output_input;
            const Operation* input_operation =
                find_input(stage.program, consumer_input);

            if (input_operation == nullptr) {
                throw FusionError(
                    "pipeline stage '" + stage.name +
                    "' has no declared input @" +
                    consumer_input +
                    " for the previous stage output"
                );
            }

            const ValueId input_value =
                *input_operation->result;
            const std::vector<bool> live =
                find_live_values(stage.program);

            if (!live[input_value]) {
                throw FusionError(
                    "pipeline stage '" + stage.name +
                    "' declares previous-output input @" +
                    consumer_input +
                    ", but that input does not contribute "
                    "to the stage output"
                );
            }

            if (
                *input_operation->result_type !=
                *previous_output_type
            ) {
                throw FusionError(
                    "cannot connect stage '" +
                    stages[stage_index - 1].name +
                    "' output " +
                    format_type(*previous_output_type) +
                    " to stage '" + stage.name +
                    "' input @" + consumer_input + " of type " +
                    format_type(*input_operation->result_type)
                );
            }

            connected_input = input_value;
            local_to_fused[input_value] =
                *previous_output_value;

            edges.push_back(FusionEdge{
                stages[stage_index - 1].name,
                stage.name,
                consumer_input,
            });
        }

        for (
            const Operation& operation :
            stage.program.operations()
        ) {
            if (operation.opcode == Opcode::Return) {
                continue;
            }

            const ValueId local_result = *operation.result;

            if (
                connected_input.has_value() &&
                local_result == *connected_input
            ) {
                continue;
            }

            if (operation.opcode == Opcode::Input) {
                const auto existing =
                    external_input_owners.find(
                        operation.input_name
                    );

                if (
                    existing !=
                    external_input_owners.end()
                ) {
                    throw FusionError(
                        "external input @" +
                        operation.input_name +
                        " appears in both stage '" +
                        existing->second +
                        "' and stage '" + stage.name +
                        "'; repeated external names are "
                        "ambiguous without explicit shared-"
                        "input identity"
                    );
                }
            }

            Operation fused = operation;
            fused.operands =
                remap_operands(
                    operation,
                    local_to_fused
                );
            fused.result = next_value;

            local_to_fused[local_result] = next_value;

            if (operation.opcode == Opcode::Input) {
                external_input_owners.emplace(
                    operation.input_name,
                    stage.name
                );
            }

            ++next_value;
            fused_operations.push_back(std::move(fused));
        }

        const ValueId local_output =
            stage.program.operations()
                .back()
                .operands.front();

        if (!local_to_fused[local_output].has_value()) {
            throw std::logic_error(
                "stage output has no fused value"
            );
        }

        previous_output_value =
            *local_to_fused[local_output];
        previous_output_type =
            stage.program.output_type();
    }

    fused_operations.push_back(Operation{
        Opcode::Return,
        std::nullopt,
        {*previous_output_value},
        std::nullopt,
        {},
        0.0f,
    });

    Program fused(std::move(fused_operations));
    verify(fused);
    require_elementwise_region(fused);

    return FusionResult{
        std::move(fused),
        std::move(stage_names),
        std::move(edges),
        stages.size(),
        1,
        stages.size() - 1,
    };
}

std::string format_fusion_report(
    const FusionResult& result
) {
    verify(result.program);

    std::ostringstream output;
    output.imbue(std::locale::classic());

    output << "elementwise fusion:\n  stages: ";

    for (
        std::size_t index = 0;
        index < result.stage_names.size();
        ++index
    ) {
        if (index != 0) {
            output << " -> ";
        }

        output << result.stage_names[index];
    }

    output << '\n';

    for (const FusionEdge& edge : result.edges) {
        output
            << "  edge: " << edge.producer_stage
            << " -> " << edge.consumer_stage
            << " via @" << edge.consumer_input << '\n';
    }

    output
        << "  dispatches: "
        << result.dispatches_before
        << " -> " << result.dispatches_after << '\n'
        << "  intermediate buffers avoided: "
        << result.intermediate_buffers_avoided
        << '\n';

    return output.str();
}

} // namespace miniir::opt
