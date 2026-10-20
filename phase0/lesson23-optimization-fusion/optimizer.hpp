#pragma once

#include "../lesson21-types-shapes-broadcasting/miniir.hpp"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace miniir::opt {

class FusionError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct ConstantFold {
    ValueId value;
    Opcode original_opcode;
    float folded_value;
};

struct ValueRemap {
    ValueId old_value;
    std::optional<ValueId> new_value;
};

struct ConstantFoldingResult {
    Program program;
    std::vector<ConstantFold> folds;
};

struct DeadCodeEliminationResult {
    Program program;
    std::vector<ValueRemap> remap;
    std::size_t removed_values;
};

struct OptimizationResult {
    Program after_constant_folding;
    Program program;
    std::vector<ConstantFold> folds;
    std::vector<ValueRemap> remap;
    std::size_t removed_values;
};

[[nodiscard]] ConstantFoldingResult fold_constants(
    const Program& program
);

[[nodiscard]] DeadCodeEliminationResult
eliminate_dead_code(const Program& program);

[[nodiscard]] OptimizationResult optimize_elementwise(
    const Program& program
);

[[nodiscard]] std::string format_optimization_report(
    const OptimizationResult& result
);

struct PipelineStage {
    std::string name;
    Program program;
    std::optional<std::string> previous_output_input;
};

struct FusionEdge {
    std::string producer_stage;
    std::string consumer_stage;
    std::string consumer_input;
};

struct FusionResult {
    Program program;
    std::vector<std::string> stage_names;
    std::vector<FusionEdge> edges;
    std::size_t dispatches_before;
    std::size_t dispatches_after;
    std::size_t intermediate_buffers_avoided;
};

[[nodiscard]] FusionResult
fuse_linear_elementwise_pipeline(
    const std::vector<PipelineStage>& stages
);

[[nodiscard]] std::string format_fusion_report(
    const FusionResult& result
);

} // namespace miniir::opt
