#pragma once

#include "../lesson14-metal-minigpu/minigpu.hpp"
#include "../lesson21-types-shapes-broadcasting/miniir.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace miniir::metal {

class CodegenError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct InputBinding {
    std::string name;
    ValueId value;
    Type type;
    std::size_t buffer_index;
    std::size_t required_elements;
};

struct LaunchMetadata {
    std::uint32_t logical_work_items;
    std::size_t threads_per_group;
};

class MetalKernelPlan {
public:
    [[nodiscard]] const std::string&
    function_name() const noexcept;

    [[nodiscard]] const std::string&
    source() const noexcept;

    [[nodiscard]] const std::vector<InputBinding>&
    inputs() const noexcept;

    [[nodiscard]] std::size_t
    output_buffer_index() const noexcept;

    [[nodiscard]] std::size_t
    element_count_index() const noexcept;

    [[nodiscard]] ValueId output_value() const noexcept;

    [[nodiscard]] const Type&
    output_type() const noexcept;

    [[nodiscard]] const LaunchMetadata&
    launch() const noexcept;

private:
    MetalKernelPlan(
        std::string function_name,
        std::string source,
        std::vector<InputBinding> inputs,
        std::size_t output_buffer_index,
        std::size_t element_count_index,
        ValueId output_value,
        Type output_type,
        LaunchMetadata launch
    );

    std::string function_name_;
    std::string source_;
    std::vector<InputBinding> inputs_;
    std::size_t output_buffer_index_;
    std::size_t element_count_index_;
    ValueId output_value_;
    Type output_type_;
    LaunchMetadata launch_;

    friend MetalKernelPlan lower_to_metal(
        const Program& program
    );
};

using RuntimeInputs =
    std::unordered_map<std::string, minigpu::Buffer>;

class MetalExecutable {
public:
    [[nodiscard]] const MetalKernelPlan&
    plan() const noexcept;

    [[nodiscard]] minigpu::Dispatch make_dispatch(
        const RuntimeInputs& inputs,
        minigpu::Buffer output
    ) const;

    [[nodiscard]] minigpu::LaunchConfig
    make_launch() const;

private:
    MetalExecutable(
        MetalKernelPlan plan,
        minigpu::Kernel kernel
    );

    MetalKernelPlan plan_;
    minigpu::Kernel kernel_;

    friend MetalExecutable compile_metal(
        const minigpu::Device& device,
        const MetalKernelPlan& plan
    );
};

[[nodiscard]] MetalKernelPlan lower_to_metal(
    const Program& program
);

[[nodiscard]] std::string format_plan(
    const MetalKernelPlan& plan
);

[[nodiscard]] MetalExecutable compile_metal(
    const minigpu::Device& device,
    const MetalKernelPlan& plan
);

} // namespace miniir::metal
