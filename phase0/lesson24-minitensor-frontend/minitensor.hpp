#pragma once

#include "../lesson22-metal-codegen/metal_backend.hpp"
#include "../lesson23-optimization-fusion/optimizer.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace minitensor {

class FrontendError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

namespace detail {

struct GraphState;

} // namespace detail

class CapturedExpression;
class Graph;

// A Tensor is a symbolic value in a captured calculation. It does not own
// runtime numbers or a GPU buffer in this lesson.
class Tensor {
public:
    [[nodiscard]] miniir::ValueId id() const noexcept;
    [[nodiscard]] const miniir::Type& type() const noexcept;

private:
    Tensor(
        std::shared_ptr<detail::GraphState> state,
        miniir::Value value
    );

    std::shared_ptr<detail::GraphState> state_;
    miniir::Value value_;

    friend class Graph;
    friend CapturedExpression capture(const Tensor& output);

    friend Tensor operator+(
        const Tensor& left,
        const Tensor& right
    );
    friend Tensor operator+(
        const Tensor& left,
        float right
    );
    friend Tensor operator+(
        float left,
        const Tensor& right
    );

    friend Tensor operator*(
        const Tensor& left,
        const Tensor& right
    );
    friend Tensor operator*(
        const Tensor& left,
        float right
    );
    friend Tensor operator*(
        float left,
        const Tensor& right
    );

    friend Tensor relu(const Tensor& input);
};

class Graph {
public:
    Graph();

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    Graph(Graph&&) = delete;
    Graph& operator=(Graph&&) = delete;

    [[nodiscard]] Tensor input(
        std::string name,
        miniir::Type type
    );

    [[nodiscard]] Tensor input(
        std::string name,
        std::vector<std::size_t> shape,
        miniir::DType dtype = miniir::DType::F32
    );

    [[nodiscard]] Tensor scalar_input(
        std::string name,
        miniir::DType dtype = miniir::DType::F32
    );

    [[nodiscard]] Tensor scalar(float value);

private:
    std::shared_ptr<detail::GraphState> state_;
};

[[nodiscard]] Tensor operator+(
    const Tensor& left,
    const Tensor& right
);
[[nodiscard]] Tensor operator+(
    const Tensor& left,
    float right
);
[[nodiscard]] Tensor operator+(
    float left,
    const Tensor& right
);

[[nodiscard]] Tensor operator*(
    const Tensor& left,
    const Tensor& right
);
[[nodiscard]] Tensor operator*(
    const Tensor& left,
    float right
);
[[nodiscard]] Tensor operator*(
    float left,
    const Tensor& right
);

[[nodiscard]] Tensor relu(const Tensor& input);

class CapturedExpression {
public:
    [[nodiscard]] const miniir::Program&
    program() const noexcept;

private:
    explicit CapturedExpression(miniir::Program program);

    miniir::Program program_;

    friend CapturedExpression capture(const Tensor& output);
};

// Adds Return, verifies the existing MiniIR, and seals the graph.
[[nodiscard]] CapturedExpression capture(
    const Tensor& output
);

// TensorData is a small host-side carrier used only at the Lesson 24
// evaluate boundary. The later storage/layout lessons will replace it with
// a real storage-bearing tensor.
class TensorData {
public:
    TensorData(
        miniir::Type type,
        std::vector<float> values
    );

    [[nodiscard]] const miniir::Type&
    type() const noexcept;

    [[nodiscard]] const std::vector<float>&
    values() const noexcept;

private:
    miniir::Type type_;
    std::vector<float> values_;
};

using InputBindings =
    std::unordered_map<std::string, TensorData>;

class CompiledExpression {
public:
    [[nodiscard]] const miniir::Program&
    captured_program() const noexcept;

    [[nodiscard]] const miniir::opt::OptimizationResult&
    optimization() const noexcept;

    [[nodiscard]] const miniir::Program&
    optimized_program() const noexcept;

    [[nodiscard]] const miniir::metal::MetalKernelPlan&
    plan() const noexcept;

private:
    CompiledExpression(
        CapturedExpression captured,
        miniir::opt::OptimizationResult optimization,
        miniir::metal::MetalExecutable executable
    );

    CapturedExpression captured_;
    miniir::opt::OptimizationResult optimization_;
    miniir::metal::MetalExecutable executable_;

    friend CompiledExpression compile(
        const minigpu::Device& device,
        const CapturedExpression& expression
    );

    friend TensorData evaluate(
        const minigpu::Device& device,
        minigpu::Queue& queue,
        const CompiledExpression& expression,
        const InputBindings& inputs
    );
};

[[nodiscard]] CompiledExpression compile(
    const minigpu::Device& device,
    const CapturedExpression& expression
);

// This operation is synchronous because it returns CPU-visible values.
[[nodiscard]] TensorData evaluate(
    const minigpu::Device& device,
    minigpu::Queue& queue,
    const CompiledExpression& expression,
    const InputBindings& inputs
);

} // namespace minitensor
