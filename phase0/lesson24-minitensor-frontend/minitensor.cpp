#include "minitensor.hpp"

#include <algorithm>
#include <cmath>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace minitensor {

namespace detail {

struct GraphState {
    miniir::ProgramBuilder builder;
    std::unordered_set<std::string> input_names;
    bool sealed = false;
};

} // namespace detail

namespace {

void require_open(
    const std::shared_ptr<detail::GraphState>& state
) {
    if (!state) {
        throw FrontendError(
            "MiniTensor has no graph state"
        );
    }

    if (state->sealed) {
        throw FrontendError(
            "MiniTensor graph is already captured and sealed"
        );
    }
}

void require_same_open_graph(
    const std::shared_ptr<detail::GraphState>& left,
    const std::shared_ptr<detail::GraphState>& right
) {
    if (left != right) {
        throw FrontendError(
            "cannot combine Tensors from different "
            "MiniTensor graphs"
        );
    }

    require_open(left);
}

void require_compatible_f32_literal(
    const Tensor& tensor,
    float value,
    miniir::Opcode opcode,
    bool literal_is_left
) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "constant_f32 requires a finite value"
        );
    }

    if (tensor.type().dtype() == miniir::DType::F32) {
        return;
    }

    const std::string tensor_type =
        miniir::format_type(tensor.type());
    const std::string left_type =
        literal_is_left ? "f32" : tensor_type;
    const std::string right_type =
        literal_is_left ? tensor_type : "f32";

    throw miniir::TypeError(
        std::string(miniir::opcode_name(opcode)) +
        " dtype mismatch: left is " + left_type +
        ", right is " + right_type +
        "; implicit casts are not supported"
    );
}

[[nodiscard]] const miniir::Operation*
find_captured_input(
    const miniir::Program& program,
    std::string_view name
) {
    const auto found = std::find_if(
        program.operations().begin(),
        program.operations().end(),
        [name](const miniir::Operation& operation) {
            return
                operation.opcode == miniir::Opcode::Input &&
                operation.input_name == name;
        }
    );

    return found == program.operations().end()
        ? nullptr
        : &*found;
}

void validate_runtime_inputs(
    const miniir::Program& captured_program,
    const miniir::metal::MetalKernelPlan& plan,
    const InputBindings& inputs
) {
    std::vector<std::string> provided_names;
    provided_names.reserve(inputs.size());

    for (const auto& [name, data] : inputs) {
        static_cast<void>(data);
        provided_names.push_back(name);
    }

    std::sort(provided_names.begin(), provided_names.end());

    // Validate against the captured frontend signature, not only the
    // optimized Metal signature. An optimizer may remove an unused input;
    // callers may still provide that declared input, in which case it is
    // validated and harmlessly ignored.
    for (const std::string& name : provided_names) {
        const miniir::Operation* declared =
            find_captured_input(captured_program, name);

        if (declared == nullptr) {
            throw std::invalid_argument(
                "unknown MiniTensor input @" + name
            );
        }

        const TensorData& data = inputs.at(name);
        const miniir::Type& declared_type =
            *declared->result_type;

        if (data.type() != declared_type) {
            throw std::invalid_argument(
                "MiniTensor input @" + name +
                " requires " +
                miniir::format_type(declared_type) +
                ", received " +
                miniir::format_type(data.type())
            );
        }
    }

    // Only inputs that survived optimization are required by the actual
    // executable.
    for (const miniir::metal::InputBinding& input :
         plan.inputs()) {
        const auto found = inputs.find(input.name);

        if (found == inputs.end()) {
            throw std::invalid_argument(
                "missing MiniTensor input @" + input.name
            );
        }

        const miniir::Operation* declared =
            find_captured_input(
                captured_program,
                input.name
            );

        if (
            declared == nullptr ||
            *declared->result_type != input.type
        ) {
            throw std::logic_error(
                "optimized MiniTensor input signature "
                "does not match its captured program"
            );
        }
    }
}

} // namespace

Tensor::Tensor(
    std::shared_ptr<detail::GraphState> state,
    miniir::Value value
)
    : state_(std::move(state)),
      value_(std::move(value)) {}

miniir::ValueId Tensor::id() const noexcept {
    return value_.id();
}

const miniir::Type& Tensor::type() const noexcept {
    return value_.type();
}

Graph::Graph()
    : state_(std::make_shared<detail::GraphState>()) {}

Tensor Graph::input(
    std::string name,
    miniir::Type type
) {
    require_open(state_);

    if (state_->input_names.contains(name)) {
        throw std::invalid_argument(
            "duplicate MiniTensor input @" + name
        );
    }

    // ProgramBuilder validates the identifier before we commit it to the
    // frontend's name set.
    miniir::Value value =
        state_->builder.input(name, std::move(type));
    state_->input_names.insert(std::move(name));

    return Tensor(state_, std::move(value));
}

Tensor Graph::input(
    std::string name,
    std::vector<std::size_t> shape,
    miniir::DType dtype
) {
    return input(
        std::move(name),
        miniir::Type::tensor(dtype, std::move(shape))
    );
}

Tensor Graph::scalar_input(
    std::string name,
    miniir::DType dtype
) {
    return input(
        std::move(name),
        miniir::Type::scalar(dtype)
    );
}

Tensor Graph::scalar(float value) {
    require_open(state_);

    return Tensor(
        state_,
        state_->builder.constant_f32(value)
    );
}

Tensor operator+(
    const Tensor& left,
    const Tensor& right
) {
    require_same_open_graph(left.state_, right.state_);

    return Tensor(
        left.state_,
        left.state_->builder.add(
            left.value_,
            right.value_
        )
    );
}

Tensor operator+(
    const Tensor& left,
    float right
) {
    require_open(left.state_);
    require_compatible_f32_literal(
        left,
        right,
        miniir::Opcode::Add,
        false
    );

    const Tensor scalar(
        left.state_,
        left.state_->builder.constant_f32(right)
    );

    return left + scalar;
}

Tensor operator+(
    float left,
    const Tensor& right
) {
    require_open(right.state_);
    require_compatible_f32_literal(
        right,
        left,
        miniir::Opcode::Add,
        true
    );

    const Tensor scalar(
        right.state_,
        right.state_->builder.constant_f32(left)
    );

    return scalar + right;
}

Tensor operator*(
    const Tensor& left,
    const Tensor& right
) {
    require_same_open_graph(left.state_, right.state_);

    return Tensor(
        left.state_,
        left.state_->builder.multiply(
            left.value_,
            right.value_
        )
    );
}

Tensor operator*(
    const Tensor& left,
    float right
) {
    require_open(left.state_);
    require_compatible_f32_literal(
        left,
        right,
        miniir::Opcode::Multiply,
        false
    );

    const Tensor scalar(
        left.state_,
        left.state_->builder.constant_f32(right)
    );

    return left * scalar;
}

Tensor operator*(
    float left,
    const Tensor& right
) {
    require_open(right.state_);
    require_compatible_f32_literal(
        right,
        left,
        miniir::Opcode::Multiply,
        true
    );

    const Tensor scalar(
        right.state_,
        right.state_->builder.constant_f32(left)
    );

    return scalar * right;
}

Tensor relu(const Tensor& input) {
    require_open(input.state_);

    return Tensor(
        input.state_,
        input.state_->builder.relu(input.value_)
    );
}

CapturedExpression::CapturedExpression(
    miniir::Program program
)
    : program_(std::move(program)) {}

const miniir::Program&
CapturedExpression::program() const noexcept {
    return program_;
}

CapturedExpression capture(const Tensor& output) {
    require_open(output.state_);

    // return_value() closes ProgramBuilder. Marking the frontend state as
    // sealed keeps later operator calls from reaching that closed builder.
    output.state_->builder.return_value(output.value_);
    output.state_->sealed = true;

    return CapturedExpression(
        output.state_->builder.build()
    );
}

TensorData::TensorData(
    miniir::Type type,
    std::vector<float> values
)
    : type_(std::move(type)),
      values_(std::move(values)) {
    if (type_.dtype() != miniir::DType::F32) {
        throw std::invalid_argument(
            "Lesson 24 TensorData supports only f32"
        );
    }

    if (values_.size() != type_.element_count()) {
        throw std::invalid_argument(
            "TensorData for " + miniir::format_type(type_) +
            " requires " +
            std::to_string(type_.element_count()) +
            " value(s), received " +
            std::to_string(values_.size())
        );
    }
}

const miniir::Type&
TensorData::type() const noexcept {
    return type_;
}

const std::vector<float>&
TensorData::values() const noexcept {
    return values_;
}

CompiledExpression::CompiledExpression(
    CapturedExpression captured,
    miniir::opt::OptimizationResult optimization,
    miniir::metal::MetalExecutable executable
)
    : captured_(std::move(captured)),
      optimization_(std::move(optimization)),
      executable_(std::move(executable)) {}

const miniir::Program&
CompiledExpression::captured_program() const noexcept {
    return captured_.program();
}

const miniir::opt::OptimizationResult&
CompiledExpression::optimization() const noexcept {
    return optimization_;
}

const miniir::Program&
CompiledExpression::optimized_program() const noexcept {
    return optimization_.program;
}

const miniir::metal::MetalKernelPlan&
CompiledExpression::plan() const noexcept {
    return executable_.plan();
}

CompiledExpression compile(
    const minigpu::Device& device,
    const CapturedExpression& expression
) {
    miniir::opt::OptimizationResult optimization =
        miniir::opt::optimize_elementwise(
            expression.program()
        );

    miniir::metal::MetalKernelPlan plan =
        miniir::metal::lower_to_metal(
            optimization.program
        );

    miniir::metal::MetalExecutable executable =
        miniir::metal::compile_metal(device, plan);

    return CompiledExpression(
        expression,
        std::move(optimization),
        std::move(executable)
    );
}

TensorData evaluate(
    const minigpu::Device& device,
    minigpu::Queue& queue,
    const CompiledExpression& expression,
    const InputBindings& inputs
) {
    const miniir::metal::MetalKernelPlan& plan =
        expression.plan();

    validate_runtime_inputs(
        expression.captured_program(),
        plan,
        inputs
    );

    miniir::metal::RuntimeInputs runtime_inputs;
    runtime_inputs.reserve(plan.inputs().size());

    for (const miniir::metal::InputBinding& input :
         plan.inputs()) {
        const TensorData& data = inputs.at(input.name);
        minigpu::Buffer buffer =
            device.allocate(data.values().size());

        for (
            std::size_t index = 0;
            index < data.values().size();
            ++index
        ) {
            buffer.store(index, data.values()[index]);
        }

        runtime_inputs.emplace(input.name, buffer);
    }

    minigpu::Buffer output = device.allocate(
        plan.output_type().element_count()
    );

    minigpu::Event finished = queue.submit(
        expression.executable_.make_dispatch(
            runtime_inputs,
            output
        ),
        expression.executable_.make_launch()
    );

    // The returned TensorData is CPU-visible, so the GPU must finish before
    // the CPU reads output.
    finished.wait();

    std::vector<float> values;
    values.reserve(output.size());

    for (
        std::size_t index = 0;
        index < output.size();
        ++index
    ) {
        values.push_back(output.load(index));
    }

    return TensorData(
        plan.output_type(),
        std::move(values)
    );
}

} // namespace minitensor
