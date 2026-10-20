#include "miniir.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <unordered_set>
#include <utility>

namespace miniir {

namespace detail {

struct BuilderIdentity {};

} // namespace detail

namespace {

[[nodiscard]] bool is_known_dtype(DType dtype) {
    switch (dtype) {
        case DType::F16:
        case DType::F32:
            return true;
    }

    return false;
}

void require_known_dtype(DType dtype) {
    if (!is_known_dtype(dtype)) {
        throw std::invalid_argument("unknown dtype");
    }
}

void validate_shape(
    const std::vector<std::size_t>& shape
) {
    std::size_t element_count = 1;

    for (std::size_t dimension : shape) {
        if (dimension == 0) {
            throw std::invalid_argument(
                "tensor dimensions must be greater than zero"
            );
        }

        if (
            element_count >
            std::numeric_limits<std::size_t>::max() /
                dimension
        ) {
            throw std::overflow_error(
                "tensor element count overflows size_t"
            );
        }

        element_count *= dimension;
    }
}

[[nodiscard]] std::size_t expected_operand_count(
    Opcode opcode
) {
    switch (opcode) {
        case Opcode::Input:
        case Opcode::ConstantF32:
            return 0;

        case Opcode::Relu:
        case Opcode::Return:
            return 1;

        case Opcode::Multiply:
        case Opcode::Add:
            return 2;
    }

    throw VerificationError("unknown IR opcode");
}

[[nodiscard]] std::string operation_prefix(
    std::size_t operation_index,
    Opcode opcode
) {
    std::ostringstream message;
    message.imbue(std::locale::classic());
    message
        << "operation " << operation_index
        << " (" << opcode_name(opcode) << "): ";
    return message.str();
}

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

[[nodiscard]] bool is_ascii_letter(char character) {
    return
        (character >= 'a' && character <= 'z') ||
        (character >= 'A' && character <= 'Z');
}

[[nodiscard]] bool is_ascii_digit(char character) {
    return character >= '0' && character <= '9';
}

[[nodiscard]] bool is_valid_input_name(
    std::string_view name
) {
    if (name.empty()) {
        return false;
    }

    if (
        !is_ascii_letter(name.front()) &&
        name.front() != '_'
    ) {
        return false;
    }

    return std::all_of(
        name.begin() + 1,
        name.end(),
        [](char character) {
            return
                is_ascii_letter(character) ||
                is_ascii_digit(character) ||
                character == '_';
        }
    );
}

void verify_unused_attributes(
    const Operation& operation,
    const std::string& prefix
) {
    if (
        operation.opcode != Opcode::Input &&
        !operation.input_name.empty()
    ) {
        throw VerificationError(
            prefix +
            "only input may have an input name"
        );
    }

    if (
        operation.opcode != Opcode::ConstantF32 &&
        operation.constant_value != 0.0f
    ) {
        throw VerificationError(
            prefix +
            "only constant_f32 may have a constant value"
        );
    }
}

[[nodiscard]] Type infer_elementwise_binary_type(
    Opcode opcode,
    const Type& left,
    const Type& right
) {
    if (
        opcode != Opcode::Multiply &&
        opcode != Opcode::Add
    ) {
        throw std::logic_error(
            "binary type inference requires add or multiply"
        );
    }

    if (left.dtype() != right.dtype()) {
        throw TypeError(
            std::string(opcode_name(opcode)) +
            " dtype mismatch: left is " +
            format_type(left) + ", right is " +
            format_type(right) +
            "; implicit casts are not supported"
        );
    }

    if (left.shape() == right.shape()) {
        return left;
    }

    if (left.is_scalar()) {
        return right;
    }

    if (right.is_scalar()) {
        return left;
    }

    throw TypeError(
        std::string(opcode_name(opcode)) +
        " shape mismatch: left is " +
        format_type(left) + ", right is " +
        format_type(right) +
        "; Lesson 21 supports only equal tensor shapes "
        "or scalar broadcasting"
    );
}

[[nodiscard]] const Type& require_result_type(
    const Operation& operation,
    const std::string& prefix
) {
    if (!operation.result_type.has_value()) {
        throw VerificationError(
            prefix +
            "value-producing operation has no result type"
        );
    }

    return *operation.result_type;
}

void require_declared_type(
    const Operation& operation,
    const Type& expected,
    const std::string& prefix
) {
    const Type& declared =
        require_result_type(operation, prefix);

    if (declared != expected) {
        throw VerificationError(
            prefix +
            "declared result type " +
            format_type(declared) +
            " does not match inferred type " +
            format_type(expected)
        );
    }
}

} // namespace

std::string_view dtype_name(DType dtype) noexcept {
    switch (dtype) {
        case DType::F16:
            return "f16";
        case DType::F32:
            return "f32";
    }

    return "unknown";
}

Type::Type(
    DType dtype,
    std::vector<std::size_t> shape
)
    : dtype_(dtype),
      shape_(std::move(shape)) {
    require_known_dtype(dtype_);
    validate_shape(shape_);
}

Type Type::scalar(DType dtype) {
    return Type(dtype, {});
}

Type Type::tensor(
    DType dtype,
    std::vector<std::size_t> shape
) {
    if (shape.empty()) {
        throw std::invalid_argument(
            "tensor type requires at least one dimension; "
            "use Type::scalar for rank zero"
        );
    }

    return Type(dtype, std::move(shape));
}

DType Type::dtype() const noexcept {
    return dtype_;
}

const std::vector<std::size_t>&
Type::shape() const noexcept {
    return shape_;
}

bool Type::is_scalar() const noexcept {
    return shape_.empty();
}

std::size_t Type::rank() const noexcept {
    return shape_.size();
}

std::size_t Type::element_count() const noexcept {
    std::size_t count = 1;

    for (std::size_t dimension : shape_) {
        count *= dimension;
    }

    return count;
}

std::string format_type(const Type& type) {
    if (type.is_scalar()) {
        return std::string(dtype_name(type.dtype()));
    }

    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "tensor<";

    for (std::size_t dimension : type.shape()) {
        output << dimension << 'x';
    }

    output
        << dtype_name(type.dtype())
        << '>';
    return output.str();
}

std::string_view opcode_name(Opcode opcode) noexcept {
    switch (opcode) {
        case Opcode::Input:
            return "input";
        case Opcode::ConstantF32:
            return "constant_f32";
        case Opcode::Multiply:
            return "multiply";
        case Opcode::Add:
            return "add";
        case Opcode::Relu:
            return "relu";
        case Opcode::Return:
            return "return";
    }

    return "unknown";
}

Program::Program(std::vector<Operation> operations)
    : operations_(std::move(operations)) {}

const std::vector<Operation>&
Program::operations() const noexcept {
    return operations_;
}

const Type& Program::value_type(ValueId value) const {
    verify(*this);

    for (const Operation& operation : operations_) {
        if (
            operation.result == value &&
            operation.result_type.has_value()
        ) {
            return *operation.result_type;
        }
    }

    throw std::out_of_range(
        "program has no typed value %" +
        std::to_string(value)
    );
}

const Type& Program::output_type() const {
    verify(*this);

    const ValueId output_value =
        operations_.back().operands.front();

    for (const Operation& operation : operations_) {
        if (
            operation.result == output_value &&
            operation.result_type.has_value()
        ) {
            return *operation.result_type;
        }
    }

    throw std::logic_error(
        "verified program has no output type"
    );
}

Value::Value(
    ValueId id,
    Type type,
    std::shared_ptr<const detail::BuilderIdentity> owner
)
    : id_(id),
      type_(std::move(type)),
      owner_(std::move(owner)) {}

ValueId Value::id() const noexcept {
    return id_;
}

const Type& Value::type() const noexcept {
    return type_;
}

ProgramBuilder::ProgramBuilder()
    : identity_(
          std::make_shared<detail::BuilderIdentity>()
      ) {}

void ProgramBuilder::require_can_append() const {
    if (has_return_) {
        throw std::logic_error(
            "cannot append an operation after return"
        );
    }
}

void ProgramBuilder::require_owned(Value value) const {
    if (value.owner_ != identity_) {
        throw std::invalid_argument(
            "value belongs to a different ProgramBuilder"
        );
    }

    if (value.id_ >= next_value_id_) {
        throw std::out_of_range(
            "value has not been defined by this builder"
        );
    }

    const Operation& defining_operation =
        operations_[value.id_];

    if (
        !defining_operation.result_type.has_value() ||
        *defining_operation.result_type != value.type_
    ) {
        throw std::logic_error(
            "value type does not match its defining operation"
        );
    }
}

Value ProgramBuilder::append_result(
    Opcode opcode,
    Type result_type,
    std::vector<ValueId> operands,
    std::string input_name,
    float constant_value
) {
    require_can_append();

    const ValueId result = next_value_id_;

    operations_.push_back(Operation{
        opcode,
        result,
        std::move(operands),
        result_type,
        std::move(input_name),
        constant_value,
    });

    ++next_value_id_;
    return Value(
        result,
        std::move(result_type),
        identity_
    );
}

Value ProgramBuilder::input(
    std::string name,
    Type type
) {
    if (!is_valid_input_name(name)) {
        throw std::invalid_argument(
            "input name must be an identifier containing "
            "letters, digits or underscores"
        );
    }

    return append_result(
        Opcode::Input,
        std::move(type),
        {},
        std::move(name)
    );
}

Value ProgramBuilder::constant_f32(float value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "constant_f32 requires a finite value"
        );
    }

    return append_result(
        Opcode::ConstantF32,
        Type::scalar(DType::F32),
        {},
        {},
        value
    );
}

Value ProgramBuilder::multiply(
    Value left,
    Value right
) {
    require_owned(left);
    require_owned(right);

    Type result_type =
        infer_elementwise_binary_type(
            Opcode::Multiply,
            left.type(),
            right.type()
        );

    return append_result(
        Opcode::Multiply,
        std::move(result_type),
        {left.id(), right.id()}
    );
}

Value ProgramBuilder::add(
    Value left,
    Value right
) {
    require_owned(left);
    require_owned(right);

    Type result_type =
        infer_elementwise_binary_type(
            Opcode::Add,
            left.type(),
            right.type()
        );

    return append_result(
        Opcode::Add,
        std::move(result_type),
        {left.id(), right.id()}
    );
}

Value ProgramBuilder::relu(Value input_value) {
    require_owned(input_value);

    return append_result(
        Opcode::Relu,
        input_value.type(),
        {input_value.id()}
    );
}

void ProgramBuilder::return_value(Value value) {
    require_can_append();
    require_owned(value);

    operations_.push_back(Operation{
        Opcode::Return,
        std::nullopt,
        {value.id()},
        std::nullopt,
        {},
        0.0f,
    });
    has_return_ = true;
}

Program ProgramBuilder::build() const {
    Program program(operations_);
    verify(program);
    return program;
}

void verify(const Program& program) {
    ValueId next_value_id = 0;
    bool saw_return = false;
    std::unordered_set<std::string> input_names;
    std::vector<Type> known_types;
    known_types.reserve(program.operations().size());

    for (
        std::size_t operation_index = 0;
        operation_index < program.operations().size();
        ++operation_index
    ) {
        const Operation& operation =
            program.operations()[operation_index];
        const std::string prefix =
            operation_prefix(
                operation_index,
                operation.opcode
            );

        verify_unused_attributes(operation, prefix);

        const std::size_t required_operands =
            expected_operand_count(operation.opcode);

        if (
            operation.operands.size() !=
            required_operands
        ) {
            std::ostringstream message;
            message
                << prefix
                << "expected " << required_operands
                << " operand(s), received "
                << operation.operands.size();
            throw VerificationError(message.str());
        }

        for (ValueId operand : operation.operands) {
            if (operand >= next_value_id) {
                std::ostringstream message;
                message
                    << prefix
                    << "operand %" << operand
                    << " is not defined before this use";
                throw VerificationError(message.str());
            }
        }

        if (operation.opcode == Opcode::Return) {
            if (operation.result.has_value()) {
                throw VerificationError(
                    prefix +
                    "return must not produce a value"
                );
            }

            if (operation.result_type.has_value()) {
                throw VerificationError(
                    prefix +
                    "return must not have a result type"
                );
            }

            if (
                operation_index + 1 !=
                program.operations().size()
            ) {
                throw VerificationError(
                    prefix +
                    "return must be the final operation"
                );
            }

            saw_return = true;
            continue;
        }

        if (!operation.result.has_value()) {
            throw VerificationError(
                prefix +
                "value-producing operation has no result"
            );
        }

        if (*operation.result != next_value_id) {
            std::ostringstream message;
            message
                << prefix
                << "expected result %" << next_value_id
                << ", received %" << *operation.result;
            throw VerificationError(message.str());
        }

        const Type& declared_type =
            require_result_type(operation, prefix);

        switch (operation.opcode) {
            case Opcode::Input:
                if (
                    !is_valid_input_name(
                        operation.input_name
                    )
                ) {
                    throw VerificationError(
                        prefix +
                        "input name must be an identifier "
                        "containing letters, digits or "
                        "underscores"
                    );
                }

                if (
                    !input_names.insert(
                        operation.input_name
                    ).second
                ) {
                    throw VerificationError(
                        prefix +
                        "duplicate input name @" +
                        operation.input_name
                    );
                }
                break;

            case Opcode::ConstantF32:
                if (!std::isfinite(
                        operation.constant_value
                    )) {
                    throw VerificationError(
                        prefix +
                        "constant must be finite"
                    );
                }

                require_declared_type(
                    operation,
                    Type::scalar(DType::F32),
                    prefix
                );
                break;

            case Opcode::Multiply:
            case Opcode::Add: {
                Type inferred =
                    Type::scalar(DType::F32);

                try {
                    inferred =
                        infer_elementwise_binary_type(
                            operation.opcode,
                            known_types[
                                operation.operands[0]
                            ],
                            known_types[
                                operation.operands[1]
                            ]
                        );
                } catch (const TypeError& error) {
                    std::ostringstream message;
                    message
                        << prefix
                        << "operands %"
                        << operation.operands[0]
                        << " and %"
                        << operation.operands[1]
                        << " are incompatible: "
                        << error.what();
                    throw VerificationError(message.str());
                }

                require_declared_type(
                    operation,
                    inferred,
                    prefix
                );
                break;
            }

            case Opcode::Relu:
                require_declared_type(
                    operation,
                    known_types[
                        operation.operands[0]
                    ],
                    prefix
                );
                break;

            case Opcode::Return:
                break;
        }

        known_types.push_back(declared_type);
        ++next_value_id;
    }

    if (!saw_return) {
        throw VerificationError(
            "program has no return operation"
        );
    }
}

std::string format_program(const Program& program) {
    verify(program);

    std::ostringstream output;
    output.imbue(std::locale::classic());

    for (const Operation& operation : program.operations()) {
        switch (operation.opcode) {
            case Opcode::Input:
                output
                    << '%' << *operation.result
                    << " = input @"
                    << operation.input_name
                    << " : "
                    << format_type(
                           *operation.result_type
                       )
                    << '\n';
                break;

            case Opcode::ConstantF32:
                output
                    << '%' << *operation.result
                    << " = constant_f32 "
                    << format_float(
                           operation.constant_value
                       )
                    << " : "
                    << format_type(
                           *operation.result_type
                       )
                    << '\n';
                break;

            case Opcode::Multiply:
                output
                    << '%' << *operation.result
                    << " = multiply %"
                    << operation.operands[0]
                    << ", %" << operation.operands[1]
                    << " : "
                    << format_type(
                           *operation.result_type
                       )
                    << '\n';
                break;

            case Opcode::Add:
                output
                    << '%' << *operation.result
                    << " = add %"
                    << operation.operands[0]
                    << ", %" << operation.operands[1]
                    << " : "
                    << format_type(
                           *operation.result_type
                       )
                    << '\n';
                break;

            case Opcode::Relu:
                output
                    << '%' << *operation.result
                    << " = relu %"
                    << operation.operands[0]
                    << " : "
                    << format_type(
                           *operation.result_type
                       )
                    << '\n';
                break;

            case Opcode::Return:
                output
                    << "return %"
                    << operation.operands[0]
                    << '\n';
                break;
        }
    }

    return output.str();
}

float evaluate_f32_element(
    const Program& program,
    const InputBindings& inputs
) {
    verify(program);

    std::vector<float> values;
    values.reserve(program.operations().size());

    for (
        std::size_t operation_index = 0;
        operation_index < program.operations().size();
        ++operation_index
    ) {
        const Operation& operation =
            program.operations()[operation_index];

        if (
            operation.result_type.has_value() &&
            operation.result_type->dtype() != DType::F32
        ) {
            throw EvaluationError(
                "evaluate_f32_element supports only f32 "
                "in Lesson 21; operation " +
                std::to_string(operation_index) +
                " produces " +
                format_type(*operation.result_type)
            );
        }

        switch (operation.opcode) {
            case Opcode::Input: {
                const auto found =
                    inputs.find(operation.input_name);

                if (found == inputs.end()) {
                    throw EvaluationError(
                        "missing input binding @" +
                        operation.input_name
                    );
                }

                values.push_back(found->second);
                break;
            }

            case Opcode::ConstantF32:
                values.push_back(
                    operation.constant_value
                );
                break;

            case Opcode::Multiply:
                values.push_back(
                    values[operation.operands[0]] *
                    values[operation.operands[1]]
                );
                break;

            case Opcode::Add:
                values.push_back(
                    values[operation.operands[0]] +
                    values[operation.operands[1]]
                );
                break;

            case Opcode::Relu:
                values.push_back(
                    std::max(
                        values[operation.operands[0]],
                        0.0f
                    )
                );
                break;

            case Opcode::Return:
                return values[operation.operands[0]];
        }
    }

    throw EvaluationError(
        "verified program unexpectedly produced no result"
    );
}

} // namespace miniir
