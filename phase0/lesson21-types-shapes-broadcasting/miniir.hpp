#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace miniir {

using ValueId = std::size_t;

enum class DType {
    F16,
    F32,
};

[[nodiscard]] std::string_view dtype_name(
    DType dtype
) noexcept;

class Type {
public:
    [[nodiscard]] static Type scalar(DType dtype);

    [[nodiscard]] static Type tensor(
        DType dtype,
        std::vector<std::size_t> shape
    );

    [[nodiscard]] DType dtype() const noexcept;

    [[nodiscard]] const std::vector<std::size_t>&
    shape() const noexcept;

    [[nodiscard]] bool is_scalar() const noexcept;
    [[nodiscard]] std::size_t rank() const noexcept;
    [[nodiscard]] std::size_t element_count() const noexcept;

    friend bool operator==(
        const Type&,
        const Type&
    ) = default;

private:
    Type(
        DType dtype,
        std::vector<std::size_t> shape
    );

    DType dtype_;
    std::vector<std::size_t> shape_;
};

[[nodiscard]] std::string format_type(const Type& type);

enum class Opcode {
    Input,
    ConstantF32,
    Multiply,
    Add,
    Relu,
    Return,
};

[[nodiscard]] std::string_view opcode_name(
    Opcode opcode
) noexcept;

struct Operation {
    Opcode opcode;
    std::optional<ValueId> result;
    std::vector<ValueId> operands;
    std::optional<Type> result_type;
    std::string input_name;
    float constant_value = 0.0f;
};

class Program {
public:
    explicit Program(std::vector<Operation> operations);

    [[nodiscard]] const std::vector<Operation>&
    operations() const noexcept;

    [[nodiscard]] const Type& value_type(
        ValueId value
    ) const;

    [[nodiscard]] const Type& output_type() const;

private:
    std::vector<Operation> operations_;
};

namespace detail {

struct BuilderIdentity;

} // namespace detail

class Value {
public:
    [[nodiscard]] ValueId id() const noexcept;
    [[nodiscard]] const Type& type() const noexcept;

private:
    Value(
        ValueId id,
        Type type,
        std::shared_ptr<const detail::BuilderIdentity> owner
    );

    ValueId id_;
    Type type_;
    std::shared_ptr<const detail::BuilderIdentity> owner_;

    friend class ProgramBuilder;
};

class ProgramBuilder {
public:
    ProgramBuilder();

    ProgramBuilder(const ProgramBuilder&) = delete;
    ProgramBuilder& operator=(const ProgramBuilder&) = delete;
    ProgramBuilder(ProgramBuilder&&) = delete;
    ProgramBuilder& operator=(ProgramBuilder&&) = delete;

    [[nodiscard]] Value input(
        std::string name,
        Type type
    );

    [[nodiscard]] Value constant_f32(float value);
    [[nodiscard]] Value multiply(Value left, Value right);
    [[nodiscard]] Value add(Value left, Value right);
    [[nodiscard]] Value relu(Value input);

    void return_value(Value value);

    [[nodiscard]] Program build() const;

private:
    [[nodiscard]] Value append_result(
        Opcode opcode,
        Type result_type,
        std::vector<ValueId> operands = {},
        std::string input_name = {},
        float constant_value = 0.0f
    );

    void require_can_append() const;
    void require_owned(Value value) const;

    std::shared_ptr<const detail::BuilderIdentity> identity_;
    std::vector<Operation> operations_;
    ValueId next_value_id_ = 0;
    bool has_return_ = false;
};

class TypeError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class VerificationError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class EvaluationError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void verify(const Program& program);

[[nodiscard]] std::string format_program(
    const Program& program
);

using InputBindings =
    std::unordered_map<std::string, float>;

[[nodiscard]] float evaluate_f32_element(
    const Program& program,
    const InputBindings& inputs
);

} // namespace miniir
