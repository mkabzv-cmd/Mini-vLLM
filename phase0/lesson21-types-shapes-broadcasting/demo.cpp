#include "miniir.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

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

[[nodiscard]] miniir::Program make_example_program() {
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

    require(
        two.type() ==
            miniir::Type::scalar(
                miniir::DType::F32
            ),
        "constant_f32 did not infer scalar f32"
    );
    require(
        scaled.type() == vector4,
        "tensor times scalar did not infer tensor shape"
    );
    require(
        added.type() == vector4,
        "equal-shape add did not preserve tensor shape"
    );
    require(
        result.type() == vector4,
        "ReLU did not preserve its operand type"
    );

    builder.return_value(result);
    return builder.build();
}

void run_positive_type_checks() {
    using miniir::DType;
    using miniir::Type;

    const Type scalar_f32 = Type::scalar(DType::F32);
    const Type vector1_f32 =
        Type::tensor(DType::F32, {1});
    const Type matrix_f32 =
        Type::tensor(DType::F32, {2, 3});
    const Type matrix_f16 =
        Type::tensor(DType::F16, {2, 3});

    require(
        scalar_f32.is_scalar() &&
            scalar_f32.rank() == 0 &&
            scalar_f32.element_count() == 1,
        "scalar metadata is incorrect"
    );
    require(
        !vector1_f32.is_scalar() &&
            vector1_f32.rank() == 1 &&
            vector1_f32.element_count() == 1,
        "tensor<1xf32> must remain a rank-one tensor"
    );
    require(
        matrix_f32.rank() == 2 &&
            matrix_f32.element_count() == 6,
        "rank-two tensor metadata is incorrect"
    );
    require(
        miniir::format_type(scalar_f32) == "f32" &&
            miniir::format_type(vector1_f32) ==
                "tensor<1xf32>" &&
            miniir::format_type(matrix_f16) ==
                "tensor<2x3xf16>",
        "type formatting is incorrect"
    );

    {
        miniir::ProgramBuilder builder;
        const miniir::Value scalar =
            builder.input("scale", scalar_f32);
        const miniir::Value matrix =
            builder.input("matrix", matrix_f32);
        const miniir::Value scaled =
            builder.multiply(scalar, matrix);
        const miniir::Value shifted =
            builder.add(scaled, scalar);

        require(
            scaled.type() == matrix_f32,
            "scalar-left multiply did not broadcast"
        );
        require(
            shifted.type() == matrix_f32,
            "scalar-right add did not broadcast"
        );

        builder.return_value(shifted);
        static_cast<void>(builder.build());
    }

    {
        miniir::ProgramBuilder builder;
        const miniir::Value left =
            builder.input("left", scalar_f32);
        const miniir::Value right =
            builder.input("right", scalar_f32);
        const miniir::Value sum =
            builder.add(left, right);

        require(
            sum.type() == scalar_f32,
            "scalar plus scalar did not remain scalar"
        );

        builder.return_value(sum);
        static_cast<void>(builder.build());
    }

    {
        miniir::ProgramBuilder builder;
        const miniir::Value left =
            builder.input("left", matrix_f16);
        const miniir::Value right =
            builder.input("right", matrix_f16);
        const miniir::Value product =
            builder.multiply(left, right);
        const miniir::Value activated =
            builder.relu(product);

        require(
            activated.type() == matrix_f16,
            "f16 operations did not preserve f16 type"
        );

        builder.return_value(activated);
        const miniir::Program program =
            builder.build();

        require_throws<miniir::EvaluationError>(
            [&] {
                static_cast<void>(
                    miniir::evaluate_f32_element(
                        program,
                        {
                            {"left", 1.0f},
                            {"right", 2.0f},
                        }
                    )
                );
            },
            "supports only f32"
        );
    }
}

void run_type_error_checks() {
    using miniir::DType;
    using miniir::Type;

    const Type scalar_f32 = Type::scalar(DType::F32);
    const Type vector1_f32 =
        Type::tensor(DType::F32, {1});
    const Type vector3_f32 =
        Type::tensor(DType::F32, {3});
    const Type vector4_f32 =
        Type::tensor(DType::F32, {4});
    const Type vector4_f16 =
        Type::tensor(DType::F16, {4});
    const Type matrix23_f32 =
        Type::tensor(DType::F32, {2, 3});
    const Type matrix21_f32 =
        Type::tensor(DType::F32, {2, 1});

    require_throws<miniir::TypeError>(
        [&] {
            miniir::ProgramBuilder builder;
            const miniir::Value left =
                builder.input("left", vector4_f32);
            const miniir::Value right =
                builder.input("right", vector4_f16);

            static_cast<void>(
                builder.add(left, right)
            );
        },
        "dtype mismatch"
    );

    require_throws<miniir::TypeError>(
        [&] {
            miniir::ProgramBuilder builder;
            const miniir::Value left =
                builder.input("left", vector4_f32);
            const miniir::Value right =
                builder.input("right", vector3_f32);

            static_cast<void>(
                builder.multiply(left, right)
            );
        },
        "shape mismatch"
    );

    require_throws<miniir::TypeError>(
        [&] {
            miniir::ProgramBuilder builder;
            const miniir::Value one_element_tensor =
                builder.input("one", vector1_f32);
            const miniir::Value vector =
                builder.input("vector", vector4_f32);

            static_cast<void>(
                builder.add(
                    one_element_tensor,
                    vector
                )
            );
        },
        "shape mismatch"
    );

    require_throws<miniir::TypeError>(
        [&] {
            miniir::ProgramBuilder builder;
            const miniir::Value matrix =
                builder.input("matrix", matrix23_f32);
            const miniir::Value vector =
                builder.input("vector", vector3_f32);

            static_cast<void>(
                builder.add(matrix, vector)
            );
        },
        "scalar broadcasting"
    );

    require_throws<miniir::TypeError>(
        [&] {
            miniir::ProgramBuilder builder;
            const miniir::Value narrow =
                builder.input("narrow", matrix21_f32);
            const miniir::Value wide =
                builder.input("wide", matrix23_f32);

            static_cast<void>(
                builder.add(narrow, wide)
            );
        },
        "shape mismatch"
    );

    require_throws<std::invalid_argument>(
        [] {
            static_cast<void>(
                Type::tensor(DType::F32, {4, 0})
            );
        },
        "greater than zero"
    );

    require_throws<std::invalid_argument>(
        [] {
            static_cast<void>(
                Type::tensor(DType::F32, {})
            );
        },
        "at least one dimension"
    );

    miniir::ProgramBuilder first_builder;
    miniir::ProgramBuilder second_builder;
    const miniir::Value first_value =
        first_builder.input("first", scalar_f32);
    const miniir::Value second_value =
        second_builder.input("second", scalar_f32);

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                second_builder.add(
                    first_value,
                    second_value
                )
            );
        },
        "different ProgramBuilder"
    );
}

void run_verifier_checks(
    const miniir::Program& valid_program
) {
    using miniir::DType;
    using miniir::Opcode;
    using miniir::Operation;
    using miniir::Program;
    using miniir::Type;
    using miniir::ValueId;

    const Type scalar_f32 = Type::scalar(DType::F32);
    const Type vector4_f32 =
        Type::tensor(DType::F32, {4});
    const Type vector3_f32 =
        Type::tensor(DType::F32, {3});
    const Type vector4_f16 =
        Type::tensor(DType::F16, {4});

    const Program stale_result_type({
        Operation{
            Opcode::Input,
            ValueId{0},
            {},
            vector4_f32,
            "input",
            0.0f,
        },
        Operation{
            Opcode::ConstantF32,
            ValueId{1},
            {},
            scalar_f32,
            {},
            2.0f,
        },
        Operation{
            Opcode::Multiply,
            ValueId{2},
            {ValueId{0}, ValueId{1}},
            scalar_f32,
            {},
            0.0f,
        },
        Operation{
            Opcode::Return,
            std::nullopt,
            {ValueId{2}},
            std::nullopt,
            {},
            0.0f,
        },
    });

    require_throws<miniir::VerificationError>(
        [&] {
            miniir::verify(stale_result_type);
        },
        "declared result type f32 does not match "
        "inferred type tensor<4xf32>"
    );

    require_throws<miniir::VerificationError>(
        [&] {
            static_cast<void>(
                stale_result_type.output_type()
            );
        },
        "declared result type f32 does not match"
    );

    const Program incompatible_dtypes({
        Operation{
            Opcode::Input,
            ValueId{0},
            {},
            vector4_f32,
            "left",
            0.0f,
        },
        Operation{
            Opcode::Input,
            ValueId{1},
            {},
            vector4_f16,
            "right",
            0.0f,
        },
        Operation{
            Opcode::Add,
            ValueId{2},
            {ValueId{0}, ValueId{1}},
            vector4_f32,
            {},
            0.0f,
        },
        Operation{
            Opcode::Return,
            std::nullopt,
            {ValueId{2}},
            std::nullopt,
            {},
            0.0f,
        },
    });

    require_throws<miniir::VerificationError>(
        [&] {
            miniir::verify(incompatible_dtypes);
        },
        "operands %0 and %1 are incompatible: "
        "add dtype mismatch"
    );

    const Program incompatible_shapes({
        Operation{
            Opcode::Input,
            ValueId{0},
            {},
            vector4_f32,
            "left",
            0.0f,
        },
        Operation{
            Opcode::Input,
            ValueId{1},
            {},
            vector3_f32,
            "right",
            0.0f,
        },
        Operation{
            Opcode::Multiply,
            ValueId{2},
            {ValueId{0}, ValueId{1}},
            vector4_f32,
            {},
            0.0f,
        },
        Operation{
            Opcode::Return,
            std::nullopt,
            {ValueId{2}},
            std::nullopt,
            {},
            0.0f,
        },
    });

    require_throws<miniir::VerificationError>(
        [&] {
            miniir::verify(incompatible_shapes);
        },
        "operands %0 and %1 are incompatible: "
        "multiply shape mismatch"
    );

    const Program wrong_constant_type({
        Operation{
            Opcode::ConstantF32,
            ValueId{0},
            {},
            Type::scalar(DType::F16),
            {},
            2.0f,
        },
        Operation{
            Opcode::Return,
            std::nullopt,
            {ValueId{0}},
            std::nullopt,
            {},
            0.0f,
        },
    });

    require_throws<miniir::VerificationError>(
        [&] {
            miniir::verify(wrong_constant_type);
        },
        "declared result type f16 does not match "
        "inferred type f32"
    );

    const Program wrong_relu_type({
        Operation{
            Opcode::Input,
            ValueId{0},
            {},
            vector4_f32,
            "input",
            0.0f,
        },
        Operation{
            Opcode::Relu,
            ValueId{1},
            {ValueId{0}},
            scalar_f32,
            {},
            0.0f,
        },
        Operation{
            Opcode::Return,
            std::nullopt,
            {ValueId{1}},
            std::nullopt,
            {},
            0.0f,
        },
    });

    require_throws<miniir::VerificationError>(
        [&] {
            miniir::verify(wrong_relu_type);
        },
        "declared result type f32 does not match "
        "inferred type tensor<4xf32>"
    );

    const Program missing_result_type({
        Operation{
            Opcode::Input,
            ValueId{0},
            {},
            std::nullopt,
            "input",
            0.0f,
        },
        Operation{
            Opcode::Return,
            std::nullopt,
            {ValueId{0}},
            std::nullopt,
            {},
            0.0f,
        },
    });

    require_throws<miniir::VerificationError>(
        [&] {
            miniir::verify(missing_result_type);
        },
        "value-producing operation has no result type"
    );

    const Program typed_return({
        Operation{
            Opcode::Input,
            ValueId{0},
            {},
            scalar_f32,
            "input",
            0.0f,
        },
        Operation{
            Opcode::Return,
            std::nullopt,
            {ValueId{0}},
            scalar_f32,
            {},
            0.0f,
        },
    });

    require_throws<miniir::VerificationError>(
        [&] {
            miniir::verify(typed_return);
        },
        "return must not have a result type"
    );

    const Program undefined_operand({
        Operation{
            Opcode::Input,
            ValueId{0},
            {},
            vector4_f32,
            "input",
            0.0f,
        },
        Operation{
            Opcode::Add,
            ValueId{1},
            {ValueId{0}, ValueId{99}},
            vector4_f32,
            {},
            0.0f,
        },
        Operation{
            Opcode::Return,
            std::nullopt,
            {ValueId{1}},
            std::nullopt,
            {},
            0.0f,
        },
    });

    require_throws<miniir::VerificationError>(
        [&] {
            miniir::verify(undefined_operand);
        },
        "operand %99 is not defined"
    );

    const Program wrong_arity({
        Operation{
            Opcode::Input,
            ValueId{0},
            {},
            scalar_f32,
            "input",
            0.0f,
        },
        Operation{
            Opcode::Add,
            ValueId{1},
            {ValueId{0}},
            scalar_f32,
            {},
            0.0f,
        },
        Operation{
            Opcode::Return,
            std::nullopt,
            {ValueId{1}},
            std::nullopt,
            {},
            0.0f,
        },
    });

    require_throws<miniir::VerificationError>(
        [&] {
            miniir::verify(wrong_arity);
        },
        "expected 2 operand(s), received 1"
    );

    const Program missing_return({
        Operation{
            Opcode::Input,
            ValueId{0},
            {},
            scalar_f32,
            "input",
            0.0f,
        },
    });

    require_throws<miniir::VerificationError>(
        [&] {
            miniir::verify(missing_return);
        },
        "program has no return"
    );

    require_throws<miniir::EvaluationError>(
        [&] {
            static_cast<void>(
                miniir::evaluate_f32_element(
                    valid_program,
                    {{"a", 1.0f}}
                )
            );
        },
        "missing input binding @b"
    );
}

} // namespace

int main() {
    try {
        const miniir::Program program =
            make_example_program();
        const std::string formatted =
            miniir::format_program(program);

        const std::string expected_ir =
            "%0 = input @a : tensor<4xf32>\n"
            "%1 = input @b : tensor<4xf32>\n"
            "%2 = constant_f32 2.0 : f32\n"
            "%3 = multiply %1, %2 : tensor<4xf32>\n"
            "%4 = add %0, %3 : tensor<4xf32>\n"
            "%5 = relu %4 : tensor<4xf32>\n"
            "return %5\n";

        require(
            formatted == expected_ir,
            "formatted typed IR did not match expectation"
        );
        require(
            miniir::format_program(program) == formatted,
            "typed IR printing was not deterministic"
        );

        const miniir::Type vector4 =
            miniir::Type::tensor(
                miniir::DType::F32,
                {4}
            );

        require(
            program.value_type(3) == vector4 &&
                program.value_type(4) == vector4 &&
                program.value_type(5) == vector4 &&
                program.output_type() == vector4,
            "program did not preserve inferred value types"
        );

        const std::vector<float> a{
            -3.0f,
            1.0f,
            5.0f,
            -2.0f,
        };
        const std::vector<float> b{
            1.0f,
            2.0f,
            -4.0f,
            10.0f,
        };
        const std::vector<float> expected{
            0.0f,
            5.0f,
            0.0f,
            18.0f,
        };

        std::vector<float> actual;
        actual.reserve(
            program.output_type().element_count()
        );

        for (
            std::size_t index = 0;
            index <
                program.output_type().element_count();
            ++index
        ) {
            actual.push_back(
                miniir::evaluate_f32_element(
                    program,
                    {
                        {"a", a[index]},
                        {"b", b[index]},
                    }
                )
            );
        }

        require(
            equal_vectors(actual, expected),
            "typed IR evaluator produced incorrect values"
        );

        run_positive_type_checks();
        run_type_error_checks();
        run_verifier_checks(program);

        std::cout
            << "Lesson 21: types, shapes and "
               "scalar broadcasting\n\n"
            << "Typed IR:\n"
            << formatted
            << '\n'
            << "The compiler inferred %3, %4 and %5 as "
            << miniir::format_type(
                   program.output_type()
               )
            << ".\n"
            << "The scalar 2.0 participates in every "
               "tensor element without becoming a "
               "tensor copy.\n\n"
            << "a      = ";
        print_vector(a);
        std::cout << "\nb      = ";
        print_vector(b);
        std::cout << "\nresult = ";
        print_vector(actual);
        std::cout
            << "\n\nType, shape, broadcasting and "
               "diagnostic checks: PASS\n"
            << "PASS: Lesson 21 typed IR succeeded.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
