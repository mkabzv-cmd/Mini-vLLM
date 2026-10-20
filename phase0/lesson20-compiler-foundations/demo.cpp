#include "miniir.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <optional>
#include <sstream>
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
    miniir::ProgramBuilder builder;

    const miniir::Value a = builder.input("a");
    const miniir::Value b = builder.input("b");
    const miniir::Value two =
        builder.constant_f32(2.0f);
    const miniir::Value scaled =
        builder.multiply(b, two);
    const miniir::Value added =
        builder.add(a, scaled);
    const miniir::Value result =
        builder.relu(added);

    builder.return_value(result);
    return builder.build();
}

void run_negative_checks(
    const miniir::Program& valid_program
) {
    using miniir::Opcode;
    using miniir::Operation;
    using miniir::Program;
    using miniir::ValueId;

    const Program undefined_operand({
        Operation{
            Opcode::Input,
            ValueId{0},
            {},
            "a",
            0.0f,
        },
        Operation{
            Opcode::Add,
            ValueId{1},
            {ValueId{0}, ValueId{99}},
            {},
            0.0f,
        },
        Operation{
            Opcode::Return,
            std::nullopt,
            {ValueId{1}},
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
            "a",
            0.0f,
        },
        Operation{
            Opcode::Relu,
            ValueId{1},
            {ValueId{0}, ValueId{0}},
            {},
            0.0f,
        },
        Operation{
            Opcode::Return,
            std::nullopt,
            {ValueId{1}},
            {},
            0.0f,
        },
    });

    require_throws<miniir::VerificationError>(
        [&] {
            miniir::verify(wrong_arity);
        },
        "expected 1 operand(s), received 2"
    );

    const Program missing_return({
        Operation{
            Opcode::Input,
            ValueId{0},
            {},
            "a",
            0.0f,
        },
    });

    require_throws<miniir::VerificationError>(
        [&] {
            miniir::verify(missing_return);
        },
        "program has no return"
    );

    const Program duplicate_input({
        Operation{
            Opcode::Input,
            ValueId{0},
            {},
            "a",
            0.0f,
        },
        Operation{
            Opcode::Input,
            ValueId{1},
            {},
            "a",
            0.0f,
        },
        Operation{
            Opcode::Return,
            std::nullopt,
            {ValueId{1}},
            {},
            0.0f,
        },
    });

    require_throws<miniir::VerificationError>(
        [&] {
            miniir::verify(duplicate_input);
        },
        "duplicate input name @a"
    );

    const Program invalid_input_name({
        Operation{
            Opcode::Input,
            ValueId{0},
            {},
            "a\nreturn %0",
            0.0f,
        },
        Operation{
            Opcode::Return,
            std::nullopt,
            {ValueId{0}},
            {},
            0.0f,
        },
    });

    require_throws<miniir::VerificationError>(
        [&] {
            miniir::verify(invalid_input_name);
        },
        "input name must be an identifier"
    );

    const Program unexpected_attribute({
        Operation{
            Opcode::Input,
            ValueId{0},
            {},
            "a",
            0.0f,
        },
        Operation{
            Opcode::Relu,
            ValueId{1},
            {ValueId{0}},
            "unused_name",
            0.0f,
        },
        Operation{
            Opcode::Return,
            std::nullopt,
            {ValueId{1}},
            {},
            0.0f,
        },
    });

    require_throws<miniir::VerificationError>(
        [&] {
            miniir::verify(unexpected_attribute);
        },
        "only input may have an input name"
    );

    miniir::ProgramBuilder first_builder;
    miniir::ProgramBuilder second_builder;
    const miniir::Value first_value =
        first_builder.input("first");
    const miniir::Value second_value =
        second_builder.input("second");

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

    require_throws<std::invalid_argument>(
        [] {
            miniir::ProgramBuilder builder;
            static_cast<void>(
                builder.input("not valid")
            );
        },
        "input name must be an identifier"
    );

    require_throws<miniir::EvaluationError>(
        [&] {
            static_cast<void>(
                miniir::evaluate_element(
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
            "%0 = input @a\n"
            "%1 = input @b\n"
            "%2 = constant_f32 2.0\n"
            "%3 = multiply %1, %2\n"
            "%4 = add %0, %3\n"
            "%5 = relu %4\n"
            "return %5\n";

        require(
            formatted == expected_ir,
            "formatted IR did not match the expected program"
        );
        require(
            miniir::format_program(program) == formatted,
            "IR printing was not deterministic"
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
        actual.reserve(a.size());

        for (
            std::size_t index = 0;
            index < a.size();
            ++index
        ) {
            actual.push_back(
                miniir::evaluate_element(
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
            "IR evaluator produced incorrect values"
        );

        run_negative_checks(program);

        std::cout
            << "Lesson 20: first in-memory compiler IR\n\n"
            << "Builder recorded this program:\n"
            << formatted
            << '\n'
            << "The CPU evaluator applies the same IR "
               "to each element:\n"
            << "a      = ";
        print_vector(a);
        std::cout << "\nb      = ";
        print_vector(b);
        std::cout << "\nresult = ";
        print_vector(actual);
        std::cout
            << "\n\nStructural and error-path checks: PASS\n"
            << "PASS: builder, SSA-like IR, verifier, "
               "printer and evaluator succeeded.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
