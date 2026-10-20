#include "contiguous.hpp"

#include "../lesson26-tensor-views/tensor_views.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using minitensor::RuntimeTensor;

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
        std::string(expected_message_part) + "'"
    );
}

[[nodiscard]] bool equal_values(
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

[[nodiscard]] std::vector<float> read_storage(
    const minigpu::Buffer& storage
) {
    std::vector<float> values;
    values.reserve(storage.size());

    for (
        std::size_t index = 0;
        index < storage.size();
        ++index
    ) {
        values.push_back(storage.load(index));
    }

    return values;
}

void print_values(const std::vector<float>& values) {
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

void require_fresh_materialization(
    const RuntimeTensor& output,
    const RuntimeTensor& input,
    const std::vector<float>& expected,
    std::string_view message
) {
    require(
        output.type() == input.type() &&
            output.is_contiguous() &&
            output.offset() == 0 &&
            output.storage_size() == output.numel() &&
            !output.buffer().shares_storage_with(
                input.buffer()
            ) &&
            equal_values(output.to_host(), expected),
        message
    );
}

[[nodiscard]] minigpu::Event
enqueue_from_temporary_source(
    const minigpu::Device& device,
    minigpu::Queue& queue,
    const minitensor::ContiguousMaterializer& materializer,
    RuntimeTensor& output
) {
    RuntimeTensor temporary =
        RuntimeTensor::from_host(
            device,
            miniir::Type::tensor(
                miniir::DType::F32,
                {3, 4}
            ),
            {
                0.0f,
                1.0f,
                2.0f,
                3.0f,
                4.0f,
                5.0f,
                6.0f,
                7.0f,
                8.0f,
                9.0f,
                10.0f,
                11.0f,
            }
        );
    RuntimeTensor view =
        minitensor::slice(temporary, 1, 1, 2);

    return materializer.enqueue_into(
        queue,
        view,
        output,
        4
    );
}

void check_errors(
    const minigpu::Device& device,
    minigpu::Queue& queue,
    const minitensor::ContiguousMaterializer& materializer,
    RuntimeTensor& matrix,
    RuntimeTensor& columns
) {
    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                materializer.run(queue, matrix, 0)
            );
        },
        "threads_per_group must be greater than zero"
    );

    RuntimeTensor ordinary_output =
        RuntimeTensor::empty(device, matrix.type());

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                materializer.enqueue_into(
                    queue,
                    matrix,
                    ordinary_output,
                    0
                )
            );
        },
        "threads_per_group must be greater than zero"
    );

    RuntimeTensor rank_five =
        RuntimeTensor::from_host(
            device,
            miniir::Type::tensor(
                miniir::DType::F32,
                {1, 1, 1, 1, 1}
            ),
            {1.0f}
        );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                materializer.run(queue, rank_five)
            );
        },
        "rank at most 4"
    );

    const std::size_t oversized_threads_per_group =
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()
        ) + 1;

    require_throws<std::overflow_error>(
        [&] {
            static_cast<void>(
                materializer.run(
                    queue,
                    matrix,
                    oversized_threads_per_group
                )
            );
        },
        "launched thread count"
    );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                materializer.enqueue_into(
                    queue,
                    matrix,
                    matrix
                )
            );
        },
        "distinct input and output storage"
    );

    RuntimeTensor square =
        RuntimeTensor::from_host(
            device,
            miniir::Type::tensor(
                miniir::DType::F32,
                {2, 2}
            ),
            {
                1.0f,
                2.0f,
                3.0f,
                4.0f,
            }
        );
    RuntimeTensor square_transposed =
        minitensor::transpose(square, 0, 1);

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                materializer.enqueue_into(
                    queue,
                    square_transposed,
                    square
                )
            );
        },
        "distinct input and output storage"
    );

    RuntimeTensor wrong_shape =
        RuntimeTensor::empty(
            device,
            miniir::Type::tensor(
                miniir::DType::F32,
                {2, 3}
            )
        );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                materializer.enqueue_into(
                    queue,
                    columns,
                    wrong_shape
                )
            );
        },
        "does not match input type"
    );

    RuntimeTensor other_matrix =
        RuntimeTensor::from_host(
            device,
            matrix.type(),
            std::vector<float>(12, 0.0f)
        );
    RuntimeTensor noncontiguous_output =
        minitensor::slice(other_matrix, 1, 1, 2);

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                materializer.enqueue_into(
                    queue,
                    columns,
                    noncontiguous_output
                )
            );
        },
        "canonical row-major strides"
    );

    minigpu::Buffer offset_storage =
        device.allocate(7);
    RuntimeTensor offset_output(
        columns.type(),
        minitensor::TensorLayout::contiguous(
            columns.shape(),
            1
        ),
        offset_storage
    );

    require(
        offset_output.is_contiguous(),
        "offset destination should still report contiguous"
    );
    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                materializer.enqueue_into(
                    queue,
                    columns,
                    offset_output
                )
            );
        },
        "offset must be zero"
    );
}

} // namespace

int main() {
    try {
        std::cout
            << "Lesson 28: contiguous materialization "
               "and buffer reuse\n\n";

        minigpu::Device device;
        std::unique_ptr<minigpu::Queue> queue =
            device.create_queue();
        minitensor::ContiguousMaterializer materializer(
            device
        );

        RuntimeTensor plain =
            RuntimeTensor::from_host(
                device,
                miniir::Type::tensor(
                    miniir::DType::F32,
                    {2, 3}
                ),
                {
                    1.0f,
                    2.0f,
                    3.0f,
                    4.0f,
                    5.0f,
                    6.0f,
                }
            );

        minigpu::Buffer copied_plain_handle =
            plain.buffer();
        minigpu::Buffer unrelated_storage =
            device.allocate(1);
        require(
            copied_plain_handle.shares_storage_with(
                plain.buffer()
            ) &&
                !unrelated_storage.shares_storage_with(
                    plain.buffer()
                ),
            "Buffer allocation identity is incorrect"
        );

        RuntimeTensor plain_copy =
            materializer.run(*queue, plain, 4);

        require_fresh_materialization(
            plain_copy,
            plain,
            {
                1.0f,
                2.0f,
                3.0f,
                4.0f,
                5.0f,
                6.0f,
            },
            "contiguous input was not independently copied"
        );

        plain_copy.store({0, 0}, 99.0f);
        plain.store({0, 1}, 88.0f);
        require(
            plain.load({0, 0}) == 1.0f &&
                plain_copy.load({0, 1}) == 2.0f,
            "fresh materialization still aliases its input"
        );

        RuntimeTensor matrix =
            RuntimeTensor::from_host(
                device,
                miniir::Type::tensor(
                    miniir::DType::F32,
                    {3, 4}
                ),
                {
                    0.0f,
                    1.0f,
                    2.0f,
                    3.0f,
                    4.0f,
                    5.0f,
                    6.0f,
                    7.0f,
                    8.0f,
                    9.0f,
                    10.0f,
                    11.0f,
                }
            );

        RuntimeTensor rows =
            minitensor::slice(matrix, 0, 1, 2);
        require(
            rows.is_contiguous() && rows.offset() == 4,
            "row view should be contiguous with offset 4"
        );

        RuntimeTensor rows_copy =
            materializer.run(*queue, rows, 5);
        require_fresh_materialization(
            rows_copy,
            rows,
            {
                4.0f,
                5.0f,
                6.0f,
                7.0f,
                8.0f,
                9.0f,
                10.0f,
                11.0f,
            },
            "nonzero-offset contiguous input copied incorrectly"
        );

        RuntimeTensor transposed =
            minitensor::transpose(matrix, 0, 1);
        RuntimeTensor transposed_copy =
            materializer.run(*queue, transposed, 7);

        require_fresh_materialization(
            transposed_copy,
            transposed,
            {
                0.0f,
                4.0f,
                8.0f,
                1.0f,
                5.0f,
                9.0f,
                2.0f,
                6.0f,
                10.0f,
                3.0f,
                7.0f,
                11.0f,
            },
            "transpose materialization is incorrect"
        );

        RuntimeTensor composed =
            minitensor::slice(
                transposed,
                0,
                1,
                2,
                2
            );
        RuntimeTensor composed_copy =
            materializer.run(*queue, composed, 4);

        require_fresh_materialization(
            composed_copy,
            composed,
            {
                1.0f,
                5.0f,
                9.0f,
                3.0f,
                7.0f,
                11.0f,
            },
            "composed-view materialization is incorrect"
        );
        composed_copy.store({0, 0}, 123.0f);
        require(
            matrix.load({0, 1}) == 1.0f,
            "materialized output modified source storage"
        );
        composed_copy.store({0, 0}, 1.0f);

        RuntimeTensor columns =
            minitensor::slice(matrix, 1, 1, 2);
        RuntimeTensor stepped =
            minitensor::slice(matrix, 1, 0, 2, 2);

        minigpu::Buffer workspace_storage =
            device.allocate(8);

        for (
            std::size_t index = 0;
            index < workspace_storage.size();
            ++index
        ) {
            workspace_storage.store(index, -777.0f);
        }

        RuntimeTensor workspace(
            columns.type(),
            minitensor::TensorLayout::contiguous(
                columns.shape()
            ),
            workspace_storage
        );

        minigpu::Event first_reuse =
            materializer.enqueue_into(
                *queue,
                columns,
                workspace,
                4
            );
        first_reuse.wait();

        require(
            equal_values(
                workspace.to_host(),
                {
                    1.0f,
                    2.0f,
                    5.0f,
                    6.0f,
                    9.0f,
                    10.0f,
                }
            ) &&
                workspace_storage.load(6) == -777.0f &&
                workspace_storage.load(7) == -777.0f,
            "first reusable copy or padded guard failed"
        );

        minigpu::Event second_reuse =
            materializer.enqueue_into(
                *queue,
                stepped,
                workspace,
                4
            );
        second_reuse.wait();

        require(
            equal_values(
                workspace.to_host(),
                {
                    0.0f,
                    2.0f,
                    4.0f,
                    6.0f,
                    8.0f,
                    10.0f,
                }
            ) &&
                workspace_storage.load(6) == -777.0f &&
                workspace_storage.load(7) == -777.0f,
            "workspace reuse or padded guard failed"
        );

        RuntimeTensor intermediate =
            RuntimeTensor::empty(device, columns.type());
        RuntimeTensor chained_output =
            RuntimeTensor::empty(device, columns.type());

        minigpu::Event first_command =
            materializer.enqueue_into(
                *queue,
                columns,
                intermediate,
                4
            );
        minigpu::Event second_command =
            materializer.enqueue_into(
                *queue,
                intermediate,
                chained_output,
                4
            );

        // No CPU wait was needed between same-queue GPU commands.
        second_command.wait();
        require(
            first_command.is_complete() &&
                equal_values(
                    chained_output.to_host(),
                    {
                        1.0f,
                        2.0f,
                        5.0f,
                        6.0f,
                        9.0f,
                        10.0f,
                    }
                ),
            "same-queue command ordering failed"
        );

        RuntimeTensor retained_output =
            RuntimeTensor::empty(device, columns.type());
        minigpu::Event retained_event =
            enqueue_from_temporary_source(
                device,
                *queue,
                materializer,
                retained_output
            );
        retained_event.wait();

        require(
            equal_values(
                retained_output.to_host(),
                {
                    1.0f,
                    2.0f,
                    5.0f,
                    6.0f,
                    9.0f,
                    10.0f,
                }
            ),
            "Event did not retain temporary input storage"
        );

        minigpu::Buffer scalar_storage =
            device.allocate(3);
        scalar_storage.store(0, -100.0f);
        scalar_storage.store(1, -200.0f);
        scalar_storage.store(2, 42.0f);

        RuntimeTensor scalar(
            miniir::Type::scalar(
                miniir::DType::F32
            ),
            minitensor::TensorLayout::contiguous(
                {},
                2
            ),
            scalar_storage
        );
        RuntimeTensor scalar_copy =
            materializer.run(*queue, scalar, 4);

        require_fresh_materialization(
            scalar_copy,
            scalar,
            {42.0f},
            "offset scalar materialization is incorrect"
        );

        RuntimeTensor rank_four =
            RuntimeTensor::from_host(
                device,
                miniir::Type::tensor(
                    miniir::DType::F32,
                    {2, 1, 2, 2}
                ),
                {
                    0.0f,
                    1.0f,
                    2.0f,
                    3.0f,
                    4.0f,
                    5.0f,
                    6.0f,
                    7.0f,
                }
            );
        RuntimeTensor rank_four_view =
            minitensor::transpose(rank_four, 0, 3);
        RuntimeTensor rank_four_copy =
            materializer.run(*queue, rank_four_view, 3);

        require_fresh_materialization(
            rank_four_copy,
            rank_four_view,
            {
                0.0f,
                4.0f,
                2.0f,
                6.0f,
                1.0f,
                5.0f,
                3.0f,
                7.0f,
            },
            "rank-four materialization is incorrect"
        );

        check_errors(
            device,
            *queue,
            materializer,
            matrix,
            columns
        );

        std::cout
            << "Device: " << device.name() << "\n\n"
            << "Composed input view\n"
            << minitensor::format_layout(
                   composed.layout()
               )
            << "\nlogical values: ";
        print_values(composed.to_host());

        std::cout
            << "\n\nFresh materialization\n"
            << minitensor::format_layout(
                   composed_copy.layout()
               )
            << "\nlogical values: ";
        print_values(composed_copy.to_host());

        std::cout
            << "\n\nReusable workspace\n"
            << "logical elements: " << workspace.numel()
            << ", buffer capacity: "
            << workspace.storage_size()
            << "\nraw storage after second copy: ";
        print_values(read_storage(workspace.buffer()));

        std::cout
            << "\n\nSame-queue chain completed after waiting "
               "only for its final Event.\n\n"
            << "PASS: arbitrary layouts were materialized "
               "into independent contiguous storage, and "
               "one output allocation was safely reused.\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
