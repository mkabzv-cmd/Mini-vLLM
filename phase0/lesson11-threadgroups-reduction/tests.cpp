#include "minigpu.hpp"

#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Function>
void require_throws(
    Function function,
    const std::string& message
) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }

    throw std::runtime_error(message);
}

void test_shared_memory_visibility_and_group_isolation() {
    minigpu::Device device;
    auto queue = device.create_queue();
    minigpu::Buffer group_sums = device.allocate(2);

    minigpu::Kernel kernel(
        "sum_each_group",
        [group_sums](minigpu::ThreadContext& context) {
            context.shared_store(
                context.local_id(),
                static_cast<float>(context.global_id() + 1)
            );

            context.barrier(100);

            if (context.local_id() == 0) {
                float sum = 0.0f;

                for (
                    std::size_t index = 0;
                    index < context.shared_size();
                    ++index
                ) {
                    sum += context.shared_load(index);
                }

                group_sums.store(context.group_id(), sum);
            }
        }
    );

    minigpu::Event finished = queue->submit(
        std::move(kernel),
        minigpu::LaunchConfig(2, 4, 4)
    );
    finished.wait();

    require(
        group_sums.load(0) == 10.0f,
        "group 0 did not observe all four shared writes"
    );
    require(
        group_sums.load(1) == 26.0f,
        "group 1 did not get isolated shared memory"
    );
}

void test_two_pass_reduction() {
    minigpu::Device device;
    auto queue = device.create_queue();
    minigpu::Buffer input = device.allocate(10);

    for (std::size_t index = 0; index < 10; ++index) {
        input.store(
            index,
            static_cast<float>(index + 1)
        );
    }

    const minigpu::ReductionResult result =
        minigpu::reduce_sum(*queue, input, 10, 4);

    require(result.value == 55.0f, "10-value sum is wrong");
    require(result.passes == 2, "expected two reduction passes");
    require(
        result.element_counts ==
            std::vector<std::size_t>{10, 3, 1},
        "expected reduction shape 10 -> 3 -> 1"
    );
}

void test_padded_lanes_still_reach_barriers() {
    minigpu::Device device;
    auto queue = device.create_queue();
    minigpu::Buffer input = device.allocate(5);

    for (std::size_t index = 0; index < 5; ++index) {
        input.store(
            index,
            static_cast<float>(index + 1)
        );
    }

    const minigpu::ReductionResult result =
        minigpu::reduce_sum(*queue, input, 5, 4);

    require(result.value == 15.0f, "padded reduction is wrong");
    require(
        result.element_counts ==
            std::vector<std::size_t>{5, 2, 1},
        "expected padded reduction shape 5 -> 2 -> 1"
    );
}

void test_reduction_is_schedule_independent() {
    minigpu::Device device;
    auto queue = device.create_queue();
    minigpu::Buffer input = device.allocate(17);

    float expected = 0.0f;

    for (std::size_t index = 0; index < 17; ++index) {
        const float value =
            static_cast<float>((index % 5) + 1);
        input.store(index, value);
        expected += value;
    }

    for (int repetition = 0; repetition < 10; ++repetition) {
        const minigpu::ReductionResult result =
            minigpu::reduce_sum(*queue, input, 17, 8);

        require(
            result.value == expected,
            "repeated reduction changed with thread scheduling"
        );
    }
}

void test_out_of_bounds_cancels_group_and_queue_recovers() {
    minigpu::Device device;
    auto queue = device.create_queue();

    minigpu::Kernel invalid(
        "shared_oob",
        [](minigpu::ThreadContext& context) {
            if (context.local_id() == 0) {
                context.shared_store(2, 7.0f);
            }

            context.barrier(200);
        }
    );

    minigpu::Event failed = queue->submit(
        std::move(invalid),
        minigpu::LaunchConfig(1, 4, 2)
    );

    require_throws<std::out_of_range>(
        [&] {
            failed.wait();
        },
        "shared-memory bounds error was not propagated"
    );

    minigpu::Buffer output = device.allocate(2);
    minigpu::Event recovered = queue->submit(
        minigpu::fill(output, 2, 9.0f),
        minigpu::LaunchConfig::cover(2, 2)
    );
    recovered.wait();

    require(
        output.load(0) == 9.0f &&
            output.load(1) == 9.0f,
        "queue did not recover after group cancellation"
    );
}

void test_early_return_is_reported_as_divergence() {
    minigpu::Device device;
    auto queue = device.create_queue();

    minigpu::Kernel invalid(
        "early_return",
        [](minigpu::ThreadContext& context) {
            if (context.local_id() == 0) {
                return;
            }

            context.barrier(300);
        }
    );

    minigpu::Event failed = queue->submit(
        std::move(invalid),
        minigpu::LaunchConfig(1, 4)
    );

    require_throws<minigpu::DivergentBarrierError>(
        [&] {
            failed.wait();
        },
        "early return did not produce a divergence error"
    );
}

void test_mismatched_barrier_sites_are_reported() {
    minigpu::Device device;
    auto queue = device.create_queue();

    minigpu::Kernel invalid(
        "mismatched_sites",
        [](minigpu::ThreadContext& context) {
            if (context.local_id() % 2 == 0) {
                context.barrier(400);
            } else {
                context.barrier(401);
            }
        }
    );

    minigpu::Event failed = queue->submit(
        std::move(invalid),
        minigpu::LaunchConfig(1, 4)
    );

    require_throws<minigpu::DivergentBarrierError>(
        [&] {
            failed.wait();
        },
        "different barrier sites were not detected"
    );
}

void test_invalid_reduction_shapes_are_rejected() {
    minigpu::Device device;
    auto queue = device.create_queue();
    minigpu::Buffer input = device.allocate(6);

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                minigpu::reduce_sum(*queue, input, 6, 1)
            );
        },
        "one-thread reduction should not be allowed"
    );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                minigpu::reduce_sum(*queue, input, 6, 3)
            );
        },
        "non-power-of-two group should not be allowed yet"
    );
}

void test_degenerate_and_boundary_reductions() {
    minigpu::Device device;
    auto queue = device.create_queue();

    minigpu::Buffer empty = device.allocate(0);
    const minigpu::ReductionResult empty_result =
        minigpu::reduce_sum(*queue, empty, 0, 2);

    require(
        empty_result.value == 0.0f &&
            empty_result.passes == 0 &&
            empty_result.element_counts ==
                std::vector<std::size_t>{0},
        "empty sum should return the identity without a pass"
    );

    minigpu::Buffer single = device.allocate(1);
    single.store(0, 42.0f);

    const minigpu::ReductionResult single_result =
        minigpu::reduce_sum(*queue, single, 1, 2);

    require(
        single_result.value == 42.0f &&
            single_result.passes == 0 &&
            single_result.element_counts ==
                std::vector<std::size_t>{1},
        "one-element sum should not launch a pass"
    );

    minigpu::Buffer three = device.allocate(3);
    three.store(0, 1.0f);
    three.store(1, 2.0f);
    three.store(2, 3.0f);

    require(
        minigpu::reduce_sum(*queue, three, 3, 2).value ==
            6.0f,
        "smallest supported group size should reduce"
    );

    const minigpu::LaunchConfig largest_valid(1, 64);
    require(
        largest_valid.threads_per_group() == 64,
        "64-thread teaching group should be accepted"
    );

    require_throws<std::invalid_argument>(
        [] {
            static_cast<void>(
                minigpu::LaunchConfig(1, 65)
            );
        },
        "group larger than the simulator cap was accepted"
    );

    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                minigpu::reduce_sum(*queue, single, 1, 0)
            );
        },
        "degenerate sum bypassed group-size validation"
    );
}

void test_barrier_reuse_across_three_generations() {
    minigpu::Device device;
    auto queue = device.create_queue();
    minigpu::Buffer output = device.allocate(1);

    minigpu::Kernel kernel(
        "three_barrier_generations",
        [output](minigpu::ThreadContext& context) {
            context.shared_store(
                context.local_id(),
                static_cast<float>(context.local_id() + 1)
            );
            context.barrier(500);

            if (context.local_id() < 2) {
                context.shared_store(
                    context.local_id(),
                    context.shared_load(context.local_id()) +
                        context.shared_load(
                            context.local_id() + 2
                        )
                );
            }
            context.barrier(501);

            if (context.local_id() == 0) {
                context.shared_store(
                    0,
                    context.shared_load(0) +
                        context.shared_load(1)
                );
            }
            context.barrier(502);

            if (context.local_id() == 3) {
                output.store(0, context.shared_load(0));
            }
        }
    );

    minigpu::Event finished = queue->submit(
        std::move(kernel),
        minigpu::LaunchConfig(1, 4, 4)
    );
    finished.wait();

    require(
        output.load(0) == 10.0f,
        "reusable barrier failed across three generations"
    );
}

template <typename Function>
void run_test(
    const std::string& name,
    Function function
) {
    function();
    std::cout << "[PASS] " << name << '\n';
}

} // namespace

int main() {
    try {
        run_test(
            "shared-memory visibility and group isolation",
            test_shared_memory_visibility_and_group_isolation
        );
        run_test(
            "two-pass reduction",
            test_two_pass_reduction
        );
        run_test(
            "padded lanes participate in barriers",
            test_padded_lanes_still_reach_barriers
        );
        run_test(
            "reduction is schedule-independent",
            test_reduction_is_schedule_independent
        );
        run_test(
            "bounds error cancels group and queue recovers",
            test_out_of_bounds_cancels_group_and_queue_recovers
        );
        run_test(
            "early return becomes divergence error",
            test_early_return_is_reported_as_divergence
        );
        run_test(
            "mismatched barrier sites are detected",
            test_mismatched_barrier_sites_are_reported
        );
        run_test(
            "invalid reduction shapes are rejected",
            test_invalid_reduction_shapes_are_rejected
        );
        run_test(
            "degenerate and boundary reductions",
            test_degenerate_and_boundary_reductions
        );
        run_test(
            "barrier reuse across three generations",
            test_barrier_reuse_across_three_generations
        );
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout
        << "All threadgroup and reduction tests passed\n";
    return 0;
}
