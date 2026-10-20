#include "minigpu.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

void test_atomic_load_store_pair_loses_updates() {
    minigpu::Device device;
    auto queue = device.create_queue();
    minigpu::AtomicU32Buffer counter =
        device.allocate_atomic_u32(1);
    counter.store(0, 0);

    minigpu::Kernel flawed(
        "atomic_load_then_store",
        [counter](minigpu::ThreadContext& context) {
            const std::uint32_t old_value = counter.load(0);

            // Force every lane to observe the same old value.
            context.barrier(600);

            // load() and store() are each atomic, but this pair is not one
            // indivisible read-modify-write operation.
            counter.store(0, old_value + 1);
            context.barrier(601);
        }
    );

    minigpu::Event finished = queue->submit(
        std::move(flawed),
        minigpu::LaunchConfig(1, 8)
    );
    finished.wait();

    require(
        counter.load(0) == 1,
        "forced load/store interleaving should lose seven updates"
    );
    require(
        counter.rmw_operation_count() == 0,
        "plain atomic load/store was counted as an RMW"
    );
}

void test_fetch_add_returns_unique_tickets() {
    minigpu::Device device;
    auto queue = device.create_queue();
    minigpu::AtomicU32Buffer counter =
        device.allocate_atomic_u32(1);
    minigpu::AtomicU32Buffer tickets =
        device.allocate_atomic_u32(32);
    counter.store(0, 7);

    minigpu::Kernel kernel(
        "unique_tickets",
        [counter, tickets](minigpu::ThreadContext& context) {
            const std::uint32_t ticket =
                counter.fetch_add(0, 1);
            tickets.store(context.global_id(), ticket);
        }
    );

    minigpu::Event finished = queue->submit(
        std::move(kernel),
        minigpu::LaunchConfig(1, 32)
    );
    finished.wait();

    std::vector<std::uint32_t> observed;
    observed.reserve(32);

    for (std::size_t index = 0; index < 32; ++index) {
        observed.push_back(tickets.load(index));
    }

    std::sort(observed.begin(), observed.end());

    for (std::size_t index = 0; index < 32; ++index) {
        require(
            observed[index] ==
                static_cast<std::uint32_t>(index + 7),
            "fetch_add tickets were not a consecutive range"
        );
    }

    require(
        counter.load(0) == 39,
        "fetch_add final counter is incorrect"
    );
    require(
        counter.rmw_operation_count() == 32,
        "fetch_add instrumentation counted the wrong RMW total"
    );
}

void test_compare_exchange_has_one_winner() {
    minigpu::Device device;
    auto queue = device.create_queue();
    minigpu::AtomicU32Buffer winner =
        device.allocate_atomic_u32(1);
    minigpu::AtomicU32Buffer successes =
        device.allocate_atomic_u32(16);

    minigpu::Kernel kernel(
        "one_cas_winner",
        [winner, successes](minigpu::ThreadContext& context) {
            std::uint32_t expected = 0;
            const std::uint32_t desired =
                static_cast<std::uint32_t>(
                    context.local_id() + 1
                );

            const bool won = winner.compare_exchange(
                0,
                expected,
                desired
            );

            successes.store(
                context.global_id(),
                won ? 1U : 0U
            );
        }
    );

    minigpu::Event finished = queue->submit(
        std::move(kernel),
        minigpu::LaunchConfig(1, 16)
    );
    finished.wait();

    std::size_t success_count = 0;

    for (std::size_t index = 0; index < 16; ++index) {
        success_count += successes.load(index);
    }

    const std::uint32_t winning_value = winner.load(0);

    require(
        success_count == 1,
        "compare_exchange produced more or fewer than one winner"
    );
    require(
        winning_value >= 1 && winning_value <= 16,
        "compare_exchange wrote an invalid winner"
    );
    require(
        successes.load(winning_value - 1) == 1,
        "winner value does not identify the successful lane"
    );
}

void test_compare_exchange_failure_updates_expected() {
    minigpu::Device device;
    minigpu::AtomicU32Buffer value =
        device.allocate_atomic_u32(1);
    value.store(0, 9);

    std::uint32_t expected = 3;
    const bool exchanged = value.compare_exchange(
        0,
        expected,
        12
    );

    require(!exchanged, "CAS unexpectedly succeeded");
    require(
        expected == 9,
        "failed CAS did not update expected"
    );
    require(
        value.load(0) == 9,
        "failed CAS changed the stored value"
    );
    require(
        value.rmw_operation_count() == 1,
        "failed CAS was not counted as an RMW attempt"
    );
}

void test_shared_rmw_race_is_canonical_and_queue_recovers() {
    minigpu::Device device;
    auto queue = device.create_queue();

    minigpu::Kernel flawed(
        "shared_load_add_store",
        [](minigpu::ThreadContext& context) {
            if (context.local_id() == 0) {
                context.shared_store(0, 0.0f);
            }
            context.barrier(700);

            const float old_value = context.shared_load(0);
            context.barrier(701);

            context.shared_store(0, old_value + 1.0f);
            context.barrier(702);
        }
    );

    minigpu::Event failed = queue->submit(
        std::move(flawed),
        minigpu::LaunchConfig(1, 4, 1)
    );

    bool caught = false;

    try {
        failed.wait();
    } catch (const minigpu::SharedMemoryRaceError& error) {
        caught = true;
        require(error.group_id() == 0, "wrong race group");
        require(error.phase() == 2, "wrong race phase");
        require(error.index() == 0, "wrong race index");
        require(
            error.first_local_id() == 0 &&
                error.second_local_id() == 1,
            "race report was not canonical by lane ID"
        );
        require(
            error.first_access() ==
                    minigpu::SharedAccessKind::store &&
                error.second_access() ==
                    minigpu::SharedAccessKind::store,
            "race report has the wrong access kinds"
        );
    }

    require(caught, "shared RMW race was not reported");

    minigpu::Buffer output = device.allocate(1);
    minigpu::Event recovered = queue->submit(
        minigpu::fill(output, 1, 12.0f),
        minigpu::LaunchConfig(1, 1)
    );
    recovered.wait();

    require(
        output.load(0) == 12.0f,
        "queue did not recover after a race error"
    );
}

void test_shared_atomic_add_is_safe_and_returns_tickets() {
    minigpu::Device device;
    auto queue = device.create_queue();
    minigpu::Buffer tickets = device.allocate(64);
    minigpu::Buffer total = device.allocate(1);

    minigpu::Kernel kernel(
        "shared_atomic_tickets",
        [tickets, total](minigpu::ThreadContext& context) {
            if (context.local_id() == 0) {
                context.shared_store(0, 0.0f);
            }
            context.barrier(710);

            const float old_value =
                context.shared_atomic_add(0, 1.0f);
            tickets.store(context.global_id(), old_value);
            context.barrier(711);

            if (context.local_id() == 0) {
                total.store(0, context.shared_load(0));
            }
        }
    );

    minigpu::Event finished = queue->submit(
        std::move(kernel),
        minigpu::LaunchConfig(1, 64, 1)
    );
    finished.wait();

    std::vector<float> observed;
    observed.reserve(64);

    for (std::size_t index = 0; index < 64; ++index) {
        observed.push_back(tickets.load(index));
    }

    std::sort(observed.begin(), observed.end());

    for (std::size_t index = 0; index < 64; ++index) {
        require(
            observed[index] == static_cast<float>(index),
            "shared atomic tickets are not unique"
        );
    }

    require(
        total.load(0) == 64.0f,
        "shared atomic add lost an update"
    );
}

void test_direct_read_write_conflict_is_reported() {
    minigpu::Device device;
    auto queue = device.create_queue();

    minigpu::Kernel flawed(
        "direct_read_write_race",
        [](minigpu::ThreadContext& context) {
            if (context.local_id() == 0) {
                context.shared_store(0, 4.0f);
            }
            context.barrier(715);

            if (context.local_id() == 0) {
                static_cast<void>(context.shared_load(0));
            } else if (context.local_id() == 1) {
                context.shared_store(0, 5.0f);
            }

            context.barrier(716);
        }
    );

    minigpu::Event failed = queue->submit(
        std::move(flawed),
        minigpu::LaunchConfig(1, 4, 1)
    );

    bool caught = false;

    try {
        failed.wait();
    } catch (const minigpu::SharedMemoryRaceError& error) {
        caught = true;
        require(error.phase() == 1, "wrong R/W race phase");
        require(error.index() == 0, "wrong R/W race index");
        require(
            error.first_local_id() == 0 &&
                error.first_access() ==
                    minigpu::SharedAccessKind::load,
            "wrong first R/W race access"
        );
        require(
            error.second_local_id() == 1 &&
                error.second_access() ==
                    minigpu::SharedAccessKind::store,
            "wrong second R/W race access"
        );
    }

    require(caught, "direct read/write race was not reported");
}

void test_atomic_add_is_not_a_barrier() {
    minigpu::Device device;
    auto queue = device.create_queue();

    minigpu::Kernel flawed(
        "atomic_then_unsynchronized_read",
        [](minigpu::ThreadContext& context) {
            if (context.local_id() == 0) {
                context.shared_store(0, 0.0f);
            }
            context.barrier(720);

            context.shared_atomic_add(0, 1.0f);

            if (context.local_id() == 0) {
                static_cast<void>(context.shared_load(0));
            }
        }
    );

    minigpu::Event failed = queue->submit(
        std::move(flawed),
        minigpu::LaunchConfig(1, 4, 1)
    );

    require_throws<minigpu::SharedMemoryRaceError>(
        [&] {
            failed.wait();
        },
        "atomic updates incorrectly acted like a barrier"
    );
}

void test_mixed_atomic_and_ordinary_access_is_a_race() {
    minigpu::Device device;
    auto queue = device.create_queue();

    minigpu::Kernel flawed(
        "mixed_atomic_ordinary",
        [](minigpu::ThreadContext& context) {
            if (context.local_id() == 0) {
                context.shared_store(0, 0.0f);
            }
            context.barrier(730);

            if (context.local_id() == 0) {
                context.shared_atomic_add(0, 1.0f);
            } else if (context.local_id() == 1) {
                context.shared_store(0, 9.0f);
            }

            context.barrier(731);
        }
    );

    minigpu::Event failed = queue->submit(
        std::move(flawed),
        minigpu::LaunchConfig(1, 4, 1)
    );

    require_throws<minigpu::SharedMemoryRaceError>(
        [&] {
            failed.wait();
        },
        "mixed atomic/non-atomic access was not rejected"
    );
}

void test_legal_shared_access_patterns_do_not_false_positive() {
    minigpu::Device device;
    auto queue = device.create_queue();
    minigpu::Buffer output = device.allocate(1);

    minigpu::Kernel kernel(
        "legal_shared_patterns",
        [output](minigpu::ThreadContext& context) {
            if (context.local_id() == 0) {
                context.shared_store(0, 5.0f);
            }
            context.barrier(740);

            const float common = context.shared_load(0);
            context.barrier(741);

            context.shared_store(
                context.local_id(),
                common +
                    static_cast<float>(context.local_id())
            );
            context.barrier(742);

            if (context.local_id() == 0) {
                float sum = 0.0f;

                for (
                    std::size_t index = 0;
                    index < context.shared_size();
                    ++index
                ) {
                    sum += context.shared_load(index);
                }

                output.store(0, sum);
            }
        }
    );

    minigpu::Event finished = queue->submit(
        std::move(kernel),
        minigpu::LaunchConfig(1, 4, 4)
    );
    finished.wait();

    require(
        output.load(0) == 26.0f,
        "legal read/read or distinct-index access was rejected"
    );
}

void test_same_lane_load_store_is_program_ordered() {
    minigpu::Device device;
    auto queue = device.create_queue();
    minigpu::Buffer output = device.allocate(1);

    minigpu::Kernel kernel(
        "same_lane_accesses",
        [output](minigpu::ThreadContext& context) {
            if (context.local_id() == 0) {
                context.shared_store(0, 2.0f);
                const float old_value =
                    context.shared_load(0);
                context.shared_store(0, old_value + 3.0f);
                output.store(0, context.shared_load(0));
            }
        }
    );

    minigpu::Event finished = queue->submit(
        std::move(kernel),
        minigpu::LaunchConfig(1, 4, 1)
    );
    finished.wait();

    require(
        output.load(0) == 5.0f,
        "same-lane program order produced a false race"
    );
}

void test_final_phase_race_is_detected_without_barrier() {
    minigpu::Device device;
    auto queue = device.create_queue();

    minigpu::Kernel flawed(
        "racy_final_phase",
        [](minigpu::ThreadContext& context) {
            context.shared_store(
                0,
                static_cast<float>(context.local_id())
            );
        }
    );

    minigpu::Event failed = queue->submit(
        std::move(flawed),
        minigpu::LaunchConfig(1, 4, 1)
    );

    require_throws<minigpu::SharedMemoryRaceError>(
        [&] {
            failed.wait();
        },
        "race after the final barrier boundary was missed"
    );
}

void test_final_phase_atomic_access_is_safe() {
    minigpu::Device device;
    auto queue = device.create_queue();
    minigpu::Buffer tickets = device.allocate(4);

    minigpu::Kernel kernel(
        "atomic_final_phase",
        [tickets](minigpu::ThreadContext& context) {
            if (context.local_id() == 0) {
                context.shared_store(0, 0.0f);
            }
            context.barrier(745);

            tickets.store(
                context.global_id(),
                context.shared_atomic_add(0, 1.0f)
            );
        }
    );

    minigpu::Event finished = queue->submit(
        std::move(kernel),
        minigpu::LaunchConfig(1, 4, 1)
    );
    finished.wait();

    std::vector<float> observed;

    for (std::size_t index = 0; index < 4; ++index) {
        observed.push_back(tickets.load(index));
    }

    std::sort(observed.begin(), observed.end());
    require(
        observed == std::vector<float>{0, 1, 2, 3},
        "final atomic-only phase was rejected or lost an update"
    );
}

void test_shared_atomic_out_of_bounds_propagates_and_recovers() {
    minigpu::Device device;
    auto queue = device.create_queue();
    minigpu::AtomicU32Buffer counter =
        device.allocate_atomic_u32(1);

    minigpu::Kernel invalid(
        "atomic_oob",
        [](minigpu::ThreadContext& context) {
            if (context.local_id() == 0) {
                context.shared_atomic_add(1, 1.0f);
            } else {
                context.barrier(750);
            }
        }
    );

    minigpu::Event failed = queue->submit(
        std::move(invalid),
        minigpu::LaunchConfig(1, 4, 1)
    );

    require_throws<std::out_of_range>(
        [&] {
            failed.wait();
        },
        "atomic bounds error was not propagated"
    );

    minigpu::Event recovered = queue->submit(
        minigpu::Kernel(
            "valid_atomic",
            [counter](minigpu::ThreadContext& context) {
                if (context.global_id() == 0) {
                    counter.fetch_add(0, 3);
                }
            }
        ),
        minigpu::LaunchConfig(1, 1)
    );
    recovered.wait();

    require(
        counter.load(0) == 3,
        "queue did not recover after atomic bounds failure"
    );
}

void test_canonical_race_selection_is_repeatable() {
    minigpu::Device device;
    auto queue = device.create_queue();

    for (int repetition = 0; repetition < 25; ++repetition) {
        minigpu::Kernel flawed(
            "two_racy_indices",
            [](minigpu::ThreadContext& context) {
                context.shared_store(
                    1,
                    static_cast<float>(context.local_id())
                );
                context.shared_store(
                    0,
                    static_cast<float>(context.local_id())
                );
                context.barrier(755);
            }
        );

        minigpu::Event failed = queue->submit(
            std::move(flawed),
            minigpu::LaunchConfig(1, 4, 2)
        );

        bool caught = false;

        try {
            failed.wait();
        } catch (
            const minigpu::SharedMemoryRaceError& error
        ) {
            caught = true;
            require(error.phase() == 0, "unstable race phase");
            require(error.index() == 0, "race index was not canonical");
            require(
                error.first_local_id() == 0 &&
                    error.second_local_id() == 1,
                "race lane pair was not canonical"
            );
            require(
                error.first_access() ==
                        minigpu::SharedAccessKind::store &&
                    error.second_access() ==
                        minigpu::SharedAccessKind::store,
                "race access pair was not canonical"
            );
        }

        require(caught, "repeatable race was not reported");
    }
}

void test_hierarchical_counting_reduces_global_atomic_operations() {
    constexpr std::size_t group_count = 4;
    constexpr std::size_t group_size = 8;
    constexpr std::size_t total_lanes =
        group_count * group_size;

    minigpu::Device device;
    auto queue = device.create_queue();

    minigpu::AtomicU32Buffer naive =
        device.allocate_atomic_u32(1);
    minigpu::Event naive_finished = queue->submit(
        minigpu::Kernel(
            "naive_global_counter",
            [naive](minigpu::ThreadContext&) {
                naive.fetch_add(0, 1);
            }
        ),
        minigpu::LaunchConfig(group_count, group_size)
    );
    naive_finished.wait();

    minigpu::AtomicU32Buffer hierarchical =
        device.allocate_atomic_u32(1);
    minigpu::Event hierarchical_finished = queue->submit(
        minigpu::Kernel(
            "hierarchical_counter",
            [hierarchical](minigpu::ThreadContext& context) {
                if (context.local_id() == 0) {
                    context.shared_store(0, 0.0f);
                }
                context.barrier(760);

                context.shared_atomic_add(0, 1.0f);
                context.barrier(761);

                if (context.local_id() == 0) {
                    hierarchical.fetch_add(
                        0,
                        static_cast<std::uint32_t>(
                            context.shared_load(0)
                        )
                    );
                }
            }
        ),
        minigpu::LaunchConfig(
            group_count,
            group_size,
            1
        )
    );
    hierarchical_finished.wait();

    require(
        naive.load(0) == total_lanes &&
            hierarchical.load(0) == total_lanes,
        "hierarchical counter changed the mathematical result"
    );
    require(
        naive.rmw_operation_count() == total_lanes,
        "naive counter did not perform one global RMW per lane"
    );
    require(
        hierarchical.rmw_operation_count() == group_count,
        "hierarchical counter did not reduce global atomics"
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
        run_test(
            "atomic load/store pair loses updates",
            test_atomic_load_store_pair_loses_updates
        );
        run_test(
            "fetch_add returns unique tickets",
            test_fetch_add_returns_unique_tickets
        );
        run_test(
            "compare_exchange has one winner",
            test_compare_exchange_has_one_winner
        );
        run_test(
            "compare_exchange failure updates expected",
            test_compare_exchange_failure_updates_expected
        );
        run_test(
            "shared RMW race is canonical and queue recovers",
            test_shared_rmw_race_is_canonical_and_queue_recovers
        );
        run_test(
            "shared atomic add is safe and returns tickets",
            test_shared_atomic_add_is_safe_and_returns_tickets
        );
        run_test(
            "direct read/write conflict is reported",
            test_direct_read_write_conflict_is_reported
        );
        run_test(
            "atomic add is not a barrier",
            test_atomic_add_is_not_a_barrier
        );
        run_test(
            "mixed atomic and ordinary access is a race",
            test_mixed_atomic_and_ordinary_access_is_a_race
        );
        run_test(
            "legal shared accesses do not false-positive",
            test_legal_shared_access_patterns_do_not_false_positive
        );
        run_test(
            "same-lane access is program-ordered",
            test_same_lane_load_store_is_program_ordered
        );
        run_test(
            "final-phase race is detected",
            test_final_phase_race_is_detected_without_barrier
        );
        run_test(
            "final-phase atomic access is safe",
            test_final_phase_atomic_access_is_safe
        );
        run_test(
            "shared atomic bounds failure propagates and recovers",
            test_shared_atomic_out_of_bounds_propagates_and_recovers
        );
        run_test(
            "canonical race selection is repeatable",
            test_canonical_race_selection_is_repeatable
        );
        run_test(
            "hierarchical counting reduces global atomic operations",
            test_hierarchical_counting_reduces_global_atomic_operations
        );
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout
        << "All MiniGPU atomics and race tests passed\n";
    return 0;
}
