#import <Foundation/Foundation.h>

#include "../lesson14-metal-minigpu/minigpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::string_view atomic_experiment_source = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void racy_increment(
    device uint* counter [[buffer(0)]],
    constant uint& increment_count [[buffer(1)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id >= increment_count) {
        return;
    }

    // Intentionally incorrect: the read, addition, and write are separate.
    const uint old_value = counter[0];
    counter[0] = old_value + 1u;
}

kernel void atomic_load_store_pair(
    device float* observed_result [[buffer(0)]],
    uint local_id [[thread_position_in_threadgroup]]
) {
    threadgroup atomic_uint counter;

    if (local_id == 0) {
        atomic_store_explicit(
            &counter,
            0u,
            memory_order_relaxed
        );
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Every operation is atomic, but the complete load/add/store sequence
    // is not one indivisible operation.
    const uint old_value = atomic_load_explicit(
        &counter,
        memory_order_relaxed
    );

    // Force every thread to finish loading zero before any thread stores.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    atomic_store_explicit(
        &counter,
        old_value + 1u,
        memory_order_relaxed
    );

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_id == 0) {
        observed_result[0] = float(
            atomic_load_explicit(
                &counter,
                memory_order_relaxed
            )
        );
    }
}

kernel void atomic_fetch_add_tickets(
    device atomic_uint* counter [[buffer(0)]],
    device float* tickets [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint global_id [[thread_position_in_grid]]
) {
    // There is no later group barrier, so padded threads may return.
    if (global_id >= element_count) {
        return;
    }

    const uint ticket = atomic_fetch_add_explicit(
        counter,
        1u,
        memory_order_relaxed
    );

    tickets[global_id] = float(ticket);
}

kernel void read_u32_counter(
    device const uint* counter [[buffer(0)]],
    device float* result [[buffer(1)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id == 0) {
        result[0] = float(counter[0]);
    }
}

kernel void read_atomic_counter(
    device atomic_uint* counter [[buffer(0)]],
    device float* result [[buffer(1)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id == 0) {
        result[0] = float(
            atomic_load_explicit(
                counter,
                memory_order_relaxed
            )
        );
    }
}
)METAL";

[[nodiscard]] bool equal_float(
    float actual,
    float expected
) {
    return std::fabs(actual - expected) < 0.00001f;
}

[[nodiscard]] float read_counter(
    minigpu::Queue& queue,
    const minigpu::Kernel& reader,
    minigpu::Buffer counter,
    minigpu::Buffer result
) {
    result.store(0, -1.0f);

    minigpu::Dispatch dispatch(reader);
    dispatch
        .bind_buffer(0, std::move(counter))
        .bind_buffer(1, result);

    minigpu::Event finished = queue.submit(
        std::move(dispatch),
        minigpu::LaunchConfig(1, 1)
    );
    finished.wait();

    return result.load(0);
}

} // namespace

int main() {
    @autoreleasepool {
        try {
            constexpr std::uint32_t racy_increment_count =
                65'536;
            constexpr std::size_t racy_threads_per_group = 256;
            constexpr std::size_t racy_trials = 5;

            constexpr std::uint32_t ticket_count = 1'000;
            constexpr std::size_t atomic_threads_per_group = 256;

            minigpu::Device device;
            auto queue = device.create_queue();

            // Metal buffers are untyped bytes. Lesson 14 allocates
            // four-byte slots, so this one slot can be viewed by MSL as a
            // uint or atomic_uint. Writing float zero initializes the same
            // four bytes to integer zero.
            minigpu::Buffer counter = device.allocate(1);
            minigpu::Buffer counter_result =
                device.allocate(1);

            minigpu::Kernel racy_increment =
                device.compile_kernel(
                    "racy_increment",
                    atomic_experiment_source
                );
            minigpu::Kernel atomic_load_store_pair =
                device.compile_kernel(
                    "atomic_load_store_pair",
                    atomic_experiment_source
                );
            minigpu::Kernel atomic_fetch_add_tickets =
                device.compile_kernel(
                    "atomic_fetch_add_tickets",
                    atomic_experiment_source
                );
            minigpu::Kernel read_u32 =
                device.compile_kernel(
                    "read_u32_counter",
                    atomic_experiment_source
                );
            minigpu::Kernel read_atomic =
                device.compile_kernel(
                    "read_atomic_counter",
                    atomic_experiment_source
                );

            std::cout
                << "Metal device: " << device.name()
                << "\n\nExperiment A: ordinary racy increment\n"
                << "Requested increments per trial: "
                << racy_increment_count << '\n';

            const minigpu::LaunchConfig racy_launch =
                minigpu::LaunchConfig::cover(
                    racy_increment_count,
                    racy_threads_per_group
                );

            std::vector<float> racy_observations;
            racy_observations.reserve(racy_trials);

            for (
                std::size_t trial = 0;
                trial < racy_trials;
                ++trial
            ) {
                counter.store(0, 0.0f);

                minigpu::Dispatch dispatch(racy_increment);
                dispatch
                    .bind_buffer(0, counter)
                    .bind_u32(1, racy_increment_count);

                minigpu::Event finished = queue->submit(
                    std::move(dispatch),
                    racy_launch
                );
                finished.wait();

                const float observed = read_counter(
                    *queue,
                    read_u32,
                    counter,
                    counter_result
                );
                racy_observations.push_back(observed);

                std::cout
                    << "trial " << trial + 1
                    << ": observed " << observed << '\n';
            }

            std::cout
                << "These values are observations, not guaranteed "
                   "answers.\n\n"
                << "Experiment B: atomic load + atomic store\n";

            minigpu::Buffer load_store_result =
                device.allocate(1);
            load_store_result.store(0, -1.0f);

            minigpu::Dispatch load_store_dispatch(
                atomic_load_store_pair
            );
            load_store_dispatch.bind_buffer(
                0,
                load_store_result
            );

            minigpu::Event load_store_finished =
                queue->submit(
                    std::move(load_store_dispatch),
                    minigpu::LaunchConfig(1, 64)
                );
            load_store_finished.wait();

            const float load_store_observed =
                load_store_result.load(0);

            std::cout
                << "64 threads attempted to increment.\n"
                << "Observed result: " << load_store_observed
                << "\nExpected from this forced schedule: 1\n\n"
                << "Experiment C: atomic fetch_add\n";

            counter.store(0, 0.0f);
            minigpu::Buffer tickets =
                device.allocate(ticket_count);

            for (
                std::size_t index = 0;
                index < ticket_count;
                ++index
            ) {
                tickets.store(index, -1.0f);
            }

            const minigpu::LaunchConfig atomic_launch =
                minigpu::LaunchConfig::cover(
                    ticket_count,
                    atomic_threads_per_group
                );

            minigpu::Dispatch atomic_dispatch(
                atomic_fetch_add_tickets
            );
            atomic_dispatch
                .bind_buffer(0, counter)
                .bind_buffer(1, tickets)
                .bind_u32(2, ticket_count);

            minigpu::Event atomic_finished =
                queue->submit(
                    std::move(atomic_dispatch),
                    atomic_launch
                );
            atomic_finished.wait();

            const float atomic_counter = read_counter(
                *queue,
                read_atomic,
                counter,
                counter_result
            );

            std::vector<float> sorted_tickets;
            sorted_tickets.reserve(ticket_count);

            for (
                std::size_t index = 0;
                index < ticket_count;
                ++index
            ) {
                sorted_tickets.push_back(
                    tickets.load(index)
                );
            }

            std::sort(
                sorted_tickets.begin(),
                sorted_tickets.end()
            );

            bool tickets_correct = true;

            for (
                std::size_t index = 0;
                index < ticket_count;
                ++index
            ) {
                if (
                    !equal_float(
                        sorted_tickets[index],
                        static_cast<float>(index)
                    )
                ) {
                    tickets_correct = false;
                    break;
                }
            }

            const bool correct =
                equal_float(load_store_observed, 1.0f) &&
                equal_float(
                    atomic_counter,
                    static_cast<float>(ticket_count)
                ) &&
                tickets_correct;

            std::cout
                << "Logical increments: " << ticket_count
                << '\n'
                << "Launch: " << atomic_launch.groups()
                << " groups x "
                << atomic_launch.threads_per_group()
                << " threads = "
                << atomic_launch.total_threads()
                << " launched threads\n"
                << "Final atomic counter: "
                << atomic_counter << '\n'
                << "Sorted ticket range: "
                << sorted_tickets.front() << " through "
                << sorted_tickets.back() << '\n';

            if (!correct) {
                std::cerr
                    << "\nFAIL: a guaranteed atomic result "
                       "was incorrect.\n";
                return 1;
            }

            std::cout
                << "\nPASS: the real GPU reproduced the "
                   "load/store lost update and preserved every "
                   "fetch_add update.\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr
                << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
}
