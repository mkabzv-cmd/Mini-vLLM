#import <Foundation/Foundation.h>

#include "../lesson14-metal-minigpu/minigpu.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::string_view group_reduce_source = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void group_reduce_sum(
    device const float* input [[buffer(0)]],
    device float* partial_sums [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint global_id [[thread_position_in_grid]],
    uint group_id [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]]
) {
    // Every threadgroup receives its own separate four-float scratch array.
    threadgroup float scratch[4];

    // Padded threads contribute the additive identity, zero. They must not
    // return because every thread still needs to reach every barrier below.
    scratch[local_id] =
        global_id < element_count
            ? input[global_id]
            : 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_id < 2) {
        scratch[local_id] += scratch[local_id + 2];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_id == 0) {
        scratch[0] += scratch[1];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_id == 0) {
        partial_sums[group_id] = scratch[0];
    }
}
)METAL";

[[nodiscard]] bool equal_float(
    float actual,
    float expected
) {
    return std::fabs(actual - expected) < 0.00001f;
}

} // namespace

int main() {
    @autoreleasepool {
        try {
            constexpr std::uint32_t element_count = 10;
            constexpr std::size_t threads_per_group = 4;

            minigpu::Device device;
            auto queue = device.create_queue();

            const minigpu::LaunchConfig launch =
                minigpu::LaunchConfig::cover(
                    element_count,
                    threads_per_group
                );

            minigpu::Buffer input =
                device.allocate(element_count);
            minigpu::Buffer partial_sums =
                device.allocate(launch.groups());

            for (
                std::size_t index = 0;
                index < element_count;
                ++index
            ) {
                input.store(
                    index,
                    static_cast<float>(index + 1)
                );
            }

            for (
                std::size_t group = 0;
                group < launch.groups();
                ++group
            ) {
                partial_sums.store(group, -1.0f);
            }

            minigpu::Kernel group_reduce =
                device.compile_kernel(
                    "group_reduce_sum",
                    group_reduce_source
                );

            minigpu::Dispatch dispatch(group_reduce);
            dispatch
                .bind_buffer(0, input)
                .bind_buffer(1, partial_sums)
                .bind_u32(2, element_count);

            minigpu::Event finished = queue->submit(
                std::move(dispatch),
                launch
            );
            finished.wait();

            const std::vector<float> expected{
                10.0f,
                26.0f,
                19.0f,
            };

            bool correct =
                launch.groups() == expected.size() &&
                launch.total_threads() == 12;
            float verified_total = 0.0f;

            std::cout
                << "Metal device: " << device.name() << '\n'
                << "Input elements: " << element_count << '\n'
                << "Launch: " << launch.groups()
                << " groups x " << threads_per_group
                << " threads = " << launch.total_threads()
                << " launched threads\n\n";

            for (
                std::size_t group = 0;
                group < launch.groups();
                ++group
            ) {
                const std::size_t first_global =
                    group * threads_per_group;
                const std::size_t last_global =
                    first_global + threads_per_group - 1;
                const float actual =
                    partial_sums.load(group);

                verified_total += actual;
                correct =
                    correct &&
                    equal_float(actual, expected.at(group));

                std::cout
                    << "group " << group
                    << " covers global IDs "
                    << first_global << "-" << last_global
                    << " -> partial sum " << actual
                    << '\n';
            }

            correct =
                correct &&
                equal_float(verified_total, 55.0f);

            std::cout
                << "\nPartial sums: ["
                << partial_sums.load(0) << ", "
                << partial_sums.load(1) << ", "
                << partial_sums.load(2) << "]\n"
                << "CPU verification total: "
                << verified_total << '\n';

            if (!correct) {
                std::cerr
                    << "FAIL: real threadgroup reduction "
                       "produced incorrect partial sums.\n";
                return 1;
            }

            std::cout
                << "PASS: real threadgroup memory and barriers "
                   "produced [10, 26, 19].\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr
                << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
}
