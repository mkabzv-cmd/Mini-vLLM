#import <Foundation/Foundation.h>

#include "../lesson14-metal-minigpu/minigpu.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <utility>

namespace {

constexpr std::string_view inspect_launch_source = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void inspect_launch(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* global_ids [[buffer(2)]],
    device float* group_ids [[buffer(3)]],
    device float* local_ids [[buffer(4)]],
    device float* active_flags [[buffer(5)]],
    constant uint& element_count [[buffer(6)]],
    uint global_id [[thread_position_in_grid]],
    uint group_id [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]]
) {
    // These diagnostic buffers have one element for every launched thread,
    // including padded threads.
    global_ids[global_id] = float(global_id);
    group_ids[global_id] = float(group_id);
    local_ids[global_id] = float(local_id);

    const bool active = global_id < element_count;
    active_flags[global_id] = active ? 1.0f : 0.0f;

    // input and output contain only element_count values. A padded thread
    // must return before it uses global_id to access those smaller buffers.
    if (!active) {
        return;
    }

    output[global_id] = input[global_id] * 2.0f;
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

            const std::size_t launched_threads =
                launch.total_threads();
            const std::size_t padded_threads =
                launched_threads - element_count;

            // These contain only real tensor elements.
            minigpu::Buffer input =
                device.allocate(element_count);
            minigpu::Buffer output =
                device.allocate(element_count);

            // These diagnostic buffers contain one slot for every launched
            // thread, including padded threads.
            minigpu::Buffer global_ids =
                device.allocate(launched_threads);
            minigpu::Buffer group_ids =
                device.allocate(launched_threads);
            minigpu::Buffer local_ids =
                device.allocate(launched_threads);
            minigpu::Buffer active_flags =
                device.allocate(launched_threads);

            for (
                std::size_t index = 0;
                index < element_count;
                ++index
            ) {
                input.store(
                    index,
                    static_cast<float>(index + 1)
                );
                output.store(index, -1.0f);
            }

            for (
                std::size_t index = 0;
                index < launched_threads;
                ++index
            ) {
                global_ids.store(index, -1.0f);
                group_ids.store(index, -1.0f);
                local_ids.store(index, -1.0f);
                active_flags.store(index, -1.0f);
            }

            minigpu::Kernel inspect_launch =
                device.compile_kernel(
                    "inspect_launch",
                    inspect_launch_source
                );

            minigpu::Dispatch dispatch(inspect_launch);
            dispatch
                .bind_buffer(0, input)
                .bind_buffer(1, output)
                .bind_buffer(2, global_ids)
                .bind_buffer(3, group_ids)
                .bind_buffer(4, local_ids)
                .bind_buffer(5, active_flags)
                .bind_u32(6, element_count);

            minigpu::Event finished = queue->submit(
                std::move(dispatch),
                launch
            );
            finished.wait();

            bool correct =
                launch.groups() == 3 &&
                launched_threads == 12 &&
                padded_threads == 2;

            std::cout
                << "Metal device: " << device.name() << '\n'
                << "Logical elements: " << element_count << '\n'
                << "Launch: " << launch.groups()
                << " groups x " << threads_per_group
                << " threads = " << launched_threads
                << " launched threads\n"
                << "Padded threads: " << padded_threads
                << "\n\n";

            std::cout
                << std::left
                << std::setw(10) << "global"
                << std::setw(10) << "group"
                << std::setw(10) << "local"
                << std::setw(12) << "status"
                << "output\n";

            for (
                std::size_t index = 0;
                index < launched_threads;
                ++index
            ) {
                const float actual_global =
                    global_ids.load(index);
                const float actual_group =
                    group_ids.load(index);
                const float actual_local =
                    local_ids.load(index);
                const float actual_active =
                    active_flags.load(index);

                const std::size_t expected_group =
                    index / threads_per_group;
                const std::size_t expected_local =
                    index % threads_per_group;
                const bool expected_active =
                    index < element_count;

                correct =
                    correct &&
                    equal_float(
                        actual_global,
                        static_cast<float>(index)
                    ) &&
                    equal_float(
                        actual_group,
                        static_cast<float>(expected_group)
                    ) &&
                    equal_float(
                        actual_local,
                        static_cast<float>(expected_local)
                    ) &&
                    equal_float(
                        actual_active,
                        expected_active ? 1.0f : 0.0f
                    ) &&
                    index ==
                        expected_group * threads_per_group +
                            expected_local;

                std::cout
                    << std::setw(10)
                    << static_cast<std::size_t>(
                           actual_global
                       )
                    << std::setw(10)
                    << static_cast<std::size_t>(
                           actual_group
                       )
                    << std::setw(10)
                    << static_cast<std::size_t>(
                           actual_local
                       )
                    << std::setw(12)
                    << (
                           expected_active
                               ? "active"
                               : "padded"
                       );

                if (expected_active) {
                    const float actual_output =
                        output.load(index);
                    const float expected_output =
                        static_cast<float>(
                            (index + 1) * 2
                        );

                    correct =
                        correct &&
                        equal_float(
                            actual_output,
                            expected_output
                        );

                    std::cout << actual_output;
                } else {
                    std::cout << "-";
                }

                std::cout << '\n';
            }

            if (!correct) {
                std::cerr
                    << "\nFAIL: launch geometry or output "
                       "was incorrect.\n";
                return 1;
            }

            std::cout
                << "\nPASS: real Metal thread IDs and padded "
                   "threads matched the expected geometry.\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr
                << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
}
