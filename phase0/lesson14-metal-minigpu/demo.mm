#import <Foundation/Foundation.h>

#include "minigpu.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <utility>

namespace {

constexpr std::string_view vector_add_source = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& element_count [[buffer(3)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id >= element_count) {
        return;
    }

    output[global_id] = a[global_id] + b[global_id];
}
)METAL";

} // namespace

int main() {
    @autoreleasepool {
        try {
            constexpr std::uint32_t element_count = 5;
            constexpr std::size_t threads_per_group = 4;

            minigpu::Device device;
            auto queue = device.create_queue();

            minigpu::Buffer a =
                device.allocate(element_count);
            minigpu::Buffer b =
                device.allocate(element_count);
            minigpu::Buffer output =
                device.allocate(element_count);

            const std::array<float, element_count> a_values{
                1.0f,
                2.0f,
                3.0f,
                4.0f,
                5.0f,
            };

            const std::array<float, element_count> b_values{
                10.0f,
                20.0f,
                30.0f,
                40.0f,
                50.0f,
            };

            const std::array<float, element_count> expected{
                11.0f,
                22.0f,
                33.0f,
                44.0f,
                55.0f,
            };

            for (
                std::size_t index = 0;
                index < element_count;
                ++index
            ) {
                a.store(index, a_values[index]);
                b.store(index, b_values[index]);
                output.store(index, -1.0f);
            }

            minigpu::Kernel vector_add =
                device.compile_kernel(
                    "vector_add",
                    vector_add_source
                );

            minigpu::Dispatch dispatch(vector_add);
            dispatch
                .bind_buffer(0, a)
                .bind_buffer(1, b)
                .bind_buffer(2, output)
                .bind_u32(3, element_count);

            const minigpu::LaunchConfig launch =
                minigpu::LaunchConfig::cover(
                    element_count,
                    threads_per_group
                );

            minigpu::Event finished = queue->submit(
                std::move(dispatch),
                launch
            );

            finished.wait();

            if (!finished.is_complete()) {
                std::cerr
                    << "FAIL: Event did not become complete.\n";
                return 1;
            }

            bool correct = true;

            std::cout
                << "Metal device: " << device.name() << '\n';
            std::cout
                << "Launch: "
                << launch.groups()
                << " groups x "
                << launch.threads_per_group()
                << " threads = "
                << launch.total_threads()
                << " launched threads\n";
            std::cout << "Output: [";

            for (
                std::size_t index = 0;
                index < element_count;
                ++index
            ) {
                const float actual = output.load(index);

                if (index != 0) {
                    std::cout << ", ";
                }

                std::cout << actual;

                if (
                    std::fabs(actual - expected[index]) >
                    0.00001f
                ) {
                    correct = false;
                }
            }

            std::cout << "]\n";

            if (!correct) {
                std::cerr
                    << "FAIL: vector_add result is incorrect.\n";
                return 1;
            }

            std::cout
                << "PASS: reusable MiniGPU API ran "
                   "vector_add on the real GPU.\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr
                << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
}
