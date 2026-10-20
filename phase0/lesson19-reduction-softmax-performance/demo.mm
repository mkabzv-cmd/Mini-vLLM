#import <Foundation/Foundation.h>

#include "../lesson14-metal-minigpu/minigpu.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kMaximumThreadsPerGroup = 256;
constexpr std::size_t kSoftmaxThreads = 256;

constexpr std::string_view lesson19_source = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void reduce_sum_pass(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint global_id [[thread_position_in_grid]],
    uint group_id [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_width [[threads_per_threadgroup]]
) {
    threadgroup float scratch[256];

    scratch[local_id] =
        global_id < element_count
            ? input[global_id]
            : 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (
        uint stride = group_width / 2u;
        stride > 0u;
        stride >>= 1u
    ) {
        if (local_id < stride) {
            scratch[local_id] +=
                scratch[local_id + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local_id == 0) {
        output[group_id] = scratch[0];
    }
}

kernel void serial_sum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& element_count [[buffer(2)]]
) {
    float total = 0.0f;

    for (uint index = 0; index < element_count; ++index) {
        total += input[index];
    }

    output[0] = total;
}

kernel void stable_softmax_one_group(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_width [[threads_per_threadgroup]]
) {
    threadgroup float maximum_scratch[256];
    threadgroup float sum_scratch[256];

    // Each thread handles local_id, local_id + group_width, and so on.
    float lane_maximum = -3.402823466e+38f;

    for (
        uint index = local_id;
        index < element_count;
        index += group_width
    ) {
        lane_maximum =
            max(lane_maximum, input[index]);
    }

    maximum_scratch[local_id] = lane_maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (
        uint stride = group_width / 2u;
        stride > 0u;
        stride >>= 1u
    ) {
        if (local_id < stride) {
            maximum_scratch[local_id] = max(
                maximum_scratch[local_id],
                maximum_scratch[local_id + stride]
            );
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float maximum = maximum_scratch[0];
    float lane_sum = 0.0f;

    for (
        uint index = local_id;
        index < element_count;
        index += group_width
    ) {
        const float unnormalized =
            exp(input[index] - maximum);

        output[index] = unnormalized;
        lane_sum += unnormalized;
    }

    sum_scratch[local_id] = lane_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (
        uint stride = group_width / 2u;
        stride > 0u;
        stride >>= 1u
    ) {
        if (local_id < stride) {
            sum_scratch[local_id] +=
                sum_scratch[local_id + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inverse_sum = 1.0f / sum_scratch[0];

    for (
        uint index = local_id;
        index < element_count;
        index += group_width
    ) {
        output[index] *= inverse_sum;
    }
}
)METAL";

struct ReductionWorkspace {
    std::uint32_t initial_count;
    std::size_t threads_per_group;
    std::vector<std::uint32_t> pass_input_counts;
    std::vector<minigpu::Buffer> outputs;
};

struct SoftmaxReference {
    std::vector<float> probabilities;
};

[[nodiscard]] bool is_power_of_two(
    std::size_t value
) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

void validate_group_size(std::size_t threads_per_group) {
    if (
        !is_power_of_two(threads_per_group) ||
        threads_per_group < 2 ||
        threads_per_group > kMaximumThreadsPerGroup
    ) {
        throw std::invalid_argument(
            "reduction group size must be a power of two "
            "between 2 and 256"
        );
    }
}

[[nodiscard]] ReductionWorkspace make_workspace(
    const minigpu::Device& device,
    std::uint32_t element_count,
    std::size_t threads_per_group
) {
    if (element_count == 0) {
        throw std::invalid_argument(
            "reduction requires at least one element"
        );
    }

    validate_group_size(threads_per_group);

    ReductionWorkspace workspace{
        element_count,
        threads_per_group,
        {},
        {},
    };

    std::uint32_t current_count = element_count;

    while (current_count > 1) {
        const minigpu::LaunchConfig launch =
            minigpu::LaunchConfig::cover(
                current_count,
                threads_per_group
            );

        if (
            launch.groups() >
            std::numeric_limits<std::uint32_t>::max()
        ) {
            throw std::overflow_error(
                "reduction partial count does not fit uint32"
            );
        }

        workspace.pass_input_counts.push_back(
            current_count
        );
        workspace.outputs.push_back(
            device.allocate(launch.groups())
        );

        current_count =
            static_cast<std::uint32_t>(
                launch.groups()
            );
    }

    return workspace;
}

[[nodiscard]] minigpu::Buffer run_reduction(
    minigpu::Queue& queue,
    const minigpu::Kernel& kernel,
    minigpu::Buffer input,
    std::uint32_t element_count,
    const ReductionWorkspace& workspace,
    std::vector<minigpu::Event>& events
) {
    if (
        workspace.initial_count != element_count ||
        workspace.pass_input_counts.size() !=
            workspace.outputs.size()
    ) {
        throw std::invalid_argument(
            "reduction workspace does not match the input"
        );
    }

    events.clear();
    minigpu::Buffer current = std::move(input);
    std::uint32_t current_count = element_count;

    for (
        std::size_t pass = 0;
        pass < workspace.outputs.size();
        ++pass
    ) {
        if (
            workspace.pass_input_counts[pass] !=
            current_count
        ) {
            throw std::logic_error(
                "reduction workspace pass count is inconsistent"
            );
        }

        minigpu::Dispatch dispatch(kernel);
        dispatch
            .bind_buffer(0, current)
            .bind_buffer(1, workspace.outputs[pass])
            .bind_u32(2, current_count);

        events.push_back(
            queue.submit(
                std::move(dispatch),
                minigpu::LaunchConfig::cover(
                    current_count,
                    workspace.threads_per_group
                )
            )
        );

        current = workspace.outputs[pass];

        if (
            current.size() >
            std::numeric_limits<std::uint32_t>::max()
        ) {
            throw std::overflow_error(
                "reduction output count does not fit uint32"
            );
        }

        current_count =
            static_cast<std::uint32_t>(
                current.size()
            );
    }

    if (!events.empty()) {
        // Commands submitted to the same queue execute in order. Waiting
        // for the last pass therefore waits for the complete dependency
        // chain. The following completed waits also surface per-pass errors.
        events.back().wait();

        for (const minigpu::Event& event : events) {
            event.wait();
        }
    }

    return current;
}

void run_serial_sum(
    minigpu::Queue& queue,
    const minigpu::Kernel& kernel,
    minigpu::Buffer input,
    minigpu::Buffer output,
    std::uint32_t element_count
) {
    minigpu::Dispatch dispatch(kernel);
    dispatch
        .bind_buffer(0, std::move(input))
        .bind_buffer(1, std::move(output))
        .bind_u32(2, element_count);

    minigpu::Event finished = queue.submit(
        std::move(dispatch),
        minigpu::LaunchConfig(1, 1)
    );
    finished.wait();
}

void run_softmax(
    minigpu::Queue& queue,
    const minigpu::Kernel& kernel,
    minigpu::Buffer input,
    minigpu::Buffer output,
    std::uint32_t element_count
) {
    minigpu::Dispatch dispatch(kernel);
    dispatch
        .bind_buffer(0, std::move(input))
        .bind_buffer(1, std::move(output))
        .bind_u32(2, element_count);

    minigpu::Event finished = queue.submit(
        std::move(dispatch),
        minigpu::LaunchConfig(1, kSoftmaxThreads)
    );
    finished.wait();
}

[[nodiscard]] SoftmaxReference reference_softmax(
    const std::vector<float>& input
) {
    const float maximum =
        *std::max_element(input.begin(), input.end());

    std::vector<double> exponentials(input.size());
    double denominator = 0.0;

    for (
        std::size_t index = 0;
        index < input.size();
        ++index
    ) {
        exponentials[index] = std::exp(
            static_cast<double>(input[index]) -
            static_cast<double>(maximum)
        );
        denominator += exponentials[index];
    }

    SoftmaxReference reference;
    reference.probabilities.resize(input.size());

    for (
        std::size_t index = 0;
        index < input.size();
        ++index
    ) {
        reference.probabilities[index] =
            static_cast<float>(
                exponentials[index] / denominator
            );
    }

    return reference;
}

template <typename Function>
[[nodiscard]] double time_once_ms(Function&& function) {
    const auto start =
        std::chrono::steady_clock::now();

    function();

    const auto finish =
        std::chrono::steady_clock::now();

    return std::chrono::duration<double, std::milli>(
        finish - start
    ).count();
}

[[nodiscard]] double median(
    std::vector<double> values
) {
    if (values.empty()) {
        throw std::invalid_argument(
            "cannot calculate an empty median"
        );
    }

    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

template <typename Function>
[[nodiscard]] double measure_median_ms(
    Function&& function,
    std::size_t warmup_count,
    std::size_t sample_count
) {
    for (
        std::size_t index = 0;
        index < warmup_count;
        ++index
    ) {
        function();
    }

    std::vector<double> samples;
    samples.reserve(sample_count);

    for (
        std::size_t index = 0;
        index < sample_count;
        ++index
    ) {
        samples.push_back(
            time_once_ms(function)
        );
    }

    return median(std::move(samples));
}

void print_reduction_shape(
    const ReductionWorkspace& workspace
) {
    std::cout << workspace.initial_count;

    for (const minigpu::Buffer& output : workspace.outputs) {
        std::cout << " -> " << output.size();
    }

    std::cout << '\n';
}

[[nodiscard]] bool equal_float(
    float actual,
    float expected,
    float tolerance = 0.00001f
) {
    return std::fabs(actual - expected) <= tolerance;
}

} // namespace

int main() {
    @autoreleasepool {
        try {
            minigpu::Device device;
            auto queue = device.create_queue();

            minigpu::Kernel reduce_sum =
                device.compile_kernel(
                    "reduce_sum_pass",
                    lesson19_source
                );
            minigpu::Kernel serial_sum =
                device.compile_kernel(
                    "serial_sum",
                    lesson19_source
                );
            minigpu::Kernel stable_softmax =
                device.compile_kernel(
                    "stable_softmax_one_group",
                    lesson19_source
                );

            std::cout
                << "Metal device: " << device.name()
                << "\n\nPart A: complete multi-pass GPU reduction\n";

            constexpr std::uint32_t small_count = 10;
            constexpr std::size_t small_group_size = 4;

            minigpu::Buffer small_input =
                device.allocate(small_count);

            for (
                std::size_t index = 0;
                index < small_count;
                ++index
            ) {
                small_input.store(
                    index,
                    static_cast<float>(index + 1)
                );
            }

            const ReductionWorkspace small_workspace =
                make_workspace(
                    device,
                    small_count,
                    small_group_size
                );
            std::vector<minigpu::Event> small_events;
            small_events.reserve(
                small_workspace.outputs.size()
            );

            minigpu::Buffer small_result =
                run_reduction(
                    *queue,
                    reduce_sum,
                    small_input,
                    small_count,
                    small_workspace,
                    small_events
                );

            const float small_sum = small_result.load(0);

            std::cout << "Reduction shape: ";
            print_reduction_shape(small_workspace);
            std::cout
                << "GPU result: " << small_sum
                << "\nExpected: 55\n\n";

            std::cout
                << "Part B: numerically stable softmax\n";

            constexpr std::uint32_t softmax_count = 4'096;

            minigpu::Buffer softmax_input =
                device.allocate(softmax_count);
            minigpu::Buffer softmax_output =
                device.allocate(softmax_count);
            std::vector<float> softmax_values(
                softmax_count
            );

            for (
                std::size_t index = 0;
                index < softmax_count;
                ++index
            ) {
                const int centered =
                    static_cast<int>(index % 257) - 128;
                const float value =
                    1'000.0f +
                    static_cast<float>(centered) *
                        0.03125f;

                softmax_values[index] = value;
                softmax_input.store(index, value);
                softmax_output.store(index, -1.0f);
            }

            const SoftmaxReference reference =
                reference_softmax(softmax_values);

            run_softmax(
                *queue,
                stable_softmax,
                softmax_input,
                softmax_output,
                softmax_count
            );

            double probability_sum = 0.0;
            float maximum_error = 0.0f;
            bool probabilities_valid = true;

            for (
                std::size_t index = 0;
                index < softmax_count;
                ++index
            ) {
                const float actual =
                    softmax_output.load(index);
                const float expected =
                    reference.probabilities[index];

                probability_sum += actual;
                maximum_error = std::max(
                    maximum_error,
                    std::fabs(actual - expected)
                );
                probabilities_valid =
                    probabilities_valid &&
                    std::isfinite(actual) &&
                    actual >= 0.0f;
            }

            std::cout
                << "Logit range: approximately 996 to 1004\n"
                << "Probability sum: "
                << probability_sum << '\n'
                << "Maximum CPU/GPU absolute error: "
                << maximum_error << "\n\n";

            std::cout
                << "Part C: end-to-end latency experiment\n";

            constexpr std::uint32_t benchmark_count =
                1u << 20u;
            constexpr std::size_t benchmark_group_size =
                256;
            constexpr std::size_t warmups = 5;
            constexpr std::size_t samples = 21;

            minigpu::Buffer benchmark_input =
                device.allocate(benchmark_count);
            minigpu::Buffer serial_output =
                device.allocate(1);

            for (
                std::size_t index = 0;
                index < benchmark_count;
                ++index
            ) {
                benchmark_input.store(index, 1.0f);
            }

            const ReductionWorkspace benchmark_workspace =
                make_workspace(
                    device,
                    benchmark_count,
                    benchmark_group_size
                );
            std::vector<minigpu::Event> benchmark_events;
            benchmark_events.reserve(
                benchmark_workspace.outputs.size()
            );

            const auto run_serial = [&] {
                run_serial_sum(
                    *queue,
                    serial_sum,
                    benchmark_input,
                    serial_output,
                    benchmark_count
                );
            };

            const auto run_parallel = [&] {
                static_cast<void>(
                    run_reduction(
                        *queue,
                        reduce_sum,
                        benchmark_input,
                        benchmark_count,
                        benchmark_workspace,
                        benchmark_events
                    )
                );
            };

            for (
                std::size_t index = 0;
                index < warmups;
                ++index
            ) {
                if (index % 2 == 0) {
                    run_serial();
                    run_parallel();
                } else {
                    run_parallel();
                    run_serial();
                }
            }

            std::vector<double> serial_samples;
            std::vector<double> parallel_samples;
            serial_samples.reserve(samples);
            parallel_samples.reserve(samples);

            for (
                std::size_t index = 0;
                index < samples;
                ++index
            ) {
                if (index % 2 == 0) {
                    serial_samples.push_back(
                        time_once_ms(run_serial)
                    );
                    parallel_samples.push_back(
                        time_once_ms(run_parallel)
                    );
                } else {
                    parallel_samples.push_back(
                        time_once_ms(run_parallel)
                    );
                    serial_samples.push_back(
                        time_once_ms(run_serial)
                    );
                }
            }

            const double serial_median =
                median(std::move(serial_samples));
            const double parallel_median =
                median(std::move(parallel_samples));

            const double softmax_median =
                measure_median_ms(
                    [&] {
                        run_softmax(
                            *queue,
                            stable_softmax,
                            softmax_input,
                            softmax_output,
                            softmax_count
                        );
                    },
                    warmups,
                    samples
                );

            run_serial();
            minigpu::Buffer benchmark_result =
                run_reduction(
                    *queue,
                    reduce_sum,
                    benchmark_input,
                    benchmark_count,
                    benchmark_workspace,
                    benchmark_events
                );

            const float serial_value =
                serial_output.load(0);
            const float parallel_value =
                benchmark_result.load(0);

            const bool correct =
                equal_float(small_sum, 55.0f) &&
                probabilities_valid &&
                std::fabs(probability_sum - 1.0) <
                    0.0005 &&
                maximum_error < 0.00005f &&
                equal_float(
                    serial_value,
                    static_cast<float>(benchmark_count)
                ) &&
                equal_float(
                    parallel_value,
                    static_cast<float>(benchmark_count)
                );

            std::cout
                << "Reduction shape: ";
            print_reduction_shape(benchmark_workspace);

            std::cout
                << std::fixed << std::setprecision(4)
                << "Serial GPU sum median: "
                << serial_median << " ms\n"
                << "Parallel multi-pass median: "
                << parallel_median << " ms\n"
                << "Serial / parallel latency ratio: "
                << serial_median / parallel_median
                << "x\n"
                << "One-group softmax median (4096 values): "
                << softmax_median << " ms\n"
                << "Both reduction results: "
                << serial_value << " and "
                << parallel_value << '\n';

            std::cout
                << "\nTiming includes command encoding, submission, "
                   "GPU execution and Event::wait().\n";

            if (!correct) {
                std::cerr
                    << "FAIL: reduction, softmax or benchmark "
                       "verification failed.\n";
                return 1;
            }

            std::cout
                << "PASS: real multi-pass reduction, stable "
                   "softmax and timing checks succeeded.\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr
                << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
}
