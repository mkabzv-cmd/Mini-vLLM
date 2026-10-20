#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>

namespace {

const char* error_message(NSError* error) {
    return error == nil
        ? "unknown error"
        : error.localizedDescription.UTF8String;
}

template <std::size_t Size>
void print_values(
    const char* name,
    const std::array<float, Size>& values
) {
    std::cout << name << " = [";

    for (std::size_t index = 0; index < Size; ++index) {
        if (index != 0) {
            std::cout << ", ";
        }

        std::cout << values[index];
    }

    std::cout << "]\n";
}

} // namespace

int main() {
    @autoreleasepool {
        // This Objective-C++ code runs on the CPU.
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        if (device == nil) {
            std::cerr << "No Metal-capable GPU was found.\n";
            return 1;
        }

        // This Metal Shading Language code will run on the GPU.
        NSString* metal_source = @R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& element_count [[buffer(3)]],
    uint global_id [[thread_position_in_grid]]
) {
    // We deliberately launch 8 threads for 5 elements.
    if (global_id >= element_count) {
        return;
    }

    output[global_id] = a[global_id] + b[global_id];
}
)METAL";

        // Compile the GPU source at runtime.
        NSError* error = nil;
        id<MTLLibrary> library =
            [device newLibraryWithSource:metal_source
                                 options:nil
                                   error:&error];

        if (library == nil) {
            std::cerr
                << "Metal source compilation failed: "
                << error_message(error)
                << '\n';
            return 2;
        }

        id<MTLFunction> function =
            [library newFunctionWithName:@"vector_add"];

        if (function == nil) {
            std::cerr
                << "The compiled library has no vector_add kernel.\n";
            return 3;
        }

        error = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function
                                                   error:&error];

        if (pipeline == nil) {
            std::cerr
                << "Compute-pipeline creation failed: "
                << error_message(error)
                << '\n';
            return 4;
        }

        constexpr std::uint32_t element_count = 5;
        constexpr NSUInteger threads_per_group = 4;
        constexpr std::size_t array_size = element_count;

        const std::array<float, array_size> input_a{
            1.0f,
            2.0f,
            3.0f,
            4.0f,
            5.0f,
        };

        const std::array<float, array_size> input_b{
            10.0f,
            20.0f,
            30.0f,
            40.0f,
            50.0f,
        };

        const std::array<float, array_size> expected{
            11.0f,
            22.0f,
            33.0f,
            44.0f,
            55.0f,
        };

        const NSUInteger byte_count =
            sizeof(float) * element_count;

        // Shared storage is accessible to both the CPU and Apple GPU.
        id<MTLBuffer> a_buffer =
            [device newBufferWithLength:byte_count
                                options:MTLResourceStorageModeShared];

        id<MTLBuffer> b_buffer =
            [device newBufferWithLength:byte_count
                                options:MTLResourceStorageModeShared];

        id<MTLBuffer> output_buffer =
            [device newBufferWithLength:byte_count
                                options:MTLResourceStorageModeShared];

        if (
            a_buffer == nil ||
            b_buffer == nil ||
            output_buffer == nil
        ) {
            std::cerr << "Metal buffer allocation failed.\n";
            return 5;
        }

        float* gpu_a = static_cast<float*>(a_buffer.contents);
        float* gpu_b = static_cast<float*>(b_buffer.contents);
        float* gpu_output =
            static_cast<float*>(output_buffer.contents);

        for (std::size_t index = 0; index < array_size; ++index) {
            gpu_a[index] = input_a[index];
            gpu_b[index] = input_b[index];
            gpu_output[index] = -1.0f;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];

        if (queue == nil) {
            std::cerr << "Metal command-queue creation failed.\n";
            return 6;
        }

        // A command buffer is a batch of recorded GPU commands.
        // It is not a data-storage MTLBuffer.
        id<MTLCommandBuffer> command_buffer =
            [queue commandBuffer];

        if (command_buffer == nil) {
            std::cerr << "Metal command-buffer creation failed.\n";
            return 7;
        }

        id<MTLComputeCommandEncoder> encoder =
            [command_buffer computeCommandEncoder];

        if (encoder == nil) {
            std::cerr << "Metal compute-encoder creation failed.\n";
            return 8;
        }

        if (
            threads_per_group >
            pipeline.maxTotalThreadsPerThreadgroup
        ) {
            std::cerr
                << "The requested threadgroup is too large.\n";
            return 9;
        }

        // These indexes must match [[buffer(0)]], [[buffer(1)]], ...
        // in the GPU kernel.
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        [encoder setBuffer:b_buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer offset:0 atIndex:2];
        [encoder setBytes:&element_count
                   length:sizeof(element_count)
                  atIndex:3];

        const NSUInteger group_count =
            (element_count + threads_per_group - 1) /
            threads_per_group;

        [encoder
            dispatchThreadgroups:MTLSizeMake(group_count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(
                                      threads_per_group,
                                      1,
                                      1
                                  )];

        [encoder endEncoding];

        // commit() submits asynchronously. The CPU must wait before
        // reading output_buffer.
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (
            command_buffer.status !=
            MTLCommandBufferStatusCompleted
        ) {
            std::cerr
                << "GPU execution failed: "
                << error_message(command_buffer.error)
                << '\n';
            return 10;
        }

        std::array<float, array_size> result{};
        bool correct = true;

        for (std::size_t index = 0; index < array_size; ++index) {
            result[index] = gpu_output[index];

            if (
                std::fabs(result[index] - expected[index]) >
                0.00001f
            ) {
                correct = false;
            }
        }

        std::cout << "Metal device: "
                  << device.name.UTF8String
                  << '\n';
        std::cout << "Elements: " << element_count << '\n';
        std::cout << "Groups: " << group_count << '\n';
        std::cout
            << "Threads per group: "
            << threads_per_group
            << '\n';
        std::cout
            << "Total launched threads: "
            << group_count * threads_per_group
            << '\n';

        print_values("a     ", input_a);
        print_values("b     ", input_b);
        print_values("output", result);

        if (!correct) {
            print_values("expect", expected);
            std::cerr << "FAIL: GPU result was incorrect.\n";
            return 11;
        }

        std::cout
            << "PASS: vector_add executed correctly on the GPU.\n";
        return 0;
    }
}
