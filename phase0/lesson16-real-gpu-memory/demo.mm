#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view memory_lesson_source = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void affine_transform(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint global_id [[thread_position_in_grid]]
) {
    if (global_id >= element_count) {
        return;
    }

    output[global_id] =
        input[global_id] * 3.0f + 1.0f;
}
)METAL";

[[nodiscard]] std::string describe_error(
    NSError* error
) {
    if (error == nil) {
        return "unknown Metal error";
    }

    const char* description =
        error.localizedDescription.UTF8String;

    return description == nullptr
        ? "unknown Metal error"
        : std::string(description);
}

[[nodiscard]] id<MTLBuffer> allocate_buffer(
    id<MTLDevice> device,
    NSUInteger byte_count,
    MTLResourceOptions options,
    NSString* label
) {
    id<MTLBuffer> buffer =
        [device
            newBufferWithLength:byte_count
                       options:options];

    if (buffer == nil) {
        throw std::runtime_error(
            "Metal buffer allocation failed"
        );
    }

    buffer.label = label;
    return buffer;
}

[[nodiscard]] id<MTLComputePipelineState>
compile_pipeline(id<MTLDevice> device) {
    NSString* source =
        [[NSString alloc]
            initWithBytes:memory_lesson_source.data()
                   length:memory_lesson_source.size()
                 encoding:NSUTF8StringEncoding];

    if (source == nil) {
        throw std::runtime_error(
            "Metal source is not valid UTF-8"
        );
    }

    NSError* error = nil;
    id<MTLLibrary> library =
        [device newLibraryWithSource:source
                            options:nil
                              error:&error];

    if (library == nil) {
        throw std::runtime_error(
            "Metal source compilation failed: " +
            describe_error(error)
        );
    }

    id<MTLFunction> function =
        [library newFunctionWithName:@"affine_transform"];

    if (function == nil) {
        throw std::runtime_error(
            "compiled library has no affine_transform kernel"
        );
    }

    error = nil;
    id<MTLComputePipelineState> pipeline =
        [device
            newComputePipelineStateWithFunction:function
                                           error:&error];

    if (pipeline == nil) {
        throw std::runtime_error(
            "Metal pipeline creation failed: " +
            describe_error(error)
        );
    }

    return pipeline;
}

void require_shared(
    id<MTLBuffer> buffer,
    std::string_view name
) {
    if (buffer.storageMode != MTLStorageModeShared) {
        throw std::logic_error(
            std::string(name) +
            " is not a shared-storage buffer"
        );
    }

    if (buffer.contents == nullptr) {
        throw std::runtime_error(
            std::string(name) +
            " has no CPU mapping"
        );
    }
}

void write_values(
    id<MTLBuffer> buffer,
    const std::vector<float>& values,
    std::string_view name
) {
    require_shared(buffer, name);

    if (
        values.size() >
        static_cast<std::size_t>(
            buffer.length / sizeof(float)
        )
    ) {
        throw std::out_of_range(
            std::string(name) +
            " is too small for the CPU write"
        );
    }

    float* destination =
        static_cast<float*>(buffer.contents);

    for (
        std::size_t index = 0;
        index < values.size();
        ++index
    ) {
        destination[index] = values[index];
    }
}

[[nodiscard]] std::vector<float> read_values(
    id<MTLBuffer> buffer,
    std::size_t element_count,
    std::string_view name
) {
    require_shared(buffer, name);

    if (
        element_count >
        static_cast<std::size_t>(
            buffer.length / sizeof(float)
        )
    ) {
        throw std::out_of_range(
            std::string(name) +
            " is too small for the CPU read"
        );
    }

    const float* source =
        static_cast<const float*>(buffer.contents);

    return std::vector<float>(
        source,
        source + element_count
    );
}

void encode_compute(
    id<MTLCommandBuffer> command_buffer,
    id<MTLComputePipelineState> pipeline,
    id<MTLBuffer> input,
    id<MTLBuffer> output,
    std::uint32_t element_count,
    NSUInteger threads_per_group
) {
    if (element_count == 0) {
        throw std::invalid_argument(
            "compute element_count must be greater than zero"
        );
    }

    if (threads_per_group == 0) {
        throw std::invalid_argument(
            "threads_per_group must be greater than zero"
        );
    }

    if (
        threads_per_group >
        pipeline.maxTotalThreadsPerThreadgroup
    ) {
        throw std::invalid_argument(
            "threads_per_group exceeds the kernel limit"
        );
    }

    const NSUInteger required_byte_count =
        static_cast<NSUInteger>(element_count) *
        sizeof(float);

    if (
        input.length < required_byte_count ||
        output.length < required_byte_count
    ) {
        throw std::out_of_range(
            "compute buffer is smaller than element_count"
        );
    }

    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];

    if (encoder == nil) {
        throw std::runtime_error(
            "Metal compute-encoder creation failed"
        );
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder
        setBytes:&element_count
          length:sizeof(element_count)
         atIndex:2];

    const NSUInteger group_count =
        (
            static_cast<NSUInteger>(element_count) +
            threads_per_group -
            1
        ) /
        threads_per_group;

    [encoder
        dispatchThreadgroups:MTLSizeMake(
                                 group_count,
                                 1,
                                 1
                             )
        threadsPerThreadgroup:MTLSizeMake(
                                  threads_per_group,
                                  1,
                                  1
                              )];

    [encoder endEncoding];
}

void encode_copy(
    id<MTLCommandBuffer> command_buffer,
    id<MTLBuffer> source,
    id<MTLBuffer> destination,
    NSUInteger byte_count
) {
    if (
        byte_count > source.length ||
        byte_count > destination.length
    ) {
        throw std::out_of_range(
            "Metal blit copy exceeds a buffer"
        );
    }

    id<MTLBlitCommandEncoder> encoder =
        [command_buffer blitCommandEncoder];

    if (encoder == nil) {
        throw std::runtime_error(
            "Metal blit-encoder creation failed"
        );
    }

    [encoder
        copyFromBuffer:source
           sourceOffset:0
               toBuffer:destination
      destinationOffset:0
                   size:byte_count];

    [encoder endEncoding];
}

void commit_and_wait(
    id<MTLCommandBuffer> command_buffer,
    std::string_view label
) {
    [command_buffer commit];

    // The command runs asynchronously after commit. CPU access to shared
    // resources participating in this command must wait for a safe handoff.
    [command_buffer waitUntilCompleted];

    if (
        command_buffer.status !=
        MTLCommandBufferStatusCompleted
    ) {
        throw std::runtime_error(
            std::string(label) +
            " failed: " +
            describe_error(command_buffer.error)
        );
    }
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

} // namespace

int main() {
    @autoreleasepool {
        try {
            constexpr std::uint32_t element_count = 8;
            constexpr NSUInteger threads_per_group = 4;
            constexpr NSUInteger byte_count =
                element_count * sizeof(float);

            const std::vector<float> input_values{
                1.0f,
                2.0f,
                3.0f,
                4.0f,
                5.0f,
                6.0f,
                7.0f,
                8.0f,
            };
            const std::vector<float> expected{
                4.0f,
                7.0f,
                10.0f,
                13.0f,
                16.0f,
                19.0f,
                22.0f,
                25.0f,
            };
            const std::vector<float> sentinel(
                element_count,
                -1.0f
            );

            id<MTLDevice> device =
                MTLCreateSystemDefaultDevice();

            if (device == nil) {
                throw std::runtime_error(
                    "no Metal-capable GPU was found"
                );
            }

            id<MTLCommandQueue> queue =
                [device newCommandQueue];

            if (queue == nil) {
                throw std::runtime_error(
                    "Metal command-queue creation failed"
                );
            }

            id<MTLComputePipelineState> pipeline =
                compile_pipeline(device);

            std::cout
                << "Metal device: "
                << device.name.UTF8String
                << '\n'
                << "Unified physical memory: "
                << (device.hasUnifiedMemory ? "yes" : "no")
                << "\n\n"
                << "Experiment A: shared storage\n";

            id<MTLBuffer> shared_input =
                allocate_buffer(
                    device,
                    byte_count,
                    MTLResourceStorageModeShared,
                    @"shared input"
                );
            id<MTLBuffer> shared_output =
                allocate_buffer(
                    device,
                    byte_count,
                    MTLResourceStorageModeShared,
                    @"shared output"
                );

            write_values(
                shared_input,
                input_values,
                "shared input"
            );
            write_values(
                shared_output,
                sentinel,
                "shared output"
            );

            const std::vector<float> shared_before =
                read_values(
                    shared_output,
                    element_count,
                    "shared output"
                );

            id<MTLCommandBuffer> shared_commands =
                [queue commandBuffer];

            if (shared_commands == nil) {
                throw std::runtime_error(
                    "shared command-buffer creation failed"
                );
            }

            encode_compute(
                shared_commands,
                pipeline,
                shared_input,
                shared_output,
                element_count,
                threads_per_group
            );

            // Do not read shared_output after commit until this returns.
            commit_and_wait(
                shared_commands,
                "shared-storage compute"
            );

            const std::vector<float> shared_after =
                read_values(
                    shared_output,
                    element_count,
                    "shared output"
                );

            std::cout
                << "CPU direct access: yes\n"
                << "Before submission: ";
            print_values(shared_before);
            std::cout << "\nAfter completion: ";
            print_values(shared_after);
            std::cout
                << "\nPath: CPU write -> GPU compute -> wait "
                   "-> CPU read\n\n"
                << "Experiment B: private storage with staging\n";

            id<MTLBuffer> upload_staging =
                allocate_buffer(
                    device,
                    byte_count,
                    MTLResourceStorageModeShared,
                    @"upload staging"
                );
            id<MTLBuffer> private_input =
                allocate_buffer(
                    device,
                    byte_count,
                    MTLResourceStorageModePrivate,
                    @"private input"
                );
            id<MTLBuffer> private_output =
                allocate_buffer(
                    device,
                    byte_count,
                    MTLResourceStorageModePrivate,
                    @"private output"
                );
            id<MTLBuffer> readback_staging =
                allocate_buffer(
                    device,
                    byte_count,
                    MTLResourceStorageModeShared,
                    @"readback staging"
                );

            if (
                private_input.storageMode !=
                    MTLStorageModePrivate ||
                private_output.storageMode !=
                    MTLStorageModePrivate
            ) {
                throw std::runtime_error(
                    "private allocation has the wrong storage mode"
                );
            }

            write_values(
                upload_staging,
                input_values,
                "upload staging"
            );
            write_values(
                readback_staging,
                sentinel,
                "readback staging"
            );

            const std::vector<float> private_before =
                read_values(
                    readback_staging,
                    element_count,
                    "readback staging"
                );

            id<MTLCommandBuffer> private_commands =
                [queue commandBuffer];

            if (private_commands == nil) {
                throw std::runtime_error(
                    "private command-buffer creation failed"
                );
            }

            // One serial GPU command list:
            // shared upload -> private input -> private output -> shared
            // readback.
            encode_copy(
                private_commands,
                upload_staging,
                private_input,
                byte_count
            );
            encode_compute(
                private_commands,
                pipeline,
                private_input,
                private_output,
                element_count,
                threads_per_group
            );
            encode_copy(
                private_commands,
                private_output,
                readback_staging,
                byte_count
            );

            // No CPU wait is needed between those GPU commands. We wait once
            // at the end before touching upload_staging or readback_staging.
            commit_and_wait(
                private_commands,
                "private-storage command chain"
            );

            const std::vector<float> private_after =
                read_values(
                    readback_staging,
                    element_count,
                    "readback staging"
                );

            std::cout
                << "Private input CPU direct access: no\n"
                << "Private output CPU direct access: no\n"
                << "Readback before submission: ";
            print_values(private_before);
            std::cout << "\nReadback after completion: ";
            print_values(private_after);
            std::cout
                << "\nPath: shared upload -> blit -> private "
                   "compute -> blit -> shared readback\n\n";

            const bool correct =
                shared_input.storageMode ==
                    MTLStorageModeShared &&
                shared_output.storageMode ==
                    MTLStorageModeShared &&
                upload_staging.storageMode ==
                    MTLStorageModeShared &&
                readback_staging.storageMode ==
                    MTLStorageModeShared &&
                equal_values(shared_before, sentinel) &&
                equal_values(private_before, sentinel) &&
                equal_values(shared_after, expected) &&
                equal_values(private_after, expected) &&
                equal_values(shared_after, private_after);

            if (!correct) {
                std::cerr
                    << "FAIL: a memory path produced incorrect "
                       "results.\n";
                return 1;
            }

            std::cout
                << "PASS: shared access and private staging "
                   "produced identical GPU results.\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr
                << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
}
