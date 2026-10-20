#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "minigpu.hpp"

#include <limits>
#include <sstream>
#include <utility>

namespace minigpu {

namespace {

std::string describe_error(NSError* error) {
    if (error == nil) {
        return "unknown Metal error";
    }

    const char* description =
        error.localizedDescription.UTF8String;

    return description == nullptr
        ? "unknown Metal error"
        : std::string(description);
}

void require_binding_index(std::size_t index) {
    if (
        index >
        static_cast<std::size_t>(
            std::numeric_limits<NSUInteger>::max()
        )
    ) {
        throw std::out_of_range(
            "binding index does not fit NSUInteger"
        );
    }
}

} // namespace

struct Buffer::Impl {
    Impl(
        id<MTLDevice> device_value,
        id<MTLBuffer> buffer_value,
        std::size_t element_count_value
    )
        : device(device_value),
          buffer(buffer_value),
          element_count(element_count_value) {}

    id<MTLDevice> device;
    id<MTLBuffer> buffer;
    std::size_t element_count;
};

struct Kernel::Impl {
    Impl(
        id<MTLDevice> device_value,
        id<MTLComputePipelineState> pipeline_value,
        std::string name_value
    )
        : device(device_value),
          pipeline(pipeline_value),
          name(std::move(name_value)) {}

    id<MTLDevice> device;
    id<MTLComputePipelineState> pipeline;
    std::string name;
};

struct Event::Impl {
    Impl(
        id<MTLCommandBuffer> command_buffer_value,
        Kernel kernel_value,
        std::vector<Buffer> buffers_value
    )
        : command_buffer(command_buffer_value),
          kernel(std::move(kernel_value)),
          buffers(std::move(buffers_value)) {}

    id<MTLCommandBuffer> command_buffer;

    // Retain the pipeline and buffers while this Event exists.
    Kernel kernel;
    std::vector<Buffer> buffers;
};

struct Queue::Impl {
    explicit Impl(id<MTLDevice> device_value)
        : device(device_value),
          queue([device_value newCommandQueue]) {
        if (queue == nil) {
            throw MetalError(
                "Metal command-queue creation failed"
            );
        }
    }

    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
};

struct Device::Impl {
    explicit Impl(id<MTLDevice> device_value)
        : device(device_value) {}

    id<MTLDevice> device;
};

Buffer::Buffer(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

std::size_t Buffer::size() const noexcept {
    return impl_->element_count;
}

bool Buffer::shares_storage_with(
    const Buffer& other
) const noexcept {
    return impl_ == other.impl_;
}

float Buffer::load(std::size_t index) const {
    if (index >= size()) {
        throw std::out_of_range(
            "Metal buffer load is out of bounds"
        );
    }

    const float* values =
        static_cast<const float*>(impl_->buffer.contents);

    return values[index];
}

void Buffer::store(
    std::size_t index,
    float value
) const {
    if (index >= size()) {
        throw std::out_of_range(
            "Metal buffer store is out of bounds"
        );
    }

    float* values =
        static_cast<float*>(impl_->buffer.contents);

    values[index] = value;
}

LaunchConfig::LaunchConfig(
    std::size_t groups,
    std::size_t threads_per_group
)
    : groups_(groups),
      threads_per_group_(threads_per_group) {
    if (threads_per_group_ == 0) {
        throw std::invalid_argument(
            "threads_per_group must be greater than zero"
        );
    }

    if (
        groups_ != 0 &&
        threads_per_group_ >
            std::numeric_limits<std::size_t>::max() /
                groups_
    ) {
        throw std::overflow_error(
            "launch size overflow"
        );
    }
}

LaunchConfig LaunchConfig::cover(
    std::size_t work_items,
    std::size_t threads_per_group
) {
    if (threads_per_group == 0) {
        throw std::invalid_argument(
            "threads_per_group must be greater than zero"
        );
    }

    const std::size_t groups =
        work_items == 0
            ? 0
            : 1 +
                (work_items - 1) /
                    threads_per_group;

    return LaunchConfig(groups, threads_per_group);
}

std::size_t LaunchConfig::groups() const noexcept {
    return groups_;
}

std::size_t
LaunchConfig::threads_per_group() const noexcept {
    return threads_per_group_;
}

std::size_t LaunchConfig::total_threads() const noexcept {
    return groups_ * threads_per_group_;
}

Kernel::Kernel(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

const std::string& Kernel::name() const noexcept {
    return impl_->name;
}

Dispatch::Dispatch(Kernel kernel)
    : kernel_(std::move(kernel)) {}

bool Dispatch::has_binding(
    std::size_t index
) const noexcept {
    for (const BufferArgument& argument : buffers_) {
        if (argument.index == index) {
            return true;
        }
    }

    for (const U32Argument& argument : u32_values_) {
        if (argument.index == index) {
            return true;
        }
    }

    for (const F32Argument& argument : f32_values_) {
        if (argument.index == index) {
            return true;
        }
    }

    return false;
}

Dispatch& Dispatch::bind_buffer(
    std::size_t index,
    Buffer buffer
) {
    require_binding_index(index);

    if (has_binding(index)) {
        throw std::invalid_argument(
            "a dispatch binding index was used twice"
        );
    }

    buffers_.push_back(BufferArgument{
        index,
        std::move(buffer),
    });

    return *this;
}

Dispatch& Dispatch::bind_u32(
    std::size_t index,
    std::uint32_t value
) {
    require_binding_index(index);

    if (has_binding(index)) {
        throw std::invalid_argument(
            "a dispatch binding index was used twice"
        );
    }

    u32_values_.push_back(U32Argument{
        index,
        value,
    });

    return *this;
}

Dispatch& Dispatch::bind_f32(
    std::size_t index,
    float value
) {
    require_binding_index(index);

    if (has_binding(index)) {
        throw std::invalid_argument(
            "a dispatch binding index was used twice"
        );
    }

    f32_values_.push_back(F32Argument{
        index,
        value,
    });

    return *this;
}

Event::Event(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

void Event::wait() const {
    [impl_->command_buffer waitUntilCompleted];

    if (
        impl_->command_buffer.status !=
        MTLCommandBufferStatusCompleted
    ) {
        std::ostringstream message;
        message
            << "Metal kernel '" << impl_->kernel.name()
            << "' failed: "
            << describe_error(
                   impl_->command_buffer.error
               );

        throw MetalError(message.str());
    }
}

bool Event::is_complete() const {
    const MTLCommandBufferStatus status =
        impl_->command_buffer.status;

    return
        status == MTLCommandBufferStatusCompleted ||
        status == MTLCommandBufferStatusError;
}

Queue::Queue(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

Queue::~Queue() = default;

Event Queue::submit(
    Dispatch dispatch,
    LaunchConfig launch
) {
    if (dispatch.kernel_.impl_->device != impl_->device) {
        throw std::invalid_argument(
            "kernel and queue belong to different devices"
        );
    }

    if (
        launch.threads_per_group() >
        dispatch.kernel_.impl_->
            pipeline.maxTotalThreadsPerThreadgroup
    ) {
        throw std::invalid_argument(
            "threads_per_group exceeds the kernel limit"
        );
    }

    for (
        const Dispatch::BufferArgument& argument :
        dispatch.buffers_
    ) {
        if (argument.buffer.impl_->device != impl_->device) {
            throw std::invalid_argument(
                "buffer and queue belong to different devices"
            );
        }
    }

    id<MTLCommandBuffer> command_buffer =
        [impl_->queue commandBuffer];

    if (command_buffer == nil) {
        throw MetalError(
            "Metal command-buffer creation failed"
        );
    }

    if (launch.groups() != 0) {
        id<MTLComputeCommandEncoder> encoder =
            [command_buffer computeCommandEncoder];

        if (encoder == nil) {
            throw MetalError(
                "Metal compute-encoder creation failed"
            );
        }

        [encoder setComputePipelineState:
                     dispatch.kernel_.impl_->pipeline];

        for (
            const Dispatch::BufferArgument& argument :
            dispatch.buffers_
        ) {
            [encoder
                setBuffer:argument.buffer.impl_->buffer
                   offset:0
                  atIndex:static_cast<NSUInteger>(
                              argument.index
                          )];
        }

        for (
            const Dispatch::U32Argument& argument :
            dispatch.u32_values_
        ) {
            [encoder
                setBytes:&argument.value
                  length:sizeof(argument.value)
                 atIndex:static_cast<NSUInteger>(
                             argument.index
                         )];
        }

        for (
            const Dispatch::F32Argument& argument :
            dispatch.f32_values_
        ) {
            [encoder
                setBytes:&argument.value
                  length:sizeof(argument.value)
                 atIndex:static_cast<NSUInteger>(
                             argument.index
                         )];
        }

        [encoder
            dispatchThreadgroups:MTLSizeMake(
                                     static_cast<NSUInteger>(
                                         launch.groups()
                                     ),
                                     1,
                                     1
                                 )
            threadsPerThreadgroup:MTLSizeMake(
                                      static_cast<NSUInteger>(
                                          launch
                                              .threads_per_group()
                                      ),
                                      1,
                                      1
                                  )];

        [encoder endEncoding];
    }

    std::vector<Buffer> retained_buffers;
    retained_buffers.reserve(dispatch.buffers_.size());

    for (
        Dispatch::BufferArgument& argument :
        dispatch.buffers_
    ) {
        retained_buffers.push_back(
            std::move(argument.buffer)
        );
    }

    auto event_impl = std::make_shared<Event::Impl>(
        command_buffer,
        std::move(dispatch.kernel_),
        std::move(retained_buffers)
    );

    [command_buffer commit];

    return Event(std::move(event_impl));
}

Device::Device() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    if (device == nil) {
        throw MetalError(
            "no Metal-capable GPU was found"
        );
    }

    impl_ = std::make_shared<Impl>(device);
}

std::string Device::name() const {
    const char* name = impl_->device.name.UTF8String;

    return name == nullptr
        ? "unnamed Metal device"
        : std::string(name);
}

Buffer Device::allocate(
    std::size_t element_count
) const {
    if (element_count == 0) {
        throw std::invalid_argument(
            "buffer element_count must be greater than zero"
        );
    }

    if (
        element_count >
        std::numeric_limits<NSUInteger>::max() /
            sizeof(float)
    ) {
        throw std::overflow_error(
            "Metal buffer size overflow"
        );
    }

    const NSUInteger byte_count =
        static_cast<NSUInteger>(
            element_count * sizeof(float)
        );

    if (byte_count > impl_->device.maxBufferLength) {
        throw std::length_error(
            "requested buffer exceeds the Metal device limit"
        );
    }

    id<MTLBuffer> buffer =
        [impl_->device
            newBufferWithLength:byte_count
                       options:
                           MTLResourceStorageModeShared];

    if (buffer == nil) {
        throw MetalError(
            "Metal buffer allocation failed"
        );
    }

    return Buffer(std::make_shared<Buffer::Impl>(
        impl_->device,
        buffer,
        element_count
    ));
}

Kernel Device::compile_kernel(
    std::string name,
    std::string_view source
) const {
    if (name.empty()) {
        throw std::invalid_argument(
            "kernel name must not be empty"
        );
    }

    if (source.empty()) {
        throw std::invalid_argument(
            "kernel source must not be empty"
        );
    }

    NSString* source_string =
        [[NSString alloc]
            initWithBytes:source.data()
                   length:source.size()
                 encoding:NSUTF8StringEncoding];

    if (source_string == nil) {
        throw std::invalid_argument(
            "kernel source is not valid UTF-8"
        );
    }

    NSError* error = nil;
    id<MTLLibrary> library =
        [impl_->device
            newLibraryWithSource:source_string
                         options:nil
                           error:&error];

    if (library == nil) {
        throw MetalError(
            "Metal source compilation failed: " +
            describe_error(error)
        );
    }

    NSString* function_name =
        [[NSString alloc]
            initWithBytes:name.data()
                   length:name.size()
                 encoding:NSUTF8StringEncoding];

    if (function_name == nil) {
        throw std::invalid_argument(
            "kernel name is not valid UTF-8"
        );
    }

    id<MTLFunction> function =
        [library newFunctionWithName:function_name];

    if (function == nil) {
        throw MetalError(
            "compiled library has no kernel named '" +
            name +
            "'"
        );
    }

    error = nil;
    id<MTLComputePipelineState> pipeline =
        [impl_->device
            newComputePipelineStateWithFunction:function
                                           error:&error];

    if (pipeline == nil) {
        throw MetalError(
            "Metal pipeline creation failed: " +
            describe_error(error)
        );
    }

    return Kernel(std::make_shared<Kernel::Impl>(
        impl_->device,
        pipeline,
        std::move(name)
    ));
}

std::unique_ptr<Queue>
Device::create_queue() const {
    return std::unique_ptr<Queue>(
        new Queue(std::make_unique<Queue::Impl>(
            impl_->device
        ))
    );
}

} // namespace minigpu
