#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace minigpu {

class MetalError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class Buffer {
public:
    [[nodiscard]] std::size_t size() const noexcept;
    // The current runtime creates one Impl for each Metal allocation.
    [[nodiscard]] bool shares_storage_with(
        const Buffer& other
    ) const noexcept;
    [[nodiscard]] float load(std::size_t index) const;
    void store(std::size_t index, float value) const;

private:
    struct Impl;

    explicit Buffer(std::shared_ptr<Impl> impl);

    std::shared_ptr<Impl> impl_;

    friend class Device;
    friend class Queue;
};

class LaunchConfig {
public:
    LaunchConfig(
        std::size_t groups,
        std::size_t threads_per_group
    );

    [[nodiscard]] static LaunchConfig cover(
        std::size_t work_items,
        std::size_t threads_per_group
    );

    [[nodiscard]] std::size_t groups() const noexcept;
    [[nodiscard]] std::size_t threads_per_group() const noexcept;
    [[nodiscard]] std::size_t total_threads() const noexcept;

private:
    std::size_t groups_;
    std::size_t threads_per_group_;
};

class Kernel {
public:
    [[nodiscard]] const std::string& name() const noexcept;

private:
    struct Impl;

    explicit Kernel(std::shared_ptr<Impl> impl);

    std::shared_ptr<Impl> impl_;

    friend class Device;
    friend class Dispatch;
    friend class Queue;
};

class Dispatch {
public:
    explicit Dispatch(Kernel kernel);

    Dispatch(const Dispatch&) = delete;
    Dispatch& operator=(const Dispatch&) = delete;
    Dispatch(Dispatch&&) noexcept = default;
    Dispatch& operator=(Dispatch&&) noexcept = default;

    Dispatch& bind_buffer(
        std::size_t index,
        Buffer buffer
    );

    Dispatch& bind_u32(
        std::size_t index,
        std::uint32_t value
    );

    Dispatch& bind_f32(
        std::size_t index,
        float value
    );

private:
    struct BufferArgument {
        std::size_t index;
        Buffer buffer;
    };

    struct U32Argument {
        std::size_t index;
        std::uint32_t value;
    };

    struct F32Argument {
        std::size_t index;
        float value;
    };

    [[nodiscard]] bool has_binding(
        std::size_t index
    ) const noexcept;

    Kernel kernel_;
    std::vector<BufferArgument> buffers_;
    std::vector<U32Argument> u32_values_;
    std::vector<F32Argument> f32_values_;

    friend class Queue;
};

class Event {
public:
    void wait() const;
    [[nodiscard]] bool is_complete() const;

private:
    struct Impl;

    explicit Event(std::shared_ptr<Impl> impl);

    std::shared_ptr<Impl> impl_;

    friend class Queue;
};

class Queue {
public:
    ~Queue();

    Queue(const Queue&) = delete;
    Queue& operator=(const Queue&) = delete;

    Event submit(
        Dispatch dispatch,
        LaunchConfig launch
    );

private:
    struct Impl;

    explicit Queue(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;

    friend class Device;
};

class Device {
public:
    Device();

    [[nodiscard]] std::string name() const;

    [[nodiscard]] Buffer allocate(
        std::size_t element_count
    ) const;

    [[nodiscard]] Kernel compile_kernel(
        std::string name,
        std::string_view source
    ) const;

    [[nodiscard]] std::unique_ptr<Queue>
    create_queue() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace minigpu
