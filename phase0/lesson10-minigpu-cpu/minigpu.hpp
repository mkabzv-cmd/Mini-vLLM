#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace minigpu {

namespace detail {
struct EventState;
}

class Buffer {
public:
    explicit Buffer(std::size_t element_count);

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] float load(std::size_t index) const;
    void store(std::size_t index, float value) const;

private:
    struct Storage;
    std::shared_ptr<Storage> storage_;
};

struct ThreadId {
    std::size_t global;
    std::size_t group;
    std::size_t local;
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
    using Body = std::function<void(const ThreadId&)>;

    Kernel(std::string name, Body body);

    void run(const ThreadId& id) const;
    [[nodiscard]] const std::string& name() const noexcept;

private:
    std::string name_;
    Body body_;
};

class Event {
public:
    void wait() const;
    [[nodiscard]] bool is_complete() const;

private:
    explicit Event(std::shared_ptr<detail::EventState> state);

    std::shared_ptr<detail::EventState> state_;

    friend class Queue;
};

class Queue {
public:
    Queue();
    ~Queue();

    Queue(const Queue&) = delete;
    Queue& operator=(const Queue&) = delete;

    Event submit(Kernel kernel, LaunchConfig launch);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class Device {
public:
    [[nodiscard]] Buffer allocate(std::size_t element_count) const;
    [[nodiscard]] std::unique_ptr<Queue> create_queue() const;
};

[[nodiscard]] Kernel vector_add(
    Buffer a,
    Buffer b,
    Buffer output,
    std::size_t element_count
);

[[nodiscard]] Kernel fill(
    Buffer output,
    std::size_t element_count,
    float value
);

[[nodiscard]] Kernel increment(
    Buffer output,
    std::size_t element_count,
    float amount
);

} // namespace minigpu
