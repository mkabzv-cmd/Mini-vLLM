#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace minigpu {

inline constexpr std::size_t kMaxSimulatedThreadsPerGroup = 64;

namespace detail {
struct EventState;
struct GroupExecutor;
}

class DivergentBarrierError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class SharedAccessKind {
    load,
    store,
    atomic_add,
};

class SharedMemoryRaceError : public std::runtime_error {
public:
    SharedMemoryRaceError(
        std::size_t group_id,
        std::size_t phase,
        std::size_t index,
        std::size_t first_local_id,
        SharedAccessKind first_access,
        std::size_t second_local_id,
        SharedAccessKind second_access
    );

    [[nodiscard]] std::size_t group_id() const noexcept;
    [[nodiscard]] std::size_t phase() const noexcept;
    [[nodiscard]] std::size_t index() const noexcept;
    [[nodiscard]] std::size_t first_local_id() const noexcept;
    [[nodiscard]] SharedAccessKind first_access() const noexcept;
    [[nodiscard]] std::size_t second_local_id() const noexcept;
    [[nodiscard]] SharedAccessKind second_access() const noexcept;

private:
    std::size_t group_id_;
    std::size_t phase_;
    std::size_t index_;
    std::size_t first_local_id_;
    SharedAccessKind first_access_;
    std::size_t second_local_id_;
    SharedAccessKind second_access_;
};

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

// An integer buffer whose individual operations are atomic across every
// simulated thread that shares the storage. The CPU backend currently gives
// these operations one simple, strongest-order total order.
class AtomicU32Buffer {
public:
    explicit AtomicU32Buffer(std::size_t element_count);

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::uint32_t load(std::size_t index) const;
    void store(std::size_t index, std::uint32_t value) const;

    // Returns the value that existed before delta was added.
    std::uint32_t fetch_add(
        std::size_t index,
        std::uint32_t delta
    ) const;

    // On failure, expected is replaced with the value that was observed.
    [[nodiscard]] bool compare_exchange(
        std::size_t index,
        std::uint32_t& expected,
        std::uint32_t desired
    ) const;

    [[nodiscard]] std::size_t rmw_operation_count() const;

private:
    struct Storage;
    std::shared_ptr<Storage> storage_;
};

struct ThreadId {
    std::size_t global;
    std::size_t group;
    std::size_t local;
};

class ThreadContext {
public:
    ThreadContext(const ThreadContext&) = delete;
    ThreadContext& operator=(const ThreadContext&) = delete;
    ThreadContext(ThreadContext&&) = delete;
    ThreadContext& operator=(ThreadContext&&) = delete;

    [[nodiscard]] const ThreadId& id() const noexcept;
    [[nodiscard]] std::size_t global_id() const noexcept;
    [[nodiscard]] std::size_t group_id() const noexcept;
    [[nodiscard]] std::size_t local_id() const noexcept;
    [[nodiscard]] std::size_t threads_per_group() const noexcept;
    [[nodiscard]] bool in_bounds(std::size_t count) const noexcept;

    [[nodiscard]] std::size_t shared_size() const noexcept;
    [[nodiscard]] float shared_load(std::size_t index) const;
    void shared_store(std::size_t index, float value);
    float shared_atomic_add(std::size_t index, float delta);

    // site_id is teaching instrumentation. All threads must provide the
    // same site ID at the same dynamic barrier.
    void barrier(std::size_t site_id);

private:
    struct RuntimeState;

    ThreadContext(ThreadId id, RuntimeState* runtime);

    ThreadId id_;
    RuntimeState* runtime_;

    friend struct detail::GroupExecutor;
};

class LaunchConfig {
public:
    LaunchConfig(
        std::size_t groups,
        std::size_t threads_per_group,
        std::size_t shared_elements = 0
    );

    [[nodiscard]] static LaunchConfig cover(
        std::size_t work_items,
        std::size_t threads_per_group,
        std::size_t shared_elements = 0
    );

    [[nodiscard]] std::size_t groups() const noexcept;
    [[nodiscard]] std::size_t threads_per_group() const noexcept;
    [[nodiscard]] std::size_t shared_elements() const noexcept;
    [[nodiscard]] std::size_t total_threads() const noexcept;

private:
    std::size_t groups_;
    std::size_t threads_per_group_;
    std::size_t shared_elements_;
};

class Kernel {
public:
    using Body = std::function<void(ThreadContext&)>;

    Kernel(std::string name, Body body);

    void run(ThreadContext& context) const;
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
    [[nodiscard]] AtomicU32Buffer allocate_atomic_u32(
        std::size_t element_count
    ) const;
    [[nodiscard]] std::unique_ptr<Queue> create_queue() const;
};

struct ReductionPass {
    Buffer output;
    Event completion;
    std::size_t output_count;
};

struct ReductionResult {
    float value;
    std::size_t passes;
    std::vector<std::size_t> element_counts;
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

[[nodiscard]] ReductionPass enqueue_reduce_sum_pass(
    Queue& queue,
    Buffer input,
    std::size_t element_count,
    std::size_t threads_per_group
);

// This beginner helper waits after every pass. That makes failure
// propagation explicit and prevents a later pass from consuming partial
// output from a failed earlier pass.
[[nodiscard]] ReductionResult reduce_sum(
    Queue& queue,
    Buffer input,
    std::size_t element_count,
    std::size_t threads_per_group
);

} // namespace minigpu
