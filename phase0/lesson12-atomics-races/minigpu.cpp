#include "minigpu.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace minigpu {

namespace {

[[nodiscard]] const char* access_kind_name(
    SharedAccessKind kind
) noexcept {
    switch (kind) {
    case SharedAccessKind::load:
        return "load";
    case SharedAccessKind::store:
        return "store";
    case SharedAccessKind::atomic_add:
        return "atomic_add";
    }

    return "unknown";
}

[[nodiscard]] std::string make_race_message(
    std::size_t group_id,
    std::size_t phase,
    std::size_t index,
    std::size_t first_local_id,
    SharedAccessKind first_access,
    std::size_t second_local_id,
    SharedAccessKind second_access
) {
    std::ostringstream message;
    message
        << "shared-memory race in group " << group_id
        << ", phase " << phase
        << ", index " << index
        << ": local thread " << first_local_id
        << ' ' << access_kind_name(first_access)
        << " conflicts with local thread "
        << second_local_id << ' '
        << access_kind_name(second_access);
    return message.str();
}

} // namespace

SharedMemoryRaceError::SharedMemoryRaceError(
    std::size_t group_id,
    std::size_t phase,
    std::size_t index,
    std::size_t first_local_id,
    SharedAccessKind first_access,
    std::size_t second_local_id,
    SharedAccessKind second_access
)
    : std::runtime_error(make_race_message(
          group_id,
          phase,
          index,
          first_local_id,
          first_access,
          second_local_id,
          second_access
      )),
      group_id_(group_id),
      phase_(phase),
      index_(index),
      first_local_id_(first_local_id),
      first_access_(first_access),
      second_local_id_(second_local_id),
      second_access_(second_access) {}

std::size_t SharedMemoryRaceError::group_id() const noexcept {
    return group_id_;
}

std::size_t SharedMemoryRaceError::phase() const noexcept {
    return phase_;
}

std::size_t SharedMemoryRaceError::index() const noexcept {
    return index_;
}

std::size_t SharedMemoryRaceError::first_local_id() const noexcept {
    return first_local_id_;
}

SharedAccessKind SharedMemoryRaceError::first_access() const noexcept {
    return first_access_;
}

std::size_t SharedMemoryRaceError::second_local_id() const noexcept {
    return second_local_id_;
}

SharedAccessKind SharedMemoryRaceError::second_access() const noexcept {
    return second_access_;
}

struct Buffer::Storage {
    explicit Storage(std::size_t element_count)
        : values(element_count) {}

    mutable std::mutex mutex;
    std::vector<float> values;
};

Buffer::Buffer(std::size_t element_count)
    : storage_(std::make_shared<Storage>(element_count)) {}

std::size_t Buffer::size() const noexcept {
    return storage_->values.size();
}

float Buffer::load(std::size_t index) const {
    std::lock_guard lock(storage_->mutex);
    return storage_->values.at(index);
}

void Buffer::store(std::size_t index, float value) const {
    std::lock_guard lock(storage_->mutex);
    storage_->values.at(index) = value;
}

struct AtomicU32Buffer::Storage {
    explicit Storage(std::size_t element_count)
        : values(element_count) {}

    mutable std::mutex mutex;
    std::vector<std::uint32_t> values;
    std::size_t rmw_operation_count = 0;
};

AtomicU32Buffer::AtomicU32Buffer(std::size_t element_count)
    : storage_(std::make_shared<Storage>(element_count)) {}

std::size_t AtomicU32Buffer::size() const noexcept {
    return storage_->values.size();
}

std::uint32_t AtomicU32Buffer::load(
    std::size_t index
) const {
    std::lock_guard lock(storage_->mutex);
    return storage_->values.at(index);
}

void AtomicU32Buffer::store(
    std::size_t index,
    std::uint32_t value
) const {
    std::lock_guard lock(storage_->mutex);
    storage_->values.at(index) = value;
}

std::uint32_t AtomicU32Buffer::fetch_add(
    std::size_t index,
    std::uint32_t delta
) const {
    std::lock_guard lock(storage_->mutex);
    std::uint32_t& value = storage_->values.at(index);
    const std::uint32_t old_value = value;
    value += delta;
    ++storage_->rmw_operation_count;
    return old_value;
}

bool AtomicU32Buffer::compare_exchange(
    std::size_t index,
    std::uint32_t& expected,
    std::uint32_t desired
) const {
    std::lock_guard lock(storage_->mutex);
    std::uint32_t& value = storage_->values.at(index);
    ++storage_->rmw_operation_count;

    if (value == expected) {
        value = desired;
        return true;
    }

    expected = value;
    return false;
}

std::size_t AtomicU32Buffer::rmw_operation_count() const {
    std::lock_guard lock(storage_->mutex);
    return storage_->rmw_operation_count;
}

namespace detail {

class SharedMemory {
public:
    SharedMemory(
        std::size_t group_id,
        std::size_t participants,
        std::size_t element_count
    )
        : group_id_(group_id),
          participants_(participants),
          values_(element_count),
          accesses_(element_count) {}

    [[nodiscard]] std::size_t size() const noexcept {
        return values_.size();
    }

    [[nodiscard]] float load(
        std::size_t local_id,
        std::size_t index
    ) {
        std::lock_guard lock(mutex_);
        const float value = values_.at(index);
        record_locked(
            accesses_.at(index),
            local_id,
            SharedAccessKind::load
        );
        return value;
    }

    void store(
        std::size_t local_id,
        std::size_t index,
        float value
    ) {
        std::lock_guard lock(mutex_);
        values_.at(index) = value;
        record_locked(
            accesses_.at(index),
            local_id,
            SharedAccessKind::store
        );
    }

    float atomic_add(
        std::size_t local_id,
        std::size_t index,
        float delta
    ) {
        std::lock_guard lock(mutex_);
        float& value = values_.at(index);
        record_locked(
            accesses_.at(index),
            local_id,
            SharedAccessKind::atomic_add
        );

        const float old_value = value;
        value += delta;
        return old_value;
    }

    void close_phase() {
        std::lock_guard lock(mutex_);
        validate_phase_locked();

        for (AccessRecord& access : accesses_) {
            access = AccessRecord{};
        }

        ++phase_;
    }

private:
    struct AccessRecord {
        std::uint64_t loads = 0;
        std::uint64_t stores = 0;
        std::uint64_t atomic_adds = 0;
    };

    [[nodiscard]] static std::uint64_t mask_for(
        const AccessRecord& access,
        SharedAccessKind kind
    ) noexcept {
        switch (kind) {
        case SharedAccessKind::load:
            return access.loads;
        case SharedAccessKind::store:
            return access.stores;
        case SharedAccessKind::atomic_add:
            return access.atomic_adds;
        }

        return 0;
    }

    [[nodiscard]] static SharedAccessKind kind_for(
        std::size_t ordinal
    ) noexcept {
        switch (ordinal) {
        case 0:
            return SharedAccessKind::load;
        case 1:
            return SharedAccessKind::store;
        default:
            return SharedAccessKind::atomic_add;
        }
    }

    [[nodiscard]] static bool conflicts(
        SharedAccessKind first,
        SharedAccessKind second
    ) noexcept {
        if (
            first == SharedAccessKind::load &&
            second == SharedAccessKind::load
        ) {
            return false;
        }

        if (
            first == SharedAccessKind::atomic_add &&
            second == SharedAccessKind::atomic_add
        ) {
            return false;
        }

        return true;
    }

    static void record_locked(
        AccessRecord& access,
        std::size_t local_id,
        SharedAccessKind kind
    ) {
        const std::uint64_t lane_bit =
            std::uint64_t{1} << local_id;

        switch (kind) {
        case SharedAccessKind::load:
            access.loads |= lane_bit;
            return;
        case SharedAccessKind::store:
            access.stores |= lane_bit;
            return;
        case SharedAccessKind::atomic_add:
            access.atomic_adds |= lane_bit;
            return;
        }
    }

    void validate_phase_locked() const {
        for (
            std::size_t index = 0;
            index < accesses_.size();
            ++index
        ) {
            const AccessRecord& access = accesses_[index];

            for (
                std::size_t first_local_id = 0;
                first_local_id < participants_;
                ++first_local_id
            ) {
                const std::uint64_t first_bit =
                    std::uint64_t{1} << first_local_id;

                for (
                    std::size_t second_local_id =
                        first_local_id + 1;
                    second_local_id < participants_;
                    ++second_local_id
                ) {
                    const std::uint64_t second_bit =
                        std::uint64_t{1} << second_local_id;

                    for (
                        std::size_t first_kind_index = 0;
                        first_kind_index < 3;
                        ++first_kind_index
                    ) {
                        const SharedAccessKind first_kind =
                            kind_for(first_kind_index);

                        if (
                            (
                                mask_for(access, first_kind) &
                                first_bit
                            ) == 0
                        ) {
                            continue;
                        }

                        for (
                            std::size_t second_kind_index = 0;
                            second_kind_index < 3;
                            ++second_kind_index
                        ) {
                            const SharedAccessKind second_kind =
                                kind_for(second_kind_index);

                            if (
                                (
                                    mask_for(
                                        access,
                                        second_kind
                                    ) &
                                    second_bit
                                ) == 0
                            ) {
                                continue;
                            }

                            if (
                                conflicts(
                                    first_kind,
                                    second_kind
                                )
                            ) {
                                throw SharedMemoryRaceError(
                                    group_id_,
                                    phase_,
                                    index,
                                    first_local_id,
                                    first_kind,
                                    second_local_id,
                                    second_kind
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    const std::size_t group_id_;
    const std::size_t participants_;
    std::mutex mutex_;
    std::vector<float> values_;
    std::vector<AccessRecord> accesses_;
    std::size_t phase_ = 0;
};

class GroupCancelled final : public std::exception {};

class ThreadgroupBarrier {
public:
    ThreadgroupBarrier(
        std::size_t participants,
        SharedMemory* shared
    )
        : participants_(participants),
          shared_(shared),
          finished_(participants, std::uint8_t{0}) {}

    void wait(
        std::size_t local_id,
        std::size_t site_id
    ) {
        std::unique_lock lock(mutex_);

        if (broken_) {
            throw GroupCancelled{};
        }

        if (finished_count_ != 0) {
            std::ostringstream message;
            message
                << "divergent barrier at site " << site_id
                << ": " << finished_count_
                << " thread(s) already exited";

            fail_locked(std::make_exception_ptr(
                DivergentBarrierError(message.str())
            ));

            lock.unlock();
            condition_.notify_all();
            throw GroupCancelled{};
        }

        if (!expected_site_.has_value()) {
            expected_site_ = site_id;
        } else if (*expected_site_ != site_id) {
            std::ostringstream message;
            message
                << "divergent barrier sites: expected "
                << *expected_site_ << " but thread "
                << local_id << " reached " << site_id;

            fail_locked(std::make_exception_ptr(
                DivergentBarrierError(message.str())
            ));

            lock.unlock();
            condition_.notify_all();
            throw GroupCancelled{};
        }

        const std::size_t my_generation = generation_;
        ++arrived_;

        if (arrived_ == participants_) {
            try {
                shared_->close_phase();
            } catch (...) {
                fail_locked(std::current_exception());

                lock.unlock();
                condition_.notify_all();
                throw GroupCancelled{};
            }

            arrived_ = 0;
            expected_site_.reset();
            ++generation_;

            lock.unlock();
            condition_.notify_all();
            return;
        }

        condition_.wait(lock, [&] {
            return broken_ || generation_ != my_generation;
        });

        if (broken_) {
            throw GroupCancelled{};
        }
    }

    void finish(std::size_t local_id) {
        bool notify = false;
        std::unique_lock lock(mutex_);

        if (finished_.at(local_id) != 0) {
            return;
        }

        finished_[local_id] = 1;
        ++finished_count_;

        if (!broken_ && arrived_ != 0) {
            std::ostringstream message;
            message
                << "divergent barrier: thread " << local_id
                << " exited while " << arrived_
                << " peer(s) were waiting";

            fail_locked(std::make_exception_ptr(
                DivergentBarrierError(message.str())
            ));
            notify = true;
        } else if (
            !broken_ &&
            finished_count_ == participants_
        ) {
            try {
                shared_->close_phase();
            } catch (...) {
                fail_locked(std::current_exception());
                notify = true;
            }
        }

        lock.unlock();

        if (notify) {
            condition_.notify_all();
        }
    }

    void cancel(std::exception_ptr error) {
        bool notify = false;

        {
            std::lock_guard lock(mutex_);

            if (!broken_) {
                fail_locked(std::move(error));
                notify = true;
            }
        }

        if (notify) {
            condition_.notify_all();
        }
    }

    void rethrow_failure() const {
        std::exception_ptr error;

        {
            std::lock_guard lock(mutex_);
            error = failure_;
        }

        if (error) {
            std::rethrow_exception(error);
        }
    }

private:
    void fail_locked(std::exception_ptr error) {
        broken_ = true;
        failure_ = std::move(error);
    }

    const std::size_t participants_;
    SharedMemory* shared_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::size_t arrived_ = 0;
    std::size_t generation_ = 0;
    std::optional<std::size_t> expected_site_;
    std::vector<std::uint8_t> finished_;
    std::size_t finished_count_ = 0;
    bool broken_ = false;
    std::exception_ptr failure_;
};

class StartGate {
public:
    [[nodiscard]] bool wait() {
        std::unique_lock lock(mutex_);

        condition_.wait(lock, [&] {
            return released_ || cancelled_;
        });

        return !cancelled_;
    }

    void release() {
        {
            std::lock_guard lock(mutex_);
            released_ = true;
        }

        condition_.notify_all();
    }

    void cancel() {
        {
            std::lock_guard lock(mutex_);
            cancelled_ = true;
        }

        condition_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    bool released_ = false;
    bool cancelled_ = false;
};

} // namespace detail

struct ThreadContext::RuntimeState {
    detail::SharedMemory* shared;
    detail::ThreadgroupBarrier* barrier;
    std::size_t threads_per_group;
};

ThreadContext::ThreadContext(
    ThreadId id,
    RuntimeState* runtime
)
    : id_(id),
      runtime_(runtime) {}

const ThreadId& ThreadContext::id() const noexcept {
    return id_;
}

std::size_t ThreadContext::global_id() const noexcept {
    return id_.global;
}

std::size_t ThreadContext::group_id() const noexcept {
    return id_.group;
}

std::size_t ThreadContext::local_id() const noexcept {
    return id_.local;
}

std::size_t ThreadContext::threads_per_group() const noexcept {
    return runtime_->threads_per_group;
}

bool ThreadContext::in_bounds(std::size_t count) const noexcept {
    return id_.global < count;
}

std::size_t ThreadContext::shared_size() const noexcept {
    return runtime_->shared->size();
}

float ThreadContext::shared_load(std::size_t index) const {
    return runtime_->shared->load(id_.local, index);
}

void ThreadContext::shared_store(
    std::size_t index,
    float value
) {
    runtime_->shared->store(id_.local, index, value);
}

float ThreadContext::shared_atomic_add(
    std::size_t index,
    float delta
) {
    return runtime_->shared->atomic_add(
        id_.local,
        index,
        delta
    );
}

void ThreadContext::barrier(std::size_t site_id) {
    runtime_->barrier->wait(id_.local, site_id);
}

LaunchConfig::LaunchConfig(
    std::size_t groups,
    std::size_t threads_per_group,
    std::size_t shared_elements
)
    : groups_(groups),
      threads_per_group_(threads_per_group),
      shared_elements_(shared_elements) {
    if (threads_per_group_ == 0) {
        throw std::invalid_argument(
            "threads_per_group must be greater than zero"
        );
    }

    if (
        threads_per_group_ >
        kMaxSimulatedThreadsPerGroup
    ) {
        throw std::invalid_argument(
            "this teaching simulator supports at most 64 "
            "threads per group"
        );
    }

    if (
        groups_ != 0 &&
        threads_per_group_ >
            std::numeric_limits<std::size_t>::max() / groups_
    ) {
        throw std::overflow_error("launch size overflow");
    }
}

LaunchConfig LaunchConfig::cover(
    std::size_t work_items,
    std::size_t threads_per_group,
    std::size_t shared_elements
) {
    if (threads_per_group == 0) {
        throw std::invalid_argument(
            "threads_per_group must be greater than zero"
        );
    }

    const std::size_t groups =
        work_items == 0
            ? 0
            : 1 + (work_items - 1) / threads_per_group;

    return LaunchConfig(
        groups,
        threads_per_group,
        shared_elements
    );
}

std::size_t LaunchConfig::groups() const noexcept {
    return groups_;
}

std::size_t LaunchConfig::threads_per_group() const noexcept {
    return threads_per_group_;
}

std::size_t LaunchConfig::shared_elements() const noexcept {
    return shared_elements_;
}

std::size_t LaunchConfig::total_threads() const noexcept {
    return groups_ * threads_per_group_;
}

Kernel::Kernel(std::string name, Body body)
    : name_(std::move(name)),
      body_(std::move(body)) {
    if (!body_) {
        throw std::invalid_argument(
            "kernel body must not be empty"
        );
    }
}

void Kernel::run(ThreadContext& context) const {
    body_(context);
}

const std::string& Kernel::name() const noexcept {
    return name_;
}

namespace detail {

struct EventState {
    void finish(std::exception_ptr execution_error) {
        {
            std::lock_guard lock(mutex);
            error = std::move(execution_error);
            complete = true;
        }

        condition.notify_all();
    }

    std::mutex mutex;
    std::condition_variable condition;
    bool complete = false;
    std::exception_ptr error;
};

struct GroupExecutor {
    static void run(
        const Kernel& kernel,
        const LaunchConfig& launch,
        std::size_t group_id
    ) {
        SharedMemory shared(
            group_id,
            launch.threads_per_group(),
            launch.shared_elements()
        );
        ThreadgroupBarrier barrier(
            launch.threads_per_group(),
            &shared
        );
        StartGate start_gate;
        std::vector<std::thread> lanes;
        lanes.reserve(launch.threads_per_group());

        try {
            for (
                std::size_t local_id = 0;
                local_id < launch.threads_per_group();
                ++local_id
            ) {
                lanes.emplace_back(
                    [
                        &kernel,
                        &launch,
                        &shared,
                        &barrier,
                        &start_gate,
                        group_id,
                        local_id
                    ] {
                        try {
                            if (!start_gate.wait()) {
                                return;
                            }

                            ThreadContext::RuntimeState runtime{
                                &shared,
                                &barrier,
                                launch.threads_per_group(),
                            };

                            ThreadContext context(
                                ThreadId{
                                    group_id *
                                            launch
                                                .threads_per_group() +
                                        local_id,
                                    group_id,
                                    local_id,
                                },
                                &runtime
                            );

                            try {
                                kernel.run(context);
                            } catch (const GroupCancelled&) {
                                // Another lane already broke the group.
                            } catch (...) {
                                barrier.cancel(
                                    std::current_exception()
                                );
                            }

                            try {
                                barrier.finish(local_id);
                            } catch (...) {
                                barrier.cancel(
                                    std::current_exception()
                                );
                            }
                        } catch (...) {
                            barrier.cancel(
                                std::current_exception()
                            );
                        }
                    }
                );
            }
        } catch (...) {
            const std::exception_ptr creation_error =
                std::current_exception();

            start_gate.cancel();

            for (std::thread& lane : lanes) {
                lane.join();
            }

            std::rethrow_exception(creation_error);
        }

        start_gate.release();

        for (std::thread& lane : lanes) {
            lane.join();
        }

        barrier.rethrow_failure();
    }
};

} // namespace detail

Event::Event(std::shared_ptr<detail::EventState> state)
    : state_(std::move(state)) {}

void Event::wait() const {
    std::exception_ptr execution_error;

    {
        std::unique_lock lock(state_->mutex);

        state_->condition.wait(lock, [&] {
            return state_->complete;
        });

        execution_error = state_->error;
    }

    if (execution_error) {
        std::rethrow_exception(execution_error);
    }
}

bool Event::is_complete() const {
    std::lock_guard lock(state_->mutex);
    return state_->complete;
}

class Queue::Impl {
public:
    Impl()
        : worker_([this] {
              worker_loop();
          }) {}

    ~Impl() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }

        condition_.notify_all();
        worker_.join();
    }

    void enqueue(
        Kernel kernel,
        LaunchConfig launch,
        std::shared_ptr<detail::EventState> event_state
    ) {
        {
            std::lock_guard lock(mutex_);

            if (stopping_) {
                throw std::runtime_error("queue is stopping");
            }

            commands_.push_back(Command{
                std::move(kernel),
                launch,
                std::move(event_state),
            });
        }

        condition_.notify_one();
    }

private:
    struct Command {
        Kernel kernel;
        LaunchConfig launch;
        std::shared_ptr<detail::EventState> event_state;
    };

    void worker_loop() {
        for (;;) {
            std::optional<Command> command;

            {
                std::unique_lock lock(mutex_);

                condition_.wait(lock, [&] {
                    return stopping_ || !commands_.empty();
                });

                if (stopping_ && commands_.empty()) {
                    return;
                }

                command.emplace(
                    std::move(commands_.front())
                );
                commands_.pop_front();
            }

            std::exception_ptr execution_error;

            try {
                for (
                    std::size_t group_id = 0;
                    group_id < command->launch.groups();
                    ++group_id
                ) {
                    detail::GroupExecutor::run(
                        command->kernel,
                        command->launch,
                        group_id
                    );
                }
            } catch (...) {
                execution_error = std::current_exception();
            }

            command->event_state->finish(
                std::move(execution_error)
            );
        }
    }

    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<Command> commands_;
    bool stopping_ = false;
    std::thread worker_;
};

Queue::Queue()
    : impl_(std::make_unique<Impl>()) {}

Queue::~Queue() = default;

Event Queue::submit(Kernel kernel, LaunchConfig launch) {
    auto event_state =
        std::make_shared<detail::EventState>();

    impl_->enqueue(
        std::move(kernel),
        launch,
        event_state
    );

    return Event(std::move(event_state));
}

Buffer Device::allocate(std::size_t element_count) const {
    return Buffer(element_count);
}

AtomicU32Buffer Device::allocate_atomic_u32(
    std::size_t element_count
) const {
    return AtomicU32Buffer(element_count);
}

std::unique_ptr<Queue> Device::create_queue() const {
    return std::make_unique<Queue>();
}

Kernel vector_add(
    Buffer a,
    Buffer b,
    Buffer output,
    std::size_t element_count
) {
    return Kernel(
        "vector_add",
        [
            a = std::move(a),
            b = std::move(b),
            output = std::move(output),
            element_count
        ](ThreadContext& context) {
            if (!context.in_bounds(element_count)) {
                return;
            }

            output.store(
                context.global_id(),
                a.load(context.global_id()) +
                    b.load(context.global_id())
            );
        }
    );
}

Kernel fill(
    Buffer output,
    std::size_t element_count,
    float value
) {
    return Kernel(
        "fill",
        [
            output = std::move(output),
            element_count,
            value
        ](ThreadContext& context) {
            if (context.in_bounds(element_count)) {
                output.store(context.global_id(), value);
            }
        }
    );
}

namespace {

[[nodiscard]] bool is_power_of_two(
    std::size_t value
) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

void validate_reduction(
    const Buffer& input,
    std::size_t element_count,
    std::size_t threads_per_group
) {
    if (element_count > input.size()) {
        throw std::out_of_range(
            "reduction element_count exceeds input size"
        );
    }

    if (threads_per_group < 2) {
        throw std::invalid_argument(
            "reduction requires at least two threads per group"
        );
    }

    if (!is_power_of_two(threads_per_group)) {
        throw std::invalid_argument(
            "this first reduction requires a power-of-two "
            "threadgroup size"
        );
    }

    if (
        threads_per_group >
        kMaxSimulatedThreadsPerGroup
    ) {
        throw std::invalid_argument(
            "reduction threadgroup exceeds simulator limit"
        );
    }
}

} // namespace

ReductionPass enqueue_reduce_sum_pass(
    Queue& queue,
    Buffer input,
    std::size_t element_count,
    std::size_t threads_per_group
) {
    if (element_count == 0) {
        throw std::invalid_argument(
            "a reduction pass needs at least one input element"
        );
    }

    validate_reduction(
        input,
        element_count,
        threads_per_group
    );

    const std::size_t group_count =
        1 + (element_count - 1) / threads_per_group;
    Buffer output(group_count);

    Kernel kernel(
        "reduce_sum_pass",
        [
            input = std::move(input),
            output,
            element_count
        ](ThreadContext& context) {
            const float value =
                context.in_bounds(element_count)
                    ? input.load(context.global_id())
                    : 0.0f;

            context.shared_store(
                context.local_id(),
                value
            );

            // Every lane, including padded lanes, reaches this barrier.
            context.barrier(0);

            std::size_t barrier_site = 1;

            for (
                std::size_t stride =
                    context.threads_per_group() / 2;
                stride > 0;
                stride /= 2
            ) {
                if (context.local_id() < stride) {
                    const float left = context.shared_load(
                        context.local_id()
                    );
                    const float right = context.shared_load(
                        context.local_id() + stride
                    );

                    context.shared_store(
                        context.local_id(),
                        left + right
                    );
                }

                // The work is conditional; the barrier is not.
                context.barrier(barrier_site);
                ++barrier_site;
            }

            if (context.local_id() == 0) {
                output.store(
                    context.group_id(),
                    context.shared_load(0)
                );
            }
        }
    );

    Event completion = queue.submit(
        std::move(kernel),
        LaunchConfig(
            group_count,
            threads_per_group,
            threads_per_group
        )
    );

    return ReductionPass{
        std::move(output),
        std::move(completion),
        group_count,
    };
}

ReductionResult reduce_sum(
    Queue& queue,
    Buffer input,
    std::size_t element_count,
    std::size_t threads_per_group
) {
    if (element_count > input.size()) {
        throw std::out_of_range(
            "reduction element_count exceeds input size"
        );
    }

    ReductionResult result{
        0.0f,
        0,
        {element_count},
    };

    validate_reduction(
        input,
        element_count,
        threads_per_group
    );

    if (element_count == 0) {
        return result;
    }

    if (element_count == 1) {
        result.value = input.load(0);
        return result;
    }

    Buffer current = std::move(input);
    std::size_t current_count = element_count;

    while (current_count > 1) {
        ReductionPass pass = enqueue_reduce_sum_pass(
            queue,
            std::move(current),
            current_count,
            threads_per_group
        );

        // FIFO ordering alone does not propagate an earlier command's
        // error. Waiting here makes each dependency explicit.
        pass.completion.wait();

        current = std::move(pass.output);
        current_count = pass.output_count;
        ++result.passes;
        result.element_counts.push_back(current_count);
    }

    result.value = current.load(0);
    return result;
}

} // namespace minigpu
