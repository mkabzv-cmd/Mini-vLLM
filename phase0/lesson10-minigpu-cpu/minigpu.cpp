#include "minigpu.hpp"

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace minigpu {

struct Buffer::Storage {
    explicit Storage(std::size_t element_count)
        : values(element_count) {}

    std::vector<float> values;
};

Buffer::Buffer(std::size_t element_count)
    : storage_(std::make_shared<Storage>(element_count)) {}

std::size_t Buffer::size() const noexcept {
    return storage_->values.size();
}

float Buffer::load(std::size_t index) const {
    return storage_->values.at(index);
}

void Buffer::store(std::size_t index, float value) const {
    storage_->values.at(index) = value;
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
            std::numeric_limits<std::size_t>::max() / groups_
    ) {
        throw std::overflow_error("launch size overflow");
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
            : 1 + (work_items - 1) / threads_per_group;

    return LaunchConfig(groups, threads_per_group);
}

std::size_t LaunchConfig::groups() const noexcept {
    return groups_;
}

std::size_t LaunchConfig::threads_per_group() const noexcept {
    return threads_per_group_;
}

std::size_t LaunchConfig::total_threads() const noexcept {
    return groups_ * threads_per_group_;
}

Kernel::Kernel(std::string name, Body body)
    : name_(std::move(name)),
      body_(std::move(body)) {
    if (!body_) {
        throw std::invalid_argument("kernel body must not be empty");
    }
}

void Kernel::run(const ThreadId& id) const {
    body_(id);
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
            Command command = take_next_command();

            if (command.event_state == nullptr) {
                return;
            }

            std::exception_ptr execution_error;

            try {
                for (
                    std::size_t group = 0;
                    group < command.launch.groups();
                    ++group
                ) {
                    for (
                        std::size_t local = 0;
                        local < command.launch.threads_per_group();
                        ++local
                    ) {
                        command.kernel.run(ThreadId{
                            group *
                                    command.launch.threads_per_group() +
                                local,
                            group,
                            local,
                        });
                    }
                }
            } catch (...) {
                execution_error = std::current_exception();
            }

            command.event_state->finish(
                std::move(execution_error)
            );
        }
    }

    Command take_next_command() {
        std::unique_lock lock(mutex_);

        condition_.wait(lock, [&] {
            return stopping_ || !commands_.empty();
        });

        if (stopping_ && commands_.empty()) {
            return Command{
                Kernel("stop", [](const ThreadId&) {}),
                LaunchConfig(0, 1),
                nullptr,
            };
        }

        Command command = std::move(commands_.front());
        commands_.pop_front();
        return command;
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
        ](const ThreadId& id) {
            if (id.global >= element_count) {
                return;
            }

            output.store(
                id.global,
                a.load(id.global) + b.load(id.global)
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
        ](const ThreadId& id) {
            if (id.global < element_count) {
                output.store(id.global, value);
            }
        }
    );
}

Kernel increment(
    Buffer output,
    std::size_t element_count,
    float amount
) {
    return Kernel(
        "increment",
        [
            output = std::move(output),
            element_count,
            amount
        ](const ThreadId& id) {
            if (id.global < element_count) {
                output.store(
                    id.global,
                    output.load(id.global) + amount
                );
            }
        }
    );
}

} // namespace minigpu
