#include "minigpu.hpp"

#include <cstddef>
#include <exception>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Function>
void require_throws(
    Function function,
    const std::string& message
) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }

    throw std::runtime_error(message);
}

void test_vector_add_and_padded_threads() {
    minigpu::Device device;
    auto queue = device.create_queue();

    minigpu::Buffer a = device.allocate(5);
    minigpu::Buffer b = device.allocate(5);
    minigpu::Buffer output = device.allocate(5);

    for (std::size_t index = 0; index < 5; ++index) {
        a.store(index, static_cast<float>(index + 1));
        b.store(index, static_cast<float>((index + 1) * 10));
    }

    const minigpu::LaunchConfig launch =
        minigpu::LaunchConfig::cover(5, 4);

    require(launch.groups() == 2, "expected two groups");
    require(
        launch.total_threads() == 8,
        "expected eight padded threads"
    );

    minigpu::Event finished = queue->submit(
        minigpu::vector_add(a, b, output, 5),
        launch
    );

    finished.wait();

    const float expected[] = {
        11.0f,
        22.0f,
        33.0f,
        44.0f,
        55.0f,
    };

    for (std::size_t index = 0; index < 5; ++index) {
        require(
            output.load(index) == expected[index],
            "vector-add result is incorrect"
        );
    }
}

void test_delayed_out_of_bounds_error() {
    minigpu::Device device;
    auto queue = device.create_queue();

    minigpu::Buffer a = device.allocate(5);
    minigpu::Buffer b = device.allocate(5);
    minigpu::Buffer output = device.allocate(5);

    minigpu::Event failed = queue->submit(
        minigpu::vector_add(a, b, output, 6),
        minigpu::LaunchConfig::cover(6, 4)
    );

    require_throws<std::out_of_range>(
        [&] {
            failed.wait();
        },
        "event.wait() did not report the kernel bounds error"
    );
}

void test_fifo_ordering() {
    minigpu::Device device;
    auto queue = device.create_queue();
    minigpu::Buffer value = device.allocate(1);

    auto started =
        std::make_shared<std::promise<void>>();
    std::future<void> started_future =
        started->get_future();

    std::promise<void> release;
    std::shared_future<void> release_future =
        release.get_future().share();

    minigpu::Kernel first(
        "gated_write",
        [
            value,
            started,
            release_future
        ](const minigpu::ThreadId& id) {
            if (id.global == 0) {
                started->set_value();
                release_future.wait();
                value.store(0, 41.0f);
            }
        }
    );

    minigpu::Event first_event = queue->submit(
        std::move(first),
        minigpu::LaunchConfig(1, 1)
    );

    minigpu::Event second_event = queue->submit(
        minigpu::increment(value, 1, 1.0f),
        minigpu::LaunchConfig(1, 1)
    );

    started_future.wait();

    require(
        !second_event.is_complete(),
        "second FIFO command completed before the first"
    );

    release.set_value();
    second_event.wait();
    first_event.wait();

    require(
        value.load(0) == 42.0f,
        "FIFO commands executed out of order"
    );
}

void test_pending_command_retains_buffers() {
    minigpu::Device device;
    auto queue = device.create_queue();

    minigpu::Buffer output = device.allocate(3);

    std::promise<void> release;
    std::shared_future<void> release_future =
        release.get_future().share();

    minigpu::Event gate = queue->submit(
        minigpu::Kernel(
            "gate",
            [release_future](const minigpu::ThreadId& id) {
                if (id.global == 0) {
                    release_future.wait();
                }
            }
        ),
        minigpu::LaunchConfig(1, 1)
    );

    std::optional<minigpu::Event> vector_add_event;

    {
        minigpu::Buffer a = device.allocate(3);
        minigpu::Buffer b = device.allocate(3);

        a.store(0, 1.0f);
        a.store(1, 2.0f);
        a.store(2, 3.0f);

        b.store(0, 10.0f);
        b.store(1, 20.0f);
        b.store(2, 30.0f);

        vector_add_event.emplace(
            queue->submit(
                minigpu::vector_add(a, b, output, 3),
                minigpu::LaunchConfig::cover(3, 2)
            )
        );
    }

    // The original input handles are gone, but the queued kernel owns
    // copies of them and therefore keeps their storage alive.
    release.set_value();
    gate.wait();
    vector_add_event->wait();

    require(output.load(0) == 11.0f, "lifetime test failed");
    require(output.load(1) == 22.0f, "lifetime test failed");
    require(output.load(2) == 33.0f, "lifetime test failed");
}

void test_failure_does_not_break_queue() {
    minigpu::Device device;
    auto queue = device.create_queue();

    minigpu::Event failed = queue->submit(
        minigpu::Kernel(
            "always_fail",
            [](const minigpu::ThreadId& id) {
                if (id.global == 0) {
                    throw std::runtime_error(
                        "intentional kernel failure"
                    );
                }
            }
        ),
        minigpu::LaunchConfig(1, 1)
    );

    require_throws<std::runtime_error>(
        [&] {
            failed.wait();
        },
        "kernel failure was not reported through the event"
    );

    minigpu::Buffer output = device.allocate(2);

    minigpu::Event recovered = queue->submit(
        minigpu::fill(output, 2, 7.0f),
        minigpu::LaunchConfig::cover(2, 2)
    );

    recovered.wait();

    require(
        output.load(0) == 7.0f &&
            output.load(1) == 7.0f,
        "queue did not recover after a failed command"
    );
}

template <typename Function>
void run_test(
    const std::string& name,
    Function function
) {
    function();
    std::cout << "[PASS] " << name << '\n';
}

} // namespace

int main() {
    try {
        run_test(
            "vector add and padded threads",
            test_vector_add_and_padded_threads
        );

        run_test(
            "delayed out-of-bounds error",
            test_delayed_out_of_bounds_error
        );

        run_test(
            "FIFO queue ordering",
            test_fifo_ordering
        );

        run_test(
            "pending command retains buffers",
            test_pending_command_retains_buffers
        );

        run_test(
            "queue recovery after failure",
            test_failure_does_not_break_queue
        );
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All MiniGPU CPU tests passed\n";
    return 0;
}
