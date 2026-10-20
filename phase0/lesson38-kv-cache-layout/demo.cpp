#include "kv_cache.hpp"

#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using minitensor::RuntimeTensor;
using minitensor::inference::KVCache;
using minitensor::inference::KVCacheConfig;

[[nodiscard]] RuntimeTensor make_tensor(
    const minigpu::Device& device,
    minitensor::Shape shape,
    std::vector<float> values
) {
    return RuntimeTensor::from_host(
        device,
        miniir::Type::tensor(
            miniir::DType::F32,
            std::move(shape)
        ),
        std::move(values)
    );
}

void require(
    bool condition,
    std::string_view message
) {
    if (!condition) {
        throw std::runtime_error(
            std::string(message)
        );
    }
}

void print_pair(
    const KVCache& cache,
    std::size_t head,
    std::size_t token,
    bool print_keys
) {
    std::cout << '[';

    for (
        std::size_t component = 0;
        component < cache.config().head_dimension;
        ++component
    ) {
        if (component != 0) {
            std::cout << ", ";
        }

        const float value =
            print_keys
                ? cache.load_key(
                      0,
                      head,
                      token,
                      component
                  )
                : cache.load_value(
                      0,
                      head,
                      token,
                      component
                  );
        std::cout << value;
    }

    std::cout << ']';
}

void print_cache(const KVCache& cache) {
    std::cout
        << "valid length = "
        << cache.sequence_length()
        << ", reserved capacity = "
        << cache.config().capacity
        << "\n\n";

    for (
        std::size_t head = 0;
        head < cache.config().key_value_head_count;
        ++head
    ) {
        std::cout << "head " << head << ":\n";

        for (
            std::size_t token = 0;
            token < cache.sequence_length();
            ++token
        ) {
            std::cout
                << "  token " << token
                << "  K = ";
            print_pair(cache, head, token, true);
            std::cout << "  V = ";
            print_pair(cache, head, token, false);
            std::cout
                << "  starts at flat index "
                << cache.storage_index(
                       head,
                       token,
                       0
                   )
                << '\n';
        }
    }
}

} // namespace

int main() {
    try {
        minigpu::Device device;
        std::unique_ptr<minigpu::Queue> queue =
            device.create_queue();

        const KVCacheConfig config{
            1, // transformer layer
            2, // key/value heads
            4, // reserved token rows
            2, // numbers in each head vector
        };
        KVCache cache(device, config);

        std::cout
            << "Lesson 38: KV-cache storage layout\n"
            << "GPU: " << device.name() << "\n\n"
            << "Each layer owns two preallocated tensors:\n"
            << "  K cache shape = [heads, capacity, "
               "head dimension]\n"
            << "  V cache shape = [heads, capacity, "
               "head dimension]\n\n"
            << "Flat address formula:\n"
            << "  ((head * capacity) + token) * "
               "head_dimension + component\n\n"
            << "Memory reserved immediately:\n"
            << "  one K or V tensor: "
            << cache.memory_usage().bytes_per_tensor
            << " bytes\n"
            << "  complete cache:   "
            << cache.memory_usage().total_bytes
            << " bytes\n\n";

        RuntimeTensor prefill_keys = make_tensor(
            device,
            {2, 2, 2},
            {
                10.0f,
                11.0f,
                12.0f,
                13.0f,
                20.0f,
                21.0f,
                22.0f,
                23.0f,
            }
        );
        RuntimeTensor prefill_values = make_tensor(
            device,
            {2, 2, 2},
            {
                110.0f,
                111.0f,
                112.0f,
                113.0f,
                120.0f,
                121.0f,
                122.0f,
                123.0f,
            }
        );

        std::cout
            << "1. Prefill appends two tokens at once.\n";
        cache.append(
            *queue,
            0,
            prefill_keys,
            prefill_values,
            3
        );
        print_cache(cache);

        RuntimeTensor decode_key = make_tensor(
            device,
            {2, 1, 2},
            {
                14.0f,
                15.0f,
                24.0f,
                25.0f,
            }
        );
        RuntimeTensor decode_value = make_tensor(
            device,
            {2, 1, 2},
            {
                114.0f,
                115.0f,
                124.0f,
                125.0f,
            }
        );

        std::cout
            << "\n2. Decode appends only the newest token.\n";
        cache.append(
            *queue,
            0,
            decode_key,
            decode_value,
            3
        );
        print_cache(cache);

        require(
            cache.sequence_length() == 3 &&
                cache.load_key(0, 0, 2, 0) ==
                    14.0f &&
                cache.load_key(0, 1, 2, 1) ==
                    25.0f &&
                cache.load_value(0, 1, 2, 0) ==
                    124.0f,
            "GPU append produced unexpected values"
        );

        std::cout
            << "\nImportant: capacity is reserved storage; "
               "valid length says which token rows may be read.\n"
            << "The cache stores past K and V vectors, not "
               "past context vectors.\n"
            << "Lesson 38 demo passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "Lesson 38 demo failed: "
            << error.what() << '\n';
        return 1;
    }
}
