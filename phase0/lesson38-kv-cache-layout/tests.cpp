#include "kv_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using minitensor::RuntimeTensor;
using minitensor::inference::KVCache;
using minitensor::inference::KVCacheConfig;
using Values = std::vector<float>;

static_assert(!std::is_copy_constructible_v<KVCache>);
static_assert(!std::is_copy_assignable_v<KVCache>);
static_assert(std::is_move_constructible_v<KVCache>);
static_assert(std::is_move_assignable_v<KVCache>);

void require(
    bool condition,
    std::string_view message
) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Exception, typename Function>
void require_throws(
    Function&& function,
    std::string_view expected_message_part
) {
    try {
        std::forward<Function>(function)();
    } catch (const Exception& error) {
        if (
            std::string_view(error.what()).find(
                expected_message_part
            ) == std::string_view::npos
        ) {
            throw std::runtime_error(
                "exception message did not contain '" +
                std::string(expected_message_part) +
                "': " + error.what()
            );
        }

        return;
    } catch (const std::exception& error) {
        throw std::runtime_error(
            "unexpected exception type: " +
            std::string(error.what())
        );
    }

    throw std::runtime_error(
        "expected an exception containing '" +
        std::string(expected_message_part) + "'"
    );
}

[[nodiscard]] RuntimeTensor make_tensor(
    const minigpu::Device& device,
    minitensor::Shape shape,
    Values values
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

[[nodiscard]] RuntimeTensor make_noncontiguous_chunk(
    const minigpu::Device& device
) {
    const minitensor::Shape shape{2, 1, 3};
    const minitensor::TensorLayout layout(
        shape,
        minitensor::Strides{4, 3, 1}
    );
    minigpu::Buffer storage =
        device.allocate(layout.minimum_storage_size());
    RuntimeTensor tensor(
        miniir::Type::tensor(
            miniir::DType::F32,
            shape
        ),
        layout,
        std::move(storage)
    );

    const Values values{
        1.0f,
        2.0f,
        3.0f,
        4.0f,
        5.0f,
        6.0f,
    };

    for (
        std::size_t index = 0;
        index < values.size();
        ++index
    ) {
        tensor.store_flat(index, values[index]);
    }

    return tensor;
}

[[nodiscard]] RuntimeTensor make_compact_alias(
    const RuntimeTensor& storage,
    minitensor::Shape shape
) {
    miniir::Type type = miniir::Type::tensor(
        miniir::DType::F32,
        shape
    );
    minitensor::TensorLayout layout =
        minitensor::TensorLayout::contiguous(
            std::move(shape)
        );

    return RuntimeTensor(
        std::move(type),
        std::move(layout),
        storage.buffer()
    );
}

void test_memory_accounting_and_config_validation() {
    const KVCacheConfig config{
        2,
        2,
        4,
        3,
    };
    const auto usage =
        minitensor::inference::
            calculate_kv_cache_memory_usage(config);

    require(
        usage.elements_per_tensor == 24 &&
            usage.bytes_per_tensor == 96 &&
            usage.elements_per_layer == 48 &&
            usage.bytes_per_layer == 192 &&
            usage.total_elements == 96 &&
            usage.total_bytes == 384,
        "KV-cache memory accounting is wrong"
    );

    require_throws<std::invalid_argument>(
        [] {
            static_cast<void>(
                minitensor::inference::
                    calculate_kv_cache_memory_usage(
                        KVCacheConfig{0, 1, 1, 1}
                    )
            );
        },
        "must be greater than zero"
    );
    require_throws<std::invalid_argument>(
        [] {
            static_cast<void>(
                minitensor::inference::
                    calculate_kv_cache_memory_usage(
                        KVCacheConfig{1, 0, 1, 1}
                    )
            );
        },
        "must be greater than zero"
    );
    require_throws<std::invalid_argument>(
        [] {
            static_cast<void>(
                minitensor::inference::
                    calculate_kv_cache_memory_usage(
                        KVCacheConfig{1, 1, 0, 1}
                    )
            );
        },
        "must be greater than zero"
    );
    require_throws<std::invalid_argument>(
        [] {
            static_cast<void>(
                minitensor::inference::
                    calculate_kv_cache_memory_usage(
                        KVCacheConfig{1, 1, 1, 0}
                    )
            );
        },
        "must be greater than zero"
    );

    require_throws<std::overflow_error>(
        [] {
            const std::size_t above_u32 =
                static_cast<std::size_t>(
                    std::numeric_limits<
                        std::uint32_t
                    >::max()
                ) +
                1;
            static_cast<void>(
                minitensor::inference::
                    calculate_kv_cache_memory_usage(
                        KVCacheConfig{
                            1,
                            above_u32,
                            1,
                            1,
                        }
                    )
            );
        },
        "uint32"
    );

    require_throws<std::overflow_error>(
        [] {
            const std::size_t maximum_u32 =
                static_cast<std::size_t>(
                    std::numeric_limits<
                        std::uint32_t
                    >::max()
                );
            static_cast<void>(
                minitensor::inference::
                    calculate_kv_cache_memory_usage(
                        KVCacheConfig{
                            1,
                            maximum_u32,
                            maximum_u32,
                            2,
                        }
                    )
            );
        },
        "overflows size_t"
    );
}

void test_exact_multihead_layout(
    const minigpu::Device& device,
    minigpu::Queue& queue
) {
    KVCache cache(
        device,
        KVCacheConfig{1, 2, 4, 3}
    );

    RuntimeTensor keys = make_tensor(
        device,
        {2, 2, 3},
        {
            1.0f,
            2.0f,
            3.0f,
            4.0f,
            5.0f,
            6.0f,
            101.0f,
            102.0f,
            103.0f,
            104.0f,
            105.0f,
            106.0f,
        }
    );
    RuntimeTensor values = make_tensor(
        device,
        {2, 2, 3},
        {
            1001.0f,
            1002.0f,
            1003.0f,
            1004.0f,
            1005.0f,
            1006.0f,
            1101.0f,
            1102.0f,
            1103.0f,
            1104.0f,
            1105.0f,
            1106.0f,
        }
    );

    cache.append(queue, 0, keys, values, 5);

    require(
        cache.config().layer_count == 1 &&
            cache.config().key_value_head_count == 2 &&
            cache.config().capacity == 4 &&
            cache.config().head_dimension == 3,
        "KV-cache did not retain its configuration"
    );
    require(
        cache.key_storage(0).shape() ==
                minitensor::Shape{2, 4, 3} &&
            cache.value_storage(0).shape() ==
                minitensor::Shape{2, 4, 3} &&
            cache.key_storage(0).is_contiguous() &&
            cache.key_storage(0).offset() == 0,
        "preallocated cache storage has the wrong layout"
    );

    require(
        cache.storage_index(0, 0, 0) == 0 &&
            cache.storage_index(0, 1, 2) == 5 &&
            cache.storage_index(1, 0, 0) == 12 &&
            cache.storage_index(1, 1, 2) == 17 &&
            cache.storage_index(1, 3, 2) == 23,
        "the [head, token, component] address formula is wrong"
    );

    require(
        cache.valid_length(0) == 2 &&
            cache.sequence_length() == 2 &&
            cache.load_key(0, 0, 0, 0) == 1.0f &&
            cache.load_key(0, 0, 1, 2) == 6.0f &&
            cache.load_key(0, 1, 0, 0) == 101.0f &&
            cache.load_key(0, 1, 1, 2) == 106.0f &&
            cache.load_value(0, 0, 1, 1) == 1005.0f &&
            cache.load_value(0, 1, 0, 2) == 1103.0f,
        "the GPU append did not preserve multi-head values"
    );

    require(
        cache.key_storage(0).buffer().load(12) ==
                101.0f &&
            cache.value_storage(0).buffer().load(17) ==
                1106.0f,
        "raw Metal storage does not match the documented layout"
    );
}

void test_prefill_then_decode_append(
    const minigpu::Device& device,
    minigpu::Queue& queue
) {
    KVCache cache(
        device,
        KVCacheConfig{1, 2, 4, 2}
    );

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
    cache.append(
        queue,
        0,
        prefill_keys,
        prefill_values,
        3
    );

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
    cache.append(
        queue,
        0,
        decode_key,
        decode_value,
        3
    );

    require(
        cache.valid_length(0) == 3 &&
            cache.load_key(0, 0, 0, 0) == 10.0f &&
            cache.load_key(0, 0, 1, 1) == 13.0f &&
            cache.load_key(0, 0, 2, 0) == 14.0f &&
            cache.load_key(0, 1, 0, 1) == 21.0f &&
            cache.load_key(0, 1, 1, 0) == 22.0f &&
            cache.load_key(0, 1, 2, 1) == 25.0f &&
            cache.load_value(0, 0, 2, 1) == 115.0f &&
            cache.load_value(0, 1, 2, 0) == 124.0f,
        "one-token decode append overwrote or misplaced cache rows"
    );
}

void test_layer_alignment_reset_and_reuse(
    const minigpu::Device& device,
    minigpu::Queue& queue
) {
    KVCache cache(
        device,
        KVCacheConfig{2, 1, 3, 2}
    );
    RuntimeTensor layer_zero_prefill = make_tensor(
        device,
        {1, 2, 2},
        {1.0f, 2.0f, 3.0f, 4.0f}
    );
    RuntimeTensor layer_one_prefill = make_tensor(
        device,
        {1, 2, 2},
        {11.0f, 12.0f, 13.0f, 14.0f}
    );
    RuntimeTensor values = make_tensor(
        device,
        {1, 2, 2},
        {101.0f, 102.0f, 103.0f, 104.0f}
    );

    cache.append(
        queue,
        0,
        layer_zero_prefill,
        values
    );
    require(
        !cache.has_uniform_length() &&
            cache.valid_length(0) == 2 &&
            cache.valid_length(1) == 0,
        "layer lengths should differ during a model pass"
    );
    require_throws<std::logic_error>(
        [&cache] {
            static_cast<void>(
                cache.sequence_length()
            );
        },
        "do not have one common sequence length"
    );

    cache.append(
        queue,
        1,
        layer_one_prefill,
        values
    );
    require(
        cache.has_uniform_length() &&
            cache.sequence_length() == 2,
        "layer lengths did not align after prefill"
    );

    RuntimeTensor layer_zero_decode = make_tensor(
        device,
        {1, 1, 2},
        {5.0f, 6.0f}
    );
    RuntimeTensor layer_one_decode = make_tensor(
        device,
        {1, 1, 2},
        {15.0f, 16.0f}
    );
    RuntimeTensor decode_values = make_tensor(
        device,
        {1, 1, 2},
        {105.0f, 106.0f}
    );
    cache.append(
        queue,
        0,
        layer_zero_decode,
        decode_values
    );
    require(
        !cache.has_uniform_length(),
        "one layer advancing should make lengths nonuniform"
    );
    cache.append(
        queue,
        1,
        layer_one_decode,
        decode_values
    );
    require(
        cache.has_uniform_length() &&
            cache.sequence_length() == 3 &&
            cache.load_key(0, 0, 2, 0) == 5.0f &&
            cache.load_key(1, 0, 2, 1) == 16.0f,
        "layer lengths or decode rows are wrong"
    );

    cache.truncate(1);
    require(
        cache.has_uniform_length() &&
            cache.sequence_length() == 1 &&
            cache.load_key(0, 0, 0, 0) == 1.0f,
        "truncate did not restore an earlier common length"
    );
    require_throws<std::out_of_range>(
        [&cache] {
            static_cast<void>(
                cache.load_key(1, 0, 1, 0)
            );
        },
        "valid length"
    );
    require_throws<std::invalid_argument>(
        [&cache] {
            cache.truncate(2);
        },
        "cannot increase"
    );
    require(
        cache.sequence_length() == 1,
        "failed truncate changed the cache length"
    );

    cache.reset();
    require(
        cache.has_uniform_length() &&
            cache.sequence_length() == 0 &&
            cache.valid_length(0) == 0 &&
            cache.valid_length(1) == 0,
        "reset did not logically empty every layer"
    );
    require_throws<std::out_of_range>(
        [&cache] {
            static_cast<void>(
                cache.load_key(0, 0, 0, 0)
            );
        },
        "valid length"
    );

    RuntimeTensor reused_key = make_tensor(
        device,
        {1, 1, 2},
        {91.0f, 92.0f}
    );
    RuntimeTensor reused_value = make_tensor(
        device,
        {1, 1, 2},
        {191.0f, 192.0f}
    );
    cache.append(
        queue,
        0,
        reused_key,
        reused_value
    );
    cache.append(
        queue,
        1,
        reused_key,
        reused_value
    );

    require(
        cache.sequence_length() == 1 &&
            cache.load_key(0, 0, 0, 0) == 91.0f &&
            cache.load_key(1, 0, 0, 1) == 92.0f &&
            cache.load_value(0, 0, 0, 1) == 192.0f,
        "cache reuse did not overwrite newly valid rows"
    );
}

void test_capacity_failure_is_transactional(
    const minigpu::Device& device,
    minigpu::Queue& queue
) {
    KVCache cache(
        device,
        KVCacheConfig{1, 1, 2, 1}
    );
    RuntimeTensor first_key = make_tensor(
        device,
        {1, 1, 1},
        {7.0f}
    );
    RuntimeTensor first_value = make_tensor(
        device,
        {1, 1, 1},
        {70.0f}
    );
    cache.append(
        queue,
        0,
        first_key,
        first_value
    );

    RuntimeTensor too_many_keys = make_tensor(
        device,
        {1, 2, 1},
        {8.0f, 9.0f}
    );
    RuntimeTensor too_many_values = make_tensor(
        device,
        {1, 2, 1},
        {80.0f, 90.0f}
    );
    require_throws<std::length_error>(
        [&] {
            cache.append(
                queue,
                0,
                too_many_keys,
                too_many_values
            );
        },
        "would exceed capacity"
    );

    require(
        cache.valid_length(0) == 1 &&
            cache.sequence_length() == 1 &&
            cache.load_key(0, 0, 0, 0) == 7.0f &&
            cache.load_value(0, 0, 0, 0) == 70.0f,
        "failed append changed the valid length or old data"
    );
}

void test_shape_layout_and_index_validation(
    const minigpu::Device& device,
    minigpu::Queue& queue
) {
    KVCache cache(
        device,
        KVCacheConfig{1, 2, 3, 3}
    );
    RuntimeTensor valid = make_tensor(
        device,
        {2, 1, 3},
        {
            1.0f,
            2.0f,
            3.0f,
            4.0f,
            5.0f,
            6.0f,
        }
    );
    RuntimeTensor rank_two = make_tensor(
        device,
        {2, 3},
        {
            1.0f,
            2.0f,
            3.0f,
            4.0f,
            5.0f,
            6.0f,
        }
    );
    require_throws<std::invalid_argument>(
        [&] {
            cache.append(
                queue,
                0,
                rank_two,
                rank_two
            );
        },
        "must have rank 3"
    );

    RuntimeTensor wrong_heads = make_tensor(
        device,
        {1, 1, 3},
        {1.0f, 2.0f, 3.0f}
    );
    require_throws<std::invalid_argument>(
        [&] {
            cache.append(
                queue,
                0,
                wrong_heads,
                wrong_heads
            );
        },
        "shape must be"
    );

    RuntimeTensor wrong_dimension = make_tensor(
        device,
        {2, 1, 2},
        {1.0f, 2.0f, 3.0f, 4.0f}
    );
    require_throws<std::invalid_argument>(
        [&] {
            cache.append(
                queue,
                0,
                wrong_dimension,
                wrong_dimension
            );
        },
        "shape must be"
    );

    RuntimeTensor two_tokens = make_tensor(
        device,
        {2, 2, 3},
        {
            1.0f,
            2.0f,
            3.0f,
            4.0f,
            5.0f,
            6.0f,
            7.0f,
            8.0f,
            9.0f,
            10.0f,
            11.0f,
            12.0f,
        }
    );
    require_throws<std::invalid_argument>(
        [&] {
            cache.append(
                queue,
                0,
                valid,
                two_tokens
            );
        },
        "identical shapes"
    );

    RuntimeTensor noncontiguous =
        make_noncontiguous_chunk(device);
    require_throws<std::invalid_argument>(
        [&] {
            cache.append(
                queue,
                0,
                noncontiguous,
                valid
            );
        },
        "compact row-major storage"
    );

    RuntimeTensor key_storage_alias =
        make_compact_alias(
            cache.key_storage(0),
            {2, 1, 3}
        );
    RuntimeTensor value_storage_alias =
        make_compact_alias(
            cache.value_storage(0),
            {2, 1, 3}
        );

    require_throws<std::invalid_argument>(
        [&] {
            cache.append(
                queue,
                0,
                key_storage_alias,
                valid
            );
        },
        "must not alias"
    );
    require_throws<std::invalid_argument>(
        [&] {
            cache.append(
                queue,
                0,
                valid,
                value_storage_alias
            );
        },
        "must not alias"
    );
    require_throws<std::invalid_argument>(
        [&] {
            cache.append(
                queue,
                0,
                value_storage_alias,
                valid
            );
        },
        "must not alias"
    );
    require_throws<std::invalid_argument>(
        [&] {
            cache.append(
                queue,
                0,
                valid,
                key_storage_alias
            );
        },
        "must not alias"
    );
    require(
        cache.valid_length(0) == 0,
        "rejected aliasing append changed valid length"
    );

    require_throws<std::out_of_range>(
        [&] {
            cache.append(
                queue,
                1,
                valid,
                valid
            );
        },
        "layer index"
    );
    require_throws<std::invalid_argument>(
        [&] {
            cache.append(
                queue,
                0,
                valid,
                valid,
                0
            );
        },
        "threads_per_group"
    );
    require_throws<std::out_of_range>(
        [&cache] {
            static_cast<void>(
                cache.key_storage(1)
            );
        },
        "layer index"
    );
    require_throws<std::out_of_range>(
        [&cache] {
            static_cast<void>(
                cache.storage_index(2, 0, 0)
            );
        },
        "head index"
    );
    require_throws<std::out_of_range>(
        [&cache] {
            static_cast<void>(
                cache.storage_index(0, 3, 0)
            );
        },
        "token index"
    );
    require_throws<std::out_of_range>(
        [&cache] {
            static_cast<void>(
                cache.storage_index(0, 0, 3)
            );
        },
        "component index"
    );
    require_throws<std::out_of_range>(
        [&cache] {
            static_cast<void>(
                cache.load_value(0, 0, 0, 0)
            );
        },
        "valid length"
    );

    cache.append(queue, 0, valid, valid);
    require_throws<std::out_of_range>(
        [&cache] {
            static_cast<void>(
                cache.load_key(0, 0, 1, 0)
            );
        },
        "valid length"
    );
}

} // namespace

int main() {
    try {
        test_memory_accounting_and_config_validation();

        minigpu::Device device;
        std::unique_ptr<minigpu::Queue> queue =
            device.create_queue();

        test_exact_multihead_layout(device, *queue);
        test_prefill_then_decode_append(device, *queue);
        test_layer_alignment_reset_and_reuse(
            device,
            *queue
        );
        test_capacity_failure_is_transactional(
            device,
            *queue
        );
        test_shape_layout_and_index_validation(
            device,
            *queue
        );

        std::cout
            << "Lesson 38 KV-cache tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "Lesson 38 test failure: "
            << error.what() << '\n';
        return 1;
    }
}
