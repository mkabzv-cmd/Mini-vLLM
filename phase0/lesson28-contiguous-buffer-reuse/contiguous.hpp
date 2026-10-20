#pragma once

#include "../lesson25-tensor-storage-layout/runtime_tensor.hpp"

#include <cstddef>

namespace minitensor {

inline constexpr std::size_t contiguous_copy_max_rank = 4;

// Copies arbitrary supported input layouts into compact row-major storage.
class ContiguousMaterializer {
public:
    explicit ContiguousMaterializer(
        const minigpu::Device& device
    );

    // Convenience path: allocate independent output, submit, wait, return.
    [[nodiscard]] RuntimeTensor run(
        minigpu::Queue& queue,
        const RuntimeTensor& input,
        std::size_t threads_per_group = 64
    ) const;

    // Reuse path: submit into caller-owned compact storage and return
    // immediately. The caller must wait before CPU access or unsafe reuse.
    [[nodiscard]] minigpu::Event enqueue_into(
        minigpu::Queue& queue,
        const RuntimeTensor& input,
        RuntimeTensor& output,
        std::size_t threads_per_group = 64
    ) const;

private:
    minigpu::Device device_;
    minigpu::Kernel kernel_;
};

} // namespace minitensor
