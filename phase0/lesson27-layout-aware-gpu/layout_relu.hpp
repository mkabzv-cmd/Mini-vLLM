#pragma once

#include "../lesson25-tensor-storage-layout/runtime_tensor.hpp"

#include <cstddef>

namespace minitensor {

// Four axes cover the common inference layout:
//
//     [batch, heads, tokens, head_dimension]
//
// A later compiler lesson can generate specialized kernels for other ranks.
inline constexpr std::size_t layout_kernel_max_rank = 4;

// Compiles one reusable Metal kernel. run() reads any supported input layout
// and returns a fresh contiguous tensor containing ReLU(input).
class LayoutAwareRelu {
public:
    explicit LayoutAwareRelu(
        const minigpu::Device& device
    );

    [[nodiscard]] RuntimeTensor run(
        minigpu::Queue& queue,
        const RuntimeTensor& input,
        std::size_t threads_per_group = 64
    ) const;

private:
    minigpu::Device device_;
    minigpu::Kernel kernel_;
};

} // namespace minitensor
