#pragma once

#include "../lesson25-tensor-storage-layout/runtime_tensor.hpp"

#include <cstddef>

namespace minitensor {

// Performs one matrix multiplication per batch.
//
// transpose_right == false:
//     left  [B, M, K]
//     right [B, K, N]
//
// transpose_right == true:
//     left  [B, M, K]
//     right [B, N, K]
//
// Both modes return:
//     output [B, M, N]
class BatchedMatmul {
public:
    explicit BatchedMatmul(
        const minigpu::Device& device
    );

    [[nodiscard]] RuntimeTensor run(
        minigpu::Queue& queue,
        const RuntimeTensor& left,
        const RuntimeTensor& right,
        bool transpose_right = false,
        std::size_t threads_per_group = 64
    ) const;

private:
    minigpu::Device device_;
    minigpu::Kernel kernel_;
};

} // namespace minitensor
