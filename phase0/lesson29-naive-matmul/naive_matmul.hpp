#pragma once

#include "../lesson25-tensor-storage-layout/runtime_tensor.hpp"

#include <cstddef>

namespace minitensor {

// A reusable, correctness-first matrix multiplication pipeline.
//
// Inputs:
//     A [M, K]
//     B [K, N]
//
// Output:
//     C [M, N]
class NaiveMatmul {
public:
    explicit NaiveMatmul(
        const minigpu::Device& device
    );

    // Submits one Metal dispatch, waits for it, and returns a compact tensor.
    [[nodiscard]] RuntimeTensor run(
        minigpu::Queue& queue,
        const RuntimeTensor& a,
        const RuntimeTensor& b,
        std::size_t threads_per_group = 64
    ) const;

private:
    minigpu::Device device_;
    minigpu::Kernel kernel_;
};

} // namespace minitensor
