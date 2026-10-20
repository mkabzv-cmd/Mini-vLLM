#pragma once

#include "../lesson25-tensor-storage-layout/runtime_tensor.hpp"

#include <cstddef>

namespace minitensor {

enum class LinearActivation {
    None,
    Relu,
};

// Computes:
//
//     output = input @ transpose(weight)
//
// then optionally adds bias and applies ReLU in the same Metal kernel.
//
// Shapes:
//
//     input   [M, K]
//     weight  [N, K]  (Hugging Face / PyTorch linear weight layout)
//     bias    [N]     (optional)
//     output  [M, N]
class LinearOperator {
public:
    explicit LinearOperator(
        const minigpu::Device& device
    );

    [[nodiscard]] RuntimeTensor run(
        minigpu::Queue& queue,
        const RuntimeTensor& input,
        const RuntimeTensor& weight,
        const RuntimeTensor* bias = nullptr,
        LinearActivation activation =
            LinearActivation::None,
        std::size_t threads_per_group = 64
    ) const;

private:
    minigpu::Device device_;
    minigpu::Kernel kernel_;
};

} // namespace minitensor
