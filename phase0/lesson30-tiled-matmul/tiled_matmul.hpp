#pragma once

#include "../lesson25-tensor-storage-layout/runtime_tensor.hpp"

#include <cstddef>

namespace minitensor {

inline constexpr std::size_t tiled_matmul_tile_size = 8;
inline constexpr std::size_t
    tiled_matmul_threads_per_group =
        tiled_matmul_tile_size *
        tiled_matmul_tile_size;

// An 8x8 teaching matmul. One 64-lane threadgroup cooperatively computes
// one 8x8 output tile using threadgroup memory.
class TiledMatmul {
public:
    explicit TiledMatmul(
        const minigpu::Device& device
    );

    [[nodiscard]] RuntimeTensor run(
        minigpu::Queue& queue,
        const RuntimeTensor& a,
        const RuntimeTensor& b
    ) const;

private:
    minigpu::Device device_;
    minigpu::Kernel kernel_;
};

} // namespace minitensor
