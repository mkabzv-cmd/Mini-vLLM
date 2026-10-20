#pragma once

#include "../lesson25-tensor-storage-layout/runtime_tensor.hpp"

#include <cstddef>

namespace minitensor {

// Return a zero-copy view with two logical axes exchanged.
[[nodiscard]] RuntimeTensor transpose(
    const RuntimeTensor& source,
    std::size_t first_axis,
    std::size_t second_axis
);

// Return a zero-copy view over `length` positions of one axis:
//
//     start, start + step, start + 2 * step, ...
//
// Lesson 26 supports only non-empty, positive-step slices.
[[nodiscard]] RuntimeTensor slice(
    const RuntimeTensor& source,
    std::size_t axis,
    std::size_t start,
    std::size_t length,
    std::size_t step = 1
);

} // namespace minitensor
