#pragma once

#include "../lesson14-metal-minigpu/minigpu.hpp"
#include "../lesson21-types-shapes-broadcasting/miniir.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace minitensor {

using Shape = std::vector<std::size_t>;
using Strides = std::vector<std::size_t>;

// TensorLayout contains no values. It only maps logical tensor coordinates
// to indexes in one flat storage buffer.
class TensorLayout {
public:
    [[nodiscard]] static TensorLayout contiguous(
        Shape shape,
        std::size_t offset = 0
    );

    TensorLayout(
        Shape shape,
        Strides strides,
        std::size_t offset = 0
    );

    [[nodiscard]] const Shape& shape() const noexcept;
    [[nodiscard]] const Strides& strides() const noexcept;
    [[nodiscard]] std::size_t offset() const noexcept;
    [[nodiscard]] std::size_t rank() const noexcept;
    [[nodiscard]] std::size_t element_count() const noexcept;

    // Smallest storage capacity that can contain every address referenced by
    // this layout. Strides and offset are measured in float elements.
    [[nodiscard]] std::size_t
    minimum_storage_size() const noexcept;

    [[nodiscard]] bool is_contiguous() const noexcept;

    [[nodiscard]] std::size_t storage_index(
        const std::vector<std::size_t>& coordinates
    ) const;

    // Interprets linear_index in logical row-major order, then applies this
    // layout's strides and offset.
    [[nodiscard]] std::size_t storage_index_from_linear(
        std::size_t linear_index
    ) const;

private:
    Shape shape_;
    Strides strides_;
    std::size_t offset_;
    std::size_t element_count_;
    std::size_t minimum_storage_size_;
    bool contiguous_;
};

[[nodiscard]] std::string format_layout(
    const TensorLayout& layout
);

// Unlike Lesson 24's symbolic Tensor, RuntimeTensor contains real
// GPU-accessible storage. This lesson's storage is f32 only.
class RuntimeTensor {
public:
    [[nodiscard]] static RuntimeTensor empty(
        const minigpu::Device& device,
        miniir::Type type
    );

    [[nodiscard]] static RuntimeTensor from_host(
        const minigpu::Device& device,
        miniir::Type type,
        std::vector<float> values
    );

    // Low-level constructor used by later view operations: a copied Buffer
    // handle shares the same underlying Metal allocation.
    RuntimeTensor(
        miniir::Type type,
        TensorLayout layout,
        minigpu::Buffer storage
    );

    [[nodiscard]] const miniir::Type& type() const noexcept;
    [[nodiscard]] const TensorLayout& layout() const noexcept;
    [[nodiscard]] const Shape& shape() const noexcept;
    [[nodiscard]] const Strides& strides() const noexcept;
    [[nodiscard]] std::size_t offset() const noexcept;
    [[nodiscard]] std::size_t rank() const noexcept;
    [[nodiscard]] std::size_t numel() const noexcept;
    [[nodiscard]] std::size_t storage_size() const noexcept;
    [[nodiscard]] bool is_contiguous() const noexcept;

    [[nodiscard]] std::size_t storage_index(
        const std::vector<std::size_t>& coordinates
    ) const;

    [[nodiscard]] float load(
        const std::vector<std::size_t>& coordinates
    ) const;

    // RuntimeTensor is a shared storage handle, like minigpu::Buffer.
    // const protects the handle/metadata, not the referenced buffer values.
    void store(
        const std::vector<std::size_t>& coordinates,
        float value
    ) const;

    [[nodiscard]] float load_flat(
        std::size_t logical_index
    ) const;

    void store_flat(
        std::size_t logical_index,
        float value
    ) const;

    [[nodiscard]] std::vector<float> to_host() const;

    [[nodiscard]] const minigpu::Buffer&
    buffer() const noexcept;

private:
    miniir::Type type_;
    TensorLayout layout_;
    minigpu::Buffer storage_;
};

} // namespace minitensor
