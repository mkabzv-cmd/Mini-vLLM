#include "linear.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace minitensor {

namespace {

constexpr std::string_view kernel_name =
    "fused_linear";

constexpr std::string_view kernel_source = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void fused_linear(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& m_size [[buffer(4)]],
    constant uint& n_size [[buffer(5)]],
    constant uint& k_size [[buffer(6)]],
    constant uint& input_offset [[buffer(7)]],
    constant uint& weight_offset [[buffer(8)]],
    constant uint& bias_offset [[buffer(9)]],
    constant uint& has_bias [[buffer(10)]],
    constant uint& activation [[buffer(11)]],
    uint global_id [[thread_position_in_grid]]
) {
    const uint output_count = m_size * n_size;

    if (global_id >= output_count) {
        return;
    }

    const uint row = global_id / n_size;
    const uint column = global_id % n_size;
    const uint input_base =
        input_offset + row * k_size;
    const uint weight_base =
        weight_offset + column * k_size;

    float sum = 0.0f;

    for (uint inner = 0; inner < k_size; ++inner) {
        sum +=
            input[input_base + inner] *
            weight[weight_base + inner];
    }

    if (has_bias != 0) {
        sum += bias[bias_offset + column];
    }

    if (activation != 0) {
        sum = max(sum, 0.0f);
    }

    output[global_id] = sum;
}
)metal";

constexpr std::size_t input_binding = 0;
constexpr std::size_t weight_binding = 1;
constexpr std::size_t bias_binding = 2;
constexpr std::size_t output_binding = 3;
constexpr std::size_t m_binding = 4;
constexpr std::size_t n_binding = 5;
constexpr std::size_t k_binding = 6;
constexpr std::size_t input_offset_binding = 7;
constexpr std::size_t weight_offset_binding = 8;
constexpr std::size_t bias_offset_binding = 9;
constexpr std::size_t has_bias_binding = 10;
constexpr std::size_t activation_binding = 11;

[[nodiscard]] std::uint32_t require_u32(
    std::size_t value,
    const char* description
) {
    if (
        value >
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()
        )
    ) {
        throw std::overflow_error(
            std::string(description) +
            " does not fit the Lesson 31 uint32 "
            "Metal indexing model"
        );
    }

    return static_cast<std::uint32_t>(value);
}

[[nodiscard]] std::size_t checked_multiply(
    std::size_t left,
    std::size_t right,
    const char* description
) {
    if (
        left != 0 &&
        right >
            std::numeric_limits<std::size_t>::max() /
                left
    ) {
        throw std::overflow_error(
            std::string(description) +
            " overflows size_t"
        );
    }

    return left * right;
}

void require_rank(
    const RuntimeTensor& tensor,
    std::size_t expected,
    const char* name
) {
    if (tensor.rank() != expected) {
        throw std::invalid_argument(
            std::string("linear ") + name +
            " must have rank " +
            std::to_string(expected) +
            "; received rank " +
            std::to_string(tensor.rank())
        );
    }
}

void require_contiguous(
    const RuntimeTensor& tensor,
    const char* name
) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(
            std::string("linear ") + name +
            " must use contiguous row-major storage; "
            "materialize the view first"
        );
    }
}

void require_addressable_u32(
    const RuntimeTensor& tensor,
    const char* description
) {
    static_cast<void>(
        require_u32(
            tensor.layout().minimum_storage_size() - 1,
            description
        )
    );
}

[[nodiscard]] std::uint32_t activation_code(
    LinearActivation activation
) {
    switch (activation) {
        case LinearActivation::None:
            return 0;
        case LinearActivation::Relu:
            return 1;
    }

    throw std::invalid_argument(
        "linear activation is not supported"
    );
}

} // namespace

LinearOperator::LinearOperator(
    const minigpu::Device& device
)
    : device_(device),
      kernel_(
          device_.compile_kernel(
              std::string(kernel_name),
              kernel_source
          )
      ) {}

RuntimeTensor LinearOperator::run(
    minigpu::Queue& queue,
    const RuntimeTensor& input,
    const RuntimeTensor& weight,
    const RuntimeTensor* bias,
    LinearActivation activation,
    std::size_t threads_per_group
) const {
    if (threads_per_group == 0) {
        throw std::invalid_argument(
            "threads_per_group must be greater than zero"
        );
    }

    require_rank(input, 2, "input");
    require_rank(weight, 2, "weight");
    require_contiguous(input, "input");
    require_contiguous(weight, "weight");

    const std::size_t m = input.shape()[0];
    const std::size_t k = input.shape()[1];
    const std::size_t n = weight.shape()[0];

    if (weight.shape()[1] != k) {
        throw std::invalid_argument(
            "linear inner dimensions do not match: "
            "input K=" + std::to_string(k) +
            ", weight K=" +
            std::to_string(weight.shape()[1])
        );
    }

    if (bias != nullptr) {
        require_rank(*bias, 1, "bias");
        require_contiguous(*bias, "bias");

        if (bias->shape()[0] != n) {
            throw std::invalid_argument(
                "linear bias length " +
                std::to_string(bias->shape()[0]) +
                " does not match output width N=" +
                std::to_string(n)
            );
        }
    }

    const std::size_t output_count =
        checked_multiply(
            m,
            n,
            "linear output element count"
        );

    const std::uint32_t m_u32 =
        require_u32(m, "linear M");
    const std::uint32_t n_u32 =
        require_u32(n, "linear N");
    const std::uint32_t k_u32 =
        require_u32(k, "linear K");
    static_cast<void>(
        require_u32(
            output_count,
            "linear output element count"
        )
    );

    require_addressable_u32(
        input,
        "highest linear input storage index"
    );
    require_addressable_u32(
        weight,
        "highest linear weight storage index"
    );

    if (bias != nullptr) {
        require_addressable_u32(
            *bias,
            "highest linear bias storage index"
        );
    }

    const minigpu::LaunchConfig launch =
        minigpu::LaunchConfig::cover(
            output_count,
            threads_per_group
        );
    static_cast<void>(
        require_u32(
            launch.total_threads(),
            "linear launched thread count"
        )
    );

    const std::uint32_t input_offset =
        require_u32(
            input.offset(),
            "linear input offset"
        );
    const std::uint32_t weight_offset =
        require_u32(
            weight.offset(),
            "linear weight offset"
        );
    const std::uint32_t bias_offset =
        bias == nullptr
            ? 0
            : require_u32(
                  bias->offset(),
                  "linear bias offset"
              );
    const std::uint32_t has_bias =
        bias == nullptr ? 0 : 1;
    const std::uint32_t activation_value =
        activation_code(activation);

    miniir::Type output_type =
        miniir::Type::tensor(
            miniir::DType::F32,
            {m, n}
        );
    RuntimeTensor output =
        RuntimeTensor::empty(
            device_,
            std::move(output_type)
        );

    minigpu::Dispatch dispatch(kernel_);
    dispatch
        .bind_buffer(input_binding, input.buffer())
        .bind_buffer(weight_binding, weight.buffer())
        // Metal still needs a valid binding when bias is disabled. The
        // kernel does not read this buffer while has_bias is zero.
        .bind_buffer(
            bias_binding,
            bias == nullptr
                ? output.buffer()
                : bias->buffer()
        )
        .bind_buffer(output_binding, output.buffer())
        .bind_u32(m_binding, m_u32)
        .bind_u32(n_binding, n_u32)
        .bind_u32(k_binding, k_u32)
        .bind_u32(
            input_offset_binding,
            input_offset
        )
        .bind_u32(
            weight_offset_binding,
            weight_offset
        )
        .bind_u32(
            bias_offset_binding,
            bias_offset
        )
        .bind_u32(has_bias_binding, has_bias)
        .bind_u32(
            activation_binding,
            activation_value
        );

    minigpu::Event finished = queue.submit(
        std::move(dispatch),
        launch
    );
    finished.wait();

    return output;
}

} // namespace minitensor
