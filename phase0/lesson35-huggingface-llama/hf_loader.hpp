#pragma once

#include "../lesson25-tensor-storage-layout/runtime_tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace minitensor::hf {

class HuggingFaceLoadError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class SafeTensorDType {
    F32,
};

struct SafeTensorInfo {
    SafeTensorDType dtype = SafeTensorDType::F32;
    Shape shape;

    // Offsets are byte offsets relative to the safetensors data section,
    // not absolute positions in the file.
    std::uint64_t data_begin = 0;
    std::uint64_t data_end = 0;

    [[nodiscard]] std::size_t element_count() const;
};

// A small, dependency-free reader for a single unsharded safetensors file.
// Lesson 35 intentionally supports F32 tensors only because the selected
// tiny Hugging Face Llama checkpoint stores every parameter as F32.
class SafeTensorFile {
public:
    [[nodiscard]] static SafeTensorFile open(
        const std::filesystem::path& path
    );

    [[nodiscard]] bool contains(
        std::string_view name
    ) const;

    [[nodiscard]] const SafeTensorInfo& info(
        std::string_view name
    ) const;

    [[nodiscard]] std::vector<std::string>
    names() const;

    [[nodiscard]] std::size_t
    tensor_count() const noexcept;

    // Decodes the requested tensor into ordinary host floats.
    [[nodiscard]] std::vector<float> read_f32(
        std::string_view name
    ) const;

    // Decodes and uploads the requested tensor into MiniTensor's shared
    // Metal storage.
    [[nodiscard]] RuntimeTensor load_tensor(
        const minigpu::Device& device,
        std::string_view name
    ) const;

private:
    using TensorMap =
        std::map<
            std::string,
            SafeTensorInfo,
            std::less<>
        >;

    SafeTensorFile(
        std::vector<std::uint8_t> file_bytes,
        std::size_t data_start,
        TensorMap tensors
    );

    std::vector<std::uint8_t> file_bytes_;
    std::size_t data_start_;
    TensorMap tensors_;
};

// The subset of Hugging Face LlamaConfig needed by our miniature runtime.
struct LlamaConfig {
    std::string model_type;
    std::string hidden_activation;

    std::size_t vocabulary_size = 0;
    std::size_t hidden_size = 0;
    std::size_t intermediate_size = 0;
    std::size_t layer_count = 0;
    std::size_t attention_head_count = 0;
    std::size_t key_value_head_count = 0;
    std::size_t maximum_position_count = 0;

    float rms_norm_epsilon = 0.0f;
    float rope_theta = 0.0f;

    std::int64_t bos_token_id = 0;
    std::int64_t eos_token_id = 0;
    std::int64_t pad_token_id = -1;

    bool attention_bias = false;
    bool tie_word_embeddings = false;
    bool use_cache = true;

    [[nodiscard]] std::size_t head_dimension() const;
};

// Parses config.json and rejects Llama variants that the current
// correctness-first runtime cannot execute yet, including GQA, attention
// bias, RoPE scaling, tensor-parallel pretraining layouts, and non-SiLU MLPs.
[[nodiscard]] LlamaConfig load_llama_config(
    const std::filesystem::path& path
);

} // namespace minitensor::hf
