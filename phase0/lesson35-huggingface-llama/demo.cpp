#include "hf_llama.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using minitensor::RuntimeTensor;
using minitensor::hf::HuggingFaceLlama;

constexpr std::array<std::uint32_t, 2> input_token_ids{
    1,
    15043,
};

constexpr std::array<std::uint32_t, 5>
    expected_top_token_ids{
        1789,
        3038,
        30564,
        14836,
        28241,
    };

void require(
    bool condition,
    std::string_view message
) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

[[nodiscard]] std::vector<float> read_reference(
    const std::filesystem::path& path,
    std::size_t expected_count
) {
    std::ifstream input(path, std::ios::binary);

    if (!input) {
        throw std::runtime_error(
            "could not open Hugging Face reference logits: " +
            path.string()
        );
    }

    input.seekg(0, std::ios::end);
    const std::streamoff byte_count = input.tellg();
    input.seekg(0, std::ios::beg);

    const std::size_t expected_bytes =
        expected_count * sizeof(float);

    if (
        byte_count < 0 ||
        static_cast<std::uint64_t>(byte_count) !=
            static_cast<std::uint64_t>(expected_bytes)
    ) {
        throw std::runtime_error(
            "reference-logit file has an unexpected size"
        );
    }

    std::vector<float> values(expected_count);
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(expected_bytes)
    );

    if (!input) {
        throw std::runtime_error(
            "could not read all reference logits"
        );
    }

    return values;
}

[[nodiscard]] std::vector<std::uint32_t> top_token_ids(
    const std::vector<float>& logits,
    std::size_t count
) {
    std::vector<std::uint32_t> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0U);

    const auto greater_logit =
        [&logits](
            std::uint32_t left,
            std::uint32_t right
        ) {
            return logits[left] > logits[right];
        };

    std::partial_sort(
        indices.begin(),
        indices.begin() +
            static_cast<std::ptrdiff_t>(count),
        indices.end(),
        greater_logit
    );
    indices.resize(count);
    return indices;
}

struct Difference {
    float maximum_absolute = 0.0f;
    double mean_absolute = 0.0;
    std::size_t maximum_index = 0;
};

[[nodiscard]] Difference compare(
    const std::vector<float>& actual,
    const std::vector<float>& expected
) {
    require(
        actual.size() == expected.size(),
        "logit vector sizes do not match"
    );

    Difference difference;
    double absolute_sum = 0.0;

    for (
        std::size_t index = 0;
        index < actual.size();
        ++index
    ) {
        require(
            std::isfinite(actual[index]),
            "Metal runtime produced a non-finite logit"
        );
        require(
            std::isfinite(expected[index]),
            "Hugging Face produced a non-finite logit"
        );

        const float absolute =
            std::fabs(actual[index] - expected[index]);
        absolute_sum += static_cast<double>(absolute);

        if (absolute > difference.maximum_absolute) {
            difference.maximum_absolute = absolute;
            difference.maximum_index = index;
        }
    }

    difference.mean_absolute =
        absolute_sum /
        static_cast<double>(actual.size());
    return difference;
}

template <typename Rep, typename Period>
[[nodiscard]] double milliseconds(
    std::chrono::duration<Rep, Period> duration
) {
    return std::chrono::duration<
        double,
        std::milli
    >(duration).count();
}

void print_ids(
    const std::vector<std::uint32_t>& ids
) {
    std::cout << "[";

    for (
        std::size_t index = 0;
        index < ids.size();
        ++index
    ) {
        if (index != 0) {
            std::cout << ", ";
        }

        std::cout << ids[index];
    }

    std::cout << "]";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr
                << "usage: lesson35_hf_llama "
                   "<pinned-tiny-model-directory> "
                   "<reference-last-logits.f32>\n";
            return 2;
        }

        const std::filesystem::path model_directory =
            argv[1];
        const std::filesystem::path reference_path =
            argv[2];
        const std::vector<std::uint32_t> token_ids(
            input_token_ids.begin(),
            input_token_ids.end()
        );

        minigpu::Device device;
        auto queue = device.create_queue();

        const auto load_start =
            std::chrono::steady_clock::now();
        HuggingFaceLlama model =
            HuggingFaceLlama::load(
                device,
                model_directory
            );
        const auto load_end =
            std::chrono::steady_clock::now();

        const auto forward_start =
            std::chrono::steady_clock::now();
        RuntimeTensor logits =
            model.forward(*queue, token_ids);
        const auto forward_end =
            std::chrono::steady_clock::now();

        require(
            logits.shape() ==
                minitensor::Shape{
                    token_ids.size(),
                    model.config().vocabulary_size,
                },
            "model returned the wrong logits shape"
        );

        const std::vector<float> all_logits =
            logits.to_host();
        const std::size_t vocabulary_size =
            model.config().vocabulary_size;
        const std::size_t last_offset =
            (token_ids.size() - 1) * vocabulary_size;
        const std::vector<float> last_logits(
            all_logits.begin() +
                static_cast<std::ptrdiff_t>(last_offset),
            all_logits.end()
        );
        const std::vector<float> reference =
            read_reference(
                reference_path,
                vocabulary_size
            );
        const Difference difference =
            compare(last_logits, reference);
        const std::vector<std::uint32_t> top_ids =
            top_token_ids(
                last_logits,
                expected_top_token_ids.size()
            );

        require(
            std::equal(
                top_ids.begin(),
                top_ids.end(),
                expected_top_token_ids.begin(),
                expected_top_token_ids.end()
            ),
            "Metal and Hugging Face disagree on top-5 IDs"
        );
        require(
            difference.maximum_absolute <= 0.0005f,
            "Metal logits differ from Hugging Face by "
            "more than the teaching tolerance"
        );

        std::cout
            << "Lesson 35: Hugging Face Llama checkpoint\n\n"
            << "Device: " << device.name() << '\n'
            << "Model directory: "
            << model_directory.string() << '\n'
            << "Layers: "
            << model.config().layer_count << '\n'
            << "Hidden size: "
            << model.config().hidden_size << '\n'
            << "Attention heads: "
            << model.config().attention_head_count << '\n'
            << "Vocabulary size: "
            << vocabulary_size << "\n\n"
            << "Prompt encoded by the Hugging Face "
               "tokenizer: \"Hello\"\n"
            << "Input token IDs: ";
        print_ids(token_ids);
        std::cout
            << "\nLogits shape: ["
            << logits.shape()[0] << ", "
            << logits.shape()[1] << "]\n"
            << "Last-token top-5 IDs: ";
        print_ids(top_ids);
        std::cout
            << "\nMaximum absolute difference from "
               "Hugging Face: "
            << std::setprecision(9)
            << difference.maximum_absolute
            << " at vocabulary index "
            << difference.maximum_index
            << "\nMean absolute difference: "
            << difference.mean_absolute
            << "\nModel load: "
            << milliseconds(load_end - load_start)
            << " ms"
            << "\nForward pass: "
            << milliseconds(
                   forward_end - forward_start
               )
            << " ms\n\n"
            << "PASS: our C++/Metal runtime loaded the "
               "Hugging Face safetensors checkpoint and "
               "matched its reference logits.\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
