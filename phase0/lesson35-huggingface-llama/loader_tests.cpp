#include "hf_loader.hpp"

#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_load_error(
    Function&& function,
    std::string_view expected_text
) {
    try {
        std::invoke(std::forward<Function>(function));
    } catch (
        const minitensor::hf::HuggingFaceLoadError& error
    ) {
        if (
            std::string_view(error.what()).find(expected_text) ==
            std::string_view::npos
        ) {
            throw std::runtime_error(
                "error message '" +
                std::string(error.what()) +
                "' does not contain '" +
                std::string(expected_text) + "'"
            );
        }

        return;
    }

    throw std::runtime_error(
        "expected HuggingFaceLoadError"
    );
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto unique_value =
            std::chrono::steady_clock::now()
                .time_since_epoch()
                .count();
        path_ =
            std::filesystem::temp_directory_path() /
            (
                "lesson35-loader-tests-" +
                std::to_string(unique_value)
            );
        std::filesystem::create_directory(path_);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        static_cast<void>(
            std::filesystem::remove_all(path_, error)
        );
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(
        const TemporaryDirectory&
    ) = delete;

    [[nodiscard]] std::filesystem::path file(
        std::string_view name
    ) const {
        return path_ / std::string(name);
    }

private:
    std::filesystem::path path_;
};

void write_bytes(
    const std::filesystem::path& path,
    const std::vector<std::uint8_t>& bytes
) {
    std::ofstream output(path, std::ios::binary);

    if (!output) {
        throw std::runtime_error(
            "cannot create test file"
        );
    }

    output.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size())
    );

    if (!output) {
        throw std::runtime_error(
            "cannot write complete test file"
        );
    }
}

void write_text(
    const std::filesystem::path& path,
    std::string_view text
) {
    std::ofstream output(path, std::ios::binary);

    if (!output) {
        throw std::runtime_error(
            "cannot create test text file"
        );
    }

    output.write(
        text.data(),
        static_cast<std::streamsize>(text.size())
    );

    if (!output) {
        throw std::runtime_error(
            "cannot write complete test text file"
        );
    }
}

void append_le_u64(
    std::vector<std::uint8_t>& bytes,
    std::uint64_t value
) {
    for (std::size_t index = 0; index < 8; ++index) {
        bytes.push_back(
            static_cast<std::uint8_t>(
                (value >> (index * 8U)) & 0xFFU
            )
        );
    }
}

void append_f32(
    std::vector<std::uint8_t>& bytes,
    float value
) {
    const std::uint32_t bits =
        std::bit_cast<std::uint32_t>(value);

    for (std::size_t index = 0; index < 4; ++index) {
        bytes.push_back(
            static_cast<std::uint8_t>(
                (bits >> (index * 8U)) & 0xFFU
            )
        );
    }
}

[[nodiscard]] std::vector<std::uint8_t>
make_safetensors(
    std::string header,
    const std::vector<float>& values
) {
    while (header.size() % 8 != 0) {
        header.push_back(' ');
    }

    std::vector<std::uint8_t> bytes;
    append_le_u64(
        bytes,
        static_cast<std::uint64_t>(header.size())
    );
    bytes.insert(
        bytes.end(),
        header.begin(),
        header.end()
    );

    for (float value : values) {
        append_f32(bytes, value);
    }

    return bytes;
}

[[nodiscard]] bool nearly_equal(
    float left,
    float right
) {
    return std::abs(left - right) <= 0.000001f;
}

void test_valid_safetensors(
    const TemporaryDirectory& temporary
) {
    const std::filesystem::path path =
        temporary.file("valid.safetensors");
    write_bytes(
        path,
        make_safetensors(
            "{"
            "\"__metadata__\":{\"format\":\"pt\"},"
            "\"weights\":{"
            "\"dtype\":\"F32\","
            "\"shape\":[2,2],"
            "\"data_offsets\":[0,16]},"
            "\"bias\":{"
            "\"dtype\":\"F32\","
            "\"shape\":[2],"
            "\"data_offsets\":[16,24]}"
            "}",
            {1.0f, -2.0f, 3.5f, 4.0f, 0.25f, -0.5f}
        )
    );

    const minitensor::hf::SafeTensorFile file =
        minitensor::hf::SafeTensorFile::open(path);

    require(file.tensor_count() == 2, "wrong tensor count");
    require(file.contains("weights"), "weights missing");
    require(!file.contains("missing"), "missing tensor found");
    require(
        file.names() ==
            std::vector<std::string>{"bias", "weights"},
        "tensor names are incorrect"
    );
    require(
        file.info("weights").shape ==
            minitensor::Shape{2, 2},
        "weights shape is incorrect"
    );

    const std::vector<float> weights =
        file.read_f32("weights");
    require(weights.size() == 4, "wrong weights size");
    require(nearly_equal(weights[0], 1.0f), "wrong value 0");
    require(nearly_equal(weights[1], -2.0f), "wrong value 1");
    require(nearly_equal(weights[2], 3.5f), "wrong value 2");
    require(nearly_equal(weights[3], 4.0f), "wrong value 3");

    const minigpu::Device device;
    const minitensor::RuntimeTensor tensor =
        file.load_tensor(device, "weights");
    require(
        tensor.shape() == minitensor::Shape{2, 2},
        "uploaded shape is incorrect"
    );
    require(
        tensor.to_host() == weights,
        "uploaded values are incorrect"
    );

    require_load_error(
        [&file] {
            static_cast<void>(file.info("missing"));
        },
        "does not contain tensor"
    );
}

void test_safetensors_rejections(
    const TemporaryDirectory& temporary
) {
    const auto test_invalid =
        [&temporary](
            std::string_view filename,
            std::string header,
            const std::vector<float>& values,
            std::string_view expected_error
        ) {
            const std::filesystem::path path =
                temporary.file(filename);
            write_bytes(
                path,
                make_safetensors(
                    std::move(header),
                    values
                )
            );
            require_load_error(
                [&path] {
                    static_cast<void>(
                        minitensor::hf::SafeTensorFile::open(
                            path
                        )
                    );
                },
                expected_error
            );
        };

    test_invalid(
        "duplicate.safetensors",
        "{"
        "\"same\":{\"dtype\":\"F32\",\"shape\":[1],"
        "\"data_offsets\":[0,4]},"
        "\"s\\u0061me\":{\"dtype\":\"F32\",\"shape\":[1],"
        "\"data_offsets\":[4,8]}"
        "}",
        {1.0f, 2.0f},
        "duplicate object key"
    );

    test_invalid(
        "dtype.safetensors",
        "{\"x\":{\"dtype\":\"F16\",\"shape\":[1],"
        "\"data_offsets\":[0,2]}}",
        {},
        "supports only F32"
    );

    test_invalid(
        "size.safetensors",
        "{\"x\":{\"dtype\":\"F32\",\"shape\":[2],"
        "\"data_offsets\":[0,4]}}",
        {1.0f},
        "shape requires 8 bytes"
    );

    test_invalid(
        "gap.safetensors",
        "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],"
        "\"data_offsets\":[4,8]}}",
        {1.0f, 2.0f},
        "unindexed gap"
    );

    test_invalid(
        "overlap.safetensors",
        "{"
        "\"a\":{\"dtype\":\"F32\",\"shape\":[2],"
        "\"data_offsets\":[0,8]},"
        "\"b\":{\"dtype\":\"F32\",\"shape\":[1],"
        "\"data_offsets\":[4,8]}"
        "}",
        {1.0f, 2.0f},
        "overlap"
    );

    const std::filesystem::path short_path =
        temporary.file("short.safetensors");
    write_bytes(short_path, {1, 2, 3});
    require_load_error(
        [&short_path] {
            static_cast<void>(
                minitensor::hf::SafeTensorFile::open(
                    short_path
                )
            );
        },
        "shorter than"
    );
}

[[nodiscard]] std::string valid_config() {
    return
        "{"
        "\"model_type\":\"llama\","
        "\"hidden_act\":\"silu\","
        "\"vocab_size\":32000,"
        "\"hidden_size\":16,"
        "\"intermediate_size\":64,"
        "\"num_hidden_layers\":2,"
        "\"num_attention_heads\":4,"
        "\"num_key_value_heads\":4,"
        "\"max_position_embeddings\":2048,"
        "\"rms_norm_eps\":1e-6,"
        "\"rope_theta\":10000.0,"
        "\"bos_token_id\":0,"
        "\"eos_token_id\":1,"
        "\"pad_token_id\":-1,"
        "\"attention_bias\":false,"
        "\"tie_word_embeddings\":false,"
        "\"use_cache\":true,"
        "\"pretraining_tp\":1,"
        "\"rope_scaling\":null"
        "}";
}

void test_valid_config(
    const TemporaryDirectory& temporary
) {
    const std::filesystem::path path =
        temporary.file("config.json");
    write_text(path, valid_config());

    const minitensor::hf::LlamaConfig config =
        minitensor::hf::load_llama_config(path);

    require(config.model_type == "llama", "wrong model type");
    require(
        config.hidden_activation == "silu",
        "wrong hidden activation"
    );
    require(config.vocabulary_size == 32000, "wrong vocabulary");
    require(config.hidden_size == 16, "wrong hidden size");
    require(
        config.intermediate_size == 64,
        "wrong intermediate size"
    );
    require(config.layer_count == 2, "wrong layer count");
    require(
        config.attention_head_count == 4,
        "wrong head count"
    );
    require(
        config.key_value_head_count == 4,
        "wrong KV head count"
    );
    require(config.head_dimension() == 4, "wrong head dimension");
    require(
        nearly_equal(config.rms_norm_epsilon, 0.000001f),
        "wrong RMS epsilon"
    );
    require(
        nearly_equal(config.rope_theta, 10000.0f),
        "wrong RoPE theta"
    );

    const std::filesystem::path tied_path =
        temporary.file("tied-config.json");
    std::string tied_config = valid_config();
    const std::string untied =
        "\"tie_word_embeddings\":false";
    const std::size_t tied_position =
        tied_config.find(untied);
    require(
        tied_position != std::string::npos,
        "tied-config replacement text not found"
    );
    tied_config.replace(
        tied_position,
        untied.size(),
        "\"tie_word_embeddings\":true"
    );
    write_text(tied_path, tied_config);

    const minitensor::hf::LlamaConfig tied =
        minitensor::hf::load_llama_config(tied_path);
    require(
        tied.tie_word_embeddings,
        "tied embeddings were not accepted"
    );
}

void test_config_rejections(
    const TemporaryDirectory& temporary
) {
    const auto replace_once = [](
        std::string source,
        std::string_view from,
        std::string_view to
    ) {
        const std::size_t position = source.find(from);

        if (position == std::string::npos) {
            throw std::runtime_error(
                "test replacement text not found"
            );
        }

        source.replace(position, from.size(), to);
        return source;
    };

    const auto test_invalid =
        [&temporary](
            std::string_view filename,
            std::string config,
            std::string_view expected_error
        ) {
            const std::filesystem::path path =
                temporary.file(filename);
            write_text(path, config);
            require_load_error(
                [&path] {
                    static_cast<void>(
                        minitensor::hf::load_llama_config(
                            path
                        )
                    );
                },
                expected_error
            );
        };

    test_invalid(
        "gqa.json",
        replace_once(
            valid_config(),
            "\"num_key_value_heads\":4",
            "\"num_key_value_heads\":2"
        ),
        "grouped-query attention"
    );

    test_invalid(
        "bias.json",
        replace_once(
            valid_config(),
            "\"attention_bias\":false",
            "\"attention_bias\":true"
        ),
        "attention projection bias"
    );

    test_invalid(
        "odd-head.json",
        replace_once(
            replace_once(
                valid_config(),
                "\"hidden_size\":16",
                "\"hidden_size\":12"
            ),
            "\"num_attention_heads\":4",
            "\"num_attention_heads\":4"
        ),
        "even head dimension"
    );

    test_invalid(
        "rope-scaling.json",
        replace_once(
            valid_config(),
            "\"rope_scaling\":null",
            "\"rope_scaling\":{\"factor\":2.0}"
        ),
        "rope_scaling"
    );

    test_invalid(
        "duplicate-config.json",
        replace_once(
            valid_config(),
            "\"model_type\":\"llama\"",
            "\"model_type\":\"llama\","
            "\"model_type\":\"llama\""
        ),
        "duplicate object key"
    );
}

void test_real_checkpoint(
    const std::filesystem::path& config_path,
    const std::filesystem::path& model_path
) {
    const minitensor::hf::LlamaConfig config =
        minitensor::hf::load_llama_config(config_path);
    const minitensor::hf::SafeTensorFile file =
        minitensor::hf::SafeTensorFile::open(model_path);

    require(config.hidden_size == 16, "real model hidden size");
    require(config.layer_count == 2, "real model layer count");
    require(file.tensor_count() == 21, "real model tensor count");
    require(
        file.info("model.embed_tokens.weight").shape ==
            minitensor::Shape{32000, 16},
        "real embedding shape"
    );
    require(
        file.info("lm_head.weight").shape ==
            minitensor::Shape{32000, 16},
        "real LM-head shape"
    );
    require(
        file.info(
            "model.layers.1.mlp.down_proj.weight"
        ).shape == minitensor::Shape{16, 64},
        "real layer weight shape"
    );
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 1 && argc != 3) {
            throw std::invalid_argument(
                "usage: loader_tests [config.json "
                "model.safetensors]"
            );
        }

        const TemporaryDirectory temporary;
        test_valid_safetensors(temporary);
        test_safetensors_rejections(temporary);
        test_valid_config(temporary);
        test_config_rejections(temporary);

        if (argc == 3) {
            test_real_checkpoint(argv[1], argv[2]);
        }

        std::cout
            << "Lesson 35 loader tests passed.\n";
    } catch (const std::exception& error) {
        std::cerr
            << "Lesson 35 loader tests failed: "
            << error.what() << '\n';
        return 1;
    }

    return 0;
}
