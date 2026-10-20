#include "hf_llama.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace minitensor::hf {

namespace {

[[nodiscard]] std::string format_shape(
    const Shape& shape
) {
    std::string result = "[";

    for (
        std::size_t axis = 0;
        axis < shape.size();
        ++axis
    ) {
        if (axis != 0) {
            result += ", ";
        }

        result += std::to_string(shape[axis]);
    }

    result += "]";
    return result;
}

void require_shape(
    const RuntimeTensor& tensor,
    Shape expected,
    std::string_view tensor_name
) {
    if (tensor.shape() != expected) {
        throw std::invalid_argument(
            "Hugging Face tensor '" +
            std::string(tensor_name) +
            "' has shape " +
            format_shape(tensor.shape()) +
            ", expected " +
            format_shape(expected)
        );
    }

    if (!tensor.is_contiguous() || tensor.offset() != 0) {
        throw std::invalid_argument(
            "Hugging Face tensor '" +
            std::string(tensor_name) +
            "' must use compact row-major storage"
        );
    }
}

[[nodiscard]] RuntimeTensor load_required_tensor(
    const minigpu::Device& device,
    const SafeTensorFile& file,
    const std::string& name
) {
    if (!file.contains(name)) {
        throw std::invalid_argument(
            "Hugging Face checkpoint is missing required "
            "tensor '" +
            name + "'"
        );
    }

    return file.load_tensor(device, name);
}

void reject_bias(
    const SafeTensorFile& file,
    const std::string& name
) {
    if (file.contains(name)) {
        throw std::invalid_argument(
            "this first Hugging Face Llama runtime does "
            "not support projection bias tensor '" +
            name + "'"
        );
    }
}

void validate_config(const LlamaConfig& config) {
    if (config.model_type != "llama") {
        throw std::invalid_argument(
            "expected a Hugging Face Llama configuration, "
            "received model_type='" +
            config.model_type + "'"
        );
    }

    if (
        config.vocabulary_size == 0 ||
        config.hidden_size == 0 ||
        config.intermediate_size == 0 ||
        config.layer_count == 0 ||
        config.attention_head_count == 0 ||
        config.key_value_head_count == 0 ||
        config.maximum_position_count == 0
    ) {
        throw std::invalid_argument(
            "Llama configuration dimensions must all be "
            "greater than zero"
        );
    }

    if (
        config.hidden_size %
            config.attention_head_count !=
        0
    ) {
        throw std::invalid_argument(
            "Llama hidden_size must be divisible by "
            "num_attention_heads"
        );
    }

    if (
        config.key_value_head_count !=
        config.attention_head_count
    ) {
        throw std::invalid_argument(
            "this first Hugging Face Llama runtime supports "
            "ordinary multi-head attention only: "
            "num_key_value_heads must equal "
            "num_attention_heads"
        );
    }

    const std::size_t head_dimension =
        config.head_dimension();

    if (head_dimension % 2 != 0) {
        throw std::invalid_argument(
            "Llama head dimension must be even for RoPE"
        );
    }

    if (
        !std::isfinite(config.rms_norm_epsilon) ||
        config.rms_norm_epsilon <= 0.0f
    ) {
        throw std::invalid_argument(
            "Llama rms_norm_eps must be finite and positive"
        );
    }

    if (
        !std::isfinite(config.rope_theta) ||
        config.rope_theta <= 0.0f
    ) {
        throw std::invalid_argument(
            "Llama rope_theta must be finite and positive"
        );
    }

    if (config.attention_bias) {
        throw std::invalid_argument(
            "this first Hugging Face Llama runtime does not "
            "support attention projection biases"
        );
    }
}

[[nodiscard]] std::string layer_prefix(
    std::size_t layer
) {
    return
        "model.layers." +
        std::to_string(layer) +
        ".";
}

[[nodiscard]] TransformerBlockWeights load_layer(
    const minigpu::Device& device,
    const SafeTensorFile& file,
    const LlamaConfig& config,
    std::size_t layer
) {
    const std::string prefix = layer_prefix(layer);

    const std::string attention_norm_name =
        prefix + "input_layernorm.weight";
    const std::string query_name =
        prefix + "self_attn.q_proj.weight";
    const std::string key_name =
        prefix + "self_attn.k_proj.weight";
    const std::string value_name =
        prefix + "self_attn.v_proj.weight";
    const std::string attention_output_name =
        prefix + "self_attn.o_proj.weight";
    const std::string ffn_norm_name =
        prefix + "post_attention_layernorm.weight";
    const std::string gate_name =
        prefix + "mlp.gate_proj.weight";
    const std::string up_name =
        prefix + "mlp.up_proj.weight";
    const std::string down_name =
        prefix + "mlp.down_proj.weight";

    reject_bias(
        file,
        prefix + "self_attn.q_proj.bias"
    );
    reject_bias(
        file,
        prefix + "self_attn.k_proj.bias"
    );
    reject_bias(
        file,
        prefix + "self_attn.v_proj.bias"
    );
    reject_bias(
        file,
        prefix + "self_attn.o_proj.bias"
    );
    reject_bias(
        file,
        prefix + "mlp.gate_proj.bias"
    );
    reject_bias(
        file,
        prefix + "mlp.up_proj.bias"
    );
    reject_bias(
        file,
        prefix + "mlp.down_proj.bias"
    );

    RuntimeTensor attention_norm =
        load_required_tensor(
            device,
            file,
            attention_norm_name
        );
    RuntimeTensor query =
        load_required_tensor(device, file, query_name);
    RuntimeTensor key =
        load_required_tensor(device, file, key_name);
    RuntimeTensor value =
        load_required_tensor(device, file, value_name);
    RuntimeTensor attention_output =
        load_required_tensor(
            device,
            file,
            attention_output_name
        );
    RuntimeTensor ffn_norm =
        load_required_tensor(
            device,
            file,
            ffn_norm_name
        );
    RuntimeTensor gate =
        load_required_tensor(device, file, gate_name);
    RuntimeTensor up =
        load_required_tensor(device, file, up_name);
    RuntimeTensor down =
        load_required_tensor(device, file, down_name);

    const std::size_t hidden = config.hidden_size;
    const std::size_t intermediate =
        config.intermediate_size;

    require_shape(
        attention_norm,
        {hidden},
        attention_norm_name
    );
    require_shape(
        query,
        {hidden, hidden},
        query_name
    );
    require_shape(
        key,
        {hidden, hidden},
        key_name
    );
    require_shape(
        value,
        {hidden, hidden},
        value_name
    );
    require_shape(
        attention_output,
        {hidden, hidden},
        attention_output_name
    );
    require_shape(
        ffn_norm,
        {hidden},
        ffn_norm_name
    );
    require_shape(
        gate,
        {intermediate, hidden},
        gate_name
    );
    require_shape(
        up,
        {intermediate, hidden},
        up_name
    );
    require_shape(
        down,
        {hidden, intermediate},
        down_name
    );

    return TransformerBlockWeights{
        std::move(attention_norm),
        std::move(query),
        std::move(key),
        std::move(value),
        std::move(attention_output),
        std::move(ffn_norm),
        std::move(gate),
        std::move(up),
        std::move(down),
    };
}

} // namespace

HuggingFaceLlama::HuggingFaceLlama(
    const minigpu::Device& device,
    LlamaConfig config,
    RuntimeTensor embedding_weight,
    std::vector<TransformerBlockWeights> layers,
    RuntimeTensor final_norm_weight,
    RuntimeTensor lm_head_weight
)
    : device_(device),
      config_(std::move(config)),
      embedding_weight_(std::move(embedding_weight)),
      layers_(std::move(layers)),
      final_norm_weight_(std::move(final_norm_weight)),
      lm_head_weight_(std::move(lm_head_weight)),
      transformer_ops_(device_),
      transformer_block_(device_),
      linear_(device_) {}

HuggingFaceLlama HuggingFaceLlama::load(
    const minigpu::Device& device,
    const std::filesystem::path& model_directory
) {
    LlamaConfig config = load_llama_config(
        model_directory / "config.json"
    );
    validate_config(config);

    SafeTensorFile weights = SafeTensorFile::open(
        model_directory / "model.safetensors"
    );

    constexpr std::size_t tensors_per_layer = 9;

    if (
        config.layer_count >
        weights.tensor_count() / tensors_per_layer
    ) {
        throw std::invalid_argument(
            "Llama config requests " +
            std::to_string(config.layer_count) +
            " layers, but the checkpoint does not contain "
            "enough tensors for that many layers"
        );
    }

    RuntimeTensor embedding_weight =
        load_required_tensor(
            device,
            weights,
            "model.embed_tokens.weight"
        );
    require_shape(
        embedding_weight,
        {
            config.vocabulary_size,
            config.hidden_size,
        },
        "model.embed_tokens.weight"
    );

    std::vector<TransformerBlockWeights> layers;
    layers.reserve(config.layer_count);

    for (
        std::size_t layer = 0;
        layer < config.layer_count;
        ++layer
    ) {
        layers.push_back(
            load_layer(
                device,
                weights,
                config,
                layer
            )
        );
    }

    RuntimeTensor final_norm_weight =
        load_required_tensor(
            device,
            weights,
            "model.norm.weight"
        );
    require_shape(
        final_norm_weight,
        {config.hidden_size},
        "model.norm.weight"
    );

    RuntimeTensor lm_head_weight =
        config.tie_word_embeddings
            ? embedding_weight
            : load_required_tensor(
                  device,
                  weights,
                  "lm_head.weight"
              );
    require_shape(
        lm_head_weight,
        {
            config.vocabulary_size,
            config.hidden_size,
        },
        "lm_head.weight"
    );

    return HuggingFaceLlama(
        device,
        std::move(config),
        std::move(embedding_weight),
        std::move(layers),
        std::move(final_norm_weight),
        std::move(lm_head_weight)
    );
}

RuntimeTensor HuggingFaceLlama::forward(
    minigpu::Queue& queue,
    const std::vector<std::uint32_t>& token_ids
) const {
    if (token_ids.empty()) {
        throw std::invalid_argument(
            "Llama forward requires at least one token ID"
        );
    }

    if (
        token_ids.size() >
        config_.maximum_position_count
    ) {
        throw std::invalid_argument(
            "Llama token count " +
            std::to_string(token_ids.size()) +
            " exceeds max_position_embeddings " +
            std::to_string(
                config_.maximum_position_count
            )
        );
    }

    for (std::uint32_t token_id : token_ids) {
        if (
            static_cast<std::size_t>(token_id) >=
            config_.vocabulary_size
        ) {
            throw std::out_of_range(
                "token ID " +
                std::to_string(token_id) +
                " is outside vocabulary size " +
                std::to_string(config_.vocabulary_size)
            );
        }
    }

    RuntimeTensor hidden = transformer_ops_.embedding(
        queue,
        token_ids,
        embedding_weight_
    );

    for (
        const TransformerBlockWeights& layer : layers_
    ) {
        hidden = transformer_block_.run(
            queue,
            hidden,
            layer,
            config_.attention_head_count,
            0,
            config_.rms_norm_epsilon,
            config_.rope_theta
        );
    }

    RuntimeTensor normalized = transformer_ops_.rms_norm(
        queue,
        hidden,
        final_norm_weight_,
        config_.rms_norm_epsilon
    );

    return linear_.run(
        queue,
        normalized,
        lm_head_weight_
    );
}

const LlamaConfig&
HuggingFaceLlama::config() const noexcept {
    return config_;
}

} // namespace minitensor::hf
