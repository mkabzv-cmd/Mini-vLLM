#include "hf_loader.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <utility>

namespace minitensor::hf {

namespace {

constexpr std::uint64_t maximum_header_bytes =
    100ULL * 1024ULL * 1024ULL;

constexpr std::size_t maximum_json_depth = 64;

struct JsonValue {
    enum class Kind {
        Null,
        Boolean,
        Number,
        String,
        Array,
        Object,
    };

    using Array = std::vector<JsonValue>;
    using Object =
        std::map<std::string, JsonValue, std::less<>>;

    Kind kind = Kind::Null;
    bool boolean = false;
    std::string text;
    Array array;
    Object object;
};

[[noreturn]] void throw_load_error(
    const std::string& message
) {
    throw HuggingFaceLoadError(message);
}

[[nodiscard]] bool is_json_whitespace(char character) {
    return
        character == ' ' ||
        character == '\t' ||
        character == '\n' ||
        character == '\r';
}

[[nodiscard]] bool is_valid_utf8(
    std::string_view text
) {
    std::size_t index = 0;

    while (index < text.size()) {
        const auto first =
            static_cast<unsigned char>(text[index]);

        if (first <= 0x7FU) {
            ++index;
            continue;
        }

        std::size_t continuation_count = 0;
        std::uint32_t code_point = 0;
        std::uint32_t minimum_code_point = 0;

        if ((first & 0xE0U) == 0xC0U) {
            continuation_count = 1;
            code_point = first & 0x1FU;
            minimum_code_point = 0x80U;
        } else if ((first & 0xF0U) == 0xE0U) {
            continuation_count = 2;
            code_point = first & 0x0FU;
            minimum_code_point = 0x800U;
        } else if ((first & 0xF8U) == 0xF0U) {
            continuation_count = 3;
            code_point = first & 0x07U;
            minimum_code_point = 0x10000U;
        } else {
            return false;
        }

        if (
            continuation_count >
            text.size() - index - 1
        ) {
            return false;
        }

        for (
            std::size_t continuation = 0;
            continuation < continuation_count;
            ++continuation
        ) {
            const auto byte = static_cast<unsigned char>(
                text[index + continuation + 1]
            );

            if ((byte & 0xC0U) != 0x80U) {
                return false;
            }

            code_point =
                (code_point << 6U) |
                static_cast<std::uint32_t>(
                    byte & 0x3FU
                );
        }

        if (
            code_point < minimum_code_point ||
            code_point > 0x10FFFFU ||
            (
                code_point >= 0xD800U &&
                code_point <= 0xDFFFU
            )
        ) {
            return false;
        }

        index += continuation_count + 1;
    }

    return true;
}

class JsonParser {
public:
    explicit JsonParser(std::string_view text)
        : text_(text) {}

    [[nodiscard]] JsonValue parse_document() {
        if (!is_valid_utf8(text_)) {
            fail("document is not valid UTF-8");
        }

        skip_whitespace();
        JsonValue result = parse_value(0);
        skip_whitespace();

        if (position_ != text_.size()) {
            fail("unexpected data after the JSON value");
        }

        return result;
    }

private:
    [[noreturn]] void fail(
        const std::string& message
    ) const {
        throw HuggingFaceLoadError(
            "JSON parse error at byte " +
            std::to_string(position_) +
            ": " + message
        );
    }

    void skip_whitespace() {
        while (
            position_ < text_.size() &&
            is_json_whitespace(text_[position_])
        ) {
            ++position_;
        }
    }

    [[nodiscard]] bool consume(char expected) {
        if (
            position_ < text_.size() &&
            text_[position_] == expected
        ) {
            ++position_;
            return true;
        }

        return false;
    }

    void require(char expected) {
        if (!consume(expected)) {
            fail(
                "expected '" +
                std::string(1, expected) + "'"
            );
        }
    }

    [[nodiscard]] JsonValue parse_value(
        std::size_t depth
    ) {
        if (depth > maximum_json_depth) {
            fail("JSON nesting is too deep");
        }

        if (position_ >= text_.size()) {
            fail("expected a JSON value");
        }

        switch (text_[position_]) {
        case 'n':
            return parse_null();
        case 't':
        case 'f':
            return parse_boolean();
        case '"': {
            JsonValue result;
            result.kind = JsonValue::Kind::String;
            result.text = parse_string();
            return result;
        }
        case '[':
            return parse_array(depth);
        case '{':
            return parse_object(depth);
        default:
            if (
                text_[position_] == '-' ||
                (
                    text_[position_] >= '0' &&
                    text_[position_] <= '9'
                )
            ) {
                return parse_number();
            }

            fail("expected a JSON value");
        }
    }

    [[nodiscard]] JsonValue parse_null() {
        constexpr std::string_view literal = "null";

        if (text_.substr(position_, literal.size()) != literal) {
            fail("invalid literal");
        }

        position_ += literal.size();
        return JsonValue{};
    }

    [[nodiscard]] JsonValue parse_boolean() {
        JsonValue result;
        result.kind = JsonValue::Kind::Boolean;

        constexpr std::string_view true_literal = "true";
        constexpr std::string_view false_literal = "false";

        if (
            text_.substr(
                position_,
                true_literal.size()
            ) == true_literal
        ) {
            result.boolean = true;
            position_ += true_literal.size();
            return result;
        }

        if (
            text_.substr(
                position_,
                false_literal.size()
            ) == false_literal
        ) {
            result.boolean = false;
            position_ += false_literal.size();
            return result;
        }

        fail("invalid boolean literal");
    }

    [[nodiscard]] static int hexadecimal_value(
        char character
    ) {
        if (character >= '0' && character <= '9') {
            return character - '0';
        }

        if (character >= 'a' && character <= 'f') {
            return character - 'a' + 10;
        }

        if (character >= 'A' && character <= 'F') {
            return character - 'A' + 10;
        }

        return -1;
    }

    [[nodiscard]] std::uint32_t parse_hexadecimal_quad() {
        if (text_.size() - position_ < 4) {
            fail("incomplete Unicode escape");
        }

        std::uint32_t result = 0;

        for (std::size_t digit = 0; digit < 4; ++digit) {
            const int value =
                hexadecimal_value(text_[position_ + digit]);

            if (value < 0) {
                fail("invalid Unicode escape");
            }

            result =
                (result << 4U) |
                static_cast<std::uint32_t>(value);
        }

        position_ += 4;
        return result;
    }

    static void append_utf8(
        std::string& output,
        std::uint32_t code_point
    ) {
        if (code_point <= 0x7FU) {
            output.push_back(
                static_cast<char>(code_point)
            );
        } else if (code_point <= 0x7FFU) {
            output.push_back(
                static_cast<char>(
                    0xC0U | (code_point >> 6U)
                )
            );
            output.push_back(
                static_cast<char>(
                    0x80U | (code_point & 0x3FU)
                )
            );
        } else if (code_point <= 0xFFFFU) {
            output.push_back(
                static_cast<char>(
                    0xE0U | (code_point >> 12U)
                )
            );
            output.push_back(
                static_cast<char>(
                    0x80U |
                    ((code_point >> 6U) & 0x3FU)
                )
            );
            output.push_back(
                static_cast<char>(
                    0x80U | (code_point & 0x3FU)
                )
            );
        } else {
            output.push_back(
                static_cast<char>(
                    0xF0U | (code_point >> 18U)
                )
            );
            output.push_back(
                static_cast<char>(
                    0x80U |
                    ((code_point >> 12U) & 0x3FU)
                )
            );
            output.push_back(
                static_cast<char>(
                    0x80U |
                    ((code_point >> 6U) & 0x3FU)
                )
            );
            output.push_back(
                static_cast<char>(
                    0x80U | (code_point & 0x3FU)
                )
            );
        }
    }

    [[nodiscard]] std::string parse_string() {
        require('"');
        std::string result;

        while (position_ < text_.size()) {
            const auto character =
                static_cast<unsigned char>(text_[position_]);
            ++position_;

            if (character == static_cast<unsigned char>('"')) {
                return result;
            }

            if (character < 0x20U) {
                fail("unescaped control character in string");
            }

            if (
                character !=
                static_cast<unsigned char>('\\')
            ) {
                result.push_back(
                    static_cast<char>(character)
                );
                continue;
            }

            if (position_ >= text_.size()) {
                fail("incomplete string escape");
            }

            const char escape = text_[position_];
            ++position_;

            switch (escape) {
            case '"':
            case '\\':
            case '/':
                result.push_back(escape);
                break;
            case 'b':
                result.push_back('\b');
                break;
            case 'f':
                result.push_back('\f');
                break;
            case 'n':
                result.push_back('\n');
                break;
            case 'r':
                result.push_back('\r');
                break;
            case 't':
                result.push_back('\t');
                break;
            case 'u': {
                std::uint32_t code_point =
                    parse_hexadecimal_quad();

                if (
                    code_point >= 0xD800U &&
                    code_point <= 0xDBFFU
                ) {
                    if (
                        text_.size() - position_ < 6 ||
                        text_[position_] != '\\' ||
                        text_[position_ + 1] != 'u'
                    ) {
                        fail(
                            "high surrogate is not followed "
                            "by a low surrogate"
                        );
                    }

                    position_ += 2;
                    const std::uint32_t low =
                        parse_hexadecimal_quad();

                    if (
                        low < 0xDC00U ||
                        low > 0xDFFFU
                    ) {
                        fail("invalid low surrogate");
                    }

                    code_point =
                        0x10000U +
                        ((code_point - 0xD800U) << 10U) +
                        (low - 0xDC00U);
                } else if (
                    code_point >= 0xDC00U &&
                    code_point <= 0xDFFFU
                ) {
                    fail("unexpected low surrogate");
                }

                append_utf8(result, code_point);
                break;
            }
            default:
                fail("invalid string escape");
            }
        }

        fail("unterminated string");
    }

    [[nodiscard]] JsonValue parse_number() {
        const std::size_t start = position_;

        static_cast<void>(consume('-'));

        if (position_ >= text_.size()) {
            fail("incomplete number");
        }

        if (consume('0')) {
            if (
                position_ < text_.size() &&
                text_[position_] >= '0' &&
                text_[position_] <= '9'
            ) {
                fail("number has a leading zero");
            }
        } else {
            if (
                text_[position_] < '1' ||
                text_[position_] > '9'
            ) {
                fail("invalid number");
            }

            while (
                position_ < text_.size() &&
                text_[position_] >= '0' &&
                text_[position_] <= '9'
            ) {
                ++position_;
            }
        }

        if (consume('.')) {
            const std::size_t fractional_start = position_;

            while (
                position_ < text_.size() &&
                text_[position_] >= '0' &&
                text_[position_] <= '9'
            ) {
                ++position_;
            }

            if (position_ == fractional_start) {
                fail("fraction has no digits");
            }
        }

        if (
            position_ < text_.size() &&
            (
                text_[position_] == 'e' ||
                text_[position_] == 'E'
            )
        ) {
            ++position_;

            if (
                position_ < text_.size() &&
                (
                    text_[position_] == '+' ||
                    text_[position_] == '-'
                )
            ) {
                ++position_;
            }

            const std::size_t exponent_start = position_;

            while (
                position_ < text_.size() &&
                text_[position_] >= '0' &&
                text_[position_] <= '9'
            ) {
                ++position_;
            }

            if (position_ == exponent_start) {
                fail("exponent has no digits");
            }
        }

        JsonValue result;
        result.kind = JsonValue::Kind::Number;
        result.text = std::string(
            text_.substr(start, position_ - start)
        );
        return result;
    }

    [[nodiscard]] JsonValue parse_array(
        std::size_t depth
    ) {
        require('[');
        JsonValue result;
        result.kind = JsonValue::Kind::Array;
        skip_whitespace();

        if (consume(']')) {
            return result;
        }

        while (true) {
            skip_whitespace();
            result.array.push_back(
                parse_value(depth + 1)
            );
            skip_whitespace();

            if (consume(']')) {
                return result;
            }

            require(',');
        }
    }

    [[nodiscard]] JsonValue parse_object(
        std::size_t depth
    ) {
        require('{');
        JsonValue result;
        result.kind = JsonValue::Kind::Object;
        skip_whitespace();

        if (consume('}')) {
            return result;
        }

        while (true) {
            skip_whitespace();

            if (
                position_ >= text_.size() ||
                text_[position_] != '"'
            ) {
                fail("object key must be a string");
            }

            std::string key = parse_string();
            skip_whitespace();
            require(':');
            skip_whitespace();

            auto [iterator, inserted] =
                result.object.emplace(
                    std::move(key),
                    parse_value(depth + 1)
                );
            static_cast<void>(iterator);

            if (!inserted) {
                fail("duplicate object key");
            }

            skip_whitespace();

            if (consume('}')) {
                return result;
            }

            require(',');
        }
    }

    std::string_view text_;
    std::size_t position_ = 0;
};

[[nodiscard]] const JsonValue::Object& require_object(
    const JsonValue& value,
    std::string_view description
) {
    if (value.kind != JsonValue::Kind::Object) {
        throw_load_error(
            std::string(description) +
            " must be a JSON object"
        );
    }

    return value.object;
}

[[nodiscard]] const JsonValue::Array& require_array(
    const JsonValue& value,
    std::string_view description
) {
    if (value.kind != JsonValue::Kind::Array) {
        throw_load_error(
            std::string(description) +
            " must be a JSON array"
        );
    }

    return value.array;
}

[[nodiscard]] const JsonValue& require_member(
    const JsonValue::Object& object,
    std::string_view name,
    std::string_view description
) {
    const auto iterator = object.find(name);

    if (iterator == object.end()) {
        throw_load_error(
            std::string(description) +
            " is missing required member '" +
            std::string(name) + "'"
        );
    }

    return iterator->second;
}

[[nodiscard]] const JsonValue* optional_member(
    const JsonValue::Object& object,
    std::string_view name
) {
    const auto iterator = object.find(name);

    return iterator == object.end()
        ? nullptr
        : &iterator->second;
}

[[nodiscard]] std::string require_string(
    const JsonValue& value,
    std::string_view description
) {
    if (value.kind != JsonValue::Kind::String) {
        throw_load_error(
            std::string(description) +
            " must be a JSON string"
        );
    }

    return value.text;
}

[[nodiscard]] bool require_boolean(
    const JsonValue& value,
    std::string_view description
) {
    if (value.kind != JsonValue::Kind::Boolean) {
        throw_load_error(
            std::string(description) +
            " must be a JSON boolean"
        );
    }

    return value.boolean;
}

[[nodiscard]] const std::string& require_number_text(
    const JsonValue& value,
    std::string_view description
) {
    if (value.kind != JsonValue::Kind::Number) {
        throw_load_error(
            std::string(description) +
            " must be a JSON number"
        );
    }

    return value.text;
}

[[nodiscard]] std::uint64_t require_u64(
    const JsonValue& value,
    std::string_view description
) {
    const std::string& text =
        require_number_text(value, description);

    if (
        text.find_first_of(".eE-") !=
        std::string::npos
    ) {
        throw_load_error(
            std::string(description) +
            " must be a non-negative integer"
        );
    }

    std::uint64_t result = 0;
    const auto parsed = std::from_chars(
        text.data(),
        text.data() + text.size(),
        result
    );

    if (
        parsed.ec != std::errc{} ||
        parsed.ptr != text.data() + text.size()
    ) {
        throw_load_error(
            std::string(description) +
            " is outside uint64 range"
        );
    }

    return result;
}

[[nodiscard]] std::int64_t require_i64(
    const JsonValue& value,
    std::string_view description
) {
    const std::string& text =
        require_number_text(value, description);

    if (
        text.find_first_of(".eE") !=
        std::string::npos
    ) {
        throw_load_error(
            std::string(description) +
            " must be an integer"
        );
    }

    std::int64_t result = 0;
    const auto parsed = std::from_chars(
        text.data(),
        text.data() + text.size(),
        result
    );

    if (
        parsed.ec != std::errc{} ||
        parsed.ptr != text.data() + text.size()
    ) {
        throw_load_error(
            std::string(description) +
            " is outside int64 range"
        );
    }

    return result;
}

[[nodiscard]] double require_f64(
    const JsonValue& value,
    std::string_view description
) {
    const std::string& text =
        require_number_text(value, description);
    double result = 0.0;

    const auto parsed = std::from_chars(
        text.data(),
        text.data() + text.size(),
        result,
        std::chars_format::general
    );

    if (
        parsed.ec != std::errc{} ||
        parsed.ptr != text.data() + text.size() ||
        !std::isfinite(result)
    ) {
        throw_load_error(
            std::string(description) +
            " must be a finite floating-point value"
        );
    }

    return result;
}

[[nodiscard]] std::size_t u64_to_size(
    std::uint64_t value,
    std::string_view description
) {
    if (
        value >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max()
        )
    ) {
        throw_load_error(
            std::string(description) +
            " does not fit size_t"
        );
    }

    return static_cast<std::size_t>(value);
}

[[nodiscard]] std::size_t require_size(
    const JsonValue& value,
    std::string_view description
) {
    return u64_to_size(
        require_u64(value, description),
        description
    );
}

[[nodiscard]] float require_f32(
    const JsonValue& value,
    std::string_view description
) {
    const double source = require_f64(value, description);

    if (
        source >
            static_cast<double>(
                std::numeric_limits<float>::max()
            ) ||
        source <
            -static_cast<double>(
                std::numeric_limits<float>::max()
            )
    ) {
        throw_load_error(
            std::string(description) +
            " is outside float range"
        );
    }

    const float result = static_cast<float>(source);

    if (!std::isfinite(result)) {
        throw_load_error(
            std::string(description) +
            " is outside float range"
        );
    }

    return result;
}

[[nodiscard]] std::size_t checked_multiply(
    std::size_t left,
    std::size_t right,
    std::string_view description
) {
    if (
        left != 0 &&
        right >
            std::numeric_limits<std::size_t>::max() / left
    ) {
        throw_load_error(
            std::string(description) +
            " overflows size_t"
        );
    }

    return left * right;
}

[[nodiscard]] std::vector<std::uint8_t> read_file(
    const std::filesystem::path& path
) {
    std::ifstream input(
        path,
        std::ios::binary | std::ios::ate
    );

    if (!input) {
        throw_load_error(
            "cannot open file '" + path.string() + "'"
        );
    }

    const std::ifstream::pos_type end_position =
        input.tellg();

    if (end_position < std::ifstream::pos_type(0)) {
        throw_load_error(
            "cannot determine size of file '" +
            path.string() + "'"
        );
    }

    const auto size_as_offset =
        static_cast<std::uintmax_t>(end_position);

    if (
        size_as_offset >
        static_cast<std::uintmax_t>(
            std::numeric_limits<std::size_t>::max()
        ) ||
        size_as_offset >
        static_cast<std::uintmax_t>(
            std::numeric_limits<std::streamsize>::max()
        )
    ) {
        throw_load_error(
            "file is too large for this process: '" +
            path.string() + "'"
        );
    }

    const std::size_t size =
        static_cast<std::size_t>(size_as_offset);
    std::vector<std::uint8_t> bytes(size);

    input.seekg(0, std::ios::beg);

    if (
        size != 0 &&
        !input.read(
            reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(size)
        )
    ) {
        throw_load_error(
            "failed to read complete file '" +
            path.string() + "'"
        );
    }

    return bytes;
}

[[nodiscard]] std::uint64_t read_le_u64(
    const std::vector<std::uint8_t>& bytes,
    std::size_t offset
) {
    if (
        offset > bytes.size() ||
        bytes.size() - offset < 8
    ) {
        throw_load_error(
            "unexpected end of file while reading uint64"
        );
    }

    std::uint64_t result = 0;

    for (std::size_t index = 0; index < 8; ++index) {
        result |=
            static_cast<std::uint64_t>(
                bytes[offset + index]
            ) << (index * 8U);
    }

    return result;
}

[[nodiscard]] std::uint32_t read_le_u32(
    const std::vector<std::uint8_t>& bytes,
    std::size_t offset
) {
    if (
        offset > bytes.size() ||
        bytes.size() - offset < 4
    ) {
        throw_load_error(
            "unexpected end of file while reading uint32"
        );
    }

    std::uint32_t result = 0;

    for (std::size_t index = 0; index < 4; ++index) {
        result |=
            static_cast<std::uint32_t>(
                bytes[offset + index]
            ) << (index * 8U);
    }

    return result;
}

[[nodiscard]] JsonValue parse_json_file(
    const std::filesystem::path& path
) {
    const std::vector<std::uint8_t> bytes =
        read_file(path);
    const char* data =
        bytes.empty()
            ? ""
            : reinterpret_cast<const char*>(
                  bytes.data()
              );
    const std::string_view text(
        data,
        bytes.size()
    );

    return JsonParser(text).parse_document();
}

[[nodiscard]] std::size_t optional_size(
    const JsonValue::Object& object,
    std::string_view name,
    std::size_t default_value
) {
    const JsonValue* value = optional_member(object, name);

    return value == nullptr
        ? default_value
        : require_size(*value, name);
}

[[nodiscard]] float optional_f32(
    const JsonValue::Object& object,
    std::string_view name,
    float default_value
) {
    const JsonValue* value = optional_member(object, name);

    return value == nullptr
        ? default_value
        : require_f32(*value, name);
}

[[nodiscard]] bool optional_boolean(
    const JsonValue::Object& object,
    std::string_view name,
    bool default_value
) {
    const JsonValue* value = optional_member(object, name);

    return value == nullptr
        ? default_value
        : require_boolean(*value, name);
}

[[nodiscard]] std::int64_t optional_i64(
    const JsonValue::Object& object,
    std::string_view name,
    std::int64_t default_value
) {
    const JsonValue* value = optional_member(object, name);

    return value == nullptr
        ? default_value
        : require_i64(*value, name);
}

void require_positive(
    std::size_t value,
    std::string_view name
) {
    if (value == 0) {
        throw_load_error(
            std::string(name) +
            " must be greater than zero"
        );
    }
}

void require_metal_u32(
    std::size_t value,
    std::string_view name
) {
    if (
        value >
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()
        )
    ) {
        throw_load_error(
            std::string(name) +
            " exceeds the current uint32 Metal "
            "indexing model"
        );
    }
}

void require_token_id(
    std::int64_t value,
    std::size_t vocabulary_size,
    std::string_view name,
    bool allow_minus_one
) {
    if (allow_minus_one && value == -1) {
        return;
    }

    if (
        value < 0 ||
        static_cast<std::uint64_t>(value) >=
            static_cast<std::uint64_t>(vocabulary_size)
    ) {
        throw_load_error(
            std::string(name) +
            " must identify a token inside the vocabulary"
        );
    }
}

void validate_supported_config(
    const JsonValue::Object& root,
    const LlamaConfig& config
) {
    if (config.model_type != "llama") {
        throw_load_error(
            "config model_type must be 'llama'"
        );
    }

    if (config.hidden_activation != "silu") {
        throw_load_error(
            "current runtime supports only Llama "
            "hidden_act='silu'"
        );
    }

    require_positive(
        config.vocabulary_size,
        "vocab_size"
    );
    require_positive(
        config.hidden_size,
        "hidden_size"
    );
    require_positive(
        config.intermediate_size,
        "intermediate_size"
    );
    require_positive(
        config.layer_count,
        "num_hidden_layers"
    );
    require_positive(
        config.attention_head_count,
        "num_attention_heads"
    );
    require_positive(
        config.key_value_head_count,
        "num_key_value_heads"
    );
    require_positive(
        config.maximum_position_count,
        "max_position_embeddings"
    );

    require_metal_u32(
        config.vocabulary_size,
        "vocab_size"
    );
    require_metal_u32(
        config.hidden_size,
        "hidden_size"
    );
    require_metal_u32(
        config.intermediate_size,
        "intermediate_size"
    );
    require_metal_u32(
        config.maximum_position_count,
        "max_position_embeddings"
    );

    require_token_id(
        config.bos_token_id,
        config.vocabulary_size,
        "bos_token_id",
        false
    );
    require_token_id(
        config.eos_token_id,
        config.vocabulary_size,
        "eos_token_id",
        false
    );
    require_token_id(
        config.pad_token_id,
        config.vocabulary_size,
        "pad_token_id",
        true
    );

    if (
        config.hidden_size %
            config.attention_head_count !=
        0
    ) {
        throw_load_error(
            "hidden_size must be divisible by "
            "num_attention_heads"
        );
    }

    if (config.head_dimension() % 2 != 0) {
        throw_load_error(
            "Llama RoPE requires an even head dimension"
        );
    }

    if (
        config.key_value_head_count !=
        config.attention_head_count
    ) {
        throw_load_error(
            "current runtime does not yet support "
            "grouped-query attention: "
            "num_key_value_heads must equal "
            "num_attention_heads"
        );
    }

    if (config.attention_bias) {
        throw_load_error(
            "current runtime does not support "
            "attention projection bias"
        );
    }

    if (
        !(config.rms_norm_epsilon > 0.0f) ||
        !std::isfinite(config.rms_norm_epsilon)
    ) {
        throw_load_error(
            "rms_norm_eps must be finite and "
            "greater than zero"
        );
    }

    if (
        !(config.rope_theta > 0.0f) ||
        !std::isfinite(config.rope_theta)
    ) {
        throw_load_error(
            "rope_theta must be finite and "
            "greater than zero"
        );
    }

    if (const JsonValue* rope_scaling =
            optional_member(root, "rope_scaling");
        rope_scaling != nullptr &&
        rope_scaling->kind != JsonValue::Kind::Null
    ) {
        throw_load_error(
            "current runtime does not support "
            "rope_scaling"
        );
    }

    if (
        optional_size(root, "pretraining_tp", 1) != 1
    ) {
        throw_load_error(
            "current runtime requires pretraining_tp=1"
        );
    }

    if (optional_boolean(root, "mlp_bias", false)) {
        throw_load_error(
            "current runtime does not support MLP bias"
        );
    }
}

} // namespace

std::size_t SafeTensorInfo::element_count() const {
    std::size_t count = 1;

    for (std::size_t dimension : shape) {
        count = checked_multiply(
            count,
            dimension,
            "safetensors element count"
        );
    }

    return count;
}

SafeTensorFile::SafeTensorFile(
    std::vector<std::uint8_t> file_bytes,
    std::size_t data_start,
    TensorMap tensors
)
    : file_bytes_(std::move(file_bytes)),
      data_start_(data_start),
      tensors_(std::move(tensors)) {}

SafeTensorFile SafeTensorFile::open(
    const std::filesystem::path& path
) {
    std::vector<std::uint8_t> bytes = read_file(path);

    if (bytes.size() < 8) {
        throw_load_error(
            "safetensors file is shorter than its "
            "8-byte header-length prefix"
        );
    }

    const std::uint64_t header_size =
        read_le_u64(bytes, 0);

    if (header_size == 0) {
        throw_load_error(
            "safetensors header must not be empty"
        );
    }

    if (header_size > maximum_header_bytes) {
        throw_load_error(
            "safetensors header exceeds the 100 MiB "
            "safety limit"
        );
    }

    if (
        header_size >
        static_cast<std::uint64_t>(bytes.size() - 8)
    ) {
        throw_load_error(
            "safetensors header extends beyond "
            "the file"
        );
    }

    const std::size_t header_size_as_size =
        u64_to_size(
            header_size,
            "safetensors header size"
        );
    const std::size_t data_start =
        8 + header_size_as_size;

    if (bytes[8] != static_cast<std::uint8_t>('{')) {
        throw_load_error(
            "safetensors header must begin with '{'"
        );
    }

    const std::string_view header(
        reinterpret_cast<const char*>(bytes.data() + 8),
        header_size_as_size
    );
    const JsonValue root =
        JsonParser(header).parse_document();
    const JsonValue::Object& entries =
        require_object(root, "safetensors header");
    const std::uint64_t data_size =
        static_cast<std::uint64_t>(
            bytes.size() - data_start
        );

    TensorMap tensors;

    for (const auto& [name, value] : entries) {
        if (name == "__metadata__") {
            const JsonValue::Object& metadata =
                require_object(
                    value,
                    "safetensors __metadata__"
                );

            for (const auto& [key, metadata_value] :
                 metadata) {
                static_cast<void>(key);
                static_cast<void>(
                    require_string(
                        metadata_value,
                        "safetensors metadata value"
                    )
                );
            }

            continue;
        }

        const JsonValue::Object& descriptor =
            require_object(
                value,
                "descriptor for tensor '" + name + "'"
            );
        const std::string dtype = require_string(
            require_member(
                descriptor,
                "dtype",
                "tensor descriptor"
            ),
            "tensor dtype"
        );

        if (dtype != "F32") {
            throw_load_error(
                "tensor '" + name + "' uses dtype '" +
                dtype +
                "'; Lesson 35 supports only F32"
            );
        }

        const JsonValue::Array& shape_values =
            require_array(
                require_member(
                    descriptor,
                    "shape",
                    "tensor descriptor"
                ),
                "tensor shape"
            );
        Shape shape;
        shape.reserve(shape_values.size());

        for (const JsonValue& dimension : shape_values) {
            shape.push_back(
                require_size(
                    dimension,
                    "tensor shape dimension"
                )
            );
        }

        const JsonValue::Array& offsets =
            require_array(
                require_member(
                    descriptor,
                    "data_offsets",
                    "tensor descriptor"
                ),
                "tensor data_offsets"
            );

        if (offsets.size() != 2) {
            throw_load_error(
                "tensor '" + name +
                "' data_offsets must contain "
                "exactly two integers"
            );
        }

        SafeTensorInfo info;
        info.dtype = SafeTensorDType::F32;
        info.shape = std::move(shape);
        info.data_begin = require_u64(
            offsets[0],
            "tensor data begin"
        );
        info.data_end = require_u64(
            offsets[1],
            "tensor data end"
        );

        if (info.data_begin > info.data_end) {
            throw_load_error(
                "tensor '" + name +
                "' has reversed data offsets"
            );
        }

        if (info.data_end > data_size) {
            throw_load_error(
                "tensor '" + name +
                "' data extends beyond the file"
            );
        }

        const std::size_t element_count =
            info.element_count();
        const std::size_t expected_byte_count =
            checked_multiply(
                element_count,
                sizeof(float),
                "tensor byte count"
            );
        const std::uint64_t actual_byte_count =
            info.data_end - info.data_begin;

        if (
            static_cast<std::uint64_t>(
                expected_byte_count
            ) != actual_byte_count
        ) {
            throw_load_error(
                "tensor '" + name +
                "' shape requires " +
                std::to_string(expected_byte_count) +
                " bytes, but its offsets describe " +
                std::to_string(actual_byte_count)
            );
        }

        tensors.emplace(name, std::move(info));
    }

    struct Interval {
        std::uint64_t begin;
        std::uint64_t end;
        std::string_view name;
    };

    std::vector<Interval> intervals;
    intervals.reserve(tensors.size());

    for (const auto& [name, info] : tensors) {
        if (info.data_begin != info.data_end) {
            intervals.push_back(Interval{
                info.data_begin,
                info.data_end,
                name,
            });
        }
    }

    std::sort(
        intervals.begin(),
        intervals.end(),
        [](const Interval& left, const Interval& right) {
            if (left.begin != right.begin) {
                return left.begin < right.begin;
            }

            if (left.end != right.end) {
                return left.end < right.end;
            }

            return left.name < right.name;
        }
    );

    std::uint64_t covered_until = 0;

    for (const Interval& interval : intervals) {
        if (interval.begin < covered_until) {
            throw_load_error(
                "safetensors data ranges overlap at "
                "tensor '" +
                std::string(interval.name) + "'"
            );
        }

        if (interval.begin > covered_until) {
            throw_load_error(
                "safetensors data has an unindexed gap "
                "before tensor '" +
                std::string(interval.name) + "'"
            );
        }

        covered_until = interval.end;
    }

    if (covered_until != data_size) {
        throw_load_error(
            "safetensors data has unindexed trailing bytes"
        );
    }

    return SafeTensorFile(
        std::move(bytes),
        data_start,
        std::move(tensors)
    );
}

bool SafeTensorFile::contains(
    std::string_view name
) const {
    return tensors_.find(name) != tensors_.end();
}

const SafeTensorInfo& SafeTensorFile::info(
    std::string_view name
) const {
    const auto iterator = tensors_.find(name);

    if (iterator == tensors_.end()) {
        throw HuggingFaceLoadError(
            "safetensors file does not contain tensor '" +
            std::string(name) + "'"
        );
    }

    return iterator->second;
}

std::vector<std::string>
SafeTensorFile::names() const {
    std::vector<std::string> result;
    result.reserve(tensors_.size());

    for (const auto& [name, tensor_info] : tensors_) {
        static_cast<void>(tensor_info);
        result.push_back(name);
    }

    return result;
}

std::size_t
SafeTensorFile::tensor_count() const noexcept {
    return tensors_.size();
}

std::vector<float> SafeTensorFile::read_f32(
    std::string_view name
) const {
    const SafeTensorInfo& tensor_info = info(name);
    const std::size_t count =
        tensor_info.element_count();
    std::vector<float> result;
    result.reserve(count);

    const std::size_t relative_begin =
        u64_to_size(
            tensor_info.data_begin,
            "tensor data begin"
        );

    if (
        relative_begin >
        file_bytes_.size() - data_start_
    ) {
        throw HuggingFaceLoadError(
            "tensor data offset is outside the file"
        );
    }

    const std::size_t absolute_begin =
        data_start_ + relative_begin;

    for (std::size_t index = 0; index < count; ++index) {
        const std::size_t byte_offset =
            absolute_begin +
            checked_multiply(
                index,
                sizeof(float),
                "tensor element byte offset"
            );
        const std::uint32_t bits =
            read_le_u32(file_bytes_, byte_offset);
        result.push_back(std::bit_cast<float>(bits));
    }

    return result;
}

RuntimeTensor SafeTensorFile::load_tensor(
    const minigpu::Device& device,
    std::string_view name
) const {
    const SafeTensorInfo& tensor_info = info(name);

    if (
        std::any_of(
            tensor_info.shape.begin(),
            tensor_info.shape.end(),
            [](std::size_t dimension) {
                return dimension == 0;
            }
        )
    ) {
        throw HuggingFaceLoadError(
            "MiniTensor does not support empty tensor '" +
            std::string(name) + "'"
        );
    }

    miniir::Type type =
        tensor_info.shape.empty()
            ? miniir::Type::scalar(miniir::DType::F32)
            : miniir::Type::tensor(
                  miniir::DType::F32,
                  tensor_info.shape
              );

    return RuntimeTensor::from_host(
        device,
        std::move(type),
        read_f32(name)
    );
}

std::size_t LlamaConfig::head_dimension() const {
    if (attention_head_count == 0) {
        throw HuggingFaceLoadError(
            "cannot calculate head dimension with "
            "zero attention heads"
        );
    }

    if (hidden_size % attention_head_count != 0) {
        throw HuggingFaceLoadError(
            "hidden_size is not divisible by "
            "num_attention_heads"
        );
    }

    return hidden_size / attention_head_count;
}

LlamaConfig load_llama_config(
    const std::filesystem::path& path
) {
    const JsonValue document = parse_json_file(path);
    const JsonValue::Object& root =
        require_object(document, "Llama config");

    LlamaConfig config;
    config.model_type = require_string(
        require_member(
            root,
            "model_type",
            "Llama config"
        ),
        "model_type"
    );
    config.hidden_activation = require_string(
        require_member(
            root,
            "hidden_act",
            "Llama config"
        ),
        "hidden_act"
    );
    config.vocabulary_size = require_size(
        require_member(
            root,
            "vocab_size",
            "Llama config"
        ),
        "vocab_size"
    );
    config.hidden_size = require_size(
        require_member(
            root,
            "hidden_size",
            "Llama config"
        ),
        "hidden_size"
    );
    config.intermediate_size = require_size(
        require_member(
            root,
            "intermediate_size",
            "Llama config"
        ),
        "intermediate_size"
    );
    config.layer_count = require_size(
        require_member(
            root,
            "num_hidden_layers",
            "Llama config"
        ),
        "num_hidden_layers"
    );
    config.attention_head_count = require_size(
        require_member(
            root,
            "num_attention_heads",
            "Llama config"
        ),
        "num_attention_heads"
    );
    config.key_value_head_count = optional_size(
        root,
        "num_key_value_heads",
        config.attention_head_count
    );
    config.maximum_position_count = require_size(
        require_member(
            root,
            "max_position_embeddings",
            "Llama config"
        ),
        "max_position_embeddings"
    );
    config.rms_norm_epsilon = require_f32(
        require_member(
            root,
            "rms_norm_eps",
            "Llama config"
        ),
        "rms_norm_eps"
    );
    config.rope_theta = optional_f32(
        root,
        "rope_theta",
        10000.0f
    );
    config.bos_token_id = require_i64(
        require_member(
            root,
            "bos_token_id",
            "Llama config"
        ),
        "bos_token_id"
    );
    config.eos_token_id = require_i64(
        require_member(
            root,
            "eos_token_id",
            "Llama config"
        ),
        "eos_token_id"
    );
    config.pad_token_id = optional_i64(
        root,
        "pad_token_id",
        -1
    );
    config.attention_bias = optional_boolean(
        root,
        "attention_bias",
        false
    );
    config.tie_word_embeddings = optional_boolean(
        root,
        "tie_word_embeddings",
        false
    );
    config.use_cache = optional_boolean(
        root,
        "use_cache",
        true
    );

    validate_supported_config(root, config);
    return config;
}

} // namespace minitensor::hf
