#include "op/encode.h"
#include "base/tiktoken.h"
#include "base/unicode.h"
#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <fstream>
#include <nlohmann/json.hpp>

namespace op {
// A matcher owns per-call state. The compiled expression can be shared by callers.
struct BpeEncodeLayer::PreTokenizer {
    std::unique_ptr<icu::RegexPattern> pattern;
    const icu::Normalizer2* normalizer = nullptr;

    PreTokenizer(const std::string& expression, bool use_nfc) {
        UErrorCode status = U_ZERO_ERROR;
        pattern.reset(icu::RegexPattern::compile(icu::UnicodeString::fromUTF8(expression), 0, status));
        if (use_nfc) normalizer = icu::Normalizer2::getNFCInstance(status);
        if (U_FAILURE(status)) throw std::runtime_error(u_errorName(status));
    }

    void encode(const std::string& segment, const tiktoken::tiktoken& bpe, std::vector<int>& ids) const {
        UErrorCode status = U_ZERO_ERROR;
        auto text = icu::UnicodeString::fromUTF8(segment);
        if (normalizer) text = normalizer->normalize(text, status);
        std::unique_ptr<icu::RegexMatcher> matcher(pattern->matcher(text, status));
        if (U_FAILURE(status)) throw std::runtime_error(u_errorName(status));
        auto append = [&](int32_t begin, int32_t end) {
            if (begin == end) return;
            std::string piece;
            text.tempSubStringBetween(begin, end).toUTF8String(piece);
            auto tokens = bpe.encode_single_piece(piece);
            ids.insert(ids.end(), tokens.begin(), tokens.end());
        };
        int32_t end = 0;
        while (matcher->find(status)) {
            int32_t begin = matcher->start(status);
            append(end, begin);
            end = matcher->end(status);
            append(begin, end);
        }
        if (U_FAILURE(status)) throw std::runtime_error(u_errorName(status));
        append(end, text.length());
    }
};

BpeEncodeLayer::~BpeEncodeLayer() = default;
static const std::string PAT_STR =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

BpeEncodeLayer::BpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos)
        : BpeEncodeLayer(std::move(token_model_path), has_bos, has_eos,
                                          "<|begin_of_text|>", "<|end_of_text|>", "<|eot_id|>") {}

BpeEncodeLayer::BpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos,
                                                              const char* bos_token, const char* eos_token, const char* stop_token)
        : EncodeLayerBase(std::move(token_model_path), has_bos, has_eos) {
    using json = nlohmann::json;
    std::ifstream f(token_model_path_);
    CHECK(f.is_open())
            << "The token model path is not valid, please check the path and type of token model.";
    json data;
    try {
        data = json::parse(f);
    } catch (json::parse_error&) {
        LOG(FATAL)
                << "The token model path is not valid, please check the path and type of token model.";
    }

    if (data.contains("pre_tokenizer") && data["pre_tokenizer"].is_object()) {
        const auto& config = data.at("pre_tokenizer");
        if (config.value("type", "") == "Sequence") {
            const auto& stages = config.at("pretokenizers");
            if (stages.size() == 2 && stages[0].value("type", "") == "Split" &&
                stages[0].value("behavior", "") == "Isolated" && !stages[0].value("invert", false) &&
                stages[1].value("type", "") == "ByteLevel" && !stages[1].value("use_regex", true) &&
                !stages[1].value("add_prefix_space", true)) {
                bool nfc = data.contains("normalizer") && data["normalizer"].is_object() &&
                           data["normalizer"].value("type", "") == "NFC";
                pre_tokenizer_ = std::make_unique<PreTokenizer>(stages[0].at("pattern").at("Regex"), nfc);
            }
        }
    }

    const auto& datas = data["added_tokens"];
    ankerl::unordered_dense::map<std::string, int> special_tokens;
    for (const auto& data1 : datas) {
        int id = data1["id"];
        std::string content = data1["content"];
        special_tokens.insert({content, id});
    }

    ankerl::unordered_dense::map<std::string, int> encoder;
    const auto& vocabs = data["model"]["vocab"];
    const auto& vocab_items = vocabs.items();
    for (const auto& v : vocab_items) {
        const auto cpts = unicode_cpts_from_utf8(v.key());
        std::string key;
        for (const auto cpt : cpts) {
            const auto utf8 = unicode_cpt_to_utf8(cpt);
            key += unicode_utf8_to_byte(utf8);
        }
        const int32_t id = v.value();
        encoder[key] = id;
    }
    bos_id_ = special_tokens.at(bos_token);
    eos_id_ = special_tokens.at(eos_token);
    stop_token1_ = eos_id_;
    stop_token2_ = special_tokens.at(stop_token);

    num_token_ = encoder.size() + special_tokens.size();
    tiktoken_ = std::make_unique<tiktoken::tiktoken>(std::move(encoder), std::move(special_tokens), PAT_STR);
}

std::vector<int32_t> BpeEncodeLayer::encode(const std::string& sentence) const {
    CHECK(tiktoken_ != nullptr);
    auto input_ids = pre_tokenizer_
        ? tiktoken_->encode(sentence, [this](const std::string& segment, std::vector<int>& ids) {
              pre_tokenizer_->encode(segment, *tiktoken_, ids);
          })
        : tiktoken_->encode(sentence);

    if (has_bos_) {
        input_ids.insert(input_ids.begin(), bos_id_);
    }
    if (has_eos_) {
        input_ids.push_back(eos_id_);
    }
    return input_ids;
}

std::string BpeEncodeLayer::decode(int32_t token_id) const { return decode(std::vector<int32_t>{token_id}); }

std::string BpeEncodeLayer::decode(const std::vector<int32_t>& token_ids) const {
    CHECK(tiktoken_ != nullptr);
    return tiktoken_->decode(token_ids);
}

bool BpeEncodeLayer::is_sentence_ending(int32_t token_id) const {
    return token_id == stop_token1_ || token_id == stop_token2_;
}

int32_t BpeEncodeLayer::vocab_size() const {
    CHECK(tiktoken_ != nullptr);
    return num_token_;
}

QwenEncodeLayer::QwenEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos)
        : BpeEncodeLayer(std::move(token_model_path), has_bos, has_eos,
                                          "<|im_start|>", "<|im_end|>", "<|endoftext|>") {}

}  // namespace op
