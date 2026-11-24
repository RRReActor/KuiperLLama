#include "op/encode.h"
#include <glog/logging.h>
#include <sentencepiece_processor.h>

namespace op {
SpeEncodeLayer::~SpeEncodeLayer() = default;
std::string SpeEncodeLayer::decode(int32_t token_id) const {
    CHECK(spe_ != nullptr);
    std::vector<int32_t> token_ids{token_id};
    return spe_->DecodeIds(token_ids);
}

std::string SpeEncodeLayer::decode(const std::vector<int32_t>& token_ids) const {
    CHECK(spe_ != nullptr);
    return spe_->DecodeIds(token_ids);
}

SpeEncodeLayer::SpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos)
        : EncodeLayerBase(std::move(token_model_path), has_bos, has_eos) {
    spe_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
    spe_->LoadOrDie(token_model_path_);
}

std::vector<int32_t> SpeEncodeLayer::encode(const std::string& sentence) const {
    CHECK(spe_ != nullptr);
    std::vector<int32_t> input_ids = spe_->EncodeAsIds(sentence);
    if (has_bos_) {
        input_ids.insert(input_ids.begin(), spe_->bos_id());
    }
    if (has_eos_) {
        input_ids.push_back(spe_->eos_id());
    }
    return input_ids;
}

bool SpeEncodeLayer::is_sentence_ending(int32_t token_id) const {
    CHECK(spe_ != nullptr);
    return token_id == spe_->eos_id();
}

int32_t SpeEncodeLayer::vocab_size() const {
    CHECK(spe_ != nullptr);
    return spe_->GetPieceSize();
}

}  // namespace op
