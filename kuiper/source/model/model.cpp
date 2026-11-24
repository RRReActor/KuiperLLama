#include "model/model.h"
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <limits>
#include <cstring>
#include <utility>
namespace model
{
    Model::Model(base::TokenizerType tokenizer_type, base::ModelType model_type,
                 std::string token_path, std::string model_path, bool is_quant_model)
        : is_quant_model_(is_quant_model), token_path_(std::move(token_path)), model_path_(std::move(model_path)),
          model_type_(model_type), tokenizer_type_(tokenizer_type) {}

    base::ModelType Model::model_type() const { return model_type_; }

    const std::string &Model::token_path() const { return token_path_; }

    const std::string &Model::model_path() const { return model_path_; }

    base::Status Model::insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor &tensor)
    {
        if (buffers_.count(buffer_idx) > 0)
        {
            return base::error::KeyHasExits(std::to_string(int(buffer_idx)) + " has exits in the buffers");
        }
        if (tensor.is_empty())
        {
            return base::error::InvalidArgument("The tensor is empty for inserting buffer.");
        }
        buffers_.insert({buffer_idx, tensor});
        return base::error::Success();
    }

    tensor::Tensor &Model::get_buffer(ModelBufferType buffer_idx)
    {
        CHECK_GT(buffers_.count(buffer_idx), 0) << int(buffer_idx);
        return buffers_.at(buffer_idx);
    }

    const tensor::Tensor &Model::get_buffer(ModelBufferType buffer_idx) const
    {
        CHECK_GT(buffers_.count(buffer_idx), 0);
        return buffers_.at(buffer_idx);
    }

    base::Status Model::generate_model_infos(const ModelConfig &config) const
    {
        if (config.dim <= 0 || config.hidden_dim <= 0 || config.layer_num <= 0 ||
            config.head_num <= 0 || config.kv_head_num <= 0 || config.seq_len <= 0 ||
            config.dim % config.head_num || config.head_num % config.kv_head_num ||
            (config.dim / config.head_num) % 2 || config.immediate_dim_ <= 0 ||
            config.vocab_size == 0 || config.vocab_size == std::numeric_limits<int32_t>::min()) {
            return base::error::ModelParseError("Invalid Qwen3 model dimensions");
        }
        config_->dim_ = config.dim;
        config_->hidden_dim_ = config.hidden_dim;
        config_->layer_num_ = config.layer_num;
        config_->head_num_ = config.head_num;
        config_->kv_head_num_ = config.kv_head_num;
        config_->seq_len_ = config.seq_len;

        config_->kv_dim_ = (config.dim / config.head_num) * config.kv_head_num;
        config_->kv_mul_ = config.head_num / config.kv_head_num;
        config_->head_size_ = config.dim / config.head_num;
        #if defined(QWEN3_SUPPORT)
        config_->immediate_dim_ = config.immediate_dim_;
        #endif
        if (config.vocab_size > 0) {
            config_->is_shared_weight_ = true;
        } else {
            config_->is_shared_weight_ = false;
        }
        config_->vocab_size_ = std::abs(config.vocab_size);
        return base::error::Success();
    }

    base::Status Model::read_model_file() {
        using namespace base;
        if (model_path_.empty()) {
            return error::PathNotValid("Failed to open the weight file, the model path is empty!");
        }
        int32_t fd = open(model_path_.data(), O_RDONLY);
        if (fd == -1) {
            return error::PathNotValid("Failed to open the weight file " + model_path_ +
                                    " may be the path does not exist!");
        }
        
        std::shared_ptr<RawModelData> mapped;
        if (is_quant_model_) mapped = std::make_shared<RawModelDataInt8>();
        else mapped = std::make_shared<RawModelDataFp32>();
        mapped->fd = fd;
        struct stat sb{};
        const size_t header_bytes = sizeof(ModelConfig) + (is_quant_model_ ? sizeof(int32_t) : 0);
        if (fstat(fd, &sb) != 0 || sb.st_size < static_cast<off_t>(header_bytes)) {
            return error::ModelParseError("Incomplete model header");
        }
        mapped->file_size = sb.st_size;
        mapped->data = mmap(nullptr, mapped->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped->data == MAP_FAILED) {
            return error::ModelParseError("Could not map " + model_path_);
        }
        ModelConfig header{};
        std::memcpy(&header, mapped->data, sizeof(header));
        auto status = generate_model_infos(header);
        if (!status) return status;
        auto* bytes = static_cast<const char*>(mapped->data);
        if (is_quant_model_) {
            std::memcpy(&group_size_, bytes + sizeof(ModelConfig), sizeof(group_size_));
            if (group_size_ <= 0) return error::ModelParseError("Invalid Qwen3 INT8 group size");
        }
        mapped->weight_data = static_cast<char*>(mapped->data) + header_bytes;
        mapped->weight_bytes = mapped->file_size - header_bytes;
        raw_model_data_ = std::move(mapped);
        return error::Success();
    }

    base::Status Model::create_encode_layer() {
        using namespace base;
        if (tokenizer_type_ == TokenizerType::kEncodeSpe) {
            encode_layer_ = std::make_unique<op::SpeEncodeLayer>(this->token_path_, true, false);
        } else if (tokenizer_type_ == TokenizerType::kEncodeBpe) {
        #if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
            encode_layer_ = std::make_unique<op::QwenEncodeLayer>(this->token_path_, false, false);
        #endif 
        }

        if (!encode_layer_) {
            return error::InternalError("Create the encode layer failed.");
        }

        config_->vocab_size_ = encode_layer_->vocab_size();
        if (config_->vocab_size_ <= 0) {
            return error::InternalError("The vocab size param read error from the model file!");
        }
        return error::Success();
    }

    base::Status Model::gen_model_from_file() {
        using namespace base;
        config_ = std::make_unique<TransformerConfig>();

        auto create_encode_status = create_encode_layer();
        if (!create_encode_status) {
            LOG(ERROR) << "Create the encode layer failed!";
            return create_encode_status;
        }

        auto mmap_status = read_model_file();
        if (!mmap_status) {
            LOG(ERROR) << "Handle model file " << model_path_ << " failed!";
            return mmap_status;
        }
        auto layer_create_status = create_layers();
        if (!layer_create_status) {
            LOG(ERROR) << "Create layers for the model file " << model_path_ << " failed!";
            return layer_create_status;
        }
        return error::Success();
    }

    std::vector<int32_t> Model::encode(const std::string& sentence) const {
        CHECK(encode_layer_ != nullptr);
        return encode_layer_->encode(sentence);
    }

    bool Model::is_sentence_ending(int32_t token_idx) const {
        CHECK(this->encode_layer_ != nullptr);
        return this->encode_layer_->is_sentence_ending(token_idx);
    }

    std::string Model::decode(int32_t token_idx) const {
        CHECK(this->encode_layer_ != nullptr);
        return this->encode_layer_->decode(token_idx);
    }

    std::string Model::decode(std::vector<int32_t> token_idxs) const {
        CHECK(this->encode_layer_ != nullptr);
        return this->encode_layer_->decode(token_idxs);
    }

    std::pair<tensor::Tensor, tensor::Tensor> Model::slice_kv_cache(int32_t layer_idx, int32_t token_pos) const {
        const size_t offset = (static_cast<size_t>(layer_idx) * config_->seq_len_ + token_pos) * config_->kv_dim_;
        auto cache_view = [&](ModelBufferType type) {
            auto cache = get_buffer(type);
            return tensor::Tensor(base::DataType::kDataTypeFp32, {config_->kv_dim_}, false, nullptr,
                                  cache.ptr<float>() + offset);
        };
        auto key = cache_view(ModelBufferType::kKeyCache);
        auto val = cache_view(ModelBufferType::kValueCache);
        key.set_device_type(device_type_);
        val.set_device_type(device_type_);
        return std::make_pair(key, val);
    }

    tensor::Tensor Model::fill_input(const tensor::Tensor& pos_tensor,
                                     const op::EmbeddingOutput& embedding_output,
                                     bool is_prompt) const {
        const int32_t pos = pos_tensor.index<int32_t>(0);
        auto embeddings = embedding_output.input_embeddings;
        const int index = is_prompt ? pos : 0;
        CHECK_GE(index, 0);
        CHECK_LT(index, embeddings.get_dim(0));
        const int width = embeddings.get_dim(1);
        tensor::Tensor input(base::DataType::kDataTypeFp32, {width}, false, nullptr,
                             embeddings.ptr<float>() + static_cast<size_t>(index) * width);
        input.set_device_type(device_type_);
        return input.clone();
    }
} // namespace model
