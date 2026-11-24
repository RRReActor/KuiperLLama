#include "model/qwen3.h"
#include "sampler/argmax_sampler.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <op/activation.h>
#include <op/mha.h>
#include <op/rmsnorm.h>
#include <algorithm>
#include <cmath>
#include "op/rope.h"
#include <fstream>
#include <limits>
#include "op/linear.h"
#include <utility>
#include "../op/kernels/cuda/rope_kernel.cuh"
#include "op/kernels/kernels_interface.h"

namespace model
{
    namespace {
    constexpr float kQwen3RopeTheta = 1000000.0f;

    void add_residual(tensor::Tensor input, const tensor::Tensor& residual, void* stream) {
        if (input.device_type() == base::DeviceType::kDeviceCUDA) {
            kernel::get_add_kernel(input.device_type())(input, input, residual, stream);
        } else {
            float* values = input.ptr<float>();
            const float* add = residual.ptr<float>();
            for (size_t i = 0; i < input.size(); ++i) values[i] += add[i];
        }
    }
    }

    void Qwen3Layers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
        auto upload = [&](const std::shared_ptr<op::Layer>& layer) {
            layer->set_cuda_config(config);
            layer->to_cuda();
        };
        for (const auto& layer : {rope_layer_, swiglu_layer_, cls_layer_, norm_layer_,
                                  embedding_layer_}) upload(layer);
        upload(mha_layer_);
        for (const auto* layers : {&wq_layers_, &wk_layers_, &wv_layers_, &wo_layers_,
                                   &w_gate_layers_, &w_down_layers_, &w_up_layers_,
                                   &rmsnorm_attn_layers_, &rmsnorm_ffn_layers_,
                                   &rmsnorm_q_layers_, &rmsnorm_k_layers_}) {
            for (const auto& layer : *layers) upload(layer);
        }
    }

    Qwen3Model::Qwen3Model(base::TokenizerType tokenizer_type, std::string token_path,
                           std::string model_path, bool is_quant_model) : Model(tokenizer_type, base::ModelType::kModelTypeQwen3, std::move(token_path),
                                                                                std::move(model_path), is_quant_model) {}

    base::Status Qwen3Model::init(base::DeviceType device_type)
    {
        using namespace base;
        if (!std::ifstream(token_path_))
        {
            return error::PathNotValid(token_path_);
        }
        if (sampler_) return error::InvalidArgument("Model is already initialized");
        if (device_type != DeviceType::kDeviceCPU && device_type != DeviceType::kDeviceCUDA)
            return error::InvalidArgument("Unknown Qwen3 device");
        device_type_ = device_type;
        if (device_type == DeviceType::kDeviceCUDA)
        {
            auto err = cudaSetDevice(0);
            if (err != cudaSuccess) return error::InternalError(cudaGetErrorString(err));
            cuda_config_ = std::make_shared<kernel::CudaConfig>();
            err = cudaStreamCreate(&cuda_config_->stream);
            if (err != cudaSuccess)
            {
                return error::InternalError("The cuda stream create failed.");
            }
        }

        Status read_status = gen_model_from_file();
        if (!read_status)
        {
            return read_status;
        }
        init_mem();
        if (device_type_ == base::DeviceType::kDeviceCPU)
        {
            const int width = config_->head_size_;
            auto* sine = get_buffer(ModelBufferType::kSinCache).ptr<float>();
            auto* cosine = get_buffer(ModelBufferType::kCosCache).ptr<float>();
            for (int col = 0; col < width; ++col) {
                const float freq = std::pow(kQwen3RopeTheta, -static_cast<float>(col) / width);
                for (int pos = 0; pos < config_->seq_len_; ++pos) {
                    const size_t offset = static_cast<size_t>(pos) * width + col;
                    sine[offset] = std::sin(pos * freq);
                    cosine[offset] = std::cos(pos * freq);
                }
            }
        }
        else
        {
            CHECK_NE(cuda_config_, nullptr);
            kernel::sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_,
                                          get_buffer(ModelBufferType::kSinCache),
                                          get_buffer(ModelBufferType::kCosCache), cuda_config_->stream, kQwen3RopeTheta);
        }

        sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
        return error::Success();
    }

    base::Status Qwen3Model::forward(const tensor::Tensor &input, const tensor::Tensor &pos_tensor,
                                     int &) const
    {
        if (!sampler_ || input.is_empty() || input.device_type() != device_type_ ||
            input.data_type() != base::DataType::kDataTypeFp32 ||
            input.dims() != std::vector<int32_t>{config_->hidden_dim_}) {
            return base::error::InvalidArgument("Invalid Qwen3 input");
        }
        if (pos_tensor.is_empty() || pos_tensor.device_type() != base::DeviceType::kDeviceCPU ||
            pos_tensor.data_type() != base::DataType::kDataTypeInt32 ||
            pos_tensor.dims() != std::vector<int32_t>{1} || pos_tensor.index<int32_t>(0) < 0 ||
            pos_tensor.index<int32_t>(0) >= config_->seq_len_) {
            return base::error::InvalidArgument("Qwen3 position outside the KV cache");
        }

        for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx)
        {
            attention_rms(layer_idx, input);
            // attention (wq wk wv @ input)
            attention_qkv(layer_idx, pos_tensor);
            // multi-head attention
            attention_mha(layer_idx, pos_tensor);
            // residual add
            add_residual(input, get_buffer(ModelBufferType::kAttnOutput), cuda_config_ ? cuda_config_->stream : nullptr);
            // feed forward
            feed_forward(layer_idx, input);
        }
        cls_logits(input);
        return base::error::Success();
    }

    void Qwen3Model::create_nonparam_layers() {
        CHECK(qwen_layers_ != nullptr);
        qwen_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(
            device_type_, config_->dim_, config_->kv_dim_, config_->head_size_);
        
        qwen_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
            device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,
            config_->head_size_);

        qwen_layers_->swiglu_layer_ =
            std::make_shared<op::Activation>(device_type_, config_->immediate_dim_);
    }

    base::Status Qwen3Model::create_param_layers() {
        const int hidden = config_->hidden_dim_;
        const int dim = config_->dim_;
        const int kv = config_->kv_dim_;
        const int head = config_->head_size_;
        const int ffn = config_->immediate_dim_;
        const int vocab = config_->vocab_size_;
        const int layers = config_->layer_num_;
        size_t offset = 0;
        const size_t payload = raw_model_data_->weight_bytes;
        auto bind = [&](const std::shared_ptr<op::Layer>& layer, const std::vector<int32_t>& shape) {
            const bool quantized = is_quant_model_ && layer->layer_type() == op::LayerType::kLayerLinear;
            const size_t remaining = payload - offset;
            size_t bytes = quantized ? sizeof(int8_t) : sizeof(float);
            for (int extent : shape) {
                if (bytes > remaining / extent) return false;
                bytes *= extent;
            }
            if (quantized) {
                if (bytes % group_size_ != 0) return false;
                const size_t groups = bytes / group_size_;
                if (groups > std::numeric_limits<int32_t>::max() ||
                    groups > (remaining - bytes) / sizeof(float)) return false;
                bytes += groups * sizeof(float);
            }
            const size_t index = is_quant_model_ ? offset : offset / sizeof(float);
            auto status = layer->set_weight(0, shape, raw_model_data_->weight(index),
                                            base::DeviceType::kDeviceCPU);
            offset += bytes;
            return static_cast<bool>(status);
        };
        auto norms = [&](auto& group, int width) {
            for (int i = 0; i < layers; ++i) {
                auto layer = std::make_shared<op::RmsNormLayer>(device_type_, width);
                if (!bind(layer, {width})) return false;
                group.push_back(layer);
            }
            return true;
        };
        auto linear = [&](int in, int out) {
            auto layer = std::make_shared<op::LinearLayer>(device_type_, in, out, is_quant_model_);
            if (is_quant_model_) layer->set_group_size(group_size_);
            return layer;
        };
        auto projections = [&](auto& group, int in, int out) {
            for (int i = 0; i < layers; ++i) {
                auto layer = linear(in, out);
                if (!bind(layer, {out, in})) return false;
                group.push_back(layer);
            }
            return true;
        };

        auto& q = *qwen_layers_;
        q.norm_layer_ = std::make_shared<op::RmsNormLayer>(device_type_, hidden);
        q.embedding_layer_ = std::make_shared<op::EmbeddingLayer>(device_type_, hidden, vocab);
        q.cls_layer_ = linear(hidden, vocab);
        // write_bin.py stores each projection for all layers before the next projection.
        if (!norms(q.rmsnorm_attn_layers_, hidden) || !norms(q.rmsnorm_ffn_layers_, hidden) ||
            !bind(q.norm_layer_, {hidden}) || !bind(q.embedding_layer_, {vocab, hidden}) ||
            !projections(q.wq_layers_, hidden, dim) || !norms(q.rmsnorm_q_layers_, head) ||
            !projections(q.wk_layers_, hidden, kv) || !norms(q.rmsnorm_k_layers_, head) ||
            !projections(q.wv_layers_, hidden, kv) || !projections(q.wo_layers_, dim, hidden) ||
            !projections(q.w_gate_layers_, hidden, ffn) || !projections(q.w_down_layers_, ffn, hidden) ||
            !projections(q.w_up_layers_, hidden, ffn) || !bind(q.cls_layer_, {vocab, hidden}) ||
            offset != payload) {
            return base::error::ModelParseError("Qwen3 weight payload does not match the header and precision");
        }
        return base::error::Success();
    }

    void Qwen3Model::init_mem() {
        std::shared_ptr<base::DeviceAllocator> alloc;
        if (device_type_ == base::DeviceType::kDeviceCPU) {
            alloc = base::CPUDeviceAllocatorFactory::get_instance();
        } else {
            alloc = base::CUDADeviceAllocatorFactory::get_instance();
        }

        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            CHECK_NE(cuda_config_, nullptr);
            qwen_layers_->to_cuda(cuda_config_);
        }

        std::shared_ptr<base::DeviceAllocator> alloc_cpu =
            base::CPUDeviceAllocatorFactory::get_instance();


        tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, {1}, true, alloc_cpu);
        tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, {1, config_->hidden_dim_}, true,
                                        alloc);
        tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, {config_->seq_len_, config_->head_size_},
                                true, alloc);
        tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, {config_->seq_len_, config_->head_size_},
                                true, alloc);
        
        CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
        CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));
        CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
        CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));
        
        tensor::Tensor rms_output(base::DataType::kDataTypeFp32, {config_->hidden_dim_}, true, alloc);
        tensor::Tensor mha_output(base::DataType::kDataTypeFp32, {config_->dim_}, true, alloc);
        CHECK(insert_buffer(ModelBufferType::kAttnRMSNorm, rms_output));
        CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));
        CHECK(insert_buffer(ModelBufferType::kMHAOutput, mha_output));

        tensor::Tensor w_gate_output(base::DataType::kDataTypeFp32, {config_->immediate_dim_}, true, alloc);
        tensor::Tensor w_down_output(base::DataType::kDataTypeFp32, {config_->hidden_dim_}, true, alloc);
        tensor::Tensor w_up_output(base::DataType::kDataTypeFp32, {config_->immediate_dim_}, true, alloc);
        CHECK(insert_buffer(ModelBufferType::kWGateOutput, w_gate_output));
        CHECK(insert_buffer(ModelBufferType::kWDownOutput, w_down_output));
        CHECK(insert_buffer(ModelBufferType::kWUpOutput, w_up_output));

        // kv cache
        tensor::Tensor key_cache(base::DataType::kDataTypeFp32, {config_->layer_num_, config_->seq_len_,
                                config_->kv_dim_}, true, alloc);
        tensor::Tensor value_cache(base::DataType::kDataTypeFp32, {config_->layer_num_, config_->seq_len_,
                                    config_->kv_dim_}, true, alloc);

        CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
        CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

        // Wq query output
        tensor::Tensor query(base::DataType::kDataTypeFp32, {config_->dim_}, true, alloc);
        CHECK(insert_buffer(ModelBufferType::kQuery, query));

        // Pos tensor
        tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, {1}, true, alloc_cpu);
        CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));
        
        // Attention output
        tensor::Tensor attn_score(base::DataType::kDataTypeFp32, {config_->head_num_, config_->seq_len_}, true,
                            alloc);
        CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn_score));
        tensor::Tensor attn_output(base::DataType::kDataTypeFp32, {config_->hidden_dim_}, true, alloc);
        CHECK(insert_buffer(ModelBufferType::kAttnOutput, attn_output));
        
        // LM Head output
        tensor::Tensor lm_output(base::DataType::kDataTypeFp32, {config_->vocab_size_}, true, alloc);
        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            tensor::Tensor lm_output_cpu(base::DataType::kDataTypeFp32, {config_->vocab_size_}, true,
                                            alloc_cpu); 
            CHECK(insert_buffer(ModelBufferType::kLMOutputCPU, lm_output_cpu));
        }

        CHECK(insert_buffer(ModelBufferType::kLMOutput, lm_output));
    }

    base::Status Qwen3Model::create_layers() {
        using namespace base;
        qwen_layers_ = std::make_unique<Qwen3Layers>();
        auto status = create_param_layers();
        if (!status) return status;
        create_nonparam_layers();
    
        return error::Success();
    }

    void Qwen3Model::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
        CHECK(qwen_layers_ != nullptr);
        tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kAttnRMSNorm);
        std::shared_ptr<op::Layer> rmsnorm_layer = qwen_layers_->rmsnorm_attn_layers_.at(layer_idx);
        if (!rmsnorm_layer) {
            LOG(FATAL) << "The attention rmsnorm layer is a null pointer in the Qwen3 model";
        }
        CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
    }

    void Qwen3Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
        CHECK(qwen_layers_ != nullptr);
        tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
        auto [key, val] = slice_kv_cache(layer_idx, pos_tensor.index<int32_t>(0));

        // query
        const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
        CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";
        
        auto rmsnorm_output = get_buffer(ModelBufferType::kAttnRMSNorm);
        CHECK(query_layer->forward(rmsnorm_output, query));

        //query norm
        auto query_norm = qwen_layers_->rmsnorm_q_layers_.at(layer_idx);
        query.reshape({(int32_t)query.size() / config_->head_size_, config_->head_size_});
        CHECK(query_norm->forward(query, query));
        query.reshape({(int32_t)query.size()});

        // key
        const auto& key_layer = qwen_layers_->wk_layers_.at(layer_idx);
        CHECK_NE(key_layer, nullptr) << "The key layer in the attention block is null pointer.";
        CHECK(key_layer->forward(rmsnorm_output, key));

        // key norm
        auto key_norm = qwen_layers_->rmsnorm_k_layers_.at(layer_idx);
        key.reshape({(int32_t)key.size() / config_->head_size_, config_->head_size_});
        CHECK(key_norm->forward(key, key));
        key.reshape({(int32_t)key.size()});

        // value
        const auto& value_layer = qwen_layers_->wv_layers_.at(layer_idx);
        CHECK_NE(value_layer, nullptr) << "The value layer in the attention block is null pointer.";
        CHECK(value_layer->forward(rmsnorm_output, val));

        // rope
        CHECK_NE(qwen_layers_->rope_layer_, nullptr)
            << "The RoPE layer in the attention block is null pointer.";
        CHECK(qwen_layers_->rope_layer_->forward(
            query, key, pos_tensor, get_buffer(ModelBufferType::kSinCache),
            get_buffer(ModelBufferType::kCosCache), tensor::Tensor{}));
    }

    void Qwen3Model::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
        CHECK(qwen_layers_ != nullptr);
        tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
        tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
        tensor::Tensor query = get_buffer(ModelBufferType::kQuery);
        tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
        tensor::Tensor mha_output = get_buffer(ModelBufferType::kMHAOutput);

        const auto& mha_layer = qwen_layers_->mha_layer_;
        CHECK_NE(mha_layer, nullptr) << "The MHA layer in the attention block is null pointer.";
        int pos = pos_tensor.index<int32_t>(0);
        mha_layer->set_pos(pos);
        mha_layer->set_layer_idx(layer_idx);
        CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));

        // wo @ attention output
        tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
        const auto& wo_layer = qwen_layers_->wo_layers_.at(layer_idx);
        CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
        CHECK(wo_layer->forward(mha_output, attn_output));
    }

    base::Status Qwen3Model::predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                 bool is_prompt, int& next) const {
        auto status = forward(input, pos_tensor, next);
        if (!status) {
            return status;
        }
        next = post_processing(pos_tensor, is_prompt);
        return base::error::Success();
    }

    int32_t Qwen3Model::post_processing(const tensor::Tensor&, bool is_prompt) const {
        tensor::Tensor forward_output = get_buffer(ModelBufferType::kLMOutput);
        const float* forward_logits = forward_output.ptr<float>();

        int32_t next = 0;
        if (is_prompt) {
            next = -1;
        } else {
            next = static_cast<int32_t>(sampler_->sample(forward_logits, forward_output.size(),
                                                        cuda_config_ ? cuda_config_->stream : nullptr));
        }
        return next;
    }

    void Qwen3Model::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
        // ffn rmsnorm
        tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
        const auto& ffn_rmsnorm = qwen_layers_->rmsnorm_ffn_layers_.at(layer_idx);
        CHECK_NE(ffn_rmsnorm, nullptr)
            << "The final rmsnorm layer in the feedforward block is null pointer";
        CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

        // gate 
        tensor::Tensor gate_output = get_buffer(ModelBufferType::kWGateOutput);
        const auto& gate_linear = qwen_layers_->w_gate_layers_.at(layer_idx);
        CHECK_NE(gate_linear, nullptr)
            << "The gate linear layer in the feedforward block is null pointer.";
        CHECK(gate_linear->forward(ffn_norm_output, gate_output));

        // up
        tensor::Tensor up_output = get_buffer(ModelBufferType::kWUpOutput);
        const auto& up_linear = qwen_layers_->w_up_layers_.at(layer_idx);
        CHECK_NE(up_linear, nullptr)
            << "The up linear layer in the feedforward block is null pointer.";
        CHECK(up_linear->forward(ffn_norm_output, up_output));

        // SwiGLU activation
        CHECK(qwen_layers_->swiglu_layer_->forward(gate_output, up_output, gate_output));

        // down
        tensor::Tensor down_output = get_buffer(ModelBufferType::kWDownOutput);
        const auto& down_linear = qwen_layers_->w_down_layers_.at(layer_idx);
        CHECK_NE(down_linear, nullptr)
            << "The down linear layer in the feedforward block is null pointer.";
        CHECK(down_linear->forward(gate_output, down_output));

        // residual add
        add_residual(input, down_output, cuda_config_ ? cuda_config_->stream : nullptr);
    }

    op::EmbeddingOutput Qwen3Model::embedding(const std::vector<int32_t>& tokens) const {
        auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
        auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
        CHECK(!tokens.empty());
        CHECK_LE(tokens.size(), static_cast<size_t>(config_->seq_len_));
        const int32_t count = static_cast<int32_t>(tokens.size());
        if (count != 1) {
            input_tokens = tensor::Tensor(base::DataType::kDataTypeInt32, {count}, true,
                                          input_tokens.get_buffer()->allocator());
            input_embeddings = tensor::Tensor(base::DataType::kDataTypeFp32, {count, config_->hidden_dim_},
                                              true, input_embeddings.get_buffer()->allocator());
        }
        std::copy(tokens.begin(), tokens.end(), input_tokens.ptr<int32_t>());
        tensor::Tensor input_token_num(base::DataType::kDataTypeInt32, {1}, true,
                                       base::CPUDeviceAllocatorFactory::get_instance());
        input_token_num.index<int32_t>(0) = count;
        LOG_IF(FATAL, !qwen_layers_->embedding_layer_)
            << "The embedding layer in the Qwen3 model is null pointer.";
        CHECK(
            qwen_layers_->embedding_layer_->forward(input_tokens, input_token_num, input_embeddings));

        op::EmbeddingOutput output(input_tokens, input_embeddings, input_token_num);
        return output;
    }

    void Qwen3Model::cls_logits(const tensor::Tensor& input) const {
        const auto& norm = qwen_layers_->norm_layer_;
        CHECK_NE(norm, nullptr) << "The normalization layer in the classification block is null pointer.";
        CHECK(norm->forward(input, input));

        tensor::Tensor lm_output = get_buffer(ModelBufferType::kLMOutput);
        CHECK(qwen_layers_->cls_layer_->forward(input, lm_output));
    }
}
