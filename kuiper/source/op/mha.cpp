#include "op/mha.h"
#include "kernels/cuda/mha_kernel.cuh"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace op
{
    MultiHeadAttention::MultiHeadAttention(base::DeviceType device_type, int32_t layer_index,
                                           int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
                                           int32_t head_num, int32_t head_size)
        : Layer(device_type, LayerType::kLayerMHA, "MultiHead"),
          layer_index_(layer_index),
          kv_mul_(kv_mul),
          kv_dim_(kv_dim),
          seq_len_(seq_len),
          head_num_(head_num),
          head_size_(head_size)
    {
        reset_input_size(4);
        reset_output_size(1);
    }

    base::Status MultiHeadAttention::check() const
    {
        base::Status status;
        const int32_t input_tensor_num = 4;
        for (int32_t i = 0; i < input_tensor_num; ++i)
        {
            // mha score tensor
            status = check_tensor(get_input(i), device_type_, data_type_);
            if (!status)
            {
                LOG(ERROR) << "The input tensor " << std::to_string(i) << " error in the matmul layer.";
                return status;
            }
        }
        status = check_tensor(get_output(0), device_type_, data_type_);
        if (!status) return status;
        const auto& cache = get_input(2);
        if (pos_ < 0 || pos_ >= seq_len_ || head_num_ <= 0 || head_size_ <= 0 ||
            kv_mul_ <= 0 || head_num_ % kv_mul_ || kv_dim_ != int64_t(head_num_ / kv_mul_) * head_size_ ||
            layer_index_ < 0 || cache.dims_size() != 3 || layer_index_ >= cache.get_dim(0) ||
            cache.get_dim(1) != seq_len_ || cache.get_dim(2) != kv_dim_ ||
            cache.dims() != get_input(3).dims() || get_input(0).size() != size_t(head_num_) * head_size_ ||
            get_input(1).size() != size_t(head_num_) * seq_len_ || get_output(0).size() != get_input(0).size()) {
            return base::error::InvalidArgument("Invalid attention cache, position or head dimensions");
        }
        return base::error::Success();
    }

    void MultiHeadAttention::set_pos(int32_t pos) { pos_ = pos; }

    void MultiHeadAttention::set_layer_idx(int32_t layer_idx) { layer_index_ = layer_idx; }

    base::Status MultiHeadAttention::forward()
    {
        auto status = check();
        if (!status)  return status;

        tensor::Tensor mha_out = get_output(0);
        const tensor::Tensor& query = get_input(0);
        tensor::Tensor& score = get_input(1);
        const tensor::Tensor& key_cache = get_input(2);
        const tensor::Tensor& val_cache = get_input(3);

        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            kernel::mha_kernel_cu(pos_, head_num_, layer_index_, seq_len_, kv_dim_, kv_mul_, head_size_,
                                  mha_out, query, score, key_cache, val_cache,
                                  cuda_config_ ? cuda_config_->stream : nullptr);
        } else if (device_type_ == base::DeviceType::kDeviceCPU) {
            const size_t layer_offset = static_cast<size_t>(layer_index_) * seq_len_ * kv_dim_;
            for (int head = 0; head < head_num_; ++head) {
                const size_t offset = layer_offset + head / kv_mul_ * head_size_;
                const float* q = query.ptr<float>() + head * head_size_;
                float* scores = score.ptr<float>() + static_cast<size_t>(head) * seq_len_;
                for (int t = 0; t <= pos_; ++t) {
                    const float* k = key_cache.ptr<float>() + offset + static_cast<size_t>(t) * kv_dim_;
                    scores[t] = std::inner_product(q, q + head_size_, k, 0.0f) / std::sqrt(float(head_size_));
                }
                const float max_score = *std::max_element(scores, scores + pos_ + 1);
                float denominator = 0;
                for (int t = 0; t <= pos_; ++t) {
                    scores[t] = std::exp(scores[t] - max_score);
                    denominator += scores[t];
                }
                float* out = mha_out.ptr<float>() + head * head_size_;
                std::fill_n(out, head_size_, 0.0f);
                for (int t = 0; t <= pos_; ++t) {
                    const float* v = val_cache.ptr<float>() + offset + static_cast<size_t>(t) * kv_dim_;
                    scores[t] /= denominator;
                    for (int d = 0; d < head_size_; ++d) out[d] += scores[t] * v[d];
                }
            }
        } else {
            return base::error::InvalidArgument("Unsupported attention device");
        }
        return base::error::Success();
    }
} // namespace op