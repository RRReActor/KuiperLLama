#include "op/rope.h"

#include "base/cuda_config.h"
#include "kernels/cuda/rope_kernel.cuh"

namespace op {
namespace {
    void rotate_heads(float* values, int32_t dim, int32_t head_size,
                      const float* sin_row, const float* cos_row) {
        const int32_t pairs_per_head = head_size / 2;
        for (int32_t offset = 0; offset < dim; offset += head_size) {
            for (int32_t pair = 0; pair < pairs_per_head; ++pair) {
                const int32_t column = 2 * pair;
#if defined(LLAMA3_SUPPORT) || defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
                const int32_t first = offset + pair;
                const int32_t second = first + pairs_per_head;
#else
                const int32_t first = offset + column;
                const int32_t second = first + 1;
#endif
                const float x = values[first];
                const float y = values[second];
                values[first] = x * cos_row[column] - y * sin_row[column];
                values[second] = x * sin_row[column] + y * cos_row[column];
            }
        }
    }
}  // namespace

    RoPELayer::RoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim,
                         int32_t head_size)
        : Layer(device_type, LayerType::kLayerRoPe, "RoPe"),
          dim_(dim), kv_dim_(kv_dim), head_size_(head_size) {
        reset_input_size(5);
        // The inherited forward overload uses this slot even though RoPE updates Q and K in place.
        reset_output_size(1);
    }

    base::Status RoPELayer::check() const {
        if (device_type_ != base::DeviceType::kDeviceCPU &&
            device_type_ != base::DeviceType::kDeviceCUDA) {
            return base::error::InvalidArgument("Unsupported RoPE device");
        }
        if (head_size_ <= 0 || head_size_ % 2 != 0 || kv_dim_ <= 0 || dim_ < kv_dim_ ||
            dim_ % head_size_ != 0 || kv_dim_ % head_size_ != 0) {
            return base::error::InvalidArgument("Invalid RoPE head dimensions");
        }

        for (int32_t index : {0, 1, 3, 4}) {
            auto status = check_tensor(get_input(index), device_type_, data_type_);
            if (!status) {
                return status;
            }
        }
        const auto& query = get_input(0);
        const auto& key = get_input(1);
        if (query.dims_size() != 1 || query.get_dim(0) != dim_ ||
            key.dims_size() != 1 || key.get_dim(0) != kv_dim_) {
            return base::error::InvalidArgument("Invalid RoPE query or key shape");
        }

        const auto& position = get_input(2);
        auto status = check_tensor(position, base::DeviceType::kDeviceCPU,
                                   base::DataType::kDataTypeInt32);
        if (!status) {
            return status;
        }
        if (position.dims_size() != 1 || position.get_dim(0) != 1) {
            return base::error::InvalidArgument("RoPE position must be a one-element CPU tensor");
        }
        const int32_t pos = *position.ptr<int32_t>();
        if (pos < 0 || static_cast<size_t>(pos) >= get_input(3).size() / head_size_ ||
            static_cast<size_t>(pos) >= get_input(4).size() / head_size_) {
            return base::error::InvalidArgument("RoPE position is outside the sine or cosine cache");
        }
        return base::error::Success();
    }

    base::Status RoPELayer::forward() {
        auto status = check();
        if (!status) {
            return status;
        }

        auto& query = get_input(0);
        auto& key = get_input(1);
        const auto& position = get_input(2);
        const auto& sin_cache = get_input(3);
        const auto& cos_cache = get_input(4);
        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            kernel::rope_kernel_cu(dim_, kv_dim_, head_size_, query, key, position,
                                   sin_cache, cos_cache, cuda_config_ ? cuda_config_->stream : nullptr);
        } else {
            const size_t offset = static_cast<size_t>(*position.ptr<int32_t>()) * head_size_;
            const float* sin_row = sin_cache.ptr<float>() + offset;
            const float* cos_row = cos_cache.ptr<float>() + offset;
            rotate_heads(query.ptr<float>(), dim_, head_size_, sin_row, cos_row);
            rotate_heads(key.ptr<float>(), kv_dim_, head_size_, sin_row, cos_row);
        }
        return base::error::Success();
    }
}  // namespace op
