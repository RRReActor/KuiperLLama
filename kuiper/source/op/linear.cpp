#include "op/linear.h"
#include <algorithm>
#include <numeric>

#include "base/cuda_config.h"
#include "op/kernels/kernels_interface.h"

namespace op {
    LinearLayer::LinearLayer(base::DeviceType device_type, int32_t input_dim, int32_t output_dim,
                             bool is_quant_layer, bool has_bias)
        : LayerParam(device_type, LayerType::kLayerLinear, is_quant_layer, "Linear"),
          input_dim_(input_dim), output_dim_(output_dim), has_bias_(has_bias) {
        reset_input_size(1);
        reset_output_size(1);
        reset_weight_size(1);
    }

    base::Status LinearLayer::check() const {
        if (input_dim_ <= 0 || output_dim_ <= 0) {
            return base::error::InvalidArgument("Invalid Linear dimensions.");
        }
        auto status = check_tensor(get_input(0), device_type_, data_type_);
        if (!status) return status;
        status = check_tensor(get_output(0), device_type_, data_type_);
        if (!status) return status;

        const auto& weight = get_weight(0);
        const auto weight_type = is_quant_layer_ ? base::DataType::kDataTypeInt8 : data_type_;
        status = check_tensor(weight, device_type_, weight_type);
        if (!status) return status;

        if (get_input(0).dims() != std::vector<int32_t>{input_dim_} ||
            get_output(0).dims() != std::vector<int32_t>{output_dim_} ||
            weight.dims() != std::vector<int32_t>{output_dim_, input_dim_}) {
            return base::error::InvalidArgument("Invalid Linear tensor shape.");
        }

        if (is_quant_layer_) {
            status = check_tensor(scales_, device_type_, base::DataType::kDataTypeFp32);
            if (!status) return status;
            if (group_size_ <= 0 || scales_.dims_size() != 1 ||
                scales_.size() != (weight.size() - 1) / group_size_ + 1) {
                return base::error::InvalidArgument("Invalid Linear quantization scales or group size.");
            }
        }

        if (has_bias_) {
            status = check_tensor(bias_, device_type_, data_type_);
            if (!status) return status;
            if (bias_.dims() != std::vector<int32_t>{output_dim_}) {
                return base::error::InvalidArgument("Invalid Linear bias shape.");
            }
        }
        return base::error::Success();
    }

    base::Status LinearLayer::forward() {
        auto status = check();
        if (!status) return status;
        if (device_type_ == base::DeviceType::kDeviceCPU) {
            const float* x = get_input(0).ptr<float>();
            const auto& weights = get_weight(0);
            float* out = get_output(0).ptr<float>();
            for (int row = 0; row < output_dim_; ++row) {
                const size_t start = static_cast<size_t>(row) * input_dim_;
                if (is_quant_layer_) {
                    const auto* codes = weights.ptr<int8_t>() + start;
                    const auto* scales = scales_.ptr<float>();
                    float sum = 0.f;
                    for (int col = 0; col < input_dim_;) {
                        const size_t index = start + col;
                        const float scale = scales[index / group_size_];
                        const int span = std::min(input_dim_ - col,
                            group_size_ - static_cast<int>(index % group_size_));
                        const int end = col + span;
                        for (; col < end; ++col) sum += (codes[col] * scale) * x[col];
                    }
                    out[row] = sum;
                } else {
                    out[row] = std::inner_product(x, x + input_dim_, weights.ptr<float>() + start, 0.0f);
                }
                if (has_bias_) out[row] += bias_.ptr<float>()[row];
            }
            return base::error::Success();
        }
        if (device_type_ != base::DeviceType::kDeviceCUDA) {
            return base::error::InvalidArgument("Unsupported Linear device.");
        }

        void* stream = cuda_config_ ? cuda_config_->stream : nullptr;
        if (is_quant_layer_) {
            kernel::get_gemv_kernel_quant(device_type_)(get_output(0), get_input(0),
                get_weight(0), group_size_, scales_, stream);
        } else {
            kernel::get_gemv_kernel(device_type_)(get_output(0), get_input(0), get_weight(0), stream);
        }

        if (has_bias_) {
            kernel::get_add_kernel(device_type_)(get_output(0), get_output(0), bias_, stream);
        }
        return base::error::Success();
    }

    base::Status LinearLayer::set_bias(int32_t idx, int32_t dims, const void* bias_ptr,
                                       base::DeviceType device_type) {
        if (idx != 0 || dims <= 0 || dims != output_dim_ || !bias_ptr) {
            return base::error::InvalidArgument("Invalid Linear bias.");
        }
        auto buffer = std::make_shared<base::Buffer>(
            static_cast<size_t>(dims) * sizeof(float), nullptr, const_cast<void*>(bias_ptr), true);
        buffer->set_device_type(device_type);

        tensor::Tensor bias(base::DataType::kDataTypeFp32, {dims});
        bias.set_device_type(device_type);
        if (!bias.assign(buffer)) {
            return base::error::InvalidArgument("Could not assign Linear bias buffer.");
        }
        bias_ = std::move(bias);
        return base::error::Success();
    }

    tensor::Tensor LinearLayer::get_bias(int32_t idx) {
        CHECK_EQ(idx, 0);
        return bias_;
    }

    const tensor::Tensor& LinearLayer::get_bias(int32_t idx) const {
        CHECK_EQ(idx, 0);
        return bias_;
    }

    void LinearLayer::to_cuda() {
        LayerParam::to_cuda();
        if (has_bias_ && !bias_.is_empty()) {
            bias_.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
        }
    }
}
