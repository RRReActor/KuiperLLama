#include "op/activation.h"
#include <cmath>
#include "base/cuda_config.h"
#include "op/kernels/kernels_interface.h"

namespace op {
    Activation::Activation(base::DeviceType device_type, int32_t hidden_dim)
        : Layer(device_type, LayerType::kLayerActivation), hidden_dim_(hidden_dim) {
        reset_input_size(2);
        reset_output_size(1);
    }

    base::Status Activation::check() const {
        for (const auto* tensor : {&get_input(0), &get_input(1), &get_output(0)}) {
            auto status = check_tensor(*tensor, device_type_, data_type_);
            if (!status) {
                return status;
            }
            if (tensor->dims_size() != 1 || tensor->get_dim(0) != hidden_dim_) {
                return base::error::InvalidArgument("Invalid Activation shape");
            }
        }
        return base::error::Success();
    }

    base::Status Activation::forward() {
        auto status = check();
        if (!status) {
            return status;
        }
        if (device_type_ == base::DeviceType::kDeviceCPU) {
            const float* gate = get_input(0).ptr<float>();
            const float* up = get_input(1).ptr<float>();
            float* output = get_output(0).ptr<float>();
            for (int i = 0; i < hidden_dim_; ++i) {
                output[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
            }
            return base::error::Success();
        }
        if (device_type_ != base::DeviceType::kDeviceCUDA) {
            return base::error::InvalidArgument("Unsupported Activation device");
        }
        kernel::get_activation_kernel(device_type_)(get_output(0), get_input(0), get_input(1),
                                                    cuda_config_ ? cuda_config_->stream : nullptr);
        return base::error::Success();
    }
}  // namespace op
