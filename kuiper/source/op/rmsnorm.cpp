#include "op/rmsnorm.h"
#include "op/kernels/kernels_interface.h"
#include "base/cuda_config.h"
namespace op {
RmsNormLayer::RmsNormLayer(base::DeviceType device_type, int32_t dim)
    : LayerParam(device_type, LayerType::kLayerRMSNorm), dim_(dim) {
    CHECK_GT(dim_, 0);
    reset_input_size(1); reset_output_size(1); reset_weight_size(1);
}
base::Status RmsNormLayer::check() const {
    if (device_type_ != base::DeviceType::kDeviceCPU && device_type_ != base::DeviceType::kDeviceCUDA)
        return base::error::InvalidArgument("Unsupported RMSNorm device");
    for (const auto* tensor : {&get_input(0), &get_output(0), &get_weight(0)}) {
        auto status = check_tensor(*tensor, device_type_, base::DataType::kDataTypeFp32);
        if (!status) return status;
    }
    const auto& input = get_input(0);
    if (input.get_dim(input.dims_size() - 1) != dim_ ||
        input.dims() != get_output(0).dims() || get_weight(0).size() != static_cast<size_t>(dim_))
        return base::error::InvalidArgument("Invalid RMSNorm shape");
    return base::error::Success();
}
base::Status RmsNormLayer::forward() {
    auto status = check();
    if (!status) return status;
    kernel::get_rmsnorm_kernel(device_type_)(get_output(0), get_input(0), get_weight(0),
                                               cuda_config_ ? cuda_config_->stream : nullptr);
    return base::error::Success();
}
}
