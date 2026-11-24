#include "op/embedding.h"

#include "base/cuda_config.h"
#include "op/kernels/kernels_interface.h"

namespace op {
    EmbeddingLayer::EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t vocab_size)
        : LayerParam(device_type, LayerType::kLayerEmbedding, false, "Embedding"),
          dim_(dim), vocab_size_(vocab_size) {
        reset_input_size(2);
        reset_weight_size(1);
        reset_output_size(1);
    }

    base::Status EmbeddingLayer::check() const {
        if (device_type_ != base::DeviceType::kDeviceCPU &&
            device_type_ != base::DeviceType::kDeviceCUDA) {
            return base::error::InvalidArgument("Unsupported embedding device");
        }

        const auto& tokens = get_input(0);
        const auto& count = get_input(1);
        for (const auto* input : {&tokens, &count}) {
            auto status = check_tensor(*input, base::DeviceType::kDeviceCPU,
                                       base::DataType::kDataTypeInt32);
            if (!status) {
                return status;
            }
        }
        if (tokens.dims_size() != 1 || count.size() != 1) {
            return base::error::InvalidArgument("Invalid embedding token input shape");
        }
        const auto token_count = count.index<int32_t>(0);
        if (token_count <= 0 || static_cast<size_t>(token_count) > tokens.size()) {
            return base::error::InvalidArgument("Invalid embedding token count");
        }

        const auto& weight = get_weight(0);
        const auto& output = get_output(0);
        for (const auto* tensor : {&weight, &output}) {
            auto status = check_tensor(*tensor, device_type_, base::DataType::kDataTypeFp32);
            if (!status) {
                return status;
            }
        }
        if (vocab_size_ <= 0 || dim_ <= 0 ||
            weight.dims() != std::vector<int32_t>{vocab_size_, dim_} ||
            output.dims() != std::vector<int32_t>{token_count, dim_}) {
            return base::error::InvalidArgument("Invalid embedding weight or output shape");
        }
        const auto* token_ids = tokens.ptr<int32_t>();
        for (int32_t i = 0; i < token_count; ++i) {
            if (token_ids[i] < 0 || token_ids[i] >= vocab_size_) {
                return base::error::InvalidArgument("Embedding token ID is out of range");
            }
        }
        return base::error::Success();
    }

    base::Status EmbeddingLayer::forward() {
        auto status = check();
        if (!status) {
            return status;
        }
        return kernel::get_emb_kernel(device_type_)(get_input(0), get_weight(0), get_output(0),
                                                    vocab_size_,
                                                    cuda_config_ ? cuda_config_->stream : nullptr);
    }
}  // namespace op
