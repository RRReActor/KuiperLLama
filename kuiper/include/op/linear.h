
#pragma once

#include "op/layer.h"

namespace op {
    class LinearLayer : public LayerParam {
        public:
            explicit LinearLayer(base::DeviceType device_type, int32_t input_dim, int32_t output_dim,
                                bool is_quant_layer = false, bool has_bias = false);
            
            base::Status check() const override;

            using Layer::forward;

            base::Status forward() override;

            base::Status set_bias(int32_t idx, int32_t dims, const void* bias_ptr,
                                    base::DeviceType device_type);

            tensor::Tensor get_bias(int32_t idx);

            const tensor::Tensor& get_bias(int32_t idx) const;

            void to_cuda() override;
        
        private:
                int32_t input_dim_ = 0;
                int32_t output_dim_ = 0;
                bool has_bias_ = false;
                tensor::Tensor bias_;
    };
} // namespace op
