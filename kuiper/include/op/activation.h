#pragma once
#include "op/layer.h"

namespace op {
    class Activation : public Layer {
        public:
            explicit Activation(base::DeviceType device_type, int32_t hidden_dim);

            base::Status check() const override;

            using Layer::forward;

            base::Status forward() override;
        private:
            int32_t hidden_dim_ = 0;
    };
}
