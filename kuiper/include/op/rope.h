#pragma once

#include "op/layer.h"

namespace op {
    class RoPELayer : public Layer {
        public:
            using Layer::forward;

            explicit RoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim,
                               int32_t head_size);

            base::Status check() const override;

            // Inputs are Q, K, a CPU position, and the sine and cosine caches.
            // The layer rotates Q and K in place.
            base::Status forward() override;

        private:
            int32_t dim_ = 0;
            int32_t kv_dim_ = 0;
            int32_t head_size_ = 0;
    };
}  // namespace op
