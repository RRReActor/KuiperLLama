#pragma once

#include "layer.h"
namespace op
{
    class RmsNormLayer : public LayerParam
    {
    public:
        explicit RmsNormLayer(base::DeviceType device_type, int32_t dim);
        using Layer::forward;

        base::Status check() const override;

        base::Status forward() override;

    private:
        int32_t dim_ = 0;
    };
} // namespace op