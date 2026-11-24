#include <gtest/gtest.h>
#include "op/layer.h"

class InspectQuantLayer : public op::LayerParam {
public:
    InspectQuantLayer() : LayerParam(base::DeviceType::kDeviceCPU,
                                    op::LayerType::kLayerLinear, true) {
        reset_weight_size(1);
        set_group_size(2);
    }
    const tensor::Tensor& scales() const { return scales_; }
};

TEST(Layer, QuantizedTensorWeightsAndScales) {
    InspectQuantLayer layer;
    int8_t values[] = {1, -2, 3, -4};
    float scales[] = {0.5f, 0.25f};
    tensor::Tensor weight(base::DataType::kDataTypeInt8, {4}, false, nullptr, values);
    tensor::Tensor scale(base::DataType::kDataTypeFp32, {2}, false, nullptr, scales);
    weight.set_device_type(base::DeviceType::kDeviceCPU);
    scale.set_device_type(base::DeviceType::kDeviceCPU);
    ASSERT_TRUE(layer.set_weight(0, weight, scale));
    EXPECT_EQ(layer.get_weight(0).ptr<int8_t>()[1], -2);
    EXPECT_EQ(layer.get_scale_num(), 2);
    EXPECT_FLOAT_EQ(layer.scales().ptr<float>()[1], 0.25f);
}

TEST(Layer, PackedQuantizedWeightsAndScales) {
    struct Packed { int8_t values[4]; float scales[2]; } packed{{1, 2, 3, 4}, {0.5f, 0.25f}};
    InspectQuantLayer layer;
    ASSERT_TRUE(layer.set_weight(0, {4}, &packed, base::DeviceType::kDeviceCPU));
    EXPECT_EQ(layer.get_weight(0).get_buffer()->byte_size(), 4u);
    EXPECT_FLOAT_EQ(layer.scales().ptr<float>()[0], 0.5f);
}

TEST(TensorRegression, ExternalCloneOwnsStorage) {
    float values[] = {1, 2};
    tensor::Tensor original(base::DataType::kDataTypeFp32, {2}, false, nullptr, values);
    original.set_device_type(base::DeviceType::kDeviceCPU);
    auto copy = original.clone();
    values[0] = 9;
    EXPECT_FLOAT_EQ(copy.ptr<float>()[0], 1);
    EXPECT_FALSE(copy.get_buffer()->is_external());
}

TEST(TensorRegressionDeathTest, InvalidDimensionsAndPointerOffset) {
    EXPECT_DEATH((tensor::Tensor(base::DataType::kDataTypeFp32, {-1})), "");
    float value = 0;
    tensor::Tensor tensor(base::DataType::kDataTypeFp32, {1}, false, nullptr, &value);
    EXPECT_DEATH(tensor.ptr<float>(-1), "");
    EXPECT_DEATH(tensor.ptr<float>(1), "");
}
