#include <algorithm>
#include <cstring>
#include <tuple>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "base/cuda_config.h"
#include "op/linear.h"

namespace {
using base::DataType;
using base::DeviceType;

TEST(LinearLayer, ValidatesShapesAndBiasOnHost) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor input(DataType::kDataTypeFp32, {3}, true, allocator);
    tensor::Tensor wrong_input(DataType::kDataTypeFp32, {3, 1}, true, allocator);
    tensor::Tensor output(DataType::kDataTypeFp32, {2}, true, allocator);
    tensor::Tensor weight(DataType::kDataTypeFp32, {2, 3}, true, allocator);
    op::LinearLayer layer(DeviceType::kDeviceCPU, 3, 2, false, true);

    ASSERT_TRUE(layer.set_weight(0, weight));
    layer.set_input(0, input);
    layer.set_output(0, output);
    EXPECT_FALSE(layer.check());

    float bias[] = {1.f, -2.f};
    EXPECT_FALSE(layer.set_bias(1, 2, bias, DeviceType::kDeviceCPU));
    EXPECT_FALSE(layer.set_bias(0, 1, bias, DeviceType::kDeviceCPU));
    EXPECT_FALSE(layer.set_bias(0, 2, nullptr, DeviceType::kDeviceCPU));
    ASSERT_TRUE(layer.set_bias(0, 2, bias, DeviceType::kDeviceCPU));
    EXPECT_EQ(layer.get_bias(0).ptr<float>(), bias);
    EXPECT_TRUE(layer.check());
    std::fill_n(input.ptr<float>(), 3, 1.0f);
    std::fill_n(weight.ptr<float>(), 6, 2.0f);
    ASSERT_TRUE(layer.forward());
    EXPECT_FLOAT_EQ(output.index<float>(0), 7.0f);
    EXPECT_FLOAT_EQ(output.index<float>(1), 4.0f);
    layer.set_input(0, wrong_input);
    EXPECT_FALSE(layer.check());
}

TEST(LinearLayer, ValidatesQuantizationScalesOnHost) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor input(DataType::kDataTypeFp32, {3}, true, allocator);
    tensor::Tensor output(DataType::kDataTypeFp32, {2}, true, allocator);
    tensor::Tensor weight(DataType::kDataTypeInt8, {2, 3}, true, allocator);
    tensor::Tensor scales(DataType::kDataTypeFp32, {2}, true, allocator);
    op::LinearLayer layer(DeviceType::kDeviceCPU, 3, 2, true);
    layer.set_input(0, input);
    layer.set_output(0, output);
    layer.set_group_size(3);
    ASSERT_TRUE(layer.set_weight(0, weight, scales));
    EXPECT_TRUE(layer.check());
    layer.set_group_size(2);
    EXPECT_FALSE(layer.check());
}

class LinearLayerCUDA : public ::testing::TestWithParam<std::tuple<bool, bool>> {};

TEST(LinearLayer, PackedInt8GroupsCrossRowsOnCPU) {
    // Five codes per scale, three columns per row, no alignment padding.
    alignas(float) int8_t packed[15 + 3 * sizeof(float)]{};
    std::fill_n(packed, 15, -2);
    const float scales[] = {0.5f, 0.f, 2.f};
    std::memcpy(packed + 15, scales, sizeof(scales));
    op::LinearLayer layer(DeviceType::kDeviceCPU, 3, 5, true);
    layer.set_group_size(5);
    ASSERT_TRUE(layer.set_weight(0, {5, 3}, packed, DeviceType::kDeviceCPU));
    auto cpu = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor input(DataType::kDataTypeFp32, {3}, true, cpu);
    tensor::Tensor output(DataType::kDataTypeFp32, {5}, true, cpu);
    std::fill_n(input.ptr<float>(), 3, 1.f);
    ASSERT_TRUE(layer.forward(input, output));
    const float expected[] = {-3.f, -2.f, 0.f, -8.f, -12.f};
    for (int i = 0; i < 5; ++i) EXPECT_FLOAT_EQ(output.index<float>(i), expected[i]);
}

TEST_P(LinearLayerCUDA, ComputesMatrixVectorProduct) {
    int device_count = 0;
    auto status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA unavailable: " << cudaGetErrorString(status);
    }
    const bool quantized = std::get<0>(GetParam());
    const bool has_bias = std::get<1>(GetParam());

    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor input(DataType::kDataTypeFp32, {3}, true, allocator);
    tensor::Tensor output(DataType::kDataTypeFp32, {2}, true, allocator);
    tensor::Tensor weight(quantized ? DataType::kDataTypeInt8 : DataType::kDataTypeFp32,
                          {2, 3}, true, allocator);
    tensor::Tensor scales(DataType::kDataTypeFp32, {2}, true, allocator);
    const float input_values[] = {2.f, -1.f, 3.f};
    const int8_t weight_values[] = {1, 2, 3, -4, 5, 2};
    float bias[] = {1.f, -2.f};
    std::copy_n(input_values, 3, input.ptr<float>());
    if (quantized) {
        std::copy_n(weight_values, 6, weight.ptr<int8_t>());
        scales.index<float>(0) = 0.5f;
        scales.index<float>(1) = 2.f;
    } else {
        std::copy_n(weight_values, 6, weight.ptr<float>());
    }

    op::LinearLayer layer(DeviceType::kDeviceCPU, 3, 2, quantized, has_bias);
    if (quantized) {
        layer.set_group_size(3);
        ASSERT_TRUE(layer.set_weight(0, weight, scales));
    } else {
        ASSERT_TRUE(layer.set_weight(0, weight));
    }
    if (has_bias) {
        ASSERT_TRUE(layer.set_bias(0, 2, bias, DeviceType::kDeviceCPU));
    }
    layer.set_input(0, input);
    layer.set_output(0, output);
    auto config = std::make_shared<kernel::CudaConfig>();
    ASSERT_EQ(cudaStreamCreateWithFlags(&config->stream, cudaStreamNonBlocking), cudaSuccess);
    layer.set_cuda_config(config);
    layer.to_cuda();
    layer.set_device_type(DeviceType::kDeviceCUDA);

    ASSERT_TRUE(layer.forward(layer.get_input(0), layer.get_output(0)));
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(config->stream), cudaSuccess);
    auto result = layer.get_output(0);
    result.to_cpu();
    EXPECT_FLOAT_EQ(result.index<float>(0), (quantized ? 4.5f : 9.f) + (has_bias ? bias[0] : 0.f));
    EXPECT_FLOAT_EQ(result.index<float>(1), (quantized ? -14.f : -7.f) + (has_bias ? bias[1] : 0.f));
}

INSTANTIATE_TEST_SUITE_P(PrecisionAndBias, LinearLayerCUDA,
                         ::testing::Combine(::testing::Bool(), ::testing::Bool()));
}  // namespace
