#include <cmath>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "base/cuda_config.h"
#include "op/activation.h"

namespace {
using base::DataType;
using base::DeviceType;

TEST(ActivationLayer, ValidatesInputsAndOutput) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor vector(DataType::kDataTypeFp32, {3}, true, allocator);
    tensor::Tensor matrix(DataType::kDataTypeFp32, {3, 1}, true, allocator);
    tensor::Tensor wrong_size(DataType::kDataTypeFp32, {2}, true, allocator);
    op::Activation layer(DeviceType::kDeviceCPU, 3);

    EXPECT_FALSE(layer.check());
    layer.set_input(0, vector);
    layer.set_input(1, vector);
    layer.set_output(0, vector);
    EXPECT_TRUE(layer.check());
    for (int i = 0; i < 3; ++i) vector.index<float>(i) = i - 1.0f;
    ASSERT_TRUE(layer.forward());
    for (int i = 0; i < 3; ++i) {
        const float x = i - 1.0f;
        EXPECT_NEAR(vector.index<float>(i), x * x / (1.0f + std::exp(-x)), 1e-6f);
    }
    EXPECT_FALSE(layer.forward(matrix, vector, vector));
    EXPECT_FALSE(layer.forward(vector, wrong_size, vector));
    EXPECT_FALSE(layer.forward(vector, vector, matrix));
}

class ActivationLayerCUDA : public ::testing::TestWithParam<bool> {};

TEST_P(ActivationLayerCUDA, ComputesSwiGLU) {
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count == 0) {
        GTEST_SKIP() << "CUDA unavailable: " << cudaGetErrorString(status);
    }
    constexpr int size = 129;
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor input(DataType::kDataTypeFp32, {size}, true, allocator);
    tensor::Tensor gate(DataType::kDataTypeFp32, {size}, true, allocator);
    tensor::Tensor output(DataType::kDataTypeFp32, {size}, true, allocator);
    for (int i = 0; i < size; ++i) {
        input.index<float>(i) = (i - 64) / 16.f;
        gate.index<float>(i) = (i % 7) - 3.f;
    }
    auto device_input = input.clone();
    auto device_gate = gate.clone();
    device_input.to_cuda();
    device_gate.to_cuda();
    output.to_cuda();

    op::Activation layer(DeviceType::kDeviceCUDA, size);
    auto config = std::make_shared<kernel::CudaConfig>();
    if (GetParam()) {
        ASSERT_EQ(cudaStreamCreateWithFlags(&config->stream, cudaStreamNonBlocking), cudaSuccess);
        layer.set_cuda_config(config);
    }
    ASSERT_TRUE(layer.forward(device_input, device_gate, output));
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(config->stream), cudaSuccess);
    output.to_cpu();

    for (int i = 0; i < size; ++i) {
        const float x = input.index<float>(i);
        const float expected = x * gate.index<float>(i) / (1.f + std::exp(-x));
        EXPECT_NEAR(output.index<float>(i), expected, 1e-5f) << "index " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(DefaultAndExplicitStream, ActivationLayerCUDA, ::testing::Bool());
}  // namespace
