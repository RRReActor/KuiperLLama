#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "base/cuda_config.h"
#include "op/embedding.h"

namespace {
using base::DataType;
using base::DeviceType;

void check_embedding(DeviceType device, int32_t token_count, bool use_stream) {
    constexpr int32_t vocab_size = 3;
    constexpr int32_t dim = 129;
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor tokens(DataType::kDataTypeInt32, {token_count + 1}, true, allocator);
    tensor::Tensor count(DataType::kDataTypeInt32, {1}, true, allocator);
    tensor::Tensor weights(DataType::kDataTypeFp32, {vocab_size, dim}, true, allocator);
    tensor::Tensor output(DataType::kDataTypeFp32, {token_count, dim}, true, allocator);
    count.index<int32_t>(0) = token_count;
    for (int32_t i = 0; i < token_count; ++i) {
        tokens.index<int32_t>(i) = (i + 2) % vocab_size;
    }
    tokens.index<int32_t>(token_count) = -1;
    for (int32_t i = 0; i < vocab_size * dim; ++i) {
        weights.index<float>(i) = static_cast<float>(i);
    }

    op::EmbeddingLayer layer(device, dim, vocab_size);
    if (device == DeviceType::kDeviceCUDA) {
        weights.to_cuda();
        output.to_cuda();
    }
    if (use_stream) {
        auto config = std::make_shared<kernel::CudaConfig>();
        ASSERT_EQ(cudaStreamCreateWithFlags(&config->stream, cudaStreamNonBlocking), cudaSuccess);
        layer.set_cuda_config(config);
    }

    ASSERT_TRUE(layer.set_weight(0, weights));
    EXPECT_FALSE(layer.forward());
    ASSERT_TRUE(layer.forward(tokens, count, output));
    if (device == DeviceType::kDeviceCUDA) {
        output.to_cpu();
    }
    for (int32_t row = 0; row < token_count; ++row) {
        for (int32_t column = 0; column < dim; ++column) {
            const float expected = tokens.index<int32_t>(row) * dim + column;
            ASSERT_EQ(output.index<float>(row * dim + column), expected);
        }
    }

    tokens.index<int32_t>(0) = -1;
    EXPECT_EQ(layer.forward(), base::StatusCode::kInvalidArgument);
    tokens.index<int32_t>(0) = vocab_size;
    EXPECT_EQ(layer.forward(), base::StatusCode::kInvalidArgument);
    count.index<int32_t>(0) = token_count + 2;
    EXPECT_EQ(layer.forward(), base::StatusCode::kInvalidArgument);
}

TEST(EmbeddingLayer, GathersActiveTokensOnCPU) {
    check_embedding(DeviceType::kDeviceCPU, 4, false);
}

class EmbeddingLayerCUDA : public ::testing::TestWithParam<bool> {};

TEST_P(EmbeddingLayerCUDA, GathersMoreThan512Tokens) {
    int device_count = 0;
    const auto status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA unavailable: " << cudaGetErrorString(status);
    }
    check_embedding(DeviceType::kDeviceCUDA, 513, GetParam());
}

INSTANTIATE_TEST_SUITE_P(DefaultAndExplicitStream, EmbeddingLayerCUDA, ::testing::Bool());
}
