#include <gtest/gtest.h>

#include <cmath>
#include <complex>

#include "base/cuda_config.h"
#include "op/kernels/cuda/rope_kernel.cuh"
#include "op/rope.h"

namespace {
    constexpr auto kFp32 = base::DataType::kDataTypeFp32;
    constexpr auto kInt32 = base::DataType::kDataTypeInt32;
    constexpr auto kCpu = base::DeviceType::kDeviceCPU;
    constexpr auto kCuda = base::DeviceType::kDeviceCUDA;

#if defined(LLAMA3_SUPPORT)
    constexpr double kTheta = 500000.0;
#elif defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    constexpr double kTheta = 1000000.0;
#else
    constexpr double kTheta = 10000.0;
#endif

    double angle(int pos, int column, int head_size) {
        return pos * std::pow(kTheta, -static_cast<double>(column) / head_size);
    }

    void expect_rotation(const tensor::Tensor& actual, const tensor::Tensor& original,
                         int dim, int head_size, int pos) {
        for (int head = 0; head < dim; head += head_size) {
            for (int pair = 0; pair < head_size / 2; ++pair) {
#if defined(LLAMA3_SUPPORT) || defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
                const int x = head + pair;
                const int y = x + head_size / 2;
#else
                const int x = head + 2 * pair;
                const int y = x + 1;
#endif
                const std::complex<double> input(original.index<float>(x), original.index<float>(y));
                const auto expected = input * std::polar(1.0, angle(pos, 2 * pair, head_size));
                EXPECT_NEAR(actual.index<float>(x), expected.real(), 3e-5) << "index " << x;
                EXPECT_NEAR(actual.index<float>(y), expected.imag(), 3e-5) << "index " << y;
            }
        }
    }

    void check_rope(base::DeviceType device, bool custom_stream) {
        // Exercise partial blocks in both kernels and the last cached position.
        constexpr int head_size = 130;
        constexpr int dim = 3 * head_size;
        constexpr int kv_dim = head_size;
        constexpr int seq_len = 130;
        constexpr int pos = seq_len - 1;
        const bool cuda = device == kCuda;
        std::shared_ptr<base::DeviceAllocator> cpu = base::CPUDeviceAllocatorFactory::get_instance();
        tensor::Tensor original_q(kFp32, {dim + head_size}, true, cpu);
        tensor::Tensor original_k(kFp32, {kv_dim + head_size}, true, cpu);
        for (int i = 0; i < dim + head_size; ++i) {
            original_q.index<float>(i) = (i % 13 - 6) * 0.125f;
        }
        for (int i = 0; i < kv_dim + head_size; ++i) {
            original_k.index<float>(i) = (i % 17 - 8) * 0.0625f;
        }
        auto query_storage = original_q.clone();
        auto key_storage = original_k.clone();
        tensor::Tensor position(kInt32, {1}, true, cpu);
        position.index<int32_t>(0) = pos;

        const auto allocator = cuda ? base::CUDADeviceAllocatorFactory::get_instance() : cpu;
        tensor::Tensor sine(kFp32, {seq_len, head_size}, true, allocator);
        tensor::Tensor cosine(kFp32, {seq_len * head_size}, true, allocator);
        auto config = std::make_shared<kernel::CudaConfig>();
        if (custom_stream) {
            ASSERT_EQ(cudaStreamCreateWithFlags(&config->stream, cudaStreamNonBlocking), cudaSuccess);
        }
        if (cuda) {
            query_storage.to_cuda();
            key_storage.to_cuda();
            kernel::sin_cos_cache_calc_cu(head_size, seq_len, sine, cosine, config->stream);
        } else {
            for (int row = 0; row < seq_len; ++row) {
                for (int column = 0; column < head_size; ++column) {
                    const double phase = angle(row, column, head_size);
                    sine.index<float>(row * head_size + column) = std::sin(phase);
                    cosine.index<float>(row * head_size + column) = std::cos(phase);
                }
            }
        }

        auto query = query_storage;
        auto key = key_storage;
        query.reshape({dim});
        key.reshape({kv_dim});
        op::RoPELayer layer(device, dim, kv_dim, head_size);
        if (custom_stream) {
            layer.set_cuda_config(config);
        }
        const auto status = layer.forward(query, key, position, sine, cosine, tensor::Tensor{});
        ASSERT_TRUE(static_cast<bool>(status)) << status;
        if (cuda) {
            ASSERT_EQ(cudaStreamSynchronize(config->stream), cudaSuccess);
            query_storage.to_cpu();
            key_storage.to_cpu();
            sine.to_cpu();
            cosine.to_cpu();
            for (int row : {0, 1, pos}) {
                for (int column = 0; column < head_size; ++column) {
                    const double phase = angle(row, column, head_size);
                    EXPECT_NEAR(sine.index<float>(row * head_size + column), std::sin(phase), 3e-5);
                    EXPECT_NEAR(cosine.index<float>(row * head_size + column), std::cos(phase), 3e-5);
                }
            }
        }
        expect_rotation(query_storage, original_q, dim, head_size, pos);
        expect_rotation(key_storage, original_k, kv_dim, head_size, pos);
        for (int i = 0; i < head_size; ++i) {
            EXPECT_EQ(query_storage.index<float>(dim + i), original_q.index<float>(dim + i));
            EXPECT_EQ(key_storage.index<float>(kv_dim + i), original_k.index<float>(kv_dim + i));
        }
    }
}  // namespace

TEST(RoPECPU, RotatesGroupedQueryAndKeyInPlace) {
    check_rope(kCpu, false);
}

TEST(RoPECPU, RejectsInvalidPositionAndTensorShapes) {
    auto cpu = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor query(kFp32, {8}, true, cpu);
    tensor::Tensor key(kFp32, {4}, true, cpu);
    tensor::Tensor position(kInt32, {1}, true, cpu);
    tensor::Tensor sine(kFp32, {8}, true, cpu);
    tensor::Tensor cosine(kFp32, {7}, true, cpu);
    op::RoPELayer layer(kCpu, 8, 4, 4);
    for (int pos : {-1, 1, 2}) {
        position.index<int32_t>(0) = pos;
        EXPECT_EQ(layer.forward(query, key, position, sine, cosine, tensor::Tensor{}),
                  base::StatusCode::kInvalidArgument);
    }
    position.index<int32_t>(0) = 0;
    query.reshape({2, 4});
    EXPECT_EQ(layer.forward(query, key, position, sine, cosine, tensor::Tensor{}),
              base::StatusCode::kInvalidArgument);
}

TEST(RoPECUDA, GeneratesCacheAndRotatesOnDefaultAndCustomStreams) {
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count == 0) {
        GTEST_SKIP() << "CUDA unavailable: " << cudaGetErrorString(status);
    }
    check_rope(kCuda, false);
    check_rope(kCuda, true);
}
