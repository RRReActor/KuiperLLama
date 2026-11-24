#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include "op/kernels/kernels_interface.h"

namespace {
void check_rmsnorm_1024_by_1024(base::DeviceType device) {
    constexpr int rows = 1024;
    constexpr int width = 1024;
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor input(base::DataType::kDataTypeFp32, {rows, width}, true, alloc);
    tensor::Tensor weight(base::DataType::kDataTypeFp32, {width}, true, alloc);
    tensor::Tensor expected(base::DataType::kDataTypeFp32, {rows, width}, true, alloc);
    for (int j = 0; j < width; ++j)
        weight.index<float>(j) = (j % 23 - 11) * 0.125f;
    for (int row = 0; row < rows; ++row) {
        double sum = 0;
        for (int j = 0; j < width; ++j) {
            const int index = row * width + j;
            // Include a zero row and different magnitudes across rows.
            const float value = row == 0 ? 0.f
                : ((row * 17 + j * 13) % 101 - 50) * (1 + row % 7) * 0.03125f;
            input.index<float>(index) = value;
            sum += static_cast<double>(value) * value;
        }
        const double scale = 1.0 / std::sqrt(sum / width + 1e-6);
        for (int j = 0; j < width; ++j) {
            const int index = row * width + j;
            expected.index<float>(index) = input.index<float>(index) * scale * weight.index<float>(j);
        }
    }
    const bool cuda = device == base::DeviceType::kDeviceCUDA;
    for (bool in_place : {false, true}) {
        for (bool custom_stream : {false, true}) {
            if (!cuda && custom_stream) continue;
            SCOPED_TRACE(::testing::Message() << "in_place=" << in_place
                                             << " custom_stream=" << custom_stream);
            auto source = input.clone();
            auto weights = weight.clone();
            tensor::Tensor output(base::DataType::kDataTypeFp32, {rows, width}, true, alloc);
            for (size_t i = 0; i < output.size(); ++i)
                output.ptr<float>()[i] = std::numeric_limits<float>::quiet_NaN();
            if (cuda) {
                source.to_cuda();
                weights.to_cuda();
                output.to_cuda();
            }
            if (in_place) output = source;
            struct StreamGuard {
                cudaStream_t stream = nullptr;
                ~StreamGuard() { if (stream) cudaStreamDestroy(stream); }
            } guard;
            if (custom_stream)
                ASSERT_EQ(cudaStreamCreateWithFlags(&guard.stream, cudaStreamNonBlocking), cudaSuccess);
            kernel::get_rmsnorm_kernel(device)(output, source, weights, guard.stream);
            if (cuda) {
                ASSERT_EQ(cudaGetLastError(), cudaSuccess);
                ASSERT_EQ(cudaStreamSynchronize(guard.stream), cudaSuccess);
                output.to_cpu();
            }
            for (size_t i = 0; i < output.size(); ++i) {
                const float reference = expected.ptr<float>()[i];
                ASSERT_NEAR(output.ptr<float>()[i], reference,
                            2e-5f * (1.f + std::fabs(reference)))
                    << "row=" << i / width << " column=" << i % width;
            }
        }
    }
}
} // namespace

TEST(RMSNormCPU, Matrix1024By1024) {
    check_rmsnorm_1024_by_1024(base::DeviceType::kDeviceCPU);
}

TEST(RMSNormCUDA, Matrix1024By1024) {
    int count = 0;
    const auto state = cudaGetDeviceCount(&count);
    if (state != cudaSuccess || count == 0)
        GTEST_SKIP() << "CUDA unavailable: " << cudaGetErrorString(state);
    check_rmsnorm_1024_by_1024(base::DeviceType::kDeviceCUDA);
}

TEST(RMSNormCPU, KnownValuesAndInPlace) {
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor input(base::DataType::kDataTypeFp32, {2, 3}, true, alloc);
    tensor::Tensor weight(base::DataType::kDataTypeFp32, {3}, true, alloc);
    for (int j = 0; j < 3; ++j) weight.index<float>(j) = j + 1;
    for (int i = 0; i < 6; ++i) input.index<float>(i) = i < 3 ? 2.f : 0.f;
    kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(input, input, weight, nullptr);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(input.index<float>(i), i < 3 ? 2.0 * (i + 1) / std::sqrt(4.0 + 1e-6) : 0, 1e-6);
}

TEST(RMSNormCUDA, MatchesCPUAcrossShapesStreamsAndInPlace) {
    int count = 0;
    const auto state = cudaGetDeviceCount(&count);
    if (state != cudaSuccess || count == 0)
        GTEST_SKIP() << "CUDA unavailable: " << cudaGetErrorString(state);
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    for (int width : {1, 3, 31, 32, 33, 255, 256, 257, 480, 1024, 4099, 72480}) {
        for (bool rows : {false, true}) {
            const std::vector<int32_t> shape = rows ? std::vector<int32_t>{3, width} : std::vector<int32_t>{width};
            tensor::Tensor input(base::DataType::kDataTypeFp32, shape, true, alloc);
            tensor::Tensor weight(base::DataType::kDataTypeFp32, {width}, true, alloc);
            tensor::Tensor expected(base::DataType::kDataTypeFp32, shape, true, alloc);
            for (size_t i = 0; i < input.size(); ++i)
                input.ptr<float>()[i] = (static_cast<int>(i % 37) - 18) * 0.13f;
            for (int j = 0; j < width; ++j) weight.index<float>(j) = (j % 11 - 5) * 0.2f;
            auto cpu = rows ? kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)
                            : kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU);
            cpu(expected, input, weight, nullptr);
            for (bool in_place : {false, true}) {
                for (bool custom_stream : {false, true}) {
                    SCOPED_TRACE(::testing::Message() << width << " rows=" << rows << " inplace=" << in_place << " stream=" << custom_stream);
                    auto gpu_input = input.clone(); gpu_input.to_cuda();
                    auto gpu_weight = weight.clone(); gpu_weight.to_cuda();
                    auto result = in_place ? gpu_input : expected.clone();
                    if (!in_place) result.to_cuda();
                    cudaStream_t stream = nullptr;
                    if (custom_stream) ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
                    auto gpu = rows ? kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)
                                    : kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA);
                    gpu(result, gpu_input, gpu_weight, stream);
                    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
                    if (custom_stream) ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
                    result.to_cpu();
                    for (size_t i = 0; i < expected.size(); ++i)
                        ASSERT_NEAR(result.ptr<float>()[i], expected.ptr<float>()[i],
                                    2e-5f * (1.f + std::fabs(expected.ptr<float>()[i]))) << i;
                }
            }
        }
    }
}

#include "op/rmsnorm.h"
TEST(RMSNormCPU, LayerForwardAndShapeValidation) {
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor input(base::DataType::kDataTypeFp32, {2, 3}, true, alloc);
    tensor::Tensor output(base::DataType::kDataTypeFp32, {2, 3}, true, alloc);
    tensor::Tensor weight(base::DataType::kDataTypeFp32, {3}, true, alloc);
    for (int i = 0; i < 6; ++i) input.index<float>(i) = 2;
    for (int i = 0; i < 3; ++i) weight.index<float>(i) = 1;
    op::RmsNormLayer layer(base::DeviceType::kDeviceCPU, 3);
    ASSERT_TRUE(layer.set_weight(0, weight));
    ASSERT_TRUE(layer.forward(input, output));
    for (int i = 0; i < 6; ++i) EXPECT_NEAR(output.index<float>(i), 1, 1e-6);
    output.reshape({6});
    EXPECT_FALSE(layer.forward(input, output));
}
