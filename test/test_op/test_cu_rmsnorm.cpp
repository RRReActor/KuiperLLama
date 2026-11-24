#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include "op/kernels/kernels_interface.h"

namespace {
class test_rmsnorm_cu : public ::testing::Test {
protected:
    void SetUp() override {
        int count = 0;
        const auto status = cudaGetDeviceCount(&count);
        if (status != cudaSuccess || count == 0)
            GTEST_SKIP() << "CUDA unavailable: " << cudaGetErrorString(status);
    }

    void compare(int rows, int width, bool use_stream, bool in_place = false) {
        auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
        const std::vector<int32_t> shape = rows == 1 ? std::vector<int32_t>{width}
                                                   : std::vector<int32_t>{rows, width};
        tensor::Tensor input(base::DataType::kDataTypeFp32, shape, true, alloc_cpu);
        tensor::Tensor weight(base::DataType::kDataTypeFp32, {width}, true, alloc_cpu);
        tensor::Tensor expected(base::DataType::kDataTypeFp32, shape, true, alloc_cpu);
        std::mt19937 generator(42);
        std::uniform_real_distribution<float> distribution(-1.f, 1.f);
        for (size_t i = 0; i < input.size(); ++i) input.ptr<float>()[i] = distribution(generator);
        for (int i = 0; i < width; ++i) weight.index<float>(i) = distribution(generator);
        // Current API is output, input, weight, stream; normalization uses the last dimension.
        kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(expected, input, weight, nullptr);
        auto input_cuda = input.clone(); input_cuda.to_cuda();
        auto weight_cuda = weight.clone(); weight_cuda.to_cuda();
        auto output_cuda = in_place ? input_cuda : expected.clone();
        if (!in_place) output_cuda.to_cuda();

        struct StreamGuard {
            cudaStream_t stream = nullptr;
            ~StreamGuard() { if (stream) cudaStreamDestroy(stream); }
        } guard;
        if (use_stream)
            ASSERT_EQ(cudaStreamCreateWithFlags(&guard.stream, cudaStreamNonBlocking), cudaSuccess);
        kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)(
            output_cuda, input_cuda, weight_cuda, guard.stream);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(guard.stream), cudaSuccess);
        output_cuda.to_cpu();
        for (size_t i = 0; i < input.size(); ++i)
            ASSERT_NEAR(output_cuda.ptr<float>()[i], expected.ptr<float>()[i],
                        1e-5f * (1.f + std::fabs(expected.ptr<float>()[i]))) << "index=" << i;
    }
};
class test_rmsnorm_cu_dim : public test_rmsnorm_cu {};

TEST_F(test_rmsnorm_cu, rmsnorm_nostream) { compare(1, 32 * 15, false); }
TEST_F(test_rmsnorm_cu, rmsnorm_stream) { compare(1, 32, true); }
TEST_F(test_rmsnorm_cu, rmsnorm_stream2) { compare(1, 32 * 151 * 15, true); }
TEST_F(test_rmsnorm_cu_dim, rmsnorm_stream) {
    compare(4, 1024, true);
    compare(4, 1024, true, true);
    compare(4, 1027, true);
    compare(4, 1027, true, true);
}
} // namespace
