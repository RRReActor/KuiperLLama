#include <algorithm>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <vector>

#include "sampler/argmax_sampler.h"
#include "tensor/tensor.h"

namespace {
TEST(ArgmaxSampler, CPUSelectsFirstMaximum) {
    sampler::ArgmaxSampler argmax(base::DeviceType::kDeviceCPU);
    sampler::Sampler& sampler = argmax;
    const float logits[] = {-5, -2, -3, -2};
    EXPECT_EQ(sampler.sample(logits, 4), 1u);
    EXPECT_EQ(sampler.sample(logits, 1), 0u);
    EXPECT_EQ(sampler.sample(nullptr, 0), 0u);
}

TEST(ArgmaxSampler, CUDASelectsFirstMaximumOnSuppliedStream) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    sampler::ArgmaxSampler argmax(base::DeviceType::kDeviceCUDA);
    std::vector<float> logits(4099, -9.0f);
    tensor::Tensor device_logits(base::DataType::kDataTypeFp32, {4099}, true,
                                 base::CUDADeviceAllocatorFactory::get_instance());
    struct StreamGuard {
        cudaStream_t stream = nullptr;
        ~StreamGuard() { if (stream) cudaStreamDestroy(stream); }
    } guard;
    ASSERT_EQ(cudaStreamCreateWithFlags(&guard.stream, cudaStreamNonBlocking), cudaSuccess);
    for (cudaStream_t stream : std::vector<cudaStream_t>{nullptr, guard.stream}) {
        for (int size : {1, 17, 512, 4099}) {
            std::fill(logits.begin(), logits.end(), -9.0f);
            const int first = size / 2;
            logits[first] = logits[size - 1] = -2.0f;
            ASSERT_EQ(cudaMemcpyAsync(device_logits.ptr<float>(), logits.data(), size * sizeof(float),
                                      cudaMemcpyHostToDevice, stream), cudaSuccess);
            EXPECT_EQ(argmax.sample(device_logits.ptr<float>(), size, stream),
                      static_cast<std::size_t>(first)) << "size=" << size;
        }
    }
}
}  // namespace
