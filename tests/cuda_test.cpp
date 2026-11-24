#include "tensor/tensor.h"
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

TEST(CUDA, AllocatorPoolCleanup) {
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count == 0) {
        GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
    }
    int original_device = -1;
    ASSERT_EQ(cudaGetDevice(&original_device), cudaSuccess);
    {
        base::CUDADeviceAllocator allocator;
        EXPECT_EQ(allocator.allocate(0), nullptr);
        allocator.release(nullptr);
        for (const size_t size : {size_t{16}, size_t{2 * 1024 * 1024}}) {
            void* ptr = allocator.allocate(size);
            ASSERT_NE(ptr, nullptr);
            allocator.memset_zero(ptr, size, nullptr, true);
            allocator.release(ptr);
            void* reused = allocator.allocate(size);
            EXPECT_EQ(reused, ptr);
            allocator.release(reused);
        }
    }
    // Compute Sanitizer --leak-check full verifies both pools were freed.
    int current_device = -1;
    ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
    EXPECT_EQ(current_device, original_device);
}

TEST(CUDA, TensorRoundTripAndAlreadyOnDevice) {
    int count = 0;
    auto status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count == 0) {
        GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
    }
    auto cpu = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor tensor(base::DataType::kDataTypeInt32, {4}, true, cpu);
    for (int i = 0; i < 4; ++i) tensor.index<int32_t>(i) = i + 10;
    tensor.to_cuda();
    ASSERT_EQ(tensor.device_type(), base::DeviceType::kDeviceCUDA);
    auto buffer = tensor.get_buffer();
    tensor.to_cuda();
    EXPECT_EQ(tensor.get_buffer(), buffer);
    auto clone = tensor.clone();
    tensor.to_cpu();
    for (int i = 0; i < 4; ++i) EXPECT_EQ(tensor.index<int32_t>(i), i + 10);
    auto cuda = base::CUDADeviceAllocatorFactory::get_instance();
    cuda->memset_zero(clone.get_buffer()->ptr(), clone.byte_size());
    clone.to_cpu();
    for (int i = 0; i < 4; ++i) EXPECT_EQ(clone.index<int32_t>(i), 0);
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    tensor.to_cuda(stream);
    tensor.to_cpu();
    for (int i = 0; i < 4; ++i) EXPECT_EQ(tensor.index<int32_t>(i), i + 10);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
