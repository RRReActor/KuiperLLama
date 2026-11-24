#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>
#include "op/kernels/cuda/gemv_kernel.cuh"

namespace {
class GemvCUDA : public ::testing::Test {
protected:
    void SetUp() override {
        int count = 0;
        const auto status = cudaGetDeviceCount(&count);
        if (status != cudaSuccess || count == 0)
            GTEST_SKIP() << "CUDA unavailable: " << cudaGetErrorString(status);
    }

    void check_result(bool quantized, int rows, int cols, int group_size,
                      bool custom_stream, bool zero_input = false) {
        SCOPED_TRACE(::testing::Message() << "rows=" << rows << " cols=" << cols
                     << " group_size=" << group_size << " stream=" << custom_stream);
        auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
        tensor::Tensor input(base::DataType::kDataTypeFp32, {cols}, true, alloc);
        tensor::Tensor weight(quantized ? base::DataType::kDataTypeInt8
                                       : base::DataType::kDataTypeFp32,
                              {rows, cols}, true, alloc);
        tensor::Tensor output(base::DataType::kDataTypeFp32, {rows}, true, alloc);
        const int groups = (rows * cols + group_size - 1) / group_size;
        tensor::Tensor scale(base::DataType::kDataTypeFp32, {groups}, true, alloc);
        for (int i = 0; i < cols; ++i)
            input.index<float>(i) = zero_input ? 0.f : (i % 19 - 9) * 0.125f;
        for (int i = 0; i < groups; ++i)
            scale.index<float>(i) = (i % 7 + 1) * 0.03125f;
        for (int i = 0; i < rows * cols; ++i) {
            if (quantized) weight.index<int8_t>(i) = static_cast<int8_t>(i % 256 - 128);
            else weight.index<float>(i) = (i % 31 - 15) * 0.0625f;
        }
        std::vector<double> expected(rows, 0.0);
        for (int row = 0; row < rows; ++row) {
            output.index<float>(row) = std::numeric_limits<float>::quiet_NaN();
            for (int col = 0; col < cols; ++col) {
                const int index = row * cols + col;
                const double w = quantized
                    ? static_cast<double>(weight.index<int8_t>(index)) * scale.index<float>(index / group_size)
                    : weight.index<float>(index);
                expected[row] += w * input.index<float>(col);
            }
        }
        input.to_cuda();
        weight.to_cuda();
        output.to_cuda();
        if (quantized) scale.to_cuda();
        struct StreamGuard {
            cudaStream_t stream = nullptr;
            ~StreamGuard() { if (stream) cudaStreamDestroy(stream); }
        } guard;
        if (custom_stream)
            ASSERT_EQ(cudaStreamCreateWithFlags(&guard.stream, cudaStreamNonBlocking), cudaSuccess);
        if (quantized)
            kernel::gemv_kernel_cuda_qint8(output, input, weight, group_size, scale, guard.stream);
        else
            kernel::gemv_kernel_cuda(output, input, weight, guard.stream);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(guard.stream), cudaSuccess);
        output.to_cpu();
        for (int row = 0; row < rows; ++row)
            ASSERT_NEAR(output.index<float>(row), expected[row],
                        2e-5 * (1.0 + std::fabs(expected[row]))) << "row=" << row;
    }
};

TEST_F(GemvCUDA, FP32MatchesReferenceAcrossShapesAndStreams) {
    for (int cols : {1, 3, 4, 31, 32, 33, 255, 256, 257, 1024, 4099})
        for (int rows : {1, 7})
            for (bool stream : {false, true})
                check_result(false, rows, cols, 1, stream);
}

TEST_F(GemvCUDA, Int8MatchesReferenceAcrossGroupsAndStreams) {
    for (int cols : {1, 3, 32, 257, 1024, 4099})
        for (int group_size : {1, 7, 32, 64, 128})
            for (bool stream : {false, true})
                check_result(true, 7, cols, group_size, stream);
}

TEST_F(GemvCUDA, ZeroInputOverwritesOutput) {
    for (bool quantized : {false, true})
        for (bool stream : {false, true})
            check_result(quantized, 7, 257, 32, stream, true);
}

TEST_F(GemvCUDA, Matrix1024By1024) {
    for (bool quantized : {false, true})
        for (bool stream : {false, true})
            check_result(quantized, 1024, 1024, 128, stream);
}
} // namespace
