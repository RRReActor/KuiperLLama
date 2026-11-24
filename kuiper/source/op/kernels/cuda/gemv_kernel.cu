#include "tensor/tensor.h"
#include "gemv_kernel.cuh"
#include <cstdint>

#define WarpSize 32

namespace kernel {
    static __device__ __forceinline__ float warp_reduce_sum(float sum) {
        for (int offset = WarpSize / 2; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }
        return sum;
    }

    template <int BlockSize>
    static __device__ __forceinline__ float block_reduce_sum(float sum) {
        const int lane_id = threadIdx.x % WarpSize;
        const int warp_id = threadIdx.x / WarpSize;
        __shared__ float shared[BlockSize / WarpSize];
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            shared[warp_id] = sum;
        }
        __syncthreads();
        sum = (threadIdx.x < BlockSize / WarpSize) ? shared[lane_id] : 0.0f;
        if (warp_id == 0) {
            sum = warp_reduce_sum(sum);
        }
        return sum;
    }
    
    // one block process one row of the matrix
    template <int BlockSize>
    __global__ void gemv_kernel_fp32(float* y, const float* x, const float* A, int rows, int cols) {
        const int row = blockIdx.x;
        const int tid = threadIdx.x;
        if (row >= rows) return;
        const float* A_row = A + row * cols;
        float sum = 0.0f;
        
        const int pack_size = 4;
        const bool aligned =
            reinterpret_cast<std::uintptr_t>(A_row) % alignof(float4) == 0 &&
            reinterpret_cast<std::uintptr_t>(x) % alignof(float4) == 0;
        const int pack_num = aligned ? cols / pack_size : 0;


        for (int i = tid; i < pack_num; i += BlockSize) {
            float4 a = reinterpret_cast<const float4*>(A_row)[i];
            float4 b = reinterpret_cast<const float4*>(x)[i];
            sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }

        for (int i = pack_num * pack_size + tid; i < cols; i += BlockSize) {
            sum += A_row[i] * x[i];
        }

        sum = block_reduce_sum<BlockSize>(sum);
        if (tid == 0) {
            y[row] = sum;
        }
    }

    template <int BlockSize>
    __global__ void gemv_kernel_fp32int8(float* y, const float* x, const int8_t* A,
        const int32_t group_size, const float* scales, int rows, int cols) {
        const int tid = threadIdx.x;
        const int row = blockIdx.x;
        if (row >= rows) return;
        float sum = 0.0f;
        for (int i = tid; i < cols; i += BlockSize) {
            const size_t weight_id = static_cast<size_t>(row) * cols + i;
            sum += (A[weight_id] * scales[weight_id / group_size]) * x[i];
        }
        sum = block_reduce_sum<BlockSize>(sum);
        if (tid == 0) {
            y[row] = sum;   
        }
    }

    // Qwen3's default layout: a warp computes one row, with four consecutive
    // weights per lane. Each packed load lies within one 64-element group.
    static __global__ void gemv_int8_group64(float* y, const float4* x, const char4* weights,
                                              const float* scales, int rows, int packs_per_row) {
        const int row = blockIdx.x * blockDim.y + threadIdx.y;
        if (row >= rows) return;
        const size_t start = static_cast<size_t>(row) * packs_per_row;
        float sum = 0.f;
        for (int pack = threadIdx.x; pack < packs_per_row; pack += WarpSize) {
            const char4 q = weights[start + pack];
            const float4 v = x[pack];
            const float scale = scales[(start + pack) / 16];
            sum += (q.x * scale) * v.x;
            sum += (q.y * scale) * v.y;
            sum += (q.z * scale) * v.z;
            sum += (q.w * scale) * v.w;
        }
        sum = warp_reduce_sum(sum);
        if (threadIdx.x == 0) y[row] = sum;
    }

    void gemv_kernel_cuda(tensor::Tensor output, const tensor::Tensor& input, 
            const tensor::Tensor& weight, void* stream) {
        CHECK(input.is_empty() == false);
        CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

        CHECK(weight.is_empty() == false && weight.dims_size() == 2);
        CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);

        const int32_t rows = weight.get_dim(0);
        const int32_t cols = weight.get_dim(1);
        const int32_t BlockSize = 256;
        CHECK_EQ(input.dims_size(), 1);
        CHECK_EQ(cols, input.get_dim(0));
        CHECK(!output.is_empty());
        CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);
        CHECK_EQ(output.dims_size(), 1);
        CHECK_EQ(output.get_dim(0), rows);
        CHECK(input.data_type() == base::DataType::kDataTypeFp32);
        CHECK(output.data_type() == base::DataType::kDataTypeFp32);
        CHECK(weight.data_type() == base::DataType::kDataTypeFp32);
        if (stream) {
            gemv_kernel_fp32<BlockSize><<<rows, BlockSize, 0, reinterpret_cast<cudaStream_t>(stream)>>>
                (output.ptr<float>(), input.ptr<float>(), weight.ptr<float>(), rows, cols);
        } else {
            gemv_kernel_fp32<BlockSize><<<rows, BlockSize>>>
                (output.ptr<float>(), input.ptr<float>(), weight.ptr<float>(), rows, cols);
        }
        CHECK_EQ(cudaGetLastError(), cudaSuccess);
    }

    void gemv_kernel_cuda_qint8(tensor::Tensor output, const tensor::Tensor& input,
             const tensor::Tensor& weight, int32_t group_size, 
             const tensor::Tensor& scale, void* stream) {
        CHECK(input.is_empty() == false);
        CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

        CHECK(weight.is_empty() == false && weight.dims_size() == 2);
        CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);

        const int32_t rows = weight.get_dim(0);
        const int32_t cols = weight.get_dim(1);
        const int32_t BlockSize = 256;
        CHECK_EQ(input.dims_size(), 1);
        CHECK_EQ(cols, input.get_dim(0));
        CHECK(!output.is_empty());
        CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);
        CHECK_EQ(output.dims_size(), 1);
        CHECK_EQ(output.get_dim(0), rows);
        CHECK(input.data_type() == base::DataType::kDataTypeFp32);
        CHECK(output.data_type() == base::DataType::kDataTypeFp32);

        CHECK(weight.data_type() == base::DataType::kDataTypeInt8);
        CHECK_GT(group_size, 0);
        CHECK(!scale.is_empty());
        CHECK(scale.device_type() == base::DeviceType::kDeviceCUDA);
        CHECK(scale.data_type() == base::DataType::kDataTypeFp32);
        CHECK_EQ(scale.dims_size(), 1);
        CHECK_EQ(scale.size(), (weight.size() - 1) / group_size + 1);
        const auto cuda_stream = static_cast<cudaStream_t>(stream);
        if (group_size == 64 && cols % 64 == 0 &&
            reinterpret_cast<std::uintptr_t>(weight.ptr<int8_t>()) % alignof(char4) == 0 &&
            reinterpret_cast<std::uintptr_t>(input.ptr<float>()) % alignof(float4) == 0) {
            constexpr int rows_per_block = 4;
            gemv_int8_group64<<<1 + (rows - 1) / rows_per_block, dim3(WarpSize, rows_per_block), 0, cuda_stream>>>(
                output.ptr<float>(), input.ptr<float4>(), weight.ptr<char4>(), scale.ptr<float>(), rows, cols / 4);
        } else {
            gemv_kernel_fp32int8<BlockSize><<<rows, BlockSize, 0, cuda_stream>>>(
                output.ptr<float>(), input.ptr<float>(), weight.ptr<int8_t>(), group_size, scale.ptr<float>(), rows, cols);
        }
        CHECK_EQ(cudaGetLastError(), cudaSuccess);
    }

}
