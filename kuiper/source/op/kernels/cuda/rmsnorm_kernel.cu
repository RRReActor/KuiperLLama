#include "rmsnorm_kernel.cuh"
#include "../rmsnorm_validation.h"
#include <cstdint>

#define WarpSize 32

namespace kernel {
    static __device__ __forceinline__ float warp_reduce_sum(float sum) {
        for(int offset = WarpSize / 2; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }
        return sum;
    }
    // only the first warp get the true result
    template < int blockSize>
    static __device__ __forceinline__ float block_reduce_sum(float sum) {
        __shared__ float smem[blockSize / WarpSize];
        const int tid = threadIdx.x;
        const int lane_id = tid % WarpSize;
        const int warp_id = tid / WarpSize;
        sum = warp_reduce_sum(sum);
        if(lane_id == 0) {
            smem[warp_id] = sum;
        }
        __syncthreads();
        sum = tid < blockSize / WarpSize ? smem[lane_id] : 0.0;
        if(warp_id == 0) {
            sum = warp_reduce_sum(sum);
        }
        return sum;
    }

    /**
     * 计算最后一维度的rmsnorm
     * each block process one row
     *  */ 
    template < int blockSize>
    static __global__ void row_rmsnorm_f32(float* dst, const float* src, const float* wei, int32_t size, float eps) {
        const int tid = threadIdx.x;
        const int row_id = blockIdx.x;
        
        const float* block_in = src + row_id * size;
        float* block_out = dst + row_id * size;

        constexpr int pack_size = 4;
        // Every thread in the block takes the same path. Rows whose starting
        // addresses are not float4-aligned use scalar accesses.
        const bool aligned =
            reinterpret_cast<std::uintptr_t>(block_in) % alignof(float4) == 0 &&
            reinterpret_cast<std::uintptr_t>(block_out) % alignof(float4) == 0 &&
            reinterpret_cast<std::uintptr_t>(wei) % alignof(float4) == 0;
        const int pack_num = aligned ? size / pack_size : 0;
        const float4* in_pack = reinterpret_cast<const float4*>(block_in);
        const float4* wei_pack = reinterpret_cast<const float4*>(wei);
        float4* out_pack = reinterpret_cast<float4*>(block_out);

        float sum = 0.0f;
        for (int i = tid; i < pack_num; i += blockSize) {
            const float4 val = in_pack[i];
            sum += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
        }
        for (int i = pack_num * pack_size + tid; i < size; i += blockSize) {
            const float val = block_in[i];
            sum += val * val;
        }
        sum = block_reduce_sum<blockSize>(sum);
        __shared__ float scale;
        if (tid == 0) scale = rsqrtf(sum / size + eps);
        __syncthreads();

        for (int i = tid; i < pack_num; i += blockSize) {
            const float4 val = in_pack[i];
            const float4 w = wei_pack[i];
            out_pack[i] = make_float4(val.x * scale * w.x, val.y * scale * w.y,
                                     val.z * scale * w.z, val.w * scale * w.w);
        }
        for (int i = pack_num * pack_size + tid; i < size; i += blockSize) {
            block_out[i] = block_in[i] * scale * wei[i];
        }
    }

    void rmsnorm_kernel_cuda(tensor::Tensor output, const tensor::Tensor& input,
                                 const tensor::Tensor& weight, void* stream) {
        validate_rmsnorm(output, input, weight, base::DeviceType::kDeviceCUDA, true);

        const float eps = 1e-6f;
        constexpr int block_size = 256;
        const int size = input.get_dim(input.dims_size() - 1);

        const float* in_ptr = input.ptr<float>();
        float* out_ptr = reinterpret_cast<float*>(output.ptr<float>());
        const float* wei_ptr = weight.ptr<float>();
    
        if (stream) {
            cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
            row_rmsnorm_f32<block_size><<<input.size() / size, block_size, 0, cuda_stream>>>(out_ptr, in_ptr, wei_ptr, size, eps);
        } else {
            row_rmsnorm_f32<block_size><<<input.size() / size, block_size>>>(out_ptr, in_ptr, wei_ptr, size, eps);
        }
        CHECK_EQ(cudaGetLastError(), cudaSuccess);
    }
}
