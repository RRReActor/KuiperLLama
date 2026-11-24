#include "mha_kernel.cuh"
#include <cfloat>

namespace kernel
{
#define WarpSize 32
    __device__ __forceinline__ float warp_reduce_sum(float sum)
    {
        for (int offset = WarpSize / 2; offset > 0; offset >>= 1)
        {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }
        return sum;
    }

    __device__ __forceinline__ float warp_reduce_max(float val)
    {
        for (int offset = WarpSize / 2; offset > 0; offset >>= 1)
        {
            val = fmax(val, __shfl_xor_sync(0xffffffff, val, offset));
        }
        return val;
    }
    template <int blockSize>
    __device__ float block_reduce_sum(float sum)
    {
        const int tid = threadIdx.x;
        const int lane_id = tid % WarpSize;
        const int warp_id = tid / WarpSize;
        __shared__ float smem[blockSize / WarpSize];
        sum = warp_reduce_sum(sum);
        if (lane_id == 0)
        {
            smem[warp_id] = sum;
        }
        __syncthreads();
        sum = tid < (blockSize / WarpSize) ? smem[lane_id] : 0;
        if (warp_id == 0)
        {
            sum = warp_reduce_sum(sum);
        }
        return sum;
    }
    // only the first warp get the result
    template <int blockSize>
    __device__ float block_reduce_max(float val)
    {
        const int tid = threadIdx.x;
        const int lane_id = tid % WarpSize;
        const int warp_id = tid / WarpSize;
        __shared__ float smem[blockSize / WarpSize];
        val = warp_reduce_max(val);
        if (lane_id == 0)
        {
            smem[warp_id] = val;
        }
        __syncthreads();
        val = tid < (blockSize / WarpSize) ? smem[lane_id] : -FLT_MAX;
        if (warp_id == 0)
        {
            val = warp_reduce_max(val);
        }
        return val;
    }
    // one block handles
    template <int blockSize>
    __device__ void softmax_kernel(float *x, int size)
    {
        const int tid = threadIdx.x;
        float val_max = -FLT_MAX;
        __shared__ float temp;
        for (int t = tid; t < size; t += blockDim.x)
        {
            val_max = fmaxf(val_max, x[t]);
        }
        val_max = block_reduce_max<blockSize>(val_max);
        if (tid == 0)
            temp = val_max;
        __syncthreads();
        val_max = temp;
        float sum = 0;
        for (int t = tid; t < size; t += blockDim.x)
        {
            float temp = expf(x[t] - val_max);
            x[t] = temp;
            sum += temp;
        }
        sum = block_reduce_sum<blockSize>(sum);
        if (tid == 0)
            temp = sum;
        __syncthreads();
        sum = temp;
        for (int t = tid; t < size; t += blockDim.x)
        {
            x[t] /= sum;
        }
    }
    // one block handles one query head
    template <int blockSize>
    __global__ void multi_head_attention_kernel(int32_t pos, int32_t seq_len, float *output,
                                                const float *query, float *score_ptr, const float *key_cache, const float *value_cache,
                                                int32_t kv_head_num, int32_t q_head_num, int32_t head_dim, size_t layer_offset)
    {
        const int head_id = blockIdx.x;
        if (head_id >= q_head_num)
        {
            return;
        }
        const int kv_head_id = head_id / (q_head_num / kv_head_num);
        const int kv_head_offset = kv_head_id * head_dim;
        const int kv_dim = kv_head_num * head_dim;
        extern __shared__ float s_query_head[];
        const int tid = threadIdx.x;
        const float *query_head = query + head_id * head_dim;
        for (int i = tid; i < head_dim; i += blockDim.x)
        {
            s_query_head[i] = query_head[i];
        }
        __syncthreads();

        float *score_head = score_ptr + static_cast<size_t>(head_id) * seq_len;
        for (int t = tid; t <= pos; t += blockDim.x)
        {
            const float *key_head = key_cache + layer_offset + static_cast<size_t>(t) * kv_dim + kv_head_offset;
            float score = 0.0f;

            for (int i = 0; i < head_dim; ++i) {
                score += key_head[i] * s_query_head[i];
            }
            score /= sqrtf(float(head_dim));
            score_head[t] = score;
        }
        __syncthreads();
        softmax_kernel<blockSize>(score_head, pos + 1);
        __syncthreads();

        float *output_head = output + head_id * head_dim;
        for (int t = tid; t < head_dim; t += blockDim.x)
        {
            float sum = 0;
            for (int i = 0; i <= pos; ++i)
            {
                const float *value_head = value_cache + layer_offset + static_cast<size_t>(i) * kv_dim + kv_head_offset;
                sum += score_head[i] * value_head[t];
            }
            output_head[t] = sum;
        }
    }

    void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                       int32_t kv_dim, int32_t kv_mul, int32_t head_size, tensor::Tensor mha_out,
                       const tensor::Tensor &query_tensor, tensor::Tensor score_tensor,
                       const tensor::Tensor &key_cache_tensor, const tensor::Tensor &value_cache_tensor,
                       cudaStream_t stream)
    {
        size_t layer_offset = static_cast<size_t>(layer_index) * seq_len * kv_dim;
        const float *query = query_tensor.ptr<float>();
        float *score = score_tensor.ptr<float>();
        float *output = mha_out.ptr<float>();

        const float* key_cache = key_cache_tensor.ptr<float>();
        const float* val_cache = value_cache_tensor.ptr<float>();


        constexpr int blockSize = 256;
        multi_head_attention_kernel<blockSize><<<head_num, blockSize, head_size * sizeof(float), stream>>>(pos, seq_len, output,
                                                query, score, key_cache, val_cache,
                                                head_num / kv_mul, head_num, head_size, layer_offset);
        CHECK_EQ(cudaGetLastError(), cudaSuccess);
    }
}