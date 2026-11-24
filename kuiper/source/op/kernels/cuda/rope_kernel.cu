#include "rope_kernel.cuh"

#include <algorithm>
#include <cuda_runtime.h>

namespace kernel {
namespace {
    constexpr int kBlockSize = 128;
#if defined(LLAMA3_SUPPORT)
    constexpr float kRopeTheta = 500000.0f;
#elif defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    constexpr float kRopeTheta = 1000000.0f;
#else
    constexpr float kRopeTheta = 10000.0f;
#endif

    __device__ void rotate_pair(float* values, int first, int second, float sine, float cosine) {
        const float x = values[first];
        const float y = values[second];
        values[first] = x * cosine - y * sine;
        values[second] = x * sine + y * cosine;
    }

    __global__ void rotate_qk(int dim, int kv_dim, int head_size, size_t cache_offset,
                              float* query, float* key, const float* sin_cache,
                              const float* cos_cache) {
        const int pair = blockIdx.x * blockDim.x + threadIdx.x;
        if (pair >= dim / 2) {
            return;
        }

        const int pairs_per_head = head_size / 2;
        const int head_offset = pair / pairs_per_head * head_size;
        const int head_pair = pair % pairs_per_head;
        const int column = 2 * head_pair;
#if defined(LLAMA3_SUPPORT) || defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
        const int first = head_offset + head_pair;
        const int second = first + pairs_per_head;
#else
        const int first = head_offset + column;
        const int second = first + 1;
#endif
        const float sine = sin_cache[cache_offset + column];
        const float cosine = cos_cache[cache_offset + column];
        rotate_pair(query, first, second, sine, cosine);
        if (head_offset < kv_dim) {
            rotate_pair(key, first, second, sine, cosine);
        }
    }

    __global__ void fill_sin_cos_cache(int head_size, int max_seq_len, float* sin_cache,
                                       float* cos_cache, float theta) {
        const int column = blockIdx.x * blockDim.x + threadIdx.x;
        if (column >= head_size) {
            return;
        }
        const float frequency = 1.0f / powf(theta, static_cast<float>(column) / head_size);
        for (size_t pos = blockIdx.y; pos < static_cast<size_t>(max_seq_len); pos += gridDim.y) {
            const size_t offset = pos * head_size + column;
            sincosf(static_cast<float>(pos) * frequency, sin_cache + offset, cos_cache + offset);
        }
    }

    void check_cuda_tensor(const tensor::Tensor& tensor, size_t min_size) {
        CHECK(!tensor.is_empty());
        CHECK(tensor.device_type() == base::DeviceType::kDeviceCUDA);
        CHECK(tensor.data_type() == base::DataType::kDataTypeFp32);
        CHECK_GE(tensor.size(), min_size);
    }
}  // namespace

    void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size,
                        const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                        const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
                        const tensor::Tensor& cos_cache, void* stream) {
        CHECK_GT(head_size, 0);
        CHECK_EQ(head_size % 2, 0);
        CHECK_GT(kv_dim, 0);
        CHECK_GE(dim, kv_dim);
        CHECK_EQ(dim % head_size, 0);
        CHECK_EQ(kv_dim % head_size, 0);
        check_cuda_tensor(input_q, dim);
        check_cuda_tensor(input_k, kv_dim);
        CHECK(input_pos.device_type() == base::DeviceType::kDeviceCPU);
        CHECK(input_pos.data_type() == base::DataType::kDataTypeInt32);
        CHECK_EQ(input_pos.size(), 1);
        const int32_t pos = *input_pos.ptr<int32_t>(0);
        CHECK_GE(pos, 0);

        check_cuda_tensor(sin_cache, head_size);
        check_cuda_tensor(cos_cache, head_size);
        CHECK_LT(static_cast<size_t>(pos), sin_cache.size() / head_size);
        CHECK_LT(static_cast<size_t>(pos), cos_cache.size() / head_size);
        const size_t cache_offset = static_cast<size_t>(pos) * head_size;

        const int blocks = (dim / 2 + kBlockSize - 1) / kBlockSize;
        rotate_qk<<<blocks, kBlockSize, 0, static_cast<cudaStream_t>(stream)>>>(
            dim, kv_dim, head_size, cache_offset, const_cast<float*>(input_q.ptr<float>()),
            const_cast<float*>(input_k.ptr<float>()), sin_cache.ptr<float>(), cos_cache.ptr<float>());
        const auto status = cudaGetLastError();
        CHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    }

    void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
                               const tensor::Tensor& cos_cache, cudaStream_t stream) {
        sin_cos_cache_calc_cu(head_size, max_seq_len, sin_cache, cos_cache, stream, kRopeTheta);
    }

    void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
                               const tensor::Tensor& cos_cache, cudaStream_t stream, float theta) {
        CHECK_GT(head_size, 0);
        CHECK_GT(max_seq_len, 0);
        const size_t cache_size = static_cast<size_t>(head_size) * max_seq_len;
        check_cuda_tensor(sin_cache, cache_size);
        check_cuda_tensor(cos_cache, cache_size);

        // Reuse each column's frequency across positions while distributing rows across blocks.
        const dim3 grid((head_size - 1) / kBlockSize + 1, std::min(max_seq_len, 128));
        fill_sin_cos_cache<<<grid, kBlockSize, 0, stream>>>(
            head_size, max_seq_len, const_cast<float*>(sin_cache.ptr<float>()),
            const_cast<float*>(cos_cache.ptr<float>()), theta);
        const auto status = cudaGetLastError();
        CHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    }
}  // namespace kernel
