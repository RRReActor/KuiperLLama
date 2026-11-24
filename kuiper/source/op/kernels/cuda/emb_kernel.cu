#include "emb_kernel.cuh"

#include <cuda_runtime.h>

namespace kernel {
    namespace {
        __global__ void gather_embeddings(const int32_t* tokens, const float* weights,
                                          float* output, size_t dim, int32_t vocab_size) {
            const auto row = static_cast<size_t>(blockIdx.x);
            const auto token = tokens[row];
            if (token < 0 || token >= vocab_size) {
                return;
            }

            const auto* weight_row = weights + static_cast<size_t>(token) * dim;
            auto* output_row = output + row * dim;
            for (size_t column = threadIdx.x; column < dim; column += blockDim.x) {
                output_row[column] = weight_row[column];
            }
        }
    }

    base::Status emb_kernel_cuda(const tensor::Tensor& input, const tensor::Tensor& weight,
                                 tensor::Tensor output, int32_t vocab_size, void* stream) {
        const auto dim = static_cast<size_t>(weight.get_dim(1));
        const auto token_count = output.size() / dim;
        const auto token_bytes = token_count * sizeof(int32_t);
        const auto cuda_stream = static_cast<cudaStream_t>(stream);

        int32_t* device_tokens = nullptr;
        auto status = cudaMalloc(&device_tokens, token_bytes);
        if (status != cudaSuccess) {
            return base::error::InternalError(cudaGetErrorString(status));
        }

        status = cudaMemcpyAsync(device_tokens, input.ptr<int32_t>(), token_bytes,
                                 cudaMemcpyHostToDevice, cuda_stream);
        if (status == cudaSuccess) {
            constexpr int threads = 128;
            gather_embeddings<<<token_count, threads, 0, cuda_stream>>>(
                device_tokens, weight.ptr<float>(), output.ptr<float>(), dim, vocab_size);
            status = cudaGetLastError();
        }

        // cudaFree waits for work using this cudaMalloc allocation before releasing it.
        const auto free_status = cudaFree(device_tokens);
        if (status == cudaSuccess) {
            status = free_status;
        }
        if (status != cudaSuccess) {
            return base::error::InternalError(cudaGetErrorString(status));
        }
        return base::error::Success();
    }
}
