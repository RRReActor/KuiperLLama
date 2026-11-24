#include "argmax_kernel.cuh"

#include <cuda_runtime.h>
#include <thrust/extrema.h>
#include <thrust/system/cuda/execution_policy.h>

namespace kernel {
std::size_t argmax_kernel_cu(const float* logits, std::size_t size, void* stream) {
    const auto policy = thrust::cuda::par.on(static_cast<cudaStream_t>(stream));
    return thrust::max_element(policy, logits, logits + size) - logits;
}
}  // namespace kernel
