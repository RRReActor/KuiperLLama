#include "sampler/argmax_sampler.h"

#include <algorithm>
#include "op/kernels/cuda/argmax_kernel.cuh"

namespace sampler {
std::size_t ArgmaxSampler::sample(const float* logits, std::size_t size, void* stream) {
    if (size == 0) {
        return 0;
    }
    CHECK(logits != nullptr);
    if (device_type_ == base::DeviceType::kDeviceCPU) {
        return std::max_element(logits, logits + size) - logits;
    }
    CHECK(device_type_ == base::DeviceType::kDeviceCUDA);
    return kernel::argmax_kernel_cu(logits, size, stream);
}
}  // namespace sampler
