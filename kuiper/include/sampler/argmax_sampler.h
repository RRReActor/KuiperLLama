#pragma once

#include "sampler/sampler.h"

namespace sampler {
class ArgmaxSampler : public Sampler {
public:
    explicit ArgmaxSampler(base::DeviceType device_type) : Sampler(device_type) {}

    // Return the first index with the largest logit, or zero for an empty range.
    std::size_t sample(const float* logits, std::size_t size, void* stream = nullptr) override;
};
}  // namespace sampler
