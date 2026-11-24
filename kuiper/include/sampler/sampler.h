#pragma once

#include <cstddef>
#include "base/base.h"

namespace sampler {
class Sampler {
public:
    explicit Sampler(base::DeviceType device_type) : device_type_(device_type) {}
    virtual ~Sampler() = default;

    virtual std::size_t sample(const float* logits, std::size_t size, void* stream = nullptr) = 0;

protected:
    base::DeviceType device_type_;
};
}  // namespace sampler
