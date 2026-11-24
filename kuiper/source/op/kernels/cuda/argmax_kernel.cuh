#pragma once

#include <cstddef>

namespace kernel {
std::size_t argmax_kernel_cu(const float* logits, std::size_t size, void* stream);
}  // namespace kernel
