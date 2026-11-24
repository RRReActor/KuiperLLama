#pragma once
#include "tensor/tensor.h"
namespace kernel
{
  void rmsnorm_kernel_cuda(tensor::Tensor output, const tensor::Tensor &input,
                           const tensor::Tensor &weight, void *stream = nullptr);
} // namespace kernel