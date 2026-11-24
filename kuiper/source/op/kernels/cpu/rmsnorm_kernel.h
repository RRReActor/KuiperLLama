#pragma once
#include "tensor/tensor.h"
namespace kernel {
void rmsnorm_kernel_cpu(tensor::Tensor, const tensor::Tensor&, const tensor::Tensor&, void*);
void rmsnorm_kernel_cpu_dim(tensor::Tensor, const tensor::Tensor&, const tensor::Tensor&, void*);
}
