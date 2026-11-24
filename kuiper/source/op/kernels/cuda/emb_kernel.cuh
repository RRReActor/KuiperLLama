#pragma once

#include "tensor/tensor.h"

namespace kernel {
    base::Status emb_kernel_cuda(const tensor::Tensor& input, const tensor::Tensor& weight,
                                 tensor::Tensor output, int32_t vocab_size, void* stream);
}
