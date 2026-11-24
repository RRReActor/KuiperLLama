#pragma once


#include "tensor/tensor.h"

namespace kernel {
    void gemv_kernel_cuda(tensor::Tensor output, const tensor::Tensor& input,
             const tensor::Tensor& weight, void* stream = nullptr);

    void gemv_kernel_cuda_qint8(tensor::Tensor output, const tensor::Tensor& input,
             const tensor::Tensor& weight, int32_t group_size, 
             const tensor::Tensor& scale, void* stream = nullptr);
}