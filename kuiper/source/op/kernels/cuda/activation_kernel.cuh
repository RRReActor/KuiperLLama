#include "tensor/tensor.h"

namespace kernel {
    void activation_kernel_cuda(tensor::Tensor output, const tensor::Tensor& input1, const tensor::Tensor& input2, void* stream);
}