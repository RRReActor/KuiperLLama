
#include "tensor/tensor.h"

namespace kernel {
    void add_kernel_cuda(tensor::Tensor output, const tensor::Tensor& input1, const tensor::Tensor& input2, void* stream = nullptr);
}