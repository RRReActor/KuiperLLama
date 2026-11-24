#pragma once
#include "tensor/tensor.h"
namespace kernel {
    typedef void (*RMSNormKernel)(tensor::Tensor output, const tensor::Tensor& intpu, 
            const tensor::Tensor& weight, void* stream);

    typedef void (*GEMVKernel)(tensor::Tensor output, const tensor::Tensor& input,
             const tensor::Tensor& weight, void* stream);
    
    typedef void (*GEMVKernelQuant)(tensor::Tensor output, const tensor::Tensor& input,
             const tensor::Tensor& weight, int32_t group_size, 
             const tensor::Tensor& scale, void* stream);

    typedef void (*AddKernel)(tensor::Tensor output, const tensor::Tensor& input1,
            const tensor::Tensor& input2, void* stream);

    typedef void (*ActivationKernel)(tensor::Tensor output, const tensor::Tensor& input1, const tensor::Tensor& input2, void* stream);

    typedef base::Status (*EmbKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                     tensor::Tensor output, int32_t vocab_size, void* stream);

    EmbKernel get_emb_kernel(base::DeviceType device_type);

    GEMVKernel get_gemv_kernel(base::DeviceType device_type);

    GEMVKernelQuant get_gemv_kernel_quant(base::DeviceType device_type);

    RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);

    AddKernel get_add_kernel(base::DeviceType device_type);

    ActivationKernel get_activation_kernel(base::DeviceType device_type);
   

}

