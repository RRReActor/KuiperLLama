#include "base/base.h"
#include "cpu/emb_kernel.h"
#include "cpu/rmsnorm_kernel.h"
#include "cuda/emb_kernel.cuh"
#include "cuda/rmsnorm_kernel.cuh"
#include "cuda/gemv_kernel.cuh"
#include "cuda/add_kernel.cuh"
#include "cuda/activation_kernel.cuh"
#include "kernels_interface.h"

namespace kernel {
    EmbKernel get_emb_kernel(base::DeviceType device_type) {
        if (device_type == base::DeviceType::kDeviceCUDA) {
            return emb_kernel_cuda;
        }
        if (device_type == base::DeviceType::kDeviceCPU) {
            return emb_kernel_cpu;
        }
        LOG(FATAL) << "Unknown device type for an embedding kernel.";
        return nullptr;
    }

    RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type) {
        if (device_type == base::DeviceType::kDeviceCUDA) {
            return rmsnorm_kernel_cuda;
        } else if (device_type == base::DeviceType::kDeviceCPU) {
            return rmsnorm_kernel_cpu;
        } else {
            LOG(FATAL) << "Unknown device type for get a rmsnorm kernel.";
            return nullptr;
        }
    }

    GEMVKernel get_gemv_kernel(base::DeviceType device_type) {
        if (device_type == base::DeviceType::kDeviceCUDA) {
            return gemv_kernel_cuda;
        } else {
            LOG(FATAL) << "Unknown device type for get a GEMV kernel.";
            return nullptr;
        }
    }

    GEMVKernelQuant get_gemv_kernel_quant(base::DeviceType device_type) {
        if (device_type == base::DeviceType::kDeviceCUDA) {
            return gemv_kernel_cuda_qint8;
        } else {
            LOG(FATAL) << "Unknown device type for get a GEMV_Quant kernel.";
            return nullptr;
        }
    }

    AddKernel get_add_kernel(base::DeviceType device_type) {
        if (device_type == base::DeviceType::kDeviceCUDA) {
            return add_kernel_cuda;
        } else {
            LOG(FATAL) << "Unknown device type for get an Add kernel.";
            return nullptr;
        }
    }

    ActivationKernel get_activation_kernel(base::DeviceType device_type) {
        if (device_type == base::DeviceType::kDeviceCUDA) {
            return activation_kernel_cuda;
        } else {
            LOG(FATAL) << "Unknown device type for get an Activation kernel.";
            return nullptr;
        }
    }


}
