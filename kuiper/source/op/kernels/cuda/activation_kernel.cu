#include "activation_kernel.cuh"

namespace kernel {
    __global__ void activation_kernel_cuda_fp32(float* output, const float* input1, const float* input2, int size) {
        int gtid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gtid >= size)
            return;
        float v1 = input1[gtid];
        float v2 = input2[gtid];
        float value = 1.0f / (1.0f + expf(-v1));
        v1 = v1 * value;
        output[gtid] = v1 * v2;
    }

    void activation_kernel_cuda(tensor::Tensor output, const tensor::Tensor& input1, const tensor::Tensor& input2, void* stream) {
        CHECK(input1.is_empty() == false);
        CHECK(input1.device_type() == base::DeviceType::kDeviceCUDA);

        CHECK(input2.is_empty() == false);
        CHECK(input2.device_type() == base::DeviceType::kDeviceCUDA);

        CHECK(output.is_empty() == false);
        CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

        const int size = static_cast<int32_t>(input1.size());
        const int block_size = 128;
        const int grid_size = (size - 1) / block_size + 1;
        activation_kernel_cuda_fp32<<<grid_size, block_size, 0, static_cast<cudaStream_t>(stream)>>>(
            output.ptr<float>(), input1.ptr<float>(), input2.ptr<float>(), size);
    }
}  // namespace kernel
