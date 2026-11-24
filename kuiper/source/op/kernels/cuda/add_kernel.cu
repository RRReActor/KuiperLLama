#include "add_kernel.cuh"

namespace kernel {

    __global__ void add_kernel_cuda_fp32(float* out, const float* in1, const float* in2, int32_t size) {
        int32_t gtid = threadIdx.x + blockDim.x * blockIdx.x;
        if(gtid >= size) return;
        out[gtid] = in1[gtid] + in2[gtid];
    }

    void add_kernel_cuda(tensor::Tensor output, const tensor::Tensor& input1, const tensor::Tensor& input2, void* stream) {
        CHECK(input1.is_empty() == false);
        CHECK(input2.is_empty() == false);
        CHECK(output.is_empty() == false);
        int32_t size = static_cast<int32_t>(input1.size());
        CHECK_EQ(size, static_cast<int32_t>(input2.size()));
        CHECK_EQ(size, static_cast<int32_t>(output.size()));
        int32_t threads = 256;
        int32_t blocks = (size + threads - 1) / threads;
        if (stream) {
            add_kernel_cuda_fp32<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
            output.ptr<float>(), input1.ptr<float>(), input2.ptr<float>(), size);
        } else {
            add_kernel_cuda_fp32<<<blocks, threads>>>(
            output.ptr<float>(), input1.ptr<float>(), input2.ptr<float>(), size);
        }
    }

} // namespace kernel