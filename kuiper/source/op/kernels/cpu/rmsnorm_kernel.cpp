#include "rmsnorm_kernel.h"
#include "../rmsnorm_validation.h"
#include <cmath>
namespace kernel {
static void normalize(tensor::Tensor output, const tensor::Tensor& input,
                      const tensor::Tensor& weight, bool rows) {
    validate_rmsnorm(output, input, weight, base::DeviceType::kDeviceCPU, rows);
    const size_t width = rows ? input.get_dim(input.dims_size() - 1) : input.size();
    for (size_t offset = 0; offset < input.size(); offset += width) {
        double sum = 0;
        for (size_t j = 0; j < width; ++j) {
            const double x = input.ptr<float>()[offset + j]; sum += x * x;
        }
        const double scale = 1.0 / std::sqrt(sum / width + 1e-6);
        for (size_t j = 0; j < width; ++j)
            output.ptr<float>()[offset + j] = input.ptr<float>()[offset + j] * scale * weight.ptr<float>()[j];
    }
}
void rmsnorm_kernel_cpu(tensor::Tensor o, const tensor::Tensor& i, const tensor::Tensor& w, void*) { normalize(o,i,w,true); }
void rmsnorm_kernel_cpu_dim(tensor::Tensor o, const tensor::Tensor& i, const tensor::Tensor& w, void*) { normalize(o,i,w,true); }
}
