#pragma once
#include "tensor/tensor.h"
#include <limits>
namespace kernel {
inline void validate_rmsnorm(const tensor::Tensor& output, const tensor::Tensor& input,
                             const tensor::Tensor& weight, base::DeviceType device, bool rows) {
    CHECK(!input.is_empty()); CHECK(!output.is_empty()); CHECK(!weight.is_empty());
    for (const auto* t : {&input, &output, &weight}) {
        CHECK(t->data_type() == base::DataType::kDataTypeFp32);
        CHECK(t->device_type() == device);
    }
    CHECK(output.dims() == input.dims());
    CHECK_LE(input.size(), static_cast<size_t>(std::numeric_limits<int32_t>::max()));
    const size_t width = rows ? input.get_dim(input.dims_size() - 1) : input.size();
    CHECK_GT(width, 0); CHECK_EQ(weight.size(), width);
}
}
