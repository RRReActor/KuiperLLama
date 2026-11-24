#include "emb_kernel.h"

#include <algorithm>

namespace kernel {
    base::Status emb_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                                tensor::Tensor output, int32_t vocab_size, void*) {
        const auto* tokens = input.ptr<int32_t>();
        const auto* weights = weight.ptr<float>();
        auto* embeddings = output.ptr<float>();
        const auto dim = static_cast<size_t>(weight.get_dim(1));
        const auto token_count = output.size() / dim;

        for (size_t i = 0; i < token_count; ++i) {
            const auto token = tokens[i];
            if (token < 0 || token >= vocab_size) {
                return base::error::InvalidArgument("Embedding token ID is out of range");
            }
            std::copy_n(weights + static_cast<size_t>(token) * dim, dim, embeddings + i * dim);
        }
        return base::error::Success();
    }
}
