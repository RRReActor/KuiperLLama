#include "model/qwen3.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <fstream>
#include <cmath>
#include <algorithm>

namespace {
void check_generation(base::DeviceType device, bool quantized = false) {
    const char* weights = quantized ? QWEN3_TEST_INT8_WEIGHTS : QWEN3_TEST_WEIGHTS;
    if (!std::ifstream(weights)) GTEST_SKIP() << weights << " unavailable";
    model::Qwen3Model model(base::TokenizerType::kEncodeBpe, QWEN3_TEST_VOCAB, weights, quantized);
    auto status = model.init(device);
    ASSERT_TRUE(status) << status;
    const auto ids = model.encode("a  b");
    ASSERT_EQ(ids, (std::vector<int>{64, 220, 293}));
    const auto prompt = model.embedding(ids);
    auto pos = model.get_buffer(model::ModelBufferType::kInputPos);
    int next = -1;

    // Independent full-sequence evaluations, using dequantized weights for INT8 (group 64).
    const int expected_ids[] = {quantized ? 1479 : 21806, 16, 220, 272};
    const float fp32_logits[] = {8.728104f, 15.615835f, 18.547077f, 20.892271f};
    const float int8_logits[] = {9.472952f, 15.611256f, 18.622770f, 21.033533f};
    const float* expected_logits = quantized ? int8_logits : fp32_logits;
    for (int step = 0; step < 4; ++step) {
        pos.index<int32_t>(0) = step;
        auto input = step < 3 ? model.fill_input(pos, prompt, true)
                              : model.fill_input(pos, model.embedding({next}), false);
        ASSERT_TRUE(model.predict(input, pos, step < 2, next));
        EXPECT_EQ(next, step < 2 ? -1 : expected_ids[step]);
        auto logits = model.get_buffer(model::ModelBufferType::kLMOutput);
        if (device == base::DeviceType::kDeviceCUDA) logits.to_cpu();
        EXPECT_NEAR(logits.index<float>(expected_ids[step]), expected_logits[step], 0.01f);
    }
    pos.index<int32_t>(0) = model.get_buffer(model::ModelBufferType::kKeyCache).get_dim(1);
    auto input = model.fill_input(pos, prompt, false);
    EXPECT_FALSE(model.predict(input, pos, false, next));
}

// The final timestep lies past the first CUDA block-sized chunk of the score row.
void check_long_attention(base::DeviceType device) {
    constexpr int length = 259;
    constexpr int width = 2;
    auto cpu = base::CPUDeviceAllocatorFactory::get_instance();
    const auto fp32 = base::DataType::kDataTypeFp32;
    tensor::Tensor query(fp32, {4}, true, cpu);
    tensor::Tensor key(fp32, {1, length, width}, true, cpu);
    tensor::Tensor value(fp32, {1, length, width}, true, cpu);
    tensor::Tensor scores(fp32, {2, length}, true, cpu);
    tensor::Tensor output(fp32, {4}, true, cpu);
    std::fill_n(query.ptr<float>(), 4, 1.f);
    std::fill_n(key.ptr<float>(), key.size(), -1000.f);
    std::fill_n(value.ptr<float>(), value.size(), 0.f);
    key.index<float>((length - 1) * width) += 16.f;
    value.index<float>((length - 1) * width) = 1.f;
    if (device == base::DeviceType::kDeviceCUDA) {
        for (auto* tensor : {&query, &key, &value, &scores, &output}) tensor->to_cuda();
    }
    op::MultiHeadAttention layer(device, 0, 2, width, length, 2, width);
    layer.set_pos(length - 1);
    ASSERT_TRUE(layer.forward(query, scores, key, value, output));
    if (device == base::DeviceType::kDeviceCUDA) output.to_cpu();
    const double expected = 1.0 / (1.0 + (length - 1) * std::exp(-16.0 / std::sqrt(2.0)));
    for (int head = 0; head < 2; ++head) {
        EXPECT_NEAR(output.index<float>(head * width), expected, 1e-5);
        EXPECT_FLOAT_EQ(output.index<float>(head * width + 1), 0.f);
    }
}

TEST(Qwen3Attention, LongNegativeScoresCPU) {
    check_long_attention(base::DeviceType::kDeviceCPU);
}

TEST(Qwen3Attention, LongNegativeScoresCUDA) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP() << "CUDA unavailable";
    check_long_attention(base::DeviceType::kDeviceCUDA);
}

TEST(Qwen3Model, CpuPrefillAndDecode) {
    check_generation(base::DeviceType::kDeviceCPU);
}

TEST(Qwen3Model, CudaPrefillAndDecode) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP() << "CUDA unavailable";
    check_generation(base::DeviceType::kDeviceCUDA);
}

TEST(Qwen3Model, CpuInt8PrefillAndDecode) {
    check_generation(base::DeviceType::kDeviceCPU, true);
}

TEST(Qwen3Model, CudaInt8PrefillAndDecode) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP() << "CUDA unavailable";
    check_generation(base::DeviceType::kDeviceCUDA, true);
}
}  // namespace
