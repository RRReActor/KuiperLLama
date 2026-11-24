#include "tensor/tensor.h"
#include "tensor/tensor.h"
#include <gtest/gtest.h>

namespace {
auto cpu = base::CPUDeviceAllocatorFactory::get_instance();
constexpr auto fp32 = base::DataType::kDataTypeFp32;
}

TEST(Tensor, ShapeStridesAndTypedAccess) {
    tensor::Tensor tensor(fp32, {2, 3, 4}, true, cpu);
    ASSERT_FALSE(tensor.is_empty());
    EXPECT_EQ(tensor.size(), 24u);
    EXPECT_EQ(tensor.byte_size(), 24 * sizeof(float));
    EXPECT_EQ(tensor.dims_size(), 3);
    EXPECT_EQ(tensor.get_dim(1), 3);
    EXPECT_EQ(tensor.strides(), (std::vector<size_t>{12, 4, 1}));
    tensor.index<float>(5) = 3.5f;
    const auto& view = tensor;
    EXPECT_FLOAT_EQ(view.index<float>(5), 3.5f);
    EXPECT_EQ(view.ptr<float>(5), view.ptr<float>() + 5);
    EXPECT_EQ(tensor.ptr<float>(5), tensor.ptr<float>() + 5);
    EXPECT_EQ(tensor.device_type(), base::DeviceType::kDeviceCPU);
    EXPECT_EQ(tensor.data_type(), fp32);
}

TEST(Tensor, ReshapeUpdatesMetadataAndRejectsGrowth) {
    tensor::Tensor tensor(fp32, {2, 3}, true, cpu);
    auto buffer = tensor.get_buffer();
    tensor.reshape({3, 2});
    EXPECT_EQ(tensor.dims(), (std::vector<int32_t>{3, 2}));
    tensor.reshape({3});
    EXPECT_EQ(tensor.size(), 3u);
    EXPECT_EQ(tensor.get_buffer(), buffer);
    tensor.reshape({7});
    EXPECT_EQ(tensor.size(), 3u);
    EXPECT_EQ(tensor.dims(), (std::vector<int32_t>{3}));
}

TEST(Tensor, CloneOwnsIndependentStorage) {
    tensor::Tensor tensor(fp32, {2}, true, cpu);
    tensor.index<float>(0) = 12;
    tensor.index<float>(1) = 23;
    auto clone = tensor.clone();
    EXPECT_NE(clone.get_buffer(), tensor.get_buffer());
    EXPECT_FLOAT_EQ(clone.index<float>(1), 23);
    clone.index<float>(0) = 99;
    EXPECT_FLOAT_EQ(tensor.index<float>(0), 12);
}

TEST(Tensor, ExternalStorageAndAssignCapacity) {
    float data[] = {1, 2, 3};
    tensor::Tensor tensor(fp32, {3}, false, nullptr, data);
    EXPECT_EQ(tensor.ptr<float>(), data);
    EXPECT_TRUE(tensor.get_buffer()->is_external());
    tensor.set_device_type(base::DeviceType::kDeviceCPU);
    EXPECT_FALSE(tensor.assign(nullptr));
    EXPECT_FALSE(tensor.assign(std::make_shared<base::Buffer>(sizeof(float), cpu)));
    EXPECT_EQ(tensor.ptr<float>(), data);
    auto buffer = std::make_shared<base::Buffer>(sizeof(data), cpu);
    EXPECT_TRUE(tensor.assign(buffer));
    EXPECT_EQ(tensor.get_buffer(), buffer);
}

TEST(Tensor, ResetAllocationAndEmptyState) {
    tensor::Tensor tensor;
    EXPECT_TRUE(tensor.is_empty());
    EXPECT_EQ(tensor.ptr<float>(), nullptr);
    EXPECT_TRUE(tensor.clone().is_empty());
    EXPECT_TRUE(tensor.strides().empty());
    EXPECT_FALSE(tensor.allocate(cpu));
    tensor.reset(fp32, {2, 2});
    EXPECT_TRUE(tensor.is_empty());
    EXPECT_FALSE(tensor.allocate(nullptr));
    EXPECT_TRUE(tensor.allocate(cpu));
    auto original = tensor.get_buffer();
    EXPECT_TRUE(tensor.allocate(cpu));
    EXPECT_EQ(tensor.get_buffer(), original);
    EXPECT_TRUE(tensor.allocate(cpu, true));
    EXPECT_NE(tensor.get_buffer(), original);
    tensor.to_cpu();
    EXPECT_EQ(tensor.device_type(), base::DeviceType::kDeviceCPU);
    tensor.reset(base::DataType::kDataTypeInt8, {5});
    EXPECT_EQ(tensor.byte_size(), 5u);
    EXPECT_TRUE(tensor.is_empty());
}

TEST(TensorDeathTest, RejectsInvalidIndices) {
    tensor::Tensor tensor(fp32, {2}, true, cpu);
    EXPECT_DEATH(tensor.index<float>(-1), "offset");
    EXPECT_DEATH(tensor.index<float>(2), "offset");
    EXPECT_DEATH(tensor.get_dim(1), "idx");
}
