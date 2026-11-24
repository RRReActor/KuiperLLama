#include "base/base.h"
#include "base/alloc.h"
#include <gtest/gtest.h>

TEST(Base, DataTypeSizes) {
    EXPECT_EQ(base::DataTypeSize(base::DataType::kDataTypeFp32), sizeof(float));
    EXPECT_EQ(base::DataTypeSize(base::DataType::kDataTypeInt32), sizeof(int32_t));
    EXPECT_EQ(base::DataTypeSize(base::DataType::kDataTypeInt8), 1u);
    EXPECT_EQ(base::DataTypeSize(base::DataType::kDataTypeUnknown), 0u);
}

TEST(Base, StatusValueAndMessage) {
    base::Status status;
    EXPECT_TRUE(static_cast<bool>(status));
    status = base::kInvalidArgument;
    status.set_err_msg("bad dimensions");
    EXPECT_FALSE(static_cast<bool>(status));
    EXPECT_EQ(status.get_err_code(), base::kInvalidArgument);
    EXPECT_EQ(status.get_err_msg(), "bad dimensions");
    EXPECT_TRUE(status == base::kInvalidArgument);
    EXPECT_TRUE(status != base::kSuccess);
    auto copy = status;
    EXPECT_EQ(static_cast<int>(copy), base::kInvalidArgument);
    EXPECT_EQ(copy.get_err_msg(), status.get_err_msg());
}

TEST(CPUAllocator, AllocateCopyAndZero) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    EXPECT_EQ(allocator, base::CPUDeviceAllocatorFactory::get_instance());
    EXPECT_EQ(allocator->allocate(0), nullptr);
    allocator->release(nullptr);
    int source[] = {1, 2, 3, 4};
    auto data = static_cast<int*>(allocator->allocate(sizeof(source)));
    ASSERT_NE(data, nullptr);
    allocator->memcpy(data, source, sizeof(source));
    for (int i = 0; i < 4; ++i) EXPECT_EQ(data[i], source[i]);
    allocator->memset_zero(data, sizeof(source));
    for (int i = 0; i < 4; ++i) EXPECT_EQ(data[i], 0);
    allocator->release(data);
    allocator->memcpy(nullptr, nullptr, 0);
    allocator->memset_zero(nullptr, 0);
}
