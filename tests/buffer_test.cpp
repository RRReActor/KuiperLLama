#include "base/buffer.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>

namespace {
class RecordingAllocator : public base::DeviceAllocator {
public:
    explicit RecordingAllocator(base::DeviceType device = base::DeviceType::kDeviceCPU)
        : DeviceAllocator(device) {}
    mutable int allocations = 0;
    mutable int releases = 0;
    mutable base::MemcpyKind last_kind = base::MemcpyKind::kMemcpyCPU2CPU;
    bool fail = false;
    void* allocate(size_t n) const override {
        ++allocations;
        return fail ? nullptr : std::malloc(n);
    }
    void release(void* ptr) const override { ++releases; std::free(ptr); }
    void memcpy(void* dst, const void* src, size_t n, base::MemcpyKind kind,
                void*, bool) const override {
        last_kind = kind;
        std::memcpy(dst, src, n);
    }
};
}

TEST(Buffer, OwnershipRepeatedAllocationAndSharedFromThis) {
    auto allocator = std::make_shared<RecordingAllocator>();
    {
        auto buffer = std::make_shared<base::Buffer>(16, allocator);
        ASSERT_NE(buffer->ptr(), nullptr);
        auto ptr = buffer->ptr();
        EXPECT_TRUE(buffer->allocate());
        EXPECT_EQ(buffer->ptr(), ptr);
        EXPECT_EQ(allocator->allocations, 1);
        EXPECT_EQ(buffer->get_shared_from_this(), buffer);
        EXPECT_EQ(buffer->device_type(), base::DeviceType::kDeviceCPU);
    }
    EXPECT_EQ(allocator->releases, 1);
}

TEST(Buffer, ExternalStorageIsNeverReleased) {
    auto allocator = std::make_shared<RecordingAllocator>();
    int value = 17;
    {
        base::Buffer buffer(sizeof(value), allocator, &value, true);
        EXPECT_TRUE(buffer.is_external());
        EXPECT_TRUE(buffer.allocate());
        EXPECT_TRUE(buffer.is_external());
        EXPECT_EQ(buffer.ptr(), &value);
    }
    EXPECT_EQ(allocator->allocations, 0);
    EXPECT_EQ(allocator->releases, 0);
    EXPECT_EQ(value, 17);
}

TEST(Buffer, AllocationFailureCanBeRetried) {
    auto allocator = std::make_shared<RecordingAllocator>();
    allocator->fail = true;
    {
        base::Buffer buffer(8, allocator);
        EXPECT_EQ(buffer.ptr(), nullptr);
        EXPECT_FALSE(buffer.allocate());
        allocator->fail = false;
        EXPECT_TRUE(buffer.allocate());
    }
    EXPECT_EQ(allocator->releases, 1);
    base::Buffer empty;
    EXPECT_FALSE(empty.allocate());
}

TEST(Buffer, BothCopyOverloadsRespectSmallerSize) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    int source[] = {11, 22, 33};
    int destination[] = {0, 0, 99};
    base::Buffer src(sizeof(source), allocator, source, true);
    base::Buffer dst(2 * sizeof(int), allocator, destination, true);
    dst.copy_from(src);
    EXPECT_EQ(destination[0], 11);
    EXPECT_EQ(destination[1], 22);
    EXPECT_EQ(destination[2], 99);
    source[0] = 44;
    dst.copy_from(&src);
    EXPECT_EQ(destination[0], 44);
}

TEST(Buffer, RoutesEveryDeviceCopyDirection) {
    using base::DeviceType;
    using base::MemcpyKind;
    const DeviceType devices[] = {DeviceType::kDeviceCPU, DeviceType::kDeviceCUDA};
    const MemcpyKind expected[2][2] = {
        {MemcpyKind::kMemcpyCPU2CPU, MemcpyKind::kMemcpyCPU2CUDA},
        {MemcpyKind::kMemcpyCUDA2CPU, MemcpyKind::kMemcpyCUDA2CUDA}};
    // Simulated devices test dispatch only; CUDA integration uses the real runtime.
    for (int src = 0; src < 2; ++src) {
        for (int dst = 0; dst < 2; ++dst) {
            auto allocator = std::make_shared<RecordingAllocator>(devices[dst]);
            int value = 7;
            base::Buffer source(sizeof(value), nullptr, &value, true);
            source.set_device_type(devices[src]);
            base::Buffer destination(sizeof(value), allocator);
            destination.copy_from(source);
            EXPECT_EQ(allocator->last_kind, expected[src][dst]);
            EXPECT_EQ(*static_cast<int*>(destination.ptr()), value);
        }
    }
}

TEST(BufferDeathTest, RejectsNullSourceAndDestination) {
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    base::Buffer valid(4, allocator);
    EXPECT_DEATH(valid.copy_from(static_cast<const base::Buffer*>(nullptr)), "other");
    base::Buffer empty;
    EXPECT_DEATH(valid.copy_from(empty), "other.ptr_");
    EXPECT_DEATH(empty.copy_from(valid), "allocator_");
}
