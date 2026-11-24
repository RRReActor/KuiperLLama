#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <vector>
#include "base.h"

namespace base {

enum class MemcpyKind {
    kMemcpyCPU2CPU = 0,
    kMemcpyCPU2CUDA = 1,
    kMemcpyCUDA2CPU = 2,
    kMemcpyCUDA2CUDA = 3
};

class DeviceAllocator {
    public:
        explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}

        virtual ~DeviceAllocator() = default;

        virtual DeviceType device_type() const {  return device_type_; }

        virtual void release(void* ptr) const = 0;

        virtual void* allocate(size_t byte_size) const = 0;

        virtual void memcpy(void* dst, const void* src, size_t byte_size, 
            MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU,
            void* stream = nullptr,
            bool need_sync = false
        ) const;

        virtual void memset_zero(void* ptr, size_t byte_size, void* stream = nullptr,
                                 bool need_sync = false) const;

    private:
        DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

class CPUDeviceAllocator : public DeviceAllocator {
    public:
        explicit CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {}

        void* allocate(size_t byte_size) const override;

        void release(void* ptr) const override;

};

struct CudaMemoryBuffer {
    void* data = nullptr;
    size_t byte_size = 0;
    bool busy = false;

    CudaMemoryBuffer() = default;

    CudaMemoryBuffer(void* data, size_t byte_size, bool busy)
        : data(data), byte_size(byte_size), busy(busy) {}
};

class CUDADeviceAllocator : public DeviceAllocator {
    public:
        explicit CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {}

        ~CUDADeviceAllocator() override;

        CUDADeviceAllocator(const CUDADeviceAllocator&) = delete;
        CUDADeviceAllocator& operator=(const CUDADeviceAllocator&) = delete;

        void* allocate(size_t byte_size) const override;

        void release(void* ptr) const override;
    
    private:
        mutable std::map<int, size_t> no_busy_cnt_;
        mutable std::map<int, size_t> big_no_busy_cnt_;
        mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;
        mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;

};

class CPUDeviceAllocatorFactory {
    public:
        static std::shared_ptr<CPUDeviceAllocator> get_instance() {
            static const auto instance = std::make_shared<CPUDeviceAllocator>();
            return instance;
        }
};


class CUDADeviceAllocatorFactory {
    public:
        static std::shared_ptr<CUDADeviceAllocator> get_instance() {
            static const auto instance = std::make_shared<CUDADeviceAllocator>();
            return instance;
        }
};

} // namespace base
