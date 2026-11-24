#include "base/buffer.h"
#include <glog/logging.h>

namespace base {
    Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator, 
        void* ptr, bool use_external):
        byte_size_(byte_size),
        allocator_(allocator),
        ptr_(ptr),
        use_external_(use_external) {
            if (allocator_) {
                device_type_ = allocator_->device_type();
            }
            // 当 ptr_ 为空且 allocator_ 不为空时， 表示需要从allocator 分配对应的资源
            if (!ptr_ && allocator_) {
                device_type_ = allocator_->device_type();
                use_external_ = false;
                ptr_ = allocator_->allocate(byte_size);
            }
        }

    Buffer::~Buffer() {
        if (!use_external_) {
            if(ptr_ && allocator_) {
                allocator_->release(ptr_);
                ptr_ = nullptr;
            }
        }
    }

    void* Buffer::ptr() {return ptr_;}

    const void* Buffer::ptr() const {return ptr_;}

    size_t Buffer::byte_size() const { return byte_size_;}

    bool Buffer::allocate() {
        if (ptr_) {
            return true;
        }
        if (allocator_ && byte_size_ ) {
            ptr_ = allocator_->allocate(byte_size_);
            if (ptr_) {
                use_external_ = false;
                device_type_ = allocator_->device_type();
            }
            return ptr_ != nullptr;
        } else {
            return false;
        }
    }

    std::shared_ptr<DeviceAllocator> Buffer::allocator() const { return allocator_;}

    DeviceType Buffer::device_type() const { return device_type_;}

    void Buffer::set_device_type(DeviceType device_type) { device_type_ = device_type;}

    bool Buffer::is_external() const { return use_external_;}

    std::shared_ptr<Buffer> Buffer::get_shared_from_this() {   return shared_from_this(); }


    void Buffer::copy_from(const Buffer& other) const {
        CHECK(allocator_ != nullptr);
        CHECK(ptr_ != nullptr);
        CHECK(other.ptr_ != nullptr);

        size_t byte_size = byte_size_ < other.byte_size_ ? byte_size_ : other.byte_size_;
        const DeviceType& buffer_device = other.device_type_;
        const DeviceType& current_device = device_type_;
        CHECK(buffer_device != DeviceType::kDeviceUnknown &&
            current_device != DeviceType::kDeviceUnknown);
        
        if(buffer_device == DeviceType::kDeviceCPU && current_device == DeviceType::kDeviceCPU) {
            return allocator_->memcpy(ptr_, other.ptr_, byte_size, MemcpyKind::kMemcpyCPU2CPU);
        } else if (buffer_device == DeviceType::kDeviceCUDA && current_device == DeviceType::kDeviceCPU) {
            return allocator_->memcpy(ptr_, other.ptr_, byte_size, MemcpyKind::kMemcpyCUDA2CPU);
        } else if (buffer_device == DeviceType::kDeviceCPU && current_device == DeviceType::kDeviceCUDA) {
            return allocator_->memcpy(ptr_, other.ptr_, byte_size, MemcpyKind::kMemcpyCPU2CUDA);
        } else {
            return allocator_->memcpy(ptr_, other.ptr_, byte_size, MemcpyKind::kMemcpyCUDA2CUDA);
        }
    }

    void Buffer::copy_from(const Buffer* other) const {
        CHECK(other != nullptr);
        copy_from(*other);
    }

}
