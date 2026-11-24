#pragma once

#include <driver_types.h>
#include <glog/logging.h>
#include <memory>
#include <vector>
#include "base/base.h"
#include "base/buffer.h"

namespace tensor {
class Tensor {
    public:
        explicit Tensor() = default;

        explicit Tensor(base::DataType data_type, std::vector<int32_t> dims, bool need_alloc = false,
            std::shared_ptr<base::DeviceAllocator> allocator = nullptr, void* ptr = nullptr);
        
        void to_cpu();

        void to_cuda(cudaStream_t stream = nullptr);

        bool is_empty() const;

        void init_buffer(std::shared_ptr<base::DeviceAllocator> allocator, base::DataType data_type,
            bool need_alloc, void* ptr);

        template <typename T>
        T* ptr();

        template <typename T>
        const T* ptr() const;

        void reshape(const std::vector<int32_t>& dims);

        std::shared_ptr<base::Buffer> get_buffer() const;

        size_t size() const;

        size_t byte_size() const;

        int32_t dims_size() const;

        base::DataType data_type() const;

        int32_t get_dim(int32_t idx) const;

        const std::vector<int32_t>& dims() const;

        std::vector<size_t> strides() const;

        bool assign(std::shared_ptr<base::Buffer> buffer);

        void reset(base::DataType data_type, const std::vector<int32_t>& dims);

        void set_device_type(base::DeviceType device_type) const;

        base::DeviceType device_type() const;

        bool allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc = false);

        template <typename T>
        T* ptr(int32_t index);

        template <typename T>
        const T* ptr(int32_t index) const;
        
        template <typename T>
        T& index(int32_t offset);

        template <typename T>
        const T& index(int32_t offset) const;

        tensor::Tensor clone() const;

        private:
            size_t size_ = 0;
            std::vector<int32_t> dims_;
            std::shared_ptr<base::Buffer> buffer_;
            base::DataType data_type_ = base::DataType::kDataTypeUnknown;
};

template <typename T>
T& Tensor::index(int32_t offset) {
    CHECK(buffer_ && buffer_->ptr());
    CHECK_GE(offset, 0);
    CHECK_LT(offset, size_);
    T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
    return val;
}

template <typename T>
const T& Tensor::index(int32_t offset) const {
    CHECK(buffer_ && buffer_->ptr());
    CHECK_GE(offset, 0);
    CHECK_LT(static_cast<size_t>(offset), size_);
    return reinterpret_cast<const T*>(buffer_->ptr())[offset];
}

template <typename T>
const T* Tensor::ptr() const {
    if(!buffer_) return nullptr;
    return const_cast<const T*>(reinterpret_cast<T*>(buffer_->ptr()));
}

template <typename T>
T* Tensor::ptr() {
    if(!buffer_) return nullptr;
    return reinterpret_cast<T*>(buffer_->ptr());
}

template <typename T>
T* Tensor::ptr(int32_t index) {
    CHECK(buffer_ && buffer_->ptr()) 
        << "The data area buffer of this tensor is empty or it points to a null pointer.";
    CHECK_GE(index, 0);
    CHECK_LT(static_cast<size_t>(index), size_);
    return reinterpret_cast<T*>(buffer_->ptr()) + index;
}

template <typename T>
const T* Tensor::ptr(int32_t index) const {
    CHECK(buffer_ && buffer_->ptr()) 
        << "The data area buffer of this tensor is empty or it points to a null pointer.";
    CHECK_GE(index, 0);
    CHECK_LT(static_cast<size_t>(index), size_);
    return reinterpret_cast<const T*>(buffer_->ptr()) + index;
}

} // namespace tensor