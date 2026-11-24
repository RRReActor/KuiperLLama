#include "tensor/tensor.h"
#include "cuda_runtime.h"
#include <glog/logging.h>
#include <numeric>
#include <functional>
#include <limits>

namespace tensor {
    template <typename T, typename Tp>
    static size_t reduce_dimension(T begin, T end, Tp init) {
        if (begin >= end) return 0;
        for (; begin != end; ++begin) {
            CHECK_GE(*begin, 0);
            if (*begin == 0) { init = 0; continue; }
            CHECK_LE(static_cast<size_t>(*begin), std::numeric_limits<size_t>::max() / (init ? init : 1));
            init *= static_cast<size_t>(*begin);
        }
        return init;
    }

    static size_t data_type_size(base::DataType data_type) {
        switch (data_type) {
            case base::DataType::kDataTypeFp32:
                return sizeof(float);
            case base::DataType::kDataTypeInt32:
                return sizeof(int32_t);
            case base::DataType::kDataTypeInt8:
                return 1;
            default:
                LOG(FATAL) << "Unknown data type size for " << int(data_type);
                return 0;
        }
    }

    Tensor::Tensor(base::DataType data_type, std::vector<int32_t> dims, bool need_alloc,
        std::shared_ptr<base::DeviceAllocator> alloc, void* ptr) :
        dims_(std::move(dims)), data_type_(data_type) {
            size_ = reduce_dimension(dims_.begin(), dims_.end(), size_t{1});
            if (need_alloc && alloc) allocate(alloc);
            else {
                init_buffer(alloc, data_type_, need_alloc, ptr);
            }
        }


    void Tensor::to_cuda(cudaStream_t stream) {
        CHECK_NE(buffer_, nullptr);
        const base::DeviceType device_type = this->device_type();
        if (device_type == base::DeviceType::kDeviceUnknown) 
            LOG(ERROR) << "The device type of the tensor is unknown.";
        else  if (device_type == base::DeviceType::kDeviceCPU) {
            size_t byte_size = this->byte_size();
            auto cuda_alloc = base::CUDADeviceAllocatorFactory::get_instance();
            auto cuda_buffer = std::make_shared<base::Buffer>(byte_size, cuda_alloc);
            cuda_alloc->memcpy(cuda_buffer->ptr(), buffer_->ptr(), byte_size, 
                base::MemcpyKind::kMemcpyCPU2CUDA, stream, true);
            this->buffer_ = cuda_buffer;
        } else {
            LOG(INFO) << "The device type of the tensor is already cuda.";
        }
    }

    void Tensor::to_cpu() {
        CHECK_NE(buffer_, nullptr);
        const base::DeviceType device_type = this->device_type();
        if (device_type == base::DeviceType::kDeviceUnknown) 
            LOG(ERROR) << "The device type of the tensor is unknown.";
        else if (device_type == base::DeviceType::kDeviceCUDA) {
            size_t byte_size = this->byte_size();
            auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
            auto cpu_buffer = std::make_shared<base::Buffer>(byte_size, cpu_alloc);
            cpu_alloc->memcpy(cpu_buffer->ptr(), buffer_->ptr(), byte_size, 
                base::MemcpyKind::kMemcpyCUDA2CPU);
            this->buffer_ = cpu_buffer;
        } else {
            LOG(INFO) << "The device type of the tensor is already cpu.";
        }
    }

    size_t Tensor::size() const { return this->size_; }

    int32_t Tensor::get_dim(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, this->dims_.size());
    return this->dims_.at(idx);
    }

    base::DeviceType Tensor::device_type() const {
    if (!buffer_) {
        return base::DeviceType::kDeviceUnknown;
    }
    return buffer_->device_type();
    }

    bool Tensor::assign(std::shared_ptr<base::Buffer> buffer) {
    if (!buffer) {
        LOG(ERROR) << "The buffer parameter in the assign function is null pointer!";
        return false;
    }
    if (buffer_) {
        if (buffer_->device_type() != buffer->device_type()) {
        LOG(ERROR) << "The device type of the new buffer is different from the original one.";
        }
    }

    if (this->byte_size() > buffer->byte_size()) {
        LOG(ERROR) << "The size of buffer is too small for the tensor!";
        return false;
    }
    buffer_ = buffer;
    return true;
    }

    bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc) {
        if (!allocator) {
            LOG(ERROR) << "The allocator parameter in the allocate function is null "
                        "pointer!";
            return false;
        }

        size_t byte_size = this->byte_size();
        if (!byte_size) {
            LOG(ERROR) << "The byte_size parameter in the allocate function is equal to zero!";
            return false;
        }

        if (buffer_ && byte_size <= buffer_->byte_size()) {
            if (!need_realloc) {
            return true;
            }
        }

        auto new_buffer = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
        if (!new_buffer->ptr()) {
            LOG(ERROR) << "The memory allocated is a null pointer!";
            return false;
        }
        buffer_ = std::move(new_buffer);
        return true;
    }

    const std::vector<int32_t>& Tensor::dims() const { return this->dims_; }

    void Tensor::set_device_type(base::DeviceType device_type) const {
        if (buffer_) { buffer_->set_device_type(device_type);   }
    }

    void Tensor::reset(base::DataType data_type, const std::vector<int32_t>& dims) {
        this->data_type_ = data_type;
        this->dims_ = dims;
        this->size_ = reduce_dimension(dims.begin(), dims.end(), size_t{1});
        this->buffer_ = nullptr;
    }

    int32_t Tensor::dims_size() const { return static_cast<int32_t>(dims_.size()); }

    base::DataType Tensor::data_type() const { return data_type_; }

    void Tensor::reshape(const std::vector<int32_t>& dims) {
        size_t size = reduce_dimension(dims.begin(), dims.end(), size_t{1});
        if (!buffer_) {
            this->dims_ = dims;
            this->size_ = size;
            return;
        }

        if (size > size_) {
            LOG(ERROR) << "the size is not equal to before";
            return;
        }
        dims_ = dims;
        size_ = size;
    }

    std::shared_ptr<base::Buffer> Tensor::get_buffer() const { return buffer_; }

    Tensor Tensor::clone() const {
        Tensor new_tensor = *this;
        if (is_empty()) return new_tensor;
        size_t byte_size = this->byte_size();

        auto allocator = buffer_->allocator();
        if (!allocator) {
            CHECK(device_type() != base::DeviceType::kDeviceUnknown);
            if (device_type() == base::DeviceType::kDeviceCPU)
                allocator = base::CPUDeviceAllocatorFactory::get_instance();
            else
                allocator = base::CUDADeviceAllocatorFactory::get_instance();
        }
        new_tensor.buffer_ = std::make_shared<base::Buffer>(byte_size, allocator);
        new_tensor.buffer_->copy_from(buffer_.get());
        return new_tensor;
    }

    size_t Tensor::byte_size() const {
        const size_t element_size = DataTypeSize(data_type_);
        if (element_size == 0) {
            CHECK_EQ(size_, 0);
            return 0;
        }
        CHECK_LE(size_, std::numeric_limits<size_t>::max() / element_size);
        return size_ * element_size;
    }

    std::vector<size_t> Tensor::strides() const {
    std::vector<size_t> strides;
    if (!dims_.empty()) {
        for (size_t i = 0; i + 1 < dims_.size(); ++i) {
        size_t stride = reduce_dimension(dims_.begin() + i + 1, dims_.end(), size_t{1});
        strides.push_back(stride);
        }
        strides.push_back(1);
    }
    return strides;
    }

    bool Tensor::is_empty() const {
    return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
    }

    void Tensor::init_buffer(std::shared_ptr<base::DeviceAllocator> allocator, base::DataType data_type,
                            bool need_alloc, void* ptr) {
        if (!allocator && !need_alloc) {
            std::shared_ptr<base::Buffer> buffer =
                std::make_shared<base::Buffer>(data_type_size(data_type) * size_, nullptr, ptr, true);
            this->buffer_ = buffer;
        } else {
            allocate(allocator, true);
        }
    }

}