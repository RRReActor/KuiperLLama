#include "base/alloc.h"
#include <cuda_runtime_api.h>

namespace base {

void DeviceAllocator::memcpy(void* dst, const void* src, size_t byte_size,
                             MemcpyKind memcpy_kind, void* stream, bool need_sync) const {
    if (byte_size == 0) return;
    CHECK(dst != nullptr);
    CHECK(src != nullptr);
    if (memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) {
        std::memcpy(dst, src, byte_size);
        return;
    }

    cudaMemcpyKind kind = cudaMemcpyDefault;
    switch (memcpy_kind) {
        case MemcpyKind::kMemcpyCPU2CUDA: kind = cudaMemcpyHostToDevice; break;
        case MemcpyKind::kMemcpyCUDA2CPU: kind = cudaMemcpyDeviceToHost; break;
        case MemcpyKind::kMemcpyCUDA2CUDA: kind = cudaMemcpyDeviceToDevice; break;
        default: LOG(FATAL) << "Unknown memory copy kind"; return;
    }

    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    const cudaError_t state = stream
        ? cudaMemcpyAsync(dst, src, byte_size, kind, cuda_stream)
        : cudaMemcpy(dst, src, byte_size, kind);
    CHECK(state == cudaSuccess) << cudaGetErrorString(state);
    if (need_sync) {
        const cudaError_t sync_state = cudaStreamSynchronize(cuda_stream);
        CHECK(sync_state == cudaSuccess) << cudaGetErrorString(sync_state);
    }
}

void DeviceAllocator::memset_zero(void* ptr, size_t byte_size, void* stream,
                                  bool need_sync) const {
    if (byte_size == 0) return;
    CHECK(ptr != nullptr);
    if (device_type() == DeviceType::kDeviceCPU) {
        std::memset(ptr, 0, byte_size);
        return;
    }
    CHECK(device_type() == DeviceType::kDeviceCUDA) << "Unknown allocator device";
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    const cudaError_t state = stream
        ? cudaMemsetAsync(ptr, 0, byte_size, cuda_stream)
        : cudaMemset(ptr, 0, byte_size);
    CHECK(state == cudaSuccess) << cudaGetErrorString(state);
    if (need_sync) {
        const cudaError_t sync_state = cudaStreamSynchronize(cuda_stream);
        CHECK(sync_state == cudaSuccess) << cudaGetErrorString(sync_state);
    }
}

} // namespace base
