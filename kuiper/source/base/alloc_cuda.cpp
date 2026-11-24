#include <cuda_runtime_api.h>
#include "base/alloc.h"

namespace base {

CUDADeviceAllocator::~CUDADeviceAllocator() {
    if (cuda_buffers_map_.empty() && big_buffers_map_.empty()) return;

    int original_device = -1;
    const auto original_state = cudaGetDevice(&original_device);
    auto free_buffers = [](const auto& buffers_map) {
        for (const auto& entry : buffers_map) {
            auto state = cudaSetDevice(entry.first);
            if (state != cudaSuccess) {
                LOG(ERROR) << "Cannot select CUDA device for pool cleanup: "
                           << cudaGetErrorString(state);
                continue;
            }
            for (const auto& buffer : entry.second) {
                state = cudaFree(buffer.data);
                if (state != cudaSuccess) {
                    LOG(ERROR) << "Cannot free CUDA pool buffer: "
                               << cudaGetErrorString(state);
                }
            }
        }
    };
    free_buffers(cuda_buffers_map_);
    free_buffers(big_buffers_map_);
    if (original_state == cudaSuccess) {
        const auto state = cudaSetDevice(original_device);
        if (state != cudaSuccess) {
            LOG(ERROR) << "Cannot restore CUDA device after pool cleanup: "
                       << cudaGetErrorString(state);
        }
    }
}

// Allocate memory on the current CUDA device, reusing big buffers if possible.
void* CUDADeviceAllocator::allocate(size_t byte_size) const {
    if (byte_size == 0) return nullptr;

    int device_id = -1;
    cudaError_t state = cudaGetDevice(&device_id);
    CHECK(state == cudaSuccess);
    if (byte_size > 1024 * 1024) {
        auto& big_buffers = big_buffers_map_[device_id];
        size_t sel_id = big_buffers.size();
        for (size_t i = 0; i < big_buffers.size(); ++i) {
            if (big_buffers[i].byte_size >= byte_size && !big_buffers[i].busy && 
                big_buffers[i].byte_size - byte_size < 1 * 1024 * 1024) {
                    if(sel_id == big_buffers.size() || big_buffers[sel_id].byte_size > big_buffers[i].byte_size)
                        sel_id = i;
                }
        }
        if(sel_id != big_buffers.size()) {
            big_buffers[sel_id].busy = true;
            big_no_busy_cnt_[device_id] -= big_buffers[sel_id].byte_size;
            return big_buffers[sel_id].data;
        }
        
        void* ptr = nullptr;
        state = cudaMalloc(&ptr, byte_size);
        if (cudaSuccess != state) {
            char buf[256];
            snprintf(buf, 256,
                    "Error: CUDA error when allocating %zu MB memory! maybe there's no enough memory "
                    "left on  device.",
                    byte_size >> 20);
            LOG(ERROR) << buf;
            return nullptr;
        }
        big_buffers.emplace_back(ptr, byte_size, true);
        return ptr;
    }

    auto& cuda_buffers = cuda_buffers_map_[device_id];
    for (size_t i = 0; i < cuda_buffers.size(); ++i) {
        if(cuda_buffers[i].byte_size >= byte_size && !cuda_buffers[i].busy) {
            cuda_buffers[i].busy = true;
            no_busy_cnt_[device_id] -= cuda_buffers[i].byte_size;
            return cuda_buffers[i].data;
        }
    }

    void* ptr = nullptr;
    state = cudaMalloc(&ptr, byte_size);
    if(state != cudaSuccess) {
        char buf[256];
        snprintf(buf, 256,
                "Error: CUDA error when allocating %zu MB memory! maybe there's no enough memory "
                "left on  device.",
                byte_size >> 20);
        LOG(ERROR) << buf;
        return nullptr;
    }
    cuda_buffers.emplace_back(ptr, byte_size, true);
    return ptr;
}

void CUDADeviceAllocator::release(void* ptr) const {
    if (!ptr) return;

    int original_device = -1;
    cudaError_t state = cudaGetDevice(&original_device);
    CHECK(state == cudaSuccess);

    // Mark a tracked allocation idle exactly once, using its allocated size.
    auto mark_idle = [ptr](auto& buffers_map, auto& idle_bytes) {
        for (auto& entry : buffers_map) {
            for (auto& buffer : entry.second) {
                if (buffer.data == ptr) {
                    CHECK(buffer.busy) << "CUDA buffer has already been released";
                    buffer.busy = false;
                    idle_bytes[entry.first] += buffer.byte_size;
                    return true;
                }
            }
        }
        return false;
    };

    const bool tracked = mark_idle(cuda_buffers_map_, no_busy_cnt_) ||
                         mark_idle(big_buffers_map_, big_no_busy_cnt_);
    if (!tracked) {
        state = cudaFree(ptr);
        CHECK(state == cudaSuccess) << "Error: CUDA error when releasing memory";
        return;
    }

    // Reclaim idle buffers after accounting for this release.
    auto reclaim = [&state](auto& buffers_map, auto& idle_bytes, size_t threshold) {
        for (auto& entry : buffers_map) {
            if (idle_bytes[entry.first] <= threshold) continue;
            state = cudaSetDevice(entry.first);
            CHECK(state == cudaSuccess);
            auto& buffers = entry.second;
            auto buffer = buffers.begin();
            while (buffer != buffers.end()) {
                if (buffer->busy) {
                    ++buffer;
                    continue;
                }
                state = cudaFree(buffer->data);
                CHECK(state == cudaSuccess)
                    << "Error: CUDA error when releasing memory on device " << entry.first;
                buffer = buffers.erase(buffer);
            }
            idle_bytes[entry.first] = 0;
        }
    };
    reclaim(cuda_buffers_map_, no_busy_cnt_, 10 * 1024 * 1024);
    reclaim(big_buffers_map_, big_no_busy_cnt_, 1024 * 1024 * 1024);

    state = cudaSetDevice(original_device);
    CHECK(state == cudaSuccess);
}

} // namespace base
