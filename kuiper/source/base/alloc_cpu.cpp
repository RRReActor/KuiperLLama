#include <glog/logging.h>
#include <cstdlib>
#include "base/alloc.h"

namespace base {

void* CPUDeviceAllocator::allocate(size_t byte_size) const {
    if(!byte_size) {
        return nullptr;
    }
    return malloc(byte_size);
}

void CPUDeviceAllocator::release(void* ptr) const {
    if(ptr) free(ptr);
}

} // namespace base