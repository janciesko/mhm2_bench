#include "upcxx_utils/memory-allocators/UPCXXAllocator.h"

#include <upcxx/upcxx.hpp>

#include "upcxx_utils/memory-allocators/upcxx_defs.h"

namespace upcxx_utils {

UPCXXAllocator::UPCXXAllocator()
    : Allocator(0) {
  Init();
}

void UPCXXAllocator::Init() {}

UPCXXAllocator::~UPCXXAllocator() {}

global_byte_ptr UPCXXAllocator::Allocate(const std::size_t size, const std::size_t alignment) {
  return upcxx::new_array<global_byte_t>(size);
}

void UPCXXAllocator::Free(global_byte_ptr &ptr) {
  upcxx::delete_array(ptr);
  ptr = nullptr;
}

};  // namespace upcxx_utils
