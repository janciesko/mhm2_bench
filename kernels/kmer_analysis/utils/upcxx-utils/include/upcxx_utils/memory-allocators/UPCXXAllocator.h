#pragma once
// UPCXXAllocator.h

#include <upcxx/upcxx.hpp>

#include "Allocator.h"
#include "upcxx_defs.h"

namespace upcxx_utils {

class UPCXXAllocator : public Allocator {
  // Example allocator of global memory
 public:
  UPCXXAllocator();

  virtual ~UPCXXAllocator();

  virtual global_byte_ptr Allocate(const std::size_t size, const std::size_t alignment = 0) override;

  virtual void Free(global_byte_ptr& ptr) override;

 protected:
  virtual void Init() override;
};

template <class T>
class UPCXXStdAllocator {
  // compatible with std::Allocator
  // Allocates and returns private memory which is allocated from a global memory
  // requires sizeof(upcxx::global_ptr) extra bytes for a header
 public:
  typedef T value_type;

  UPCXXStdAllocator() = default;

  template <class U>
  constexpr UPCXXStdAllocator(const UPCXXStdAllocator<U>&) noexcept {}

  T* allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::bad_alloc();

    // allocate header + request
    global_byte_ptr ptr = upcxx::new_array<global_byte_t>(sizeof(global_byte_ptr) + n * sizeof(T));
    if (ptr) {
      // store the global_byte_ptr in the header
      memcpy(ptr.local(), &ptr, sizeof(global_byte_ptr));
      return static_cast<T*>(ptr.local() + sizeof(global_byte_ptr));
    }
    throw std::bad_alloc();
  }

  void deallocate(T* p, std::size_t) noexcept {
    global_byte_ptr ptr;
    // restore global_ptr from header
    global_byte_t* x = ((global_byte_t*)p) - sizeof(global_byte_ptr);
    memcpy(&ptr, x, sizeof(global_byte_ptr));
    memset(x, 0, sizeof(global_byte_ptr));
    upcxx::delete_array(ptr);
  }
};

template <class T, class U>
bool operator==(const UPCXXStdAllocator<T>&, const UPCXXStdAllocator<U>&) {
  return true;
}

template <class T, class U>
bool operator!=(const UPCXXStdAllocator<T>&, const UPCXXStdAllocator<U>&) {
  return false;
}

};  // namespace upcxx_utils
