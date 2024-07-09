#pragma once
// Allocator.h

#include <cstddef>  // size_t
#include <upcxx/upcxx.hpp>

#include "upcxx_defs.h"

namespace upcxx_utils {

class Allocator {
  // virtual base class for global memory allocations
 protected:
  std::size_t m_totalSize;

 public:
  Allocator(const std::size_t totalSize);

  virtual ~Allocator();

  virtual global_byte_ptr Allocate(const std::size_t size, const std::size_t alignment = 0) = 0;

  virtual void Free(global_byte_ptr &ptr) = 0;

  UPCXX_SERIALIZED_FIELDS(m_totalSize);

  void swap(Allocator &other) { std::swap(m_totalSize, other.m_totalSize); }

 protected:
  virtual void Init() = 0;
};

// A simple allocator without state that allocates a global ptr
// and stores its value in a header within the allocation
template <typename T>
struct GlobalPtrAllocator {
  typedef T value_type;
  GlobalPtrAllocator() = default;
  template <class U>
  constexpr GlobalPtrAllocator(const GlobalPtrAllocator<U> &) noexcept {}

  static const int extra = (sizeof(upcxx::global_ptr<T>) + sizeof(T) - 1) / sizeof(T);

  T *allocate(std::size_t n) {
    if (n + extra >= std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::bad_alloc();
    upcxx::global_ptr<T> gptr = upcxx::new_array<T>(n + extra);
    if (gptr.is_null()) throw std::bad_alloc();
    auto p = gptr.local();
    memcpy(&gptr, p, sizeof(upcxx::global_ptr<T>));  // record the gptr as the header
    return p + extra;
  }

  void deallocate(T *p, std::size_t n) noexcept {
    auto pgptr = p - extra;
    upcxx::global_ptr<T> gptr{};
    memcpy(pgptr, &gptr, sizeof(upcxx::global_ptr<T>));
    assert(!gptr.is_null());
    assert(gptr.local() == pgptr);
    assert(gptr.where() == upcxx::rank_me());
    memset(pgptr, 0, sizeof(upcxx::global_ptr<T>));  // clear the header
    upcxx::delete_array(gptr);
  }
};

template <class T, class U>
bool operator==(const GlobalPtrAllocator<T> &, const GlobalPtrAllocator<U> &) {
  return true;
}
template <class T, class U>
bool operator!=(const GlobalPtrAllocator<T> &, const GlobalPtrAllocator<U> &) {
  return false;
}

};  // namespace upcxx_utils
