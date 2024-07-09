#pragma once

/*
 * File:   SharedArray.hpp
 * Author: regan
 *
 * Created on April 25, 2020, 7:14 PM
 */

#include <atomic>
#include <cassert>
#include <exception>
#include <functional>
#include <memory>
#include <upcxx/upcxx.hpp>

#include "Allocators.hpp"

using upcxx::global_ptr;

namespace upcxx_utils {

class AtomicOffsetSizeCapacity {
  size_t _capacity;             // maximum offset/size. should never change after construction
  std::atomic<size_t> _size;    // == min(offset,capacity) after all writes have completed always <= capacity (never >capacity)
  std::atomic<size_t> _offset;  // first empty idx - may be >= capacity signaling full state
 protected:
  void set_capacity(size_t capacity, size_t start_offset = 0);

 public:
  using CallbackFunc = const std::function<void(size_t offset, size_t len)>;
  AtomicOffsetSizeCapacity(size_t capacity = 0, size_t start_offset = 0);

  // disable copy and move that would affect the atomics
  AtomicOffsetSizeCapacity(const AtomicOffsetSizeCapacity &copy) = delete;
  AtomicOffsetSizeCapacity &operator=(const AtomicOffsetSizeCapacity &copy) = delete;
  AtomicOffsetSizeCapacity(AtomicOffsetSizeCapacity &&move) = delete;
  AtomicOffsetSizeCapacity &&operator=(AtomicOffsetSizeCapacity &&move) = delete;

  inline size_t capacity() const { return _capacity; }
  inline size_t size() const { return _size.load(); }

  // returns the old size
  size_t reset(size_t new_offset = 0);

  // returns if the state is clean (size == offset or capacity)
  bool ready() const;
  bool full() const;
  bool empty() const;

  // first checks offset for any remaining capacity
  // then increments offset, and if possible:
  //   appends all or a portion of data
  //   increments size after the apppend has completed
  // returns >len, when append fully succeeded, and more capacity remains
  //         with the assumption no action is required after this operation
  // returns the number appended (i.e >=0 & <=len ) when capacity was reached
  //         if the returned number is >0 it was reached by this operation
  //         with the assumption that this rank will perform an action about this full container

  size_t append(size_t len, CallbackFunc &callback);
};

template <typename T>
struct SharedArray {
  // an static capacity array of global memory array of T

  AtomicOffsetSizeCapacity oss;
  global_ptr<T> ptr;

  SharedArray()
      : oss()
      , ptr() {}

  SharedArray(size_t capacity_)
      : oss(capacity_)
      , ptr() {}

  SharedArray(global_ptr<T> ptr_, size_t capacity_)
      : oss(capacity_)
      , ptr() {
    set_new_ptr(ptr_, 0);
  }

  // disable copy and move which would affect the atomics
  SharedArray(const SharedArray &copy) = delete;
  SharedArray &operator=(const SharedArray &copy) = delete;
  SharedArray(SharedArray &&move) = delete;
  SharedArray &&operator=(SharedArray &&move) = delete;
  ~SharedArray() { clear(); }

  global_ptr<T> get_global_ptr() { return ptr; }
  // sets a new pointer, returns the old size.
  size_t set_new_ptr(global_ptr<T> ptr_, size_t new_offset = 0) {
    assert(ready());
    assert(ptr_.is_local());
    // first ensure no append operations will succeed
    size_t oldSize = set(capacity());
    // assign the new pointer
    ptr = ptr_;
    DBG("Zeroing new_offset=", new_offset, " oldSize=", oldSize, " capacity=", capacity(), " on this=", this, " ptr=", ptr, "\n");
    if (!ptr.is_null()) memset((void *)(ptr.local() + new_offset), 0, (capacity() - new_offset) * sizeof(T));
    // set the new offset
    size_t oldSize2 = set(new_offset);
    if (oldSize2 != capacity()) throw runtime_error("Invalid change of state while set_new_ptr is executing");
    return oldSize;
  }

  // pass through accessors
  inline bool empty() const { return oss.empty(); }
  inline bool ready() const { return oss.ready(); }
  inline bool full() const { return oss.full(); }
  inline size_t capacity() const { return oss.capacity(); }
  inline size_t size() const { return oss.size(); }

  // returns the old value
  size_t set(size_t val = 0) {
    assert(capacity() >= val);
    return oss.reset(val);
  }

  void clear() {
    destroy_all();
    oss.reset(0);
  }

  void destroy_all() {
    DBG("destroy_all size=", size(), " capacity=", capacity(), " to this=", this, "\n");
    for (T &t : *this) {
      t.~T();
    }
  }

  const T *begin() const {
    assert(capacity() > 0);
    assert(!ptr.is_null());
    assert(ptr.is_local());
    return ptr.local();
  }

  const T *end() const { return begin() + oss.size(); }

  T *begin() { return const_cast<T *>(((const SharedArray *)this)->begin()); }

  T *end() { return const_cast<T *>(((const SharedArray *)this)->end()); }

  // first checks size for any remaining capacity
  // then increments offset, and if possible:
  //   appends all or a portion of data
  //   increments size after the append has completed
  // returns (len + 1), when append fully succeeded, and more capacity remains
  // return 0 when capacity has already been reached (and no elements were appended)
  // returns then number appended (i.e >=1 && <=len ) when capacity was reached by this operation
  //         with the assumption that this rank will do something about this full SharedArray

  template <typename iter>
  size_t append(iter source, size_t len) {
    assert(capacity() > 0);
    assert(len > 0);
    assert(ptr.is_local());

    return oss.append(len, [source, this](size_t offset, size_t append_len) {
      DBG("append offset=", offset, " append_len=", append_len, " to this=", this, "\n");
      T *dest = this->begin() + offset;
      iter _source = source;
      for (size_t i = 0; i < append_len; i++) {
        // dest[i] = *(_source++);
        auto ptr = new (dest + i) T(*(_source++));
        assert(ptr == dest + i);
      }
    });
  }
};

// SharedArray with two types, in separate contiguous blocks of memory...

template <typename T1, typename T2>
struct SharedArray2 : public SharedArray<T1> {
  SharedArray2()
      : SharedArray<T1>() {}
  SharedArray2(size_t capacity_)
      : SharedArray<T1>(capacity_) {}
  SharedArray2(global_ptr<T1> ptr_, size_t capacity_)
      : SharedArray<T1>(ptr_, capacity_) {}

  // disable copy and move operations
  SharedArray2(const SharedArray2 &copy) = delete;
  SharedArray2(SharedArray2 &&copy) = delete;
  SharedArray2 &operator=(const SharedArray2 &) = delete;
  SharedArray2 &operator=(SharedArray2 &&) = delete;

  ~SharedArray2() { clear(); }

  void clear() {
    destroy_all2();
    ((SharedArray<T1> *)this)->clear();
  }

  void destroy_all2() {
    for (T2 *t2 = begin2(); t2 != end2(); t2++) {
      (*t2).~T2();
    }
  }

  // block of T2 starts at the end/capacity of T1

  const T2 *begin2() const { return (T2 *)(this->begin() + this->capacity()); }

  const T2 *end2() const { return begin2() + this->size(); }

  T2 *begin2() { return const_cast<T2 *>(((const SharedArray2 *)this)->begin2()); }

  T2 *end2() { return const_cast<T2 *>(((const SharedArray2 *)this)->end2()); }

  // FIXME to move, not copy
  template <typename iter1, typename iter2>
  size_t append(iter1 source1, iter2 source2, size_t len) {
    assert(this->capacity() > 0);

    return this->oss.append(len, [source1, source2, this](size_t offset, size_t add_len) {
      DBG("append offset=", offset, " add_len=", add_len, " to this=", this, "\n");
      T1 *dest1 = this->begin() + offset;
      iter1 _source1 = source1;
      T2 *dest2 = this->begin2() + offset;
      iter2 _source2 = source2;
      for (size_t i = 0; i < add_len; i++) {
        // dest1[i] = *(_source1++);
        // dest2[i] = *(_source2++);
        auto ptr1 = new (dest1 + i) T1(*(_source1++));
        assert(ptr1 == dest1 + i);
        auto ptr2 = new (dest2 + i) T2(*(_source2++));
        assert(ptr2 == dest2 + i);
      }
    });
  }
};

};  // namespace upcxx_utils
