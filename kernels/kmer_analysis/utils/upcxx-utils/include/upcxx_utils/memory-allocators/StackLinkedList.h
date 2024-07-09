#pragma once
// StackLinkedList.h

#include <atomic>
#include <upcxx/upcxx.hpp>

#include "upcxx_defs.h"
#include "upcxx_utils/log.hpp"
#include "upcxx_utils/shared_global_ptr.hpp"

using upcxx::global_ptr;
using upcxx::local_team;

namespace upcxx_utils {

class OffsetAndCount {
  // class to assist in ABA problem
  // size_t is tagged with an iteration count
  // so only updates of head from the same iteration are compare exchanged
 private:
  struct _ {
    size_t m_offset : 48;   // 6 bytes up to 256 TB
    size_t iteration : 16;  // 2 bytes up to 64k unique iterations
  };

  union __ {
    size_t val;
    _ offsetRank;
  };
  __ m_val;

  static std::atomic<size_t> &getIter() {
    static std::atomic<size_t> iter((size_t)local_team().rank_me());  // stagger iteration space across local team
    return iter;
  }

  static uint16_t getIteration(size_t val = local_team().rank_n()) { return (uint16_t)(getIter().fetch_add(val)); }

 public:
  static const size_t end_offset = 0xffffffffffff;

  OffsetAndCount(size_t offset = end_offset) {
    assert(offset <= end_offset);
    m_val.offsetRank.m_offset = offset;
    m_val.offsetRank.iteration = getIteration();
  }

  OffsetAndCount(const std::atomic<size_t> &aptr) { m_val.val = aptr.load(); }

  // stores this offset in an atomic

  void store(std::atomic<size_t> &aptr) const { aptr.store(m_val.val); }

  // loads and sets this offset from an atomic

  void load(const std::atomic<size_t> &aptr) { m_val.val = aptr.load(); }

  inline size_t get_offset() const { return m_val.offsetRank.m_offset; }

  // void increment() {
  //    m_val.offsetRank.iteration = getIteration();
  //}

  inline operator bool() const { return m_val.offsetRank.m_offset != end_offset; }

  inline explicit operator const size_t &() const { return m_val.val; }

  inline explicit operator size_t &() { return m_val.val; }

  inline size_t &getVal() { return m_val.val; }

  inline const size_t &getVal() const { return m_val.val; }

  static bool atomic_cswap(std::atomic<size_t> &aptr, OffsetAndCount &oldVal, const OffsetAndCount &newVal) {
    return aptr.compare_exchange_weak(static_cast<size_t &>(oldVal), static_cast<const size_t>(newVal));
  }

  friend std::ostream &operator<<(std::ostream &os, const OffsetAndCount &oac) {
    os << "OffsetAndCount(offset=" << oac.get_offset() << ", iter=" << oac.m_val.offsetRank.iteration << ", val=" << oac.m_val.val
       << ", static_iter=" << getIter().load() << ")";
    return os;
  }
};

template <class LLType>
class StackLinkedList {
  // lockfree thread safe stack (LIFO) linked list
  // copy/move safe using the same free list as the original
  // copy only works between ranks on the local team (to support dist_object)
 public:
  struct Node {
    LLType data;
    OffsetAndCount next_offset;
  };
  using global_node_ptr_t = global_ptr<Node>;

  struct HeadStartSize {
    // HeadStartSize is always within a fixed block of memory so that the offset encoded in head is always positive.
    std::atomic<size_t> head;
    const global_node_ptr_t start_ptr;
    const size_t size;

    HeadStartSize()
        : head(0)
        , start_ptr(nullptr)
        , size(0) {}
    HeadStartSize(global_node_ptr_t start_, size_t size_)
        : head(0)
        , start_ptr(start_)
        , size(size_) {}
    friend std::ostream &operator<<(std::ostream &os, const HeadStartSize &hss) {
      return os << "HeadStartSize(head=" << hss.head.load() << ", start_ptr=" << hss.start_ptr << ", size=" << hss.size << ")";
    }
  };
  using shared_head_start_size_ptr_t = shared_global_ptr<HeadStartSize>;

 protected:
  shared_head_start_size_ptr_t m_head_start_size_ptr;

 public:
  StackLinkedList();
  StackLinkedList(const StackLinkedList &copy);
  StackLinkedList(StackLinkedList &&move);

  StackLinkedList &operator=(const StackLinkedList &copy) {
    DBG_VERBOSE("StackLinkedList", this, " &= copy=", copy, "\n");
    StackLinkedList cpy(copy);
    this->swap(cpy);
    DBG_VERBOSE("StackLinkedList", this, " &= this=", *this, "\n");
    return *this;
  }
  StackLinkedList &operator=(StackLinkedList &&move) {
    DBG_VERBOSE("StackLinkedList", this, " &&= move=", move, "\n");
    StackLinkedList moved(std::move(move));
    this->swap(moved);
    DBG_VERBOSE("StackLinkedList", this, " &&= this=", *this, "\n");
    return *this;
  }

  // NOTE care must be taken to destruct and clear only after all ranks are ready
  ~StackLinkedList();
  void reset(void *start_ptr, size_t size);
  void push(global_node_ptr_t newNode);
  global_node_ptr_t pop();
  UPCXX_SERIALIZED_FIELDS(m_head_start_size_ptr);

  friend std::ostream &operator<<(std::ostream &os, const StackLinkedList &sll) {
    os << "StackLinkedList(" << &sll << " m_head_start_size=" << sll.m_head_start_size_ptr;
    if (sll.m_head_start_size_ptr && sll.m_head_start_size_ptr.is_local()) {
      os << ":" << *(sll.m_head_start_size_ptr.local());
    } else {
      os << "(nullptr)";
    }
    os << ")";
    return os;
  }

  void clear();

 protected:
  void swap(StackLinkedList &other);
};

#include "StackLinkedListImpl.h"

}  // namespace upcxx_utils
