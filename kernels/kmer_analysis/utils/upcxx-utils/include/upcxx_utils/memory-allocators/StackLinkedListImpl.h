#pragma once
// StackLinkedListImpl.h

#include <atomic>
#include <stdexcept>
#include <upcxx/upcxx.hpp>

#include "StackLinkedList.h"
#include "upcxx_defs.h"
#include "upcxx_utils/log.hpp"

using std::runtime_error;

using upcxx::global_ptr;
using upcxx::local_team;
using upcxx::local_team_contains;
using upcxx::to_global_ptr;

// construct / destruct

template <class LLType>
StackLinkedList<LLType>::StackLinkedList()
    : m_head_start_size_ptr() {
  // DBG_VERBOSE("StackLinkedList() this=", *this, "\n");
}

template <class LLType>
StackLinkedList<LLType>::StackLinkedList(const StackLinkedList<LLType> &copy)
    : m_head_start_size_ptr(copy.m_head_start_size_ptr) {
  if (m_head_start_size_ptr) {
    int where = m_head_start_size_ptr.where();
    if (!local_team_contains(where)) {
      // does not support copy out of the local_team
      throw runtime_error("Can not make a copy outside the local_team");
    }
  }
  // DBG_VERBOSE("StackLinkedList(copy) this=", *this, " copied=", copy, "\n");
}

template <class LLType>
StackLinkedList<LLType>::StackLinkedList(StackLinkedList &&move)
    : m_head_start_size_ptr(std::move(move.m_head_start_size_ptr)) {
  move.m_head_start_size_ptr.reset();
  // DBG_VERBOSE("StackLinkedList(move) this=", *this, " moved=", move, "\n");
}

template <class LLType>
void StackLinkedList<LLType>::reset(void *start_ptr, size_t size) {
  assert(start_ptr);
  assert(size > 0);
  Node *node_start = (Node *)start_ptr;
  bool needs_new = true;
  if (m_head_start_size_ptr) {
    assert(m_head_start_size_ptr.is_local());
    auto hss = m_head_start_size_ptr.local();
    if (node_start != hss->start_ptr.local() || size != hss->size) {
      // HeadStartSize pointers are okay already.  Just reset the head
      needs_new = false;
    }
  }
  if (needs_new) {
    // need a new head_start_size ptr
    m_head_start_size_ptr.reset(upcxx::new_<HeadStartSize>(to_global_ptr(node_start), size));
  }
  OffsetAndCount offset;
  assert(!offset);  // end_offset aka nullptr
  offset.store(m_head_start_size_ptr.local()->head);
  // DBG_VERBOSE("StackLinkedList::Init this=", *this, "\n");
}

template <class LLType>
void StackLinkedList<LLType>::swap(StackLinkedList &other) {
  // DBG_VERBOSE("StackLinkedList::swap this=", *this, " other=", other, "\n");
  std::swap(m_head_start_size_ptr, other.m_head_start_size_ptr);
}

template <class LLType>
StackLinkedList<LLType>::~StackLinkedList() {
  // DBG_VERBOSE("~StackLinkedList() ", *this, "\n");
  // necessary to ensure all shared_global_ptr entries are freed
  while (upcxx::progress_required()) upcxx::progress();
  clear();
};

template <class LLType>
void StackLinkedList<LLType>::clear() {
  // DBG_VERBOSE("StackLinkedList::clear() ", *this, "\n");
  m_head_start_size_ptr.reset();
};

// Thread safe atomic push and pop operations

template <class LLType>
void StackLinkedList<LLType>::push(global_ptr<Node> newNode_gptr) {
  assert(m_head_start_size_ptr);
  assert(m_head_start_size_ptr.is_local());
  assert(newNode_gptr);
  if (!newNode_gptr.is_local()) throw runtime_error("Call to push from non-local process to pushed mem");
  assert(newNode_gptr.is_local());
  if (!m_head_start_size_ptr.is_local()) {
    DIE("Call to push(", newNode_gptr, ") from non-local process: ", *this, "\n");
  }

  auto &hss = *(m_head_start_size_ptr.local());
  if (hss.size == 0 || !hss.start_ptr || newNode_gptr.where() != hss.start_ptr.where()) {
    DIE("push(", newNode_gptr, ") to zero or uninitialized or other rank's list: ", *this, "!\n");
  }
  assert(hss.start_ptr);
  assert(hss.size > 0);

  if (newNode_gptr < hss.start_ptr ||
      ((global_byte_t *)newNode_gptr.local()) >= (((global_byte_t *)hss.start_ptr.local()) + hss.size)) {
    DIE("push(", newNode_gptr, ") out of bounds for this list: ", *this, "!\n");
  }
  assert(newNode_gptr >= hss.start_ptr);
  assert(((global_byte_t *)newNode_gptr.local()) < (((global_byte_t *)hss.start_ptr.local()) + hss.size));

  Node *newNode = newNode_gptr.local();
  OffsetAndCount &oldHeadOffset = newNode->next_offset;  // address of this pushed node::next_offset

  // address of this pushed node with new iteration
  const OffsetAndCount newHeadOffset(((global_byte_t *)newNode) - ((global_byte_t *)hss.start_ptr.local()));

  size_t iterations = 0;
  oldHeadOffset.load(hss.head);  // sets head in newNode->next_offset
  do {
    if (OffsetAndCount::atomic_cswap(hss.head, oldHeadOffset, newHeadOffset)) {
      // DBG_VERBOSE("push set offset=", newHeadOffset, " old ", oldHeadOffset, ". ", m_start_global_ptr, "\n");
      return;
    }  // else oldHeadOffset has the latest value of head
    iterations++;
    DBG_VERBOSE("push missed ", iterations, " times. head now is ", oldHeadOffset, "\n");
  } while (true);
}

template <class LLType>
typename StackLinkedList<LLType>::global_node_ptr_t StackLinkedList<LLType>::pop() {
  assert(m_head_start_size_ptr);
  assert(m_head_start_size_ptr.is_local());
  if (m_head_start_size_ptr.is_null() || !m_head_start_size_ptr.is_local()) {
    DIE("Call to pop from non-local process: ", *this, "\n");
  }

  auto &hss = *(m_head_start_size_ptr.local());
  if (hss.size == 0 || hss.start_ptr.is_null() || !hss.start_ptr.is_local()) {
    DIE("pop from zero-sized or uninitialized or non-local list! ", *this, "\n");
  }
  assert(hss.start_ptr);
  assert(hss.start_ptr.is_local());
  assert(hss.size > 0);

  size_t iterations = 0;

  global_node_ptr_t poppedNode = nullptr;
  Node *testNode = nullptr;
  global_byte_t *start_ptr = (global_byte_t *)hss.start_ptr.local();

  // get the latest copy of head
  OffsetAndCount oldHeadOffset(hss.head);

  do {
    if (!oldHeadOffset) {
      break;  // stack list is empty
    }

    // calculate the head node ptr
    size_t old_offset = oldHeadOffset.get_offset();
    if (old_offset >= hss.size) {
      DIE("Invalid StackLinkedList - old_offset=", old_offset, ", oldHeadOffset=", oldHeadOffset, ", head=", hss.head.load(),
          ", m_head_start_size_ptr=", m_head_start_size_ptr, "\n");
    }
    assert(old_offset < hss.size);
    testNode = (Node *)(start_ptr + old_offset);

    size_t new_offset = testNode->next_offset.get_offset();
    if (new_offset != OffsetAndCount::end_offset && new_offset >= hss.size) {
      // got garbage try again
      oldHeadOffset.load(hss.head);
      continue;
    }

    // re-tag existing (or null) next_offset with new iteration count from this thread
    OffsetAndCount newHeadOffset(new_offset);

    if (OffsetAndCount::atomic_cswap(hss.head, oldHeadOffset, newHeadOffset)) {
      if (!newHeadOffset) {
        // DBG_VERBOSE("pop set head to empty list ", newHeadOffset, " m_start_global_ptr= ", m_start_global_ptr, "\n");
      }
      // DBG_VERBOSE(": pop got offset=", oldHeadOffset, " set ", newHeadOffset, ". ",m_start_global_ptr, "\n");
      poppedNode = to_global_ptr<Node>(testNode);
      break;
    }  // else oldHeadOffset now has latest value of head

    iterations++;
    DBG_VERBOSE("pop missed ", iterations, " times. head is now: ", oldHeadOffset, "\n");
  } while (true);

  return poppedNode;
}
