// PoolAllocator.cpp

#include <assert.h>
#include <stdint.h>

#include <upcxx/upcxx.hpp>

#ifdef _DEBUG
#include <iostream>
#endif

#include "upcxx_utils/log.hpp"
#include "upcxx_utils/memory-allocators/PoolAllocator.h"
#include "upcxx_utils/memory-allocators/upcxx_defs.h"

using upcxx::delete_array;
using upcxx::new_array;

namespace upcxx_utils {

PoolAllocatorImpl::PoolAllocatorImpl()
    : Allocator(0)
    , m_freeList()
    , m_start_ptr()
    , m_chunkSize(0) {
  DBG_VERBOSE("PoolAllocatorImpl()", *this, "\n");
}

PoolAllocatorImpl::PoolAllocatorImpl(const std::size_t totalSize, const std::size_t chunkSize)
    : Allocator(totalSize)
    , m_freeList()
    , m_start_ptr()
    , m_chunkSize(chunkSize) {
  padChunk();
  assert(chunkSize >= 8 && "Chunk size must be greater or equal to 8");
  assert(totalSize % chunkSize == 0 && "Total Size must be a multiple of Chunk Size");
  assert(totalSize >= chunkSize && "totalSize must be >= chunkSize");
  Init();
  DBG_VERBOSE("PoolAllocatorImpl(...)", *this, "\n");
}

PoolAllocatorImpl::PoolAllocatorImpl(const PoolAllocatorImpl &copy)
    : Allocator((Allocator &)copy)
    , m_freeList(copy.m_freeList)
    , m_start_ptr(copy.m_start_ptr)
    , m_chunkSize(copy.m_chunkSize) {
  DBG_VERBOSE("PoolAllocatorImpl(copy)", *this, ", copy=", copy, "\n");
}

PoolAllocatorImpl &PoolAllocatorImpl::operator=(const PoolAllocatorImpl &copy) {
  DBG_VERBOSE("PoolAllocatorImpl(copy)=", *this, ", copy=", copy, "\n");
  PoolAllocatorImpl c(copy);
  this->swap(c);
  return *this;
}

PoolAllocatorImpl::PoolAllocatorImpl(PoolAllocatorImpl &&move)
    : Allocator(std::move((Allocator &&) move))
    , m_freeList(std::move(move.m_freeList))
    , m_start_ptr(std::move(move.m_start_ptr))
    , m_chunkSize(std::move(move.m_chunkSize)) {
  move.m_start_ptr.reset();
  move.m_chunkSize = 0;
  DBG_VERBOSE("PoolAllocatorImpl(move)", *this, ", move=", move, "\n");
}

PoolAllocatorImpl &PoolAllocatorImpl::operator=(PoolAllocatorImpl &&move) {
  ((Allocator &)*this) = ((Allocator &)move);
  m_freeList = std::move(move.m_freeList);
  m_start_ptr = std::move(move.m_start_ptr);
  m_chunkSize = std::move(move.m_chunkSize);
  DBG_VERBOSE("PoolAllocatorImpl(move)=", *this, ", move=", move, "\n");
  return *this;
}

void PoolAllocatorImpl::swap(PoolAllocatorImpl &other) {
  DBG_VERBOSE("PoolAllocatorImpl(move)=", *this, ", other=", other, "\n");
  ((Allocator &)*this).swap((Allocator &)other);
  std::swap(m_freeList, other.m_freeList);
  std::swap(m_start_ptr, other.m_start_ptr);
  std::swap(m_chunkSize, other.m_chunkSize);
}

void PoolAllocatorImpl::padChunk() {
  if (m_totalSize == 0) return;
  size_t num_chunks = (m_totalSize + m_chunkSize - 1) / m_chunkSize;
  size_t offset = m_chunkSize % 8;
  if (offset != 0) {
    m_chunkSize += 8 - offset;
  }
  m_totalSize = m_chunkSize * num_chunks;
}

void PoolAllocatorImpl::Init() {
  assert(m_chunkSize % 8 == 0 && "PoolAllocatorImpl::Init() has aligned chunkSize");
  assert(m_chunkSize > 0 && "PoolAllocatorImpl::Init() has nonzero chunkSize");
  assert(m_totalSize % m_chunkSize == 0 && "PoolAllocatorImpl::Init() has aligned totalSize");
  assert(m_totalSize > 0 && "PoolAllocatorImpl::Init() has nonzero totalSize");
  assert(m_start_ptr.is_null() && "PoolAllocatorImpl::Init() found non null start");
  m_start_ptr.reset(new_array<uint8_t>(m_totalSize), true);
  this->Reset();
  DBG_VERBOSE("PoolAllocatorImpl::Init() allocated ", m_totalSize, " - ", m_start_ptr, "\n");
}

PoolAllocatorImpl::~PoolAllocatorImpl() {
  DBG_VERBOSE("~PoolAllocatorImpl()\n");
  if (m_start_ptr.is_null()) return;
  m_freeList.clear();
  assert(m_start_ptr.is_local() && "~PoolAllocatorImpl found non local start");
  DBG_VERBOSE("~PoolAllocatorImpl(", *this, ")\n");

  int where = m_start_ptr.where();
  if (rank_me() != where) return;

  m_start_ptr.reset();  // delete_array(m_start_ptr);       m_start_ptr = nullptr;
  assert(!m_start_ptr);

  assert(m_start_ptr.is_null() && "~PoolAllocatorImpl could not set start to be null");

  // necessary to assure shared_global_ptrs are freed
  while (upcxx::progress_required()) upcxx::progress();
}

global_byte_ptr PoolAllocatorImpl::Allocate(const std::size_t allocationSize, const std::size_t alignment) {
  assert(this->m_totalSize > 0 && "Allocate must be to an intialized PoolAllocatorImpl");
  assert(allocationSize == this->m_chunkSize && "Allocation size must be equal to chunk size");
  assert(m_start_ptr.is_local() && "start ptr is local");

  global_byte_ptr ret = nullptr;
  Node *freePosition = m_freeList.pop().local();

  if (freePosition == nullptr) {
    DBG("Allocator is full\n");
    // The pool allocator is full return nullptr
  } else {
    int64_t offset = ((global_byte_t *)freePosition) - ((global_byte_t *)m_start_ptr.local());
    assert(offset >= 0 && "offset >= 0 in Allocate");
    ret = m_start_ptr.get() + offset;
    assert(IsPooled(ret));
  }
  return ret;
}

void PoolAllocatorImpl::Free(global_byte_ptr &ptr) {
  assert(m_start_ptr.is_local() && "ptr is local");

  if (!ptr) return;

  if (!ptr.is_local()) {
    DIE("Free called on a pointer that is not local: ", ptr);
  }
  if (!IsPooled(ptr)) {
    DIE("Free called on a pointer that is not from this pool: ", ptr);
  }

  m_freeList.push(to_global_ptr<Node>((Node *)ptr.local()));
  ptr = nullptr;
}

void PoolAllocatorImpl::Reset() {
  // Create a linked-list with all free positions
  assert(m_start_ptr);
  assert(m_start_ptr.is_local());
  m_freeList.reset(m_start_ptr.local(), m_totalSize);
  const int nChunks = m_totalSize / m_chunkSize;
  DBG_VERBOSE("pushing ", nChunks, " to freeList\n");
  for (int i = 0; i < nChunks; ++i) {
    std::size_t address = (std::size_t)m_start_ptr.local() + i * m_chunkSize;
    m_freeList.push(to_global_ptr<Node>((Node *)address));
  }
}

bool PoolAllocatorImpl::IsPooled(const global_byte_ptr ptr) const {
  if (!ptr) return false;
  assert(ptr.is_local() && "is_pooled_chunk called on non-local global ptr");
  return (ptr >= m_start_ptr.get() && ptr < m_start_ptr.get() + m_totalSize && (ptr - m_start_ptr.get()) % m_chunkSize == 0);
}

//
// DistPoolAllocatorImpl
//

intrank_t DistPoolAllocatorImpl::myteam_rank_from_anyteam(intrank_t other_rank, upcxx::team &anyteam) {
  const upcxx::team &myteam = this->team();
  assert(other_rank < anyteam.rank_n());
  return myteam.from_world(anyteam[other_rank], rank_n());
}

intrank_t DistPoolAllocatorImpl::local_rank_from_myteam(intrank_t myteam_rank) {
  const upcxx::team &myteam = this->team();
  assert(myteam_rank < myteam.rank_n());
  return local_team().from_world(myteam[myteam_rank], rank_n());
}

DistPoolAllocatorImpl::DistPoolAllocatorImpl(const upcxx::team &someteam)
    : dist_object<PoolAllocatorImpl>(someteam)
    , m_local_copies() {
  Init();
  DBG_VERBOSE("DistPoolAllocator()", *this, "\n");
}

DistPoolAllocatorImpl::DistPoolAllocatorImpl(const std::size_t totalSize, const std::size_t chunkSize, const upcxx::team &someteam)
    : dist_object<PoolAllocatorImpl>(someteam, totalSize, chunkSize)
    , m_local_copies() {
  Init();
  DBG_VERBOSE("DistPoolAllocator(...)", *this, "\n");
}

void DistPoolAllocatorImpl::Init() {
  DBG_VERBOSE("DistPoolAllocator::Init()", *this, "\n");
  m_local_copies.clear();
  m_local_copies.reserve(this->team().rank_n());
  // wait for entire team to call Init
  DBG("Init() - my pool size:", (*this)->getTotalSize(), "\n");
  barrier(this->team());
  for (intrank_t team_rank = 0; team_rank < this->team().rank_n(); team_rank++) {
    if (team_rank == this->team().rank_me()) {
      m_local_copies.push_back((*(*this)));
    } else {
      auto copy = ((_DistPoolAllocatorImpl *)this)->fetch(team_rank).wait();
      m_local_copies.push_back(std::move(copy));
    }
    DBG("m_local_copies for team_rank=", team_rank, " is ", m_local_copies[team_rank],
        " pool's total size:", m_local_copies[team_rank].getTotalSize(), "\n");
  }
  // wait for entire team to finish init
  barrier(this->team());
}

DistPoolAllocatorImpl::~DistPoolAllocatorImpl() {
  DBG_VERBOSE("~DistPoolAllocator", *this, "\n");
  // ensure all shared_global_ptrs are freed
  while (upcxx::progress_required()) upcxx::progress();
  barrier(this->team());
  // m_local_copies will be destroyed normally
}

};  // namespace upcxx_utils
