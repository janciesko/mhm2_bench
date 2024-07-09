// PoolAllocator.h
#pragma once

#include <upcxx/upcxx.hpp>

#include "Allocator.h"
#include "StackLinkedList.h"
#include "upcxx_defs.h"
#include "upcxx_utils/log.hpp"
#include "upcxx_utils/shared_global_ptr.hpp"

using upcxx::dist_object;
using upcxx::global_ptr;
using upcxx::intrank_t;
using upcxx::local_team;
using upcxx::make_future;
using upcxx::progress;
using upcxx::to_future;
using upcxx::to_global_ptr;
using upcxx::world;

namespace upcxx_utils {

class PoolAllocatorImpl : public Allocator {
  // Pooled allocator of fixed size chunk of global memory
  // works as a dist_object as it is copy safe between ranks of the same local_team
 protected:
  struct FreeHeader {};
  using Node = StackLinkedList<FreeHeader>::Node;

  StackLinkedList<FreeHeader> m_freeList;
  shared_global_ptr<global_byte_t> m_start_ptr;
  std::size_t m_chunkSize;

 public:
  PoolAllocatorImpl();
  PoolAllocatorImpl(const std::size_t totalSize, const std::size_t chunkSize);
  PoolAllocatorImpl(const PoolAllocatorImpl &copy);
  PoolAllocatorImpl &operator=(const PoolAllocatorImpl &copy);
  PoolAllocatorImpl(PoolAllocatorImpl &&move);
  PoolAllocatorImpl &operator=(PoolAllocatorImpl &&move);
  void swap(PoolAllocatorImpl &other);
  virtual ~PoolAllocatorImpl();

  operator bool() const { return m_totalSize > 0 && m_chunkSize > 0; }

  virtual global_byte_ptr Allocate(const std::size_t size, const std::size_t alignment = 0) override;

  virtual void Free(global_byte_ptr &ptr) override;

  virtual bool IsPooled(const global_byte_ptr ptr) const;

  inline size_t getTotalSize() const { return m_totalSize; }

  inline size_t getChunkSize() const { return m_chunkSize; }

  UPCXX_SERIALIZED_FIELDS(m_freeList, m_start_ptr, m_chunkSize, m_totalSize);

  friend std::ostream &operator<<(std::ostream &os, const PoolAllocatorImpl &pai) {
    os << "PoolAllocatorImpl(m_freeList=" << pai.m_freeList << ", m_start_ptr=" << pai.m_start_ptr
       << ", m_chunkSize=" << pai.m_chunkSize << ", Allocator::m_totalSize=" << pai.m_totalSize << ")";
    return os;
  }

 protected:
  void padChunk();
  virtual void Init() override;
  virtual void Reset();
};

template <class T>
class PoolAllocator : public PoolAllocatorImpl {
  // Allocates and returns a chunkCount array of global_ptr<T> which is allocated from a PoolAllocatorImpl
 public:
  PoolAllocator()
      : PoolAllocatorImpl() {}

  PoolAllocator(const std::size_t numChunks, const std::size_t chunkCount)
      : PoolAllocatorImpl(numChunks * chunkCount * sizeof(T), chunkCount * sizeof(T)) {
    DBG_VERBOSE("PoolAllocator(", numChunks, ", ", chunkCount, ") ", *this, "\n");
  }

  PoolAllocator(const PoolAllocator &copy)
      : PoolAllocatorImpl((const PoolAllocatorImpl &)copy) {
    DBG_VERBOSE("PoolAllocator(copy=", &copy, " ", copy, ") ", *this, "\n");
  }

  PoolAllocator(PoolAllocator &&move)
      : PoolAllocatorImpl((PoolAllocatorImpl &&) move) {
    DBG_VERBOSE("PoolAllocator(move=", &move, ") ", *this, "\n");
  }

  PoolAllocator &operator=(const PoolAllocator &copy) {
    *((PoolAllocatorImpl *)this) = (PoolAllocatorImpl &)copy;
    return *this;
  }
  PoolAllocator &operator=(PoolAllocator &&move) {
    *((PoolAllocatorImpl *)this) = std::move((PoolAllocatorImpl &&) move);
    return *this;
  }

  global_ptr<T> allocate() {
    assert(this->m_chunkSize % sizeof(T) == 0);
    assert(this->m_chunkSize >= sizeof(T));
    global_byte_ptr ptr = this->Allocate(this->m_chunkSize);
    if (!ptr) return nullptr;
    assert(ptr.is_local());
    return to_global_ptr<T>((T *)ptr.local());
  }

  void deallocate(global_ptr<T> &p) noexcept {
    assert(p.is_local());
    global_byte_ptr ptr = to_global_ptr<global_byte_t>((global_byte_t *)p.local());
    this->Free(ptr);
    p = nullptr;
  }

  size_t get_total_count() const { return this->getTotalSize() / sizeof(T); }

  size_t get_chunk_count() const { return this->getChunkSize() / sizeof(T); }

  friend std::ostream &operator<<(std::ostream &os, const PoolAllocator &pa) {
    os << "PoolAllocator(PoolAllocatorImpl=" << (PoolAllocatorImpl &)pa << ")";
    return os;
  }
};

class DistPoolAllocatorImpl : public dist_object<PoolAllocatorImpl> {
  // A distributed Pool Allocator with cached copies of local team instances
  // can allocate and free global memory from a pool from any rank via rpc and local_team by atomic operations
 public:
  using _DistPoolAllocatorImpl = dist_object<PoolAllocatorImpl>;
  DistPoolAllocatorImpl(const upcxx::team &myteam = world());
  DistPoolAllocatorImpl(const std::size_t totalSize, const std::size_t chunkSize, const upcxx::team &myteam = world());
  DistPoolAllocatorImpl(const DistPoolAllocatorImpl &copy) = delete;
  DistPoolAllocatorImpl(DistPoolAllocatorImpl &&move)
      : _DistPoolAllocatorImpl(std::move((_DistPoolAllocatorImpl &)move))
      , m_local_copies(std::move(move.m_local_copies)) {
    DBG_VERBOSE("DistPoolAllocatorImpl(move=", &move, ") ", *this, "\n");
    move.m_local_copies.clear();
  }

  DistPoolAllocatorImpl &operator=(const DistPoolAllocatorImpl &copy) = delete;
  DistPoolAllocatorImpl &operator=(DistPoolAllocatorImpl &&move) {
    DBG_VERBOSE("DistPoolAllocatorImpl = DistPoolAllocatorImpl(move=", &move, " ", move, ")\n");
    DistPoolAllocatorImpl m(std::move(move));
    std::swap(*this, m);
    DBG_VERBOSE("DistPoolAllocatorImpl = DistPoolAllocatorImpl(move=) is now ", *this, "\n");
    return *this;
  }

  virtual ~DistPoolAllocatorImpl();

  operator bool() const {
    // return PoolAllocatorImpl's bool operator
    return static_cast<bool>(*(*this));
  }

  // will return >= myteam.rank_n() if not a member or anyteam
  intrank_t myteam_rank_from_anyteam(intrank_t other_rank, upcxx::team &anyteam = upcxx::local_team());

  // will return >= local_team().rank_n() if not a member of local_team()
  intrank_t local_rank_from_myteam(intrank_t myteam_rank);

  /*
  // attempts to allocate from the team_rank but only if it is ready immediately
  // may return null, from common members of the local_team
  // will always return null from other ranks
  template<class U>
  global_ptr<U> tryAllocate(const std::size_t size, intrank_t myteam_rank);
   */
  /*
  // attempts to allocate from only the local_team members, but only if it is ready immediately
  // first from local_team and myteam members of anyteam (if any)
  // then attempts any remaining local_team members of anyteam (if any)
  // may return null, from common members of the local_team
  // will always return null if no members overlap
  template<class U>
  global_ptr<U> tryAllocateLocal(const std::size_t size, const upcxx::team &anyteam);

   */

  /*
          // allocate from any member of the supplied team that overlaps with this dist_object team
          // strongly preferring in this order:
          //   1) myself
          //   2) then myteam within the local_team
          //   3) local_team members from anyteam
          // then try while no allocations from the local team have succeeded:
          //     myteam not in local_team, then anyteam not in local_team
          //     with progress() and retries of local_team ranks between attempts
          //     cancelling any off node requests that may come back late
          // may return null eventually even after some blocking
          template<class U>
          upcxx::future< global_ptr<U> > AllocateRemote(const std::size_t size, const upcxx::team &anyteam);

          // allocates from a specific team member
          template<class U>
          upcxx::future< global_ptr<U> > AllocateRemote(const std::size_t size, intrank_t my_team_rank);

          // allocates from any member of myteam, possibly outside the local_team

          template<class U>
          inline upcxx::future< global_ptr<U> > AllocateRemote(const std::size_t size) {
              return AllocateRemote<U>(size, this->team().rank_me());
          }
  */
  // allocate from any member of the supplied team that overlaps with my_team and the local_team
  // strongly preferring in this order:
  //   1) myself
  //   2) then myteam within the local_team
  //   3) local_team members from anyteam
  template <class U>
  global_ptr<U> Allocate(const std::size_t size, const upcxx::team &any_team);

  // allocates from a specific team member -- must also be a member of local_team
  template <class U>
  global_ptr<U> Allocate(const std::size_t size, intrank_t myteam_rank);

  // allocates only from from me

  template <class U>
  inline global_ptr<U> Allocate(const std::size_t size) {
    return Allocate<U>(size, this->team().rank_me());
  }

  // frees an allocation (originating from any team or non-team rank)
  template <class U>
  void Free(global_ptr<U> &ptr);

  friend std::ostream &operator<<(std::ostream &os, const DistPoolAllocatorImpl &dpa) {
    os << "PoolAllocator(_DistPoolAllocatorImpl=" << (const _DistPoolAllocatorImpl *)&dpa << ")";
    return os;
  }

 protected:
  void Init();

 private:
  std::vector<PoolAllocatorImpl> m_local_copies;
};

// allocates from a specific team member -- must also be a member of local_team
template <class U>
global_ptr<U> DistPoolAllocatorImpl::Allocate(const std::size_t size, intrank_t myteam_rank) {
  // DBG_VERBOSE("DPAI::Allocate(size=", size, " myteam_rank=", myteam_rank, " this=", *this, ", using local_copy:",
  // m_local_copies[myteam_rank], "\n");
  const upcxx::team &myteam = this->team();
  assert(myteam_rank < myteam.rank_n());
  global_ptr<U> uptr;
  if (!local_team_contains(myteam[myteam_rank])) return uptr;
  global_byte_ptr ptr;
  size_t alignment = 0;
  ptr = m_local_copies[myteam_rank].Allocate(size, alignment);
  if (ptr) {
    uptr = to_global_ptr<U>((U *)ptr.local());
  }
  // DBG_VERBOSE("got uptr=", uptr, " team_rank=", myteam_rank, "\n");
  return uptr;
}

// allocate from any member of the supplied team that overlaps with my_team and the local_team
// strongly preferring in this order:
//   1) myself
//   2) then myteam within the local_team
//   3) local_team members from anyteam

template <class U>
global_ptr<U> DistPoolAllocatorImpl::Allocate(const std::size_t size, const upcxx::team &any_team) {
  const upcxx::team &myteam = this->team();
  intrank_t my_local_rank = local_team().rank_me();
  global_ptr<U> uptr;
  for (int i = 0; i < local_team().rank_n(); i++) {
    intrank_t local_rank = (i + my_local_rank) % local_team().rank_n();
    intrank_t world_rank = local_team()[local_rank];
    intrank_t myteam_rank = myteam.from_world(world_rank, rank_n());
    intrank_t any_team_rank = (any_team.id() == local_team().id()) ? local_rank : any_team.from_world(world_rank, rank_n());
    if (myteam_rank < myteam.rank_n() && any_team_rank < any_team.rank_n()) {
      uptr = Allocate<U>(size, myteam_rank);
      if (uptr) break;
    }
  }
  return uptr;
}

template <class U>
void DistPoolAllocatorImpl::Free(global_ptr<U> &ptr) {
  const upcxx::team &myteam = this->team();
  intrank_t world_rank = ptr.where();
  intrank_t team_rank = myteam.from_world(world_rank, rank_n());
  if (team_rank < myteam.rank_n()) {
    assert(ptr.is_local());
    global_byte_ptr bptr = to_global_ptr<global_byte_t>((global_byte_t *)ptr.local());
    assert(bptr);
    assert(bptr.where() == ptr.where());
    // DBG_VERBOSE("Freeing bptr=", bptr, " from PoolAllocatorImpl copy of local_team_rank=", team_rank, " rank=", world_rank, ": ",
    // m_local_copies[team_rank], "\n");
    m_local_copies[team_rank].Free(bptr);
  } else {
    // Not sure if this works, when dpa is of a different team than the destination world_rank...
    // DBG_VERBOSE("Freeing an off-team pointer by rpc: ", ptr, "\n");
    rpc_ff(
        world_rank,
        [](_DistPoolAllocatorImpl &dpa, global_ptr<U> ptr) {
          assert(ptr.is_local());
          global_byte_ptr bptr = to_global_ptr<global_byte_t>((global_byte_t *)ptr.local());
          dpa->Free(bptr);
        },
        *((_DistPoolAllocatorImpl *)this), ptr);
  }

  ptr = nullptr;
}

template <class T>
class DistPoolAllocator : public DistPoolAllocatorImpl {
  // Allocates and returns a raw pointer which is allocated from a PoolAllocatorImpl
 public:
  using PA = PoolAllocator<T>;
  using PAI = PoolAllocatorImpl;

  DistPoolAllocator(const upcxx::team &someteam = world())
      : DistPoolAllocatorImpl(someteam) {}

  DistPoolAllocator(const std::size_t numChunks, const std::size_t chunkCount, const upcxx::team &someteam = world())
      : DistPoolAllocatorImpl(numChunks * chunkCount * sizeof(T), chunkCount * sizeof(T), someteam) {}

  DistPoolAllocator(const DistPoolAllocator &copy) = delete;
  DistPoolAllocator(DistPoolAllocator &&move) = delete;

  DistPoolAllocator &operator=(const PA &copy) = delete;
  DistPoolAllocator &operator=(PA &&move) {
    DBG_VERBOSE("DistPoolAllocator = PoolAllocator(move=", &move, " ", move, "\n");
    *(*((DistPoolAllocatorImpl *)this)) = (PAI &&) move;
    assert((*this)->getTotalSize() == move.getTotalSize());
    assert((*this)->getChunkSize() == move.getChunkSize());
    this->Init();
    DBG_VERBOSE("DistPoolAllocator = PoolAllocator(move) is now", *this, "\n");
    return *this;
  }

  virtual ~DistPoolAllocator() {}

  // allocate from any valid member of the team (must overlap with this team)

  inline global_ptr<T> allocate(const upcxx::team &someteam) {
    size_t chunkSize = (*((DistPoolAllocatorImpl *)this))->getChunkSize();
    return ((DistPoolAllocatorImpl *)this)->Allocate<T>(chunkSize, someteam);
  }

  // allocate from a specific team rank

  inline global_ptr<T> allocate(intrank_t team_rank) {
    size_t chunkSize = (*((DistPoolAllocatorImpl *)this))->getChunkSize();
    return ((DistPoolAllocatorImpl *)this)->Allocate<T>(chunkSize, team_rank);
  }

  // allocate from me

  inline global_ptr<T> allocate() { return allocate(this->team().rank_me()); }

  inline void deallocate(global_ptr<T> &ptr) { ((DistPoolAllocatorImpl *)this)->Free<T>(ptr); }

  size_t get_total_count() const { return (*this)->getTotalSize() / sizeof(T); }

  size_t get_chunk_count() const { return (*this)->getChunkSize() / sizeof(T); }

  friend std::ostream &operator<<(std::ostream &os, const DistPoolAllocator &dpa) {
    os << "DistPoolAllocator(DistPoolAllocatorImpl=" << (const DistPoolAllocatorImpl &)dpa << ", PA=" << *dpa << ")";
    return os;
  }
};

};  // namespace upcxx_utils
