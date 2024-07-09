#include <cassert>
#include <iostream>
#include <stdexcept>
#include <upcxx/upcxx.hpp>

#include "upcxx_utils/Allocators.hpp"
#include "upcxx_utils/timers.hpp"

using namespace upcxx;
using namespace upcxx_utils;

#ifndef mult_factor
#ifdef DEBUG
#define mult_factor 1
#else
#define mult_factor 100
#endif
#endif

int test_allocators(int argc, char **argv) {
  upcxx_utils::open_dbg("test_allocators");

  return 0;  // FIXME
  SLOG("Found upcxx_utils version ", UPCXX_UTILS_VERSION, "\n");

  {
    BarrierTimer bt("First stage");
    size_t num_chunks = 1000 * mult_factor, chunk_count = 100;

    SLOG("PoolAllocator num_chunks=", num_chunks, " chunk_count=", chunk_count, "\n");
    upcxx_utils::PoolAllocator<int> pa(num_chunks, chunk_count);

    global_ptr<int> first, last, tmp;
    for (int i = 0; i < num_chunks; i++) {
      global_ptr<int> ptr = pa.allocate();
      assert(ptr && "ptr is not null");
      if (!first) first = ptr;
      last = ptr;
      assert(ptr.is_local());
      assert((ptr + chunk_count - 1).is_local());
      int *lptr = ptr.local();
      for (int j = 0; j < chunk_count; j++) {
        lptr[j] = i * chunk_count + j;
      }
    }
    if (pa.allocate()) {
      DIE("Pool should be empty");
    }
    tmp = last;
    pa.deallocate(tmp);
    assert(!tmp);
    tmp = pa.allocate();
    assert(tmp == last);

    tmp = first;
    pa.deallocate(tmp);
    assert(!tmp);
    tmp = pa.allocate();
    assert(tmp == first);
  }

  {
    BarrierTimer bt("Second stage");
    int rounds = 10 * mult_factor;
    {
      SLOG("DistPoolAllocator dummy1 (empty)\n");
      upcxx_utils::DistPoolAllocator<int> dummy1(upcxx::world());  // empty dist_object
      assert(!dummy1);

      {
        BarrierTimer bt2("PoolAllocator init");
        int num_chunks = rounds * upcxx::local_team().rank_n(), chunk_count = 5;
        SLOG("PoolAllocator alloc1 (num_chunks=", num_chunks, " chunk_count=", chunk_count, ")\n");
        upcxx_utils::PoolAllocator<int> alloc1(num_chunks, chunk_count);
        DBG("Constructed alloc1\n");
        // ensure dist object can be move-assigned with a different allocator.
        dummy1 = std::move(alloc1);
        DBG("*dummy1 - ", *dummy1, "\n");
        assert(dummy1);
        DBG("Assertion\n");
      }
      DBG("Destructed temps\n");
      // and remains active at least after the move-assigned PoolAllocator is destroyed
      assert(dummy1);
      SLOG("dummy1 should be allocated with alloc1: ", dummy1, "\n");

      barrier();
      SLOG("Starting allocations from dummy1=", dummy1, "\n");
      std::vector<global_ptr<int> > allocated;

      // force an imbalance
      if (local_team().rank_me() % 2 == 0) barrier(local_team());

      SLOG("Starting allocations first from even ranks\n");
      for (int i = 0; i < rounds; i++) {
        for (intrank_t r = 0; r < upcxx::local_team().rank_n(); r++) {
          auto ptr = dummy1.allocate(r);
          if (ptr.is_null()) DIE("Could not allocate! i=", i, " r=", r, " of ", upcxx::local_team().rank_n(), "\n");
          assert(ptr.where() == r);
          allocated.push_back(ptr);
        }
      }
      SLOG("Allocated across ranks, \n");
      for (auto &ptr : allocated) {
        if (ptr.is_local()) {
          int *p = ptr.local();
          for (int i = 0; i < dummy1.get_chunk_count(); i++) {
            p[i] = upcxx::local_team().rank_me() + i * upcxx::local_team().rank_n();
          }
        } else {
          assert(false && "allocated pointers are not local!\n");
        }
      }
      SLOG("asserting data across ranks, \n");
      for (auto &ptr : allocated) {
        if (ptr.is_local()) {
          int *p = ptr.local();
          for (int i = 0; i < dummy1.get_chunk_count(); i++) {
            assert(p[i] == upcxx::local_team().rank_me() + i * upcxx::local_team().rank_n() && "memory corruption");
          }
        } else {
          assert(false && "allocated pointers are not local!\n");
        }
      }

      // forced imbalance
      if (local_team().rank_me() % 2 == 1) barrier(local_team());
      SLOG("Finished imbalanced allocations\n");

      barrier();

      // everyone return half
      SLOG("deallocating disproportionately\n");
      int returnsome = (allocated.size() + 2 - 1) / 2;
      while (returnsome--) {
        dummy1.deallocate(allocated.back());
        allocated.pop_back();
      }
      barrier();

      SLOG("Even node reallocate the available blocks\n");
      // now have every other rank allocate the remaining across the local team leaving an imbalance
      if (upcxx::local_team().rank_n() % 2 == 0 && upcxx::local_team().rank_me() % 2 == 0) {
        for (int i = 0; i < 10; i++) {
          for (intrank_t r = 0; r < upcxx::local_team().rank_n(); r++) {
            auto ptr = dummy1.allocate(local_team());
            if (ptr.is_null()) DIE("Could not allocate! r=", r, " of ", upcxx::local_team().rank_n(), "\n");
            assert(local_team_contains(ptr.where()));
            allocated.push_back(ptr);
          }
        }
      }
      // no need to free.
      barrier();  // necessary to keep dummy1 in scope for other ranks
    }
    SLOG("Destroyed.\n");
    size_t num_chunks = 1000, chunk_count = 5;
    upcxx_utils::DistPoolAllocator<int> pa(num_chunks, chunk_count, upcxx::local_team());
    std::vector<upcxx::global_ptr<int> > myptrs;
    myptrs.reserve(num_chunks + 1);
    SLOG("allocating with contention num_chunks=", num_chunks, " chunk_count=", chunk_count,
         " total_size=", upcxx_utils::get_size_str(upcxx::local_team().rank_n() * num_chunks * chunk_count * sizeof(int)), "\n");

    for (upcxx::intrank_t rank = 0; rank < upcxx::local_team().rank_n(); rank++) {
      SLOG("Allocating from rank ", rank, "\n");
      upcxx::barrier();

      for (int c = upcxx::local_team().rank_me(); c < num_chunks; c += upcxx::local_team().rank_n()) {
        auto ptr = pa.allocate(rank);
        assert(ptr);
        myptrs.push_back(ptr);
      }

      upcxx::barrier();
    }
    upcxx::barrier();
    SLOG("writing to allocated memory\n");
    // populate them all
    int i = 0;
    for (auto ptr : myptrs) {
      int *block = ptr.local();
      for (int j = 1; j < chunk_count; j++) {
        block[j] = (i * chunk_count + j) * upcxx::local_team().rank_n() + upcxx::local_team().rank_me();
      }
      i++;
    }
    upcxx::barrier();
    SLOG("validating allocated memory\n");
    // validate no wrong values
    i = 0;
    for (auto ptr : myptrs) {
      int *block = ptr.local();
      for (int j = 1; j < chunk_count; j++) {
        assert(block[j] % upcxx::local_team().rank_n() == upcxx::local_team().rank_me());
        assert(block[j] == (i * chunk_count + j) * upcxx::local_team().rank_n() + upcxx::local_team().rank_me());
      }
      i++;
    }
    upcxx::barrier();
    SLOG("deallocating them all\n");
    // deallocate them all
    for (auto ptr : myptrs) {
      pa.deallocate(ptr);
    }
    upcxx::barrier();  // keep all pool allocators in scope for other ranks
  }

  upcxx_utils::close_dbg();
  return 0;
}
