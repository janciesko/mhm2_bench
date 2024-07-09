#include <cassert>
#include <iostream>
#include <upcxx/upcxx.hpp>

#include "upcxx_utils/gasnetvars.hpp"

using namespace upcxx_utils;
using namespace upcxx;

int test_gasnetvars(int argc, char **argv) {
  upcxx_utils::open_dbg("test_gasnetvars");

  if (!upcxx::rank_me()) std::cout << "Found upcxx_utils version " << UPCXX_UTILS_VERSION << std::endl;

  size_t shared_heap_size, user_allocations, internal_rdzv, internal_misc;
  size_t shared_heap_size2, user_allocations2, internal_rdzv2, internal_misc2;
  barrier();
  auto status = GasNetVars::getSharedHeapInfoByBadAlloc(shared_heap_size, user_allocations, internal_rdzv, internal_misc);
  barrier(local_team());
  if (!status) {
    assert(UPCXX_VERSION < 20200800 && "Older versions do not support these gasnet queries to the shared heap");
    return 0;
  }
  assert(status && "got shared heap");

  size_t sh_total, used, sh_total2, used2;
  barrier(local_team());
  status = GasNetVars::getSharedHeapInfo(sh_total, used);
  barrier(local_team());
  INFO("status=", status, " sh_total=", sh_total, " shared_heap_size=", shared_heap_size, " user_alloc=", user_allocations,
       " internal_rdzv=", internal_rdzv, " internal_misc=", internal_misc, " used=", used,
       " tot=", user_allocations + internal_rdzv + internal_misc, "\n");
  assert(status && "got shared heap");
  assert(sh_total == shared_heap_size);
  // FIXME  assert(used == user_allocations + internal_rdzv + internal_misc);

  barrier();
  auto ptr = upcxx::new_array<int64_t>(1024);
  barrier();
  status = GasNetVars::getSharedHeapInfoByBadAlloc(shared_heap_size2, user_allocations2, internal_rdzv2, internal_misc2);
  assert(status && "got shared heap");
  assert(user_allocations2 - user_allocations >= 1024 * 8);
  assert(GasNetVars::getSharedHeapUsed() > user_allocations);
  assert(shared_heap_size2 == shared_heap_size);

  status = GasNetVars::getSharedHeapInfo(sh_total2, used2);
  assert(status && "got shared heap");
  assert(sh_total2 == shared_heap_size);
  assert(used2 >= used + 1024 * 8);

  barrier();

  upcxx::delete_array(ptr);
  barrier();
  status = GasNetVars::getSharedHeapInfoByBadAlloc(shared_heap_size2, user_allocations2, internal_rdzv2, internal_misc2);
  assert(status && "got shared heap");
  assert(user_allocations2 == user_allocations && "same as start after delete");
  assert(shared_heap_size2 == shared_heap_size);
  assert(GasNetVars::getSharedHeapSize() == shared_heap_size2);
  assert(GasNetVars::getSharedHeapUsed() >= 0);
  barrier();

  upcxx_utils::close_dbg();

  return 0;
}
