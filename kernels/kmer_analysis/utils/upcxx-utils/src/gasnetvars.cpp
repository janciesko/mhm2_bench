#include "upcxx_utils/gasnetvars.hpp"

#include <exception>
#include <sstream>
#include <string>
#include <upcxx/upcxx.hpp>

#include "upcxx_utils/log.hpp"

#undef NDEBUG
#undef __OPTIMIZE__

#include <gasnet.h>

using std::string;
using std::to_string;

namespace upcxx_utils {

size_t GasNetVars::getAMMaxArgs() { return gasnet_AMMaxArgs(); }
size_t GasNetVars::getAMMaxMedium() { return gasnet_AMMaxMedium(); }
size_t GasNetVars::getAMMaxLongRequest() { return gasnet_AMMaxLongRequest(); }
size_t GasNetVars::getAMMaxLongReply() { return gasnet_AMMaxLongReply(); }
size_t GasNetVars::getMaxLocalSegmentSize() { return gasnet_getMaxLocalSegmentSize(); }
size_t GasNetVars::getMaxGlobalSegmentSize() { return gasnet_getMaxGlobalSegmentSize(); }
size_t GasNetVars::getMyNode() { return gasnet_mynode(); }
size_t GasNetVars::getNodes() { return gasnet_nodes(); }
char *GasNetVars::getEnv(const char *name) { return gasnet_getenv(name); }

size_t multiply_unit(size_t val, char unit) {
  switch (unit) {
    case 'E':
    case 'e': return val * ONE_EB;
    case 'T':
    case 't': return val * ONE_TB;
    case 'G':
    case 'g': return val * ONE_GB;
    case 'M':
    case 'm': return val * ONE_MB;
    case 'K':
    case 'k': return val * ONE_KB;
    case 'B':
    case 'b': return val * ONE_B;
    default: throw std::invalid_argument(string("Unit is invalid: ") + to_string(unit));
  }
}

static int64_t &_getSharedHeapSizeCache() {
  static int64_t _ = (UPCXX_SPEC_VERSION < 20200800L) ? -1 : -2;
  return _;
}

int64_t GasNetVars::getSharedHeapSize() {
  int64_t &_ = _getSharedHeapSizeCache();
#if UPCXX_VERSION >= 20201105
  _ = upcxx::shared_segment_size();
#else
  if (_ == -2) {
    size_t shared_heap_size, user_allocations, internal_rdzv, internal_misc;
    if (getSharedHeapInfoByBadAlloc(shared_heap_size, user_allocations, internal_rdzv, internal_misc)) {
      _ = shared_heap_size;
    }
  }
#endif
  return _;
}

int64_t GasNetVars::getSharedHeapUsed() {
#if UPCXX_VERSION >= 20201105
  return upcxx::shared_segment_used();
#else
  if (UPCXX_SPEC_VERSION < 20200800L) return -1;
  size_t shared_heap_size, user_allocations, internal_rdzv, internal_misc;
  if (getSharedHeapInfoByBadAlloc(shared_heap_size, user_allocations, internal_rdzv, internal_misc)) {
    int64_t &cached_size = _getSharedHeapSizeCache();
    if (cached_size < 0) cached_size = shared_heap_size;
    return user_allocations + internal_rdzv + internal_misc;
  } else {
    return -1;
  }
#endif
}

bool GasNetVars::getSharedHeapInfo(size_t &shared_heap_size, size_t &used_size) {
  used_size = getSharedHeapUsed();
  shared_heap_size = getSharedHeapSize();
  return shared_heap_size >= 0;
}
bool GasNetVars::getSharedHeapInfoByBadAlloc(size_t &shared_heap_size, size_t &user_allocations, size_t &internal_rdzv,
                                             size_t &internal_misc) {
  size_t user_objs, rdzv_objs, misc_objs;
  return getSharedHeapInfoByBadAlloc(shared_heap_size, user_objs, user_allocations, rdzv_objs, internal_rdzv, misc_objs,
                                     internal_misc);
}
bool GasNetVars::getSharedHeapInfoByBadAlloc(size_t &shared_heap_size, size_t &user_objs, size_t &user_allocations,
                                             size_t &rdzv_objs, size_t &internal_rdzv, size_t &misc_objs, size_t &internal_misc) {
  if (UPCXX_SPEC_VERSION < 20200800L) return false;
  bool success = false;
  try {
    shared_heap_size = 0;
    user_allocations = 0;
    internal_rdzv = 0;
    internal_misc = 0;
    auto ptr = upcxx::new_array<char>(1ULL << 60);
    WARN("Got unrealistic pointer: ", ptr, "\n");
    /*
  to produce this nice human readable output :

upcxx::bad_shared_alloc: UPC++ shared heap is out of memory on process 0
 inside upcxx::new_array while trying to allocate 1152921504606846984 more bytes
 Local shared heap statistics:
  Shared heap size on process 0:             128 MB
  User allocations:               0 objects, 0
  Internal rdzv buffers:          0 objects, 0
  Internal misc buffers:          0 objects, 0

 You may need to request a larger shared heap with `upcxx-run -shared-heap` or $UPCXX_SHARED_HEAP_SIZE.

        */
  } catch (std::bad_alloc &e) {
    success |= parseBadAlloc(e, shared_heap_size, user_objs, user_allocations, rdzv_objs, internal_rdzv, misc_objs, internal_misc);
  }
  assert(success);
  DBG("SharedHeapSize=", get_size_str(shared_heap_size), " UserAllocations=", get_size_str(user_allocations),
      " RDZV=", get_size_str(internal_rdzv), " MISC=", get_size_str(internal_rdzv), "\n");
  return success;
}

bool GasNetVars::parseBadAlloc(std::bad_alloc &e, size_t &shared_heap_size, size_t &user_allocations, size_t &internal_rdzv,
                               size_t &internal_misc) {
  size_t user_objects, internal_rdzv_objects, internal_misc_objects;
  return parseBadAlloc(e, shared_heap_size, user_objects, user_allocations, internal_rdzv_objects, internal_rdzv,
                       internal_misc_objects, internal_misc);
}
bool GasNetVars::parseBadAlloc(std::bad_alloc &e, size_t &shared_heap_size, size_t &user_objects, size_t &user_allocations,
                               size_t &internal_rdzv_objects, size_t &internal_rdzv, size_t &internal_misc_objects,
                               size_t &internal_misc) {
  assert(&shared_heap_size != &user_allocations);
  assert(&user_allocations != &internal_rdzv);
  assert(&internal_rdzv != &internal_misc);
  bool success = true;
  auto msg = string(e.what());
  DBG_VERBOSE("Got information from memory exception: ", msg, "\n");
  size_t val_pos = msg.find("Shared heap size ");
  success &= val_pos != string::npos;
  size_t colon_pos = msg.find(":", val_pos);
  success &= colon_pos != string::npos && colon_pos > val_pos;
  size_t after_int = 0;
  shared_heap_size = stoll(msg.substr(colon_pos + 1), &after_int, 10);
  assert(shared_heap_size > 0);
  shared_heap_size = multiply_unit(shared_heap_size, msg[colon_pos + 1 + after_int + 1]);

  val_pos = msg.find("User alloc", colon_pos + 1 + after_int + 2);
  success &= val_pos != string::npos;
  colon_pos = msg.find(":", val_pos);
  success &= colon_pos != string::npos && colon_pos > val_pos;
  size_t obj_count_pos = msg.find_first_of("0123456789", colon_pos);
  size_t next_space_pos = msg.find_first_of(" ", obj_count_pos);
  user_objects = stoll(msg.substr(obj_count_pos, next_space_pos - obj_count_pos));
  colon_pos = msg.find(",", colon_pos);
  success &= colon_pos != string::npos && colon_pos > val_pos;
  after_int = 0;
  user_allocations = stoll(msg.substr(colon_pos + 1), &after_int, 10);
  if (user_allocations > 0) user_allocations = multiply_unit(user_allocations, msg[colon_pos + 1 + after_int + 1]);

  val_pos = msg.find("Internal rdzv", colon_pos + 1 + after_int + 2);
  success &= val_pos != string::npos;
  colon_pos = msg.find(":", val_pos);
  success &= colon_pos != string::npos && colon_pos > val_pos;
  obj_count_pos = msg.find_first_of("0123456789", colon_pos);
  next_space_pos = msg.find_first_of(" ", obj_count_pos);
  internal_rdzv_objects = stoll(msg.substr(obj_count_pos, next_space_pos - obj_count_pos));
  colon_pos = msg.find(",", colon_pos);
  success &= colon_pos != string::npos && colon_pos > val_pos;
  after_int = 0;
  internal_rdzv = stoll(msg.substr(colon_pos + 1), &after_int, 10);
  if (internal_rdzv > 0) internal_rdzv = multiply_unit(internal_rdzv, msg[colon_pos + 1 + after_int + 1]);

  val_pos = msg.find("Internal misc", colon_pos + 1 + after_int + 2);
  success &= val_pos != string::npos;
  colon_pos = msg.find(":", val_pos);
  success &= colon_pos != string::npos && colon_pos > val_pos;
  success &= colon_pos != string::npos && colon_pos > val_pos;
  obj_count_pos = msg.find_first_of("0123456789", colon_pos);
  next_space_pos = msg.find_first_of(" ", obj_count_pos);
  internal_misc_objects = stoll(msg.substr(obj_count_pos, next_space_pos - obj_count_pos));
  colon_pos = msg.find(",", colon_pos);
  success &= colon_pos != string::npos && colon_pos > val_pos;
  after_int = 0;
  size_t internal_misc2 = stoll(msg.substr(colon_pos + 1), &after_int, 10);
  if (internal_misc2 > 0) internal_misc2 = multiply_unit(internal_misc2, msg[colon_pos + 1 + after_int + 1]);
  internal_misc = internal_misc2;
  success &= shared_heap_size > 0;
  return success;
}

std::string GasNetVars::getUsedShmMsg() {
  size_t shared_heap_size, user_objs, user_allocations, rdzv_objs, internal_rdzv, misc_objs, internal_misc;
  auto status = getSharedHeapInfoByBadAlloc(shared_heap_size, user_objs, user_allocations, rdzv_objs, internal_rdzv, misc_objs,
                                            internal_misc);
  return getUsedShmMsg(shared_heap_size, user_objs, user_allocations, rdzv_objs, internal_rdzv, misc_objs, internal_misc);
}
std::string GasNetVars::getUsedShmMsg(const size_t shared_heap_size, const size_t user_objs, const size_t user_allocations,
                                      const size_t rdzv_objs, const size_t internal_rdzv, const size_t misc_objs,
                                      const size_t internal_misc) {
  std::ostringstream oss;
  oss << "GasNetVars: ";
  upcxx_utils::_logger_recurse(oss, "SharedHeapSize=", get_size_str(shared_heap_size), " UserAllocations=", user_objs, " ",
                               get_size_str(user_allocations), " RDZV=", rdzv_objs, " ", get_size_str(internal_rdzv),
                               " MISC=", misc_objs, " ", get_size_str(internal_misc));
  return oss.str();
}

bool TrackedSharedMemory::check(size_t proposed_allocation, bool force_update, float min_pct) {
  if (total == 0) {
    // no shared heap inspection is available
    assert(UPCXX_SPEC_VERSION < 20200800L);
    return true;
  }
  size_t test_min_free = (min_pct == 0.0) ? min_free : min_pct * total;
  assert(proposed_allocation < total * 4 / 5 && "proposed is <80% of total!");
  // use caution and double the new allocation for first test
  size_t new_used = estimated_used + proposed_allocation * 2;
  if (force_update || total < new_used + test_min_free) {
    // refresh with a new measurement
    update();
  }
  new_used = estimated_used + proposed_allocation;
  if (total < new_used + test_min_free) {
    return false;
  }
#ifdef UPCXX_UTILS_CHECK_WITH_TEST
  if (!test(proposed_allocation * 1.2)) {
    return false;
  }
#endif
  estimated_used += proposed_allocation;
  return true;
}

void TrackedSharedMemory::wait(size_t proposed_allocation, bool force_update, float min_pct) {
  int iter = 0;
  do {
    if (check(proposed_allocation, force_update, min_pct)) return;
    upcxx::progress();
    upcxx::discharge();
    upcxx::progress();
  } while (iter++ < 1000);
}

bool TrackedSharedMemory::test(size_t proposed_allocation) {
  upcxx::global_ptr<size_t> ptr;
  try {
    ptr = upcxx::new_array<size_t>(proposed_allocation / sizeof(size_t));
  } catch (std::bad_alloc &e) {
    size_t shared_heap_size, user_objs, user_allocations, rdzv_objs, internal_rdvz, misc_objs, internal_misc;
    if (GasNetVars::parseBadAlloc(e, shared_heap_size, user_objs, user_allocations, rdzv_objs, internal_rdvz, misc_objs,
                                  internal_misc)) {
      estimated_used.store(user_allocations + internal_rdvz + internal_rdvz);
      LOG("Failed test of ", get_size_str(proposed_allocation), " ",
          GasNetVars::getUsedShmMsg(shared_heap_size, user_objs, user_allocations, rdzv_objs, internal_rdvz, misc_objs,
                                    internal_misc),
          "\n");
    }
    assert(!ptr);
    return false;
  }
  assert(ptr);
  upcxx::delete_array(ptr);
  return true;
}

void TrackedSharedMemory::update() {
  size_t newval;
  if (GasNetVars::getSharedHeapInfo(total, newval) && newval >= 0) {
    auto oldval = estimated_used.load();
    if (oldval != newval) {
      estimated_used.store(newval);
      // FIXME reduce verbosity
      LOG("Updated used to ", get_size_str(newval), " from estimated ", get_size_str(oldval), "\n");
    }
  }
}

};  // namespace upcxx_utils
