#pragma once
#include <atomic>
#include <cstddef>
#include <exception>
#include <upcxx/upcxx.hpp>

#include "upcxx_utils/log.hpp"

namespace upcxx_utils {
class GasNetVars {
 public:
  static size_t getAMMaxArgs();
  static size_t getAMMaxMedium();
  static size_t getAMMaxLongRequest();
  static size_t getAMMaxLongReply();
  static size_t getMaxLocalSegmentSize();
  static size_t getMaxGlobalSegmentSize();
  static size_t getMyNode();
  static size_t getNodes();
  static char *getEnv(const char *name);
  static int64_t getSharedHeapSize();
  static int64_t getSharedHeapUsed();
  static bool getSharedHeapInfo(size_t &shared_heap_size, size_t &used_size);
  static bool getSharedHeapInfoByBadAlloc(size_t &shared_heap_size, size_t &user_allocations, size_t &internal_rdzv,
                                          size_t &internal_misc);
  static bool getSharedHeapInfoByBadAlloc(size_t &shared_heap_size, size_t &user_objs, size_t &user_allocations, size_t &rdzv_objs,
                                          size_t &internal_rdzv, size_t &misc_objs, size_t &internal_misc);
  static bool parseBadAlloc(std::bad_alloc &e, size_t &shared_heap_size, size_t &user_allocations, size_t &internal_rdzv,
                            size_t &internal_misc);
  static bool parseBadAlloc(std::bad_alloc &e, size_t &shared_heap_size, size_t &user_objects, size_t &user_allocations,
                            size_t &internal_rdzv_objects, size_t &internal_rdzv, size_t &internal_misc_objects,
                            size_t &internal_misc);
  static std::string getUsedShmMsg();
  static std::string getUsedShmMsg(const size_t shared_heap_size, const size_t user_objs, const size_t user_allocations,
                                   const size_t rdzv_objs, const size_t internal_rdzv, const size_t misc_objs,
                                   const size_t internal_misc);
};

class TrackedSharedMemory {
  // A class that helps decide when shared help allocations is nearing exhaustion
  // (say from upcxx::new_array or RendezVous allocations or ...)
  size_t total;
  size_t min_free;
  std::atomic<size_t> estimated_used;

 public:
  static TrackedSharedMemory &singleton() {
    static TrackedSharedMemory _tsm;
    return _tsm;
  }

  TrackedSharedMemory(float min_pct = 0.25)
      : total{}
      , min_free{}
      , estimated_used{} {
    update();
    min_free = total * min_pct;
  }

  bool check(size_t proposed_allocation, bool force_update = false, float min_pct = 0.0);

  void wait(size_t proposed_allocation, bool force_update = false, float min_pct = 0.0);

  template <typename Func>
  static upcxx::future<> delay_execution(size_t required_allocation, Func &&func, bool force_update = false, float min_pct = 0.0,
                                         int max_iter = 100000, int iter = 0) {
    using return_t = typename std::invoke_result<Func>::type;
    static_assert(std::is_void<return_t>::value, "void is the required return type for delay_execution");
    if (iter < max_iter && !singleton().check(required_allocation, force_update, min_pct)) {
      iter++;
      return upcxx::current_persona().lpc([required_allocation, func = std::move(func), force_update, min_pct, max_iter, iter]() {
        return delay_execution(required_allocation, func, force_update, min_pct, max_iter, iter);
      });
    }
    if (iter) LOG("delayed for ", iter, " attempts\n");  // FIXME verbosity
    func();
    return upcxx::make_future();
  };

  bool test(size_t proposed_allocation);

 protected:
  void update();
};  // namespace upcxx_utils

// experimental functions that retry rpcs if they should throw exception as backpressure
template <typename Func, typename... Args>
void safe_rpc_ff(const upcxx::team &tm, upcxx::intrank_t rank, Func &&func, Args &&...args) {
  int iter = 0;
  do {
    try {
      upcxx::rpc_ff(tm, rank, func, args...);
    } catch (std::bad_alloc &e) {
      if (iter++ > 100) throw e;
      upcxx::progress();
      upcxx::discharge();
      upcxx::progress();
    }
  } while (true);
};

template <typename Func, typename... Args>
auto safe_rpc(const upcxx::team &tm, upcxx::intrank_t rank, Func &&func, Args &&...args) {
  int iter = 0;
  do {
    try {
      return upcxx::rpc(tm, rank, func, args...);
    } catch (std::bad_alloc &e) {
      if (iter++ > 100) throw e;
      upcxx::progress();
      upcxx::discharge();
      upcxx::progress();
    }
  } while (true);
};
};  // namespace upcxx_utils
