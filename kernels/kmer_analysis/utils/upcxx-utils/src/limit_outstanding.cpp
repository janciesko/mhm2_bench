#include <cassert>
#include <upcxx/upcxx.hpp>
#include <upcxx_utils/limit_outstanding.hpp>
#include <upcxx_utils/log.hpp>

using upcxx::when_all;
using LimitedFutureQueue = std::deque<upcxx::future<> >;

LimitedFutureQueue &upcxx_utils::_get_outstanding_queue() {
  static LimitedFutureQueue outstanding_queue;
  return outstanding_queue;
}

upcxx::future<> upcxx_utils::collapse_outstanding_futures(int limit, LimitedFutureQueue &outstanding_queue, int max_check) {
  // DBG("limit=", limit, " outstanding=", outstanding_queue.size(), " max_check=", max_check, "\n");
  // pop any ready futures at the front of the queue presuming that FIFO ordering is dominant
  while (outstanding_queue.size() && outstanding_queue.front().is_ready()) outstanding_queue.pop_front();
  upcxx::future<> returned_future = upcxx::make_future();
  if (outstanding_queue.size() >= limit) {
    // reduce to limit when over
    // collapsing first queued futures into the returned future, presuming that FIFO ordering is dominant
    while (outstanding_queue.size() > limit) {
      auto fut = outstanding_queue.front();
      outstanding_queue.pop_front();
      if (!fut.is_ready()) returned_future = upcxx::when_all(fut, returned_future);
    }
    // DBG("limit=", limit, " outstanding=", outstanding_queue.size(), " max_check=", max_check, "\n");
    if (limit == 0) {
      assert(outstanding_queue.empty());
    } else {
      assert(outstanding_queue.size() <= limit);
      // check the queue for any futures that are now ready and return it
      static auto fast_random_lambda = [](uint32_t a) {
        static uint32_t seed = upcxx::rank_me() + 2;
        a += seed;
        // google fast hash for a quick pseudo random number generator
        a = (a ^ 61) ^ (a >> 16);
        a = a + (a << 3);
        a = a ^ (a >> 4);
        a = a * 0x27d4eb2d;
        a = a ^ (a >> 15);
        seed = a;
        return seed;
      };
      bool check_randomly = true;
      auto sz = outstanding_queue.size();
      if (max_check >= 3 * sz / 4) {
        check_randomly = false;
        if (max_check > sz) max_check = sz;
      }
      int i = 0;
      while (i < max_check && !returned_future.is_ready() && i < sz) {
        // randomly check for a ready future in the queue to swap with. queue size will not change
        int idx = check_randomly ? fast_random_lambda(i) % sz : i;
        auto &test_fut = outstanding_queue[idx];
        if (test_fut.is_ready()) {
          std::swap(test_fut, returned_future);
          assert(returned_future.is_ready());
          break;
        }
        i++;
      }
    }
  }
  // DBG("limit=", limit, " outstanding=", outstanding_queue.size(), " max_check=", max_check, ", ret=", returned_future.is_ready(),
  // "\n");
  return returned_future;
}

void upcxx_utils::add_outstanding_future(upcxx::future<> fut, LimitedFutureQueue &outstanding_queue) {
  if (!fut.is_ready()) outstanding_queue.push_back(fut);
}

upcxx::future<> upcxx_utils::limit_outstanding_futures(int limit, LimitedFutureQueue &outstanding_queue) {
  return collapse_outstanding_futures(limit, outstanding_queue, std::min(limit / 4, (int)16));
}

upcxx::future<> upcxx_utils::limit_outstanding_futures(upcxx::future<> fut, int limit, LimitedFutureQueue &outstanding_queue) {
  if (limit < 0) limit = upcxx::local_team().rank_n() * 2;
  // DBG("limit=", limit, " outstanding=", outstanding_queue.size(), "\n");
  if (limit == 0) {
    if (outstanding_queue.empty()) return fut;
    return upcxx::when_all(collapse_outstanding_futures(limit, outstanding_queue), fut);
  }
  if (fut.is_ready()) {
    if (outstanding_queue.size() <= limit) return fut;
  } else {
    outstanding_queue.push_back(fut);
  }
  return limit_outstanding_futures(limit, outstanding_queue);
}

upcxx::future<> upcxx_utils::flush_outstanding_futures_async(LimitedFutureQueue &outstanding_queue) {
  // DBG_VERBOSE(" outstanding=", outstanding_queue.size(), "\n");
  upcxx::future<> all_fut = collapse_outstanding_futures(0, outstanding_queue);
  return all_fut;
}

void upcxx_utils::flush_outstanding_futures(LimitedFutureQueue &outstanding_queue) {
  assert(!upcxx::in_progress());
  while (!outstanding_queue.empty()) flush_outstanding_futures_async(outstanding_queue).wait();
  assert(outstanding_queue.empty());
}
