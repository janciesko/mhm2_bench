#pragma once
#include <cassert>
#include <deque>
#include <upcxx/upcxx.hpp>

namespace upcxx_utils {

// Limit Outstanding Futures is used instead of a future chain where too many rpcs could accumulate on the queue
// by default the upper limit is local_team().rank_n() * 2
// also by default there is a singleton queue, though that can be overridden by supplying a different LimitedFutureQueue
//
// usage:
//
// for(...) {
//   upcxx::future<> fut = rpc(...);
//   progress(); // optional but recommended
//   limit_outstanding_futures(fut).wait();
// }
// flush_outstanding_futures(); // or later in the code: flush_outstanding_futures_async().wait()
//

using LimitedFutureQueue = std::deque<upcxx::future<> >;

LimitedFutureQueue &_get_outstanding_queue();

upcxx::future<> collapse_outstanding_futures(int limit = -1,
                                             LimitedFutureQueue &outstanding_queue = upcxx_utils::_get_outstanding_queue(),
                                             int max_check = 8);

void add_outstanding_future(upcxx::future<> fut, LimitedFutureQueue &outstanding_queue);

upcxx::future<> limit_outstanding_futures(int limit = -1,
                                          LimitedFutureQueue &outstanding_queue = upcxx_utils::_get_outstanding_queue());

upcxx::future<> limit_outstanding_futures(upcxx::future<> fut, int limit = -1,
                                          LimitedFutureQueue &outstanding_queue = upcxx_utils::_get_outstanding_queue());

upcxx::future<> flush_outstanding_futures_async(LimitedFutureQueue &outstanding_queue = upcxx_utils::_get_outstanding_queue());

void flush_outstanding_futures(LimitedFutureQueue &outstanding_queue = upcxx_utils::_get_outstanding_queue());

template <typename Result, typename Future>
upcxx::future<> assign_oustanding_future_result(Result &res, Future fut, int limit = -1,
                                                LimitedFutureQueue &outstanding_queue = upcxx_utils::_get_outstanding_queue()) {
  upcxx::future<> res_fut = fut.then([&res](const Result &val) { res = val; });
  return limit_outstanding_futures(res_fut, limit);
}

};  // namespace upcxx_utils
