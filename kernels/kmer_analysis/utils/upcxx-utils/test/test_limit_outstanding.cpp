#include <cassert>
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>
#include <upcxx/upcxx.hpp>

using std::cout;
using std::endl;
using std::flush;
using std::shared_ptr;

using upcxx::barrier;
using upcxx::broadcast;
using upcxx::future;
using upcxx::global_ptr;
using upcxx::intrank_t;
using upcxx::local_team;
using upcxx::make_future;
using upcxx::progress;
using upcxx::rank_me;
using upcxx::rank_n;
using upcxx::rpc;
using upcxx::when_all;

#include "upcxx_utils/limit_outstanding.hpp"
#include "upcxx_utils/log.hpp"

using namespace upcxx_utils;

#ifndef mult_factor
#ifdef DEBUG
#define mult_factor 1
#else
#define mult_factor 100
#endif
#endif

int should_hang_summit(int argc, char **argv) {
  barrier();

  int iterations = 10;
  int chunk_size = 1000 * mult_factor;

  if (argc > 2) iterations = atoi(argv[1]);
  if (argc > 3) chunk_size = atoi(argv[2]);

  if (!rank_me()) cout << "iterations=" << iterations << " chunk_size=" << chunk_size << "\n" << flush;
  barrier();

  future<> fut_all = make_future();
  for (int i = 0; i < iterations; i++) {
    future<> fut_chain = make_future();
    for (int j = 0; j < chunk_size; j++) {
      auto fut = rpc((i + j) % rank_n(),
                     [](int j) {
                       std::this_thread::sleep_for(std::chrono::microseconds(10));
                       return make_future(j);
                     },
                     j)
                     .then([j](int j_ret) { assert(j == j_ret); });
      progress();
      fut_chain = when_all(fut, fut_chain);
      if (!rank_me()) {
        if (j % 100 == 0) {
          cout << j << flush;
        } else {
          cout << "." << flush;
        }
      };
    }
    if (!rank_me()) cout << "*" << flush;
    fut_all = when_all(fut_chain, fut_all);
    fut_chain.wait();
    if (!rank_me()) cout << "\n" << flush;
  }
  if (!rank_me()) cout << "All sent\n" << flush;
  fut_all.wait();
  if (!rank_me()) cout << "All returned\n" << flush;

  barrier();
  return 0;
}

void test_limit(int count, int limit = 100, int version = 0) {
  SLOG_VERBOSE("Testing limit count=", count, " limit=", limit, "\n");
  barrier();
  assert(_get_outstanding_queue().empty());
  int count_returned = 0;

  for (int i = 0; i < count; i++) {
    future<> fut;
    switch (version) {
      case 0: fut = rpc(i % rank_n(), []() { return make_future(); }); break;
      case 1: fut = rpc(i % rank_n(), []() { return; }); break;
      case 2:
        fut = rpc(
                  i % rank_n(), [](int j) { return j; }, i + rank_me())
                  .then([](int ignored) {});
        break;
      case 3:
        fut = rpc(
            i % rank_n(), [](int j) { return; }, i + rank_me());
        break;
    }
    fut = fut.then([&count_returned]() { count_returned++; });
    progress();
    limit_outstanding_futures(fut, limit).wait();
    if (limit >= 0)
      assert(_get_outstanding_queue().size() <= limit);
    else
      assert(_get_outstanding_queue().size() <= local_team().rank_n() * 2);
    assert(count_returned <= count);
  }

  flush_outstanding_futures();
  assert(flush_outstanding_futures_async().is_ready());

  assert(count_returned == count);

  assert(_get_outstanding_queue().empty());
  barrier();
  SLOG_VERBOSE("Done limit count=", count, " limit=", limit, "\n");
}

int test_limit_outstanding(int argc, char **argv) {
  open_dbg("test_limit_outstanding");
  barrier();

  for (int ver = 0; ver < 4; ver++) {
    test_limit(10, -1, ver);
    test_limit(100, -1, ver);
    test_limit(10, 0, ver);
    test_limit(100, 0, ver);
    test_limit(100, 10, ver);
#ifndef DEBUG
    test_limit(200, 100, ver);
    test_limit(1000, -1, ver);
    test_limit(1000, 0, ver);
    test_limit(1000, 10, ver);
    test_limit(10000, 10, ver);
    test_limit(1000, 100, ver);
    test_limit(10000, 100, ver);
#endif
  }

  // should_hang_summit(argc, argv);

  barrier();
  close_dbg();
  return 0;
}
