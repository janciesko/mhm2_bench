#include <unistd.h>

#include <atomic>
#include <iostream>
#include <string>
#include <upcxx/upcxx.hpp>

#include "upcxx_utils/thread_pool.hpp"
#include "upcxx_utils/timers.hpp"
#include "upcxx_utils/version.h"

using upcxx::future;
using upcxx::promise;
using upcxx_utils::ThreadPool;

void test_threads(int num_threads) {
  future<> all_done = make_future();
  {
    DBG("Construct destruct\n");
    ThreadPool tp(num_threads);
  }
  barrier();
  int iterations = 250;
  std::atomic<size_t> sum(0);
  {
    DBG("construct send jobs and join\n");
    ThreadPool tp(num_threads);
    for (int i = 0; i < iterations; i++) {
      auto fut = tp.enqueue(
          [&sum, i, num_threads](int i_arg) {
            assert(i == i_arg);
            if (num_threads == 1) assert(i == sum.load() && "one thread is serial");
            sum++;
          },
          i);
      all_done = when_all(all_done, fut);
    }
    all_done.wait();
    assert(sum.load() == iterations && "all tasks completed after wait");
  }
  assert(all_done.is_ready());
  barrier();

  sum.store(0);
  {
    DBG("construct send jobs and join\n");
    ThreadPool tp(num_threads);
    for (int i = 0; i < iterations; i++) {
      auto fut = tp.enqueue(
          [&sum, i, num_threads](int i_arg) {
            assert(i == i_arg);
            if (num_threads == 1) assert(i == sum.load() && "one thread is serial");
            sum++;
          },
          i);
      all_done = when_all(all_done, fut);
    }
    // no wait necessary here
  }

  assert(sum.load() == iterations && "all tasks completed after deconstruct");
  all_done.wait();  // futures may not be is_ready, however
  DBG("all tasks are complete.\n");
  barrier();

  // enqueue with return
  sum.store(0);
  {
    DBG("construct send jobs and join with return\n");
    ThreadPool tp(num_threads);
    for (int i = 0; i < iterations; i++) {
      auto fut = tp.enqueue([&sum, i, num_threads]() {
                     if (num_threads == 1) assert(i == sum.load() && "one thread is serial");
                     sum++;
                     return i;
                   }).then([i](int v) { assert(i == v && "return v == i"); });
      all_done = when_all(all_done, fut);
    }
    all_done.wait();
    assert(sum.load() == iterations && "all tasks completed after wait");
  }
  assert(all_done.is_ready());
  barrier();

  sum.store(0);
  {
    DBG("construct send jobs and join with return\n");
    ThreadPool tp(num_threads);
    for (int i = 0; i < iterations; i++) {
      auto fut = tp.enqueue([&sum, i, num_threads]() {
                     if (num_threads == 1) assert(i == sum.load() && "one thread is serial");
                     sum++;
                     return i;
                   }).then([i](int v) { assert(i == v && "return v == i"); });
      all_done = when_all(all_done, fut);
    }
    // no wait necessary here
  }

  assert(sum.load() == iterations && "all tasks completed after destruct");
  all_done.wait();  // futures may not be however
  DBG("all tasks are complete.\n");
  barrier();

  sum.store(0);
  {
    promise prom1(1);
    DBG("construct within progress callback serially send jobs and join\n");
    ThreadPool tp(num_threads);
    for (int i = 0; i < iterations; i++) {
      all_done = when_all(all_done, prom1.get_future()).then([&sum, &tp, i, num_threads]() {
        return tp.enqueue([&sum, i, num_threads]() {
          if (num_threads == 1) assert(i == sum.load() && "one thread is serial");
          sum++;
        });
      });
    }
    prom1.fulfill_anonymous(1);
    all_done.wait();
    assert(sum.load() == iterations && "all tasks completed after wait");
  }

  barrier();

  sum.store(0);
  {
    DBG("construct within progress callback\n");
    promise prom1(1);
    ThreadPool tp(num_threads);
    for (int i = 0; i < iterations; i++) {
      auto fut = prom1.get_future().then([&sum, &tp, i, num_threads]() { return tp.enqueue([&sum, i, num_threads]() { sum++; }); });
      all_done = when_all(all_done, fut);
    }
    prom1.fulfill_anonymous(1);
    all_done.wait();
    assert(sum.load() == iterations && "all tasks completed after wait");
  }
  barrier();
  DBG("Done with test_threads(", num_threads, "\n");
}

int test_serial_threads() {
  ThreadPool &tp1 = ThreadPool::get_single_pool(1);
  ThreadPool &tp2 = ThreadPool::get_single_pool(5);
  assert(&tp1 == &tp2);
  int64_t val = 0, max = 10000;
  DBG("Enqueue no return\n");
  future<> fut = make_future();
  for (int i = 0; i < max; i++) {
    fut = fut.then([&val, &tp1, i]() {
      return tp1.enqueue_no_return(
          [&val](int iter) {
            static int64_t iteration = 0;
            if (iter != iteration) DIE("Out of order execution iter=", iter, " iteration=", iteration, "\n");
            assert(iter == iteration);
            assert(val == iteration);
            iteration++;
            val++;
          },
          i);
    });
  }
  fut.wait();
  DBG("done\n");
  assert(val == max);
  barrier();

  DBG("Enqueue serially\n");
  val = 0;
  for (int i = 0; i < max; i++) {
    fut = ThreadPool::get_single_pool().enqueue_serially(
        [&val](int iter) {
          static int64_t iteration = 0;
          if (iter != iteration) DIE("Out of order execution iter=", iter, " iteration=", iteration, "\n");
          assert(iter == iteration);
          assert(val == iteration);
          iteration++;
          val++;
        },
        i);
  }
  fut.wait();
  DBG("done\n");
  assert(val == max);
  ThreadPool::join_single_pool();
  return 0;
}

int test_thread_pool(int argc, char **argv) {
  if (!upcxx::rank_me()) std::cout << "Found upcxx_utils version " << UPCXX_UTILS_VERSION << std::endl;
  char hnbuf[64];
  gethostname(hnbuf, sizeof(hnbuf) - 1);
  if (upcxx::local_team().rank_me() == 0) {
    std::cout << "proc " << upcxx::rank_me() << " on " << hnbuf << std::endl;
  }
  upcxx_utils::open_dbg("test_thread_pool");

  for (int threads = 0; threads <= 10; threads++) {
    barrier();
    SOUT("Running tests with ", threads, " thread(s) in the pool... \n");
    upcxx_utils::BaseTimer t("");
    t.start();
    test_threads(threads);
    barrier();
    t.stop();
    SOUT(t.get_elapsed(), " s.\n");
  }
  SOUT("Running serial threads test\n");
  barrier();
  test_serial_threads();
  barrier();
  DBG("Done with tests\n");

  upcxx_utils::close_dbg();
  return 0;
}
