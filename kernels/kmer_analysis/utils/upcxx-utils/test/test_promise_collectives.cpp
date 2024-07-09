#include <random>
#include <vector>

#include "upcxx/upcxx.hpp"
#include "upcxx_utils/log.hpp"
#include "upcxx_utils/promise_collectives.hpp"

using std::vector;

using upcxx::barrier;
using upcxx::future;
using upcxx::make_future;
using upcxx::when_all;

using upcxx_utils::op_PromiseReduce;
using upcxx_utils::PromiseBarrier;
using upcxx_utils::PromiseBroadcast;
using upcxx_utils::PromiseReduce;
using upcxx_utils::roundup_log2;

#define ASSERT_FLOAT(a, b, c) assert((a) >= ((b)-0.000001) && (a) <= ((b) + 0.000001) && c)

int test_promise_barrier(int argc, char **argv) {
  barrier();
  upcxx_utils::open_dbg("test_promise_barrier");

  assert(roundup_log2(0) == -1);
  assert(roundup_log2(1) == 0);
  assert(roundup_log2(2) == 1);
  assert(roundup_log2(3) == 2);
  assert(roundup_log2(4) == 2);
  assert(roundup_log2(5) == 3);
  assert(roundup_log2(6) == 3);
  assert(roundup_log2(7) == 3);
  assert(roundup_log2(8) == 3);
  assert(roundup_log2(9) == 4);
  assert(roundup_log2(15) == 4);
  assert(roundup_log2(16) == 4);
  assert(roundup_log2(17) == 5);
  {
    PromiseBarrier pb;
    assert(!pb.get_future().is_ready());
    pb.fulfill();
    pb.get_future().wait();
  }
  barrier();
  {
    DBG("1s 2s 1e 2e\n");
    barrier();
    PromiseBarrier pb1, pb2;
    barrier();
    assert(!pb1.get_future().is_ready());
    pb1.fulfill();
    barrier();
    assert(!pb2.get_future().is_ready());
    pb2.fulfill();
    barrier();
    pb1.get_future().wait();
    barrier();
    pb2.get_future().wait();
    barrier();
  }
  {
    DBG("1s 1e 2s 2e\n");
    barrier();
    PromiseBarrier pb1, pb2;
    barrier();
    assert(!pb1.get_future().is_ready());
    pb1.fulfill();
    barrier();
    pb1.get_future().wait();
    barrier();
    assert(!pb2.get_future().is_ready());
    pb2.fulfill();
    barrier();
    pb2.get_future().wait();
    barrier();
  }
  {
    DBG("1s 2s 2e 1e\n");
    barrier();
    PromiseBarrier pb1, pb2;
    barrier();
    assert(!pb1.get_future().is_ready());
    pb1.fulfill();
    barrier();
    assert(!pb2.get_future().is_ready());
    pb2.fulfill();
    barrier();
    pb2.get_future().wait();
    barrier();
    pb1.get_future().wait();
    barrier();
  }

  {
    DBG("2s 1s 2e 1e\n");
    barrier();
    PromiseBarrier pb1, pb2;
    barrier();
    assert(!pb2.get_future().is_ready());
    pb2.fulfill();
    barrier();
    assert(!pb1.get_future().is_ready());
    pb1.fulfill();
    barrier();
    pb2.get_future().wait();
    barrier();
    pb1.get_future().wait();
    barrier();
  }
  {
    DBG("2s 2e 1s 1e\n");
    barrier();
    PromiseBarrier pb1, pb2;
    barrier();
    assert(!pb2.get_future().is_ready());
    pb2.fulfill();
    barrier();
    pb2.get_future().wait();
    barrier();
    assert(!pb1.get_future().is_ready());
    pb1.fulfill();
    barrier();
    pb1.get_future().wait();
    barrier();
  }
  {
    DBG("2s 1s 1e 2e\n");
    barrier();
    PromiseBarrier pb1, pb2;
    barrier();
    assert(!pb2.get_future().is_ready());
    pb2.fulfill();
    barrier();
    assert(!pb1.get_future().is_ready());
    pb1.fulfill();
    barrier();
    pb1.get_future().wait();
    barrier();
    pb2.get_future().wait();
    barrier();
  }

  std::mt19937 g(rank_n());  // seed all ranks the same
  {
    DBG("fulfill all barrier wait all same order\n");
    int iterations = 1000;
    vector<PromiseBarrier> pbs(iterations);
    vector<intrank_t> fulfill_order(iterations);
    vector<intrank_t> &wait_order = fulfill_order;
    for (int i = 0; i < iterations; i++) {
      fulfill_order[i] = i;
      wait_order[i] = i;
      assert(!pbs[i].get_future().is_ready());
    }
    std::shuffle(fulfill_order.begin(), fulfill_order.end(), g);
    barrier();
    // initiate all
    for (int i = 0; i < iterations; i++) {
      assert(!pbs[fulfill_order[i]].get_future().is_ready());
      pbs[fulfill_order[i]].fulfill();
    }
    // wait all
    future<> all_fut = make_future();
    for (int i = 0; i < iterations; i++) {
      auto fut = pbs[wait_order[i]].get_future();
      all_fut = when_all(all_fut, fut);
    }
    all_fut.wait();
    barrier();
  }

  {
    DBG("fulfill all barrier wait all different order\n");
    int iterations = 1000;
    vector<PromiseBarrier> pbs(iterations);
    vector<intrank_t> fulfill_order(iterations);
    vector<intrank_t> wait_order(iterations);
    for (int i = 0; i < iterations; i++) {
      fulfill_order[i] = i;
      wait_order[i] = i;
      assert(!pbs[i].get_future().is_ready());
    }
    std::shuffle(fulfill_order.begin(), fulfill_order.end(), g);
    std::shuffle(wait_order.begin(), wait_order.end(), g);
    barrier();
    // initiate all
    for (int i = 0; i < iterations; i++) {
      assert(!pbs[fulfill_order[i]].get_future().is_ready());
      pbs[fulfill_order[i]].fulfill();
    }
    barrier();
    // wait all
    future<> all_fut = make_future();
    for (int i = 0; i < iterations; i++) {
      auto fut = pbs[wait_order[i]].get_future();
      all_fut = when_all(all_fut, fut);
    }
    all_fut.wait();
    barrier();
  }

  {
    DBG("fulfill all barrier wait each same order\n");
    int iterations = 1000;
    vector<PromiseBarrier> pbs(iterations);
    vector<intrank_t> fulfill_order(iterations);
    vector<intrank_t> &wait_order = fulfill_order;
    for (int i = 0; i < iterations; i++) {
      fulfill_order[i] = i;
      wait_order[i] = i;
    }
    std::shuffle(fulfill_order.begin(), fulfill_order.end(), g);
    barrier();
    // initiate all
    for (int i = 0; i < iterations; i++) {
      assert(!pbs[fulfill_order[i]].get_future().is_ready());
      pbs[fulfill_order[i]].fulfill();
    }
    barrier();
    // wait each
    for (int i = 0; i < iterations; i++) {
      pbs[wait_order[i]].get_future().wait();
    }
    barrier();
  }

  {
    DBG("fulfill all barrier wait each different order\n");
    int iterations = 1000;
    vector<PromiseBarrier> pbs(iterations);
    vector<intrank_t> fulfill_order(iterations);
    vector<intrank_t> wait_order(iterations);
    for (int i = 0; i < iterations; i++) {
      fulfill_order[i] = i;
      wait_order[i] = i;
    }
    std::shuffle(fulfill_order.begin(), fulfill_order.end(), g);
    std::shuffle(wait_order.begin(), wait_order.end(), g);
    barrier();
    // initiate all
    for (int i = 0; i < iterations; i++) {
      assert(!pbs[fulfill_order[i]].get_future().is_ready());
      pbs[fulfill_order[i]].fulfill();
    }
    barrier();
    // wait each
    for (int i = 0; i < iterations; i++) {
      pbs[wait_order[i]].get_future().wait();
    }
    barrier();
  }

  {
    DBG("fulfill then wait each same order\n");
    int iterations = 1000;
    vector<PromiseBarrier> pbs(iterations);
    vector<intrank_t> fulfill_order(iterations);
    vector<intrank_t> &wait_order = fulfill_order;
    for (int i = 0; i < iterations; i++) {
      fulfill_order[i] = i;
      wait_order[i] = i;
    }
    std::shuffle(fulfill_order.begin(), fulfill_order.end(), g);
    barrier();
    // initiate all
    for (int i = 0; i < iterations; i++) {
      assert(!pbs[fulfill_order[i]].get_future().is_ready());
      pbs[fulfill_order[i]].fulfill();
      pbs[wait_order[i]].get_future().wait();
    }
    barrier();
  }

  DBG("fulfill then wait each different order would deadlock\n");

  {
    DBG("fulfill all then wait all same order\n");
    int iterations = 1000;
    vector<PromiseBarrier> pbs(iterations);
    vector<intrank_t> fulfill_order(iterations);
    vector<intrank_t> &wait_order = fulfill_order;
    for (int i = 0; i < iterations; i++) {
      fulfill_order[i] = i;
      wait_order[i] = i;
    }
    std::shuffle(fulfill_order.begin(), fulfill_order.end(), g);
    barrier();
    // initiate all
    future<> all_fut = make_future();
    for (int i = 0; i < iterations; i++) {
      assert(!pbs[fulfill_order[i]].get_future().is_ready());
      pbs[fulfill_order[i]].fulfill();
      auto fut = pbs[wait_order[i]].get_future();
      all_fut = when_all(all_fut, fut);
    }
    all_fut.wait();
    barrier();
  }

  {
    DBG("fulfill all then wait all different order\n");
    int iterations = 1000;
    vector<PromiseBarrier> pbs(iterations);
    vector<intrank_t> fulfill_order(iterations);
    vector<intrank_t> wait_order(iterations);
    for (int i = 0; i < iterations; i++) {
      fulfill_order[i] = i;
      wait_order[i] = i;
    }
    std::shuffle(fulfill_order.begin(), fulfill_order.end(), g);
    std::shuffle(wait_order.begin(), wait_order.end(), g);
    barrier();
    // initiate all
    future<> all_fut = make_future();
    for (int i = 0; i < iterations; i++) {
      assert(!pbs[fulfill_order[i]].get_future().is_ready());
      pbs[fulfill_order[i]].fulfill();
      auto fut = pbs[wait_order[i]].get_future();
      all_fut = when_all(all_fut, fut);
    }
    all_fut.wait();
    barrier();
  }

  upcxx_utils::close_dbg();
  barrier();
  return 0;
}

template <typename T>
int test_promise_broadcast(int argc, char **argv) {
  barrier();
  upcxx_utils::open_dbg("test_promise_broadcast");
  vector<T> data;

  upcxx::future<> fut = make_future();
  for (int i = 0; i < 2; i++) {
    auto &tm = i == 0 ? upcxx::world() : upcxx::local_team();

    int root = 0;
    int last_sz = 1, last_sz2 = 1;
    for (int sz = 1; sz < 4000; sz += last_sz) {
      last_sz = last_sz2;
      last_sz2 = sz;
      DBG("i=", i, " sz=", sz, " root=", root, "\n");
      T val = sz + i + root;
      data.clear();
      data.resize(sz, val);
      auto sh_results = make_shared<vector<T>>(sz);
      DBG("made sh_results=", sh_results.get(), " sz=", sz, " val=", val, "\n");
      PromiseBroadcast<T> pb(root, tm);
      fut = when_all(fut, pb.get_future()).then([sh_results, i, sz, val, root]() {
        DBG("Checking i=", i, " sz=", sz, " root=", root, "\n");
        assert(sh_results->size() == sz);
        for (auto &x : *sh_results) {
          DBG_VERBOSE("x=", x, " val=", val, "\n");
          assert(x == val);
        }
        DBG("i=", i, " sz=", sz, " root=", root, " checks\n");
      });
      assert(sh_results->size() == data.size());
      assert(data.size() == sz);
      assert(data[0] == val);
      if (root == tm.rank_me()) *sh_results = data;
      DBG("Fulfilling sh_results=", sh_results.get(), " sz=", sz, "\n");
      pb.fulfill(sh_results->data(), sz);

      sh_results = make_shared<vector<T>>(sz);
      auto sh_pb2 = make_shared<PromiseBroadcast<T>>(root, tm);
      auto &pb2 = *sh_pb2;
      fut = when_all(fut, pb2.get_future()).then([sh_results, i, sz, val, root]() {
        DBG("Checking2 i=", i, " sz=", sz, " root=", root, "\n");
        assert(sh_results->size() == sz);
        for (auto &x : *sh_results) {
          DBG_VERBOSE("2 x=", x, " val=", val, "\n");
          assert(x == val);
        }
        DBG("2 i=", i, " sz=", sz, " root=", root, " checks\n");
      });
      assert(sh_results->size() == data.size());
      assert(data.size() == sz);
      assert(data[0] == val);
      if (root == tm.rank_me()) *sh_results = data;
      DBG("Fulfilling2 sh_results=", sh_results.get(), " sz=", sz, "\n");
      auto fut2 = pb.get_future().then([&pb2, sh_pb2, sh_results, sz]() { pb2.fulfill(sh_results->data(), sz); });
      fut = when_all(fut, fut2);

      root = (root + 1) % tm.rank_n();
    }
    PromiseBroadcast<T> pb_never_fulfilled(root, tm);
  }
  fut.wait();

  upcxx_utils::close_dbg();
  barrier();
  return 0;
}

int test_promise_reduce(int argc, char **argv) {
  barrier();
  upcxx_utils::open_dbg("test_promise_reduce");

  {
    double d1 = 2.0, d2 = 3.14159265358979;
    int64_t tmp = op_PromiseReduce::double2T(d1);
    ASSERT_FLOAT(op_PromiseReduce::T2double(tmp), d1, "");
    tmp = op_PromiseReduce::double2T(d2);
    ASSERT_FLOAT(op_PromiseReduce::T2double(tmp), d2, "");
  }

  {
    PromiseReduce pr /*mixed*/, pr_2 /*only reduce_all*/, pr_3 /*only reduce_one*/;
    auto fut_min = pr.reduce_one(rank_me(), op_fast_min);
    auto fut_a_min = pr_2.reduce_all(rank_me(), op_fast_min);
    auto fut_max = pr.reduce_one(rank_me(), op_fast_max);
    auto fut_a_max = pr.reduce_all(rank_me(), op_fast_max);
    auto fut_sum = pr.reduce_one(rank_me(), op_fast_add);
    auto fut_a_sum = pr.reduce_all(rank_me(), op_fast_add);

    auto fut_min_d = pr.reduce_one(rank_me() * 1.0 / rank_n(), op_fast_min);
    auto fut_a_min_d = pr.reduce_all(rank_me() * 1.0 / rank_n(), op_fast_min);
    auto fut_max_d = pr.reduce_one(rank_me() * 1.0 / rank_n(), op_fast_max);
    auto fut_a_max_d = pr.reduce_all(rank_me() * 1.0 / rank_n(), op_fast_max);
    auto fut_sum_d = pr.reduce_one(rank_me() * 1.0 / rank_n(), op_fast_add);
    auto fut_a_sum_d = pr_2.reduce_all(rank_me() * 1.0 / rank_n(), op_fast_add);
    auto fut_max_d2 = pr.reduce_one(rank_me() * 1.0 / rank_n(), op_fast_max);

    auto fut_min1 = pr_3.reduce_one(rank_me(), op_fast_min);
    auto fut_max1 = pr_3.reduce_one(rank_me(), op_fast_max);
    auto fut_sum1 = pr_3.reduce_one(rank_me(), op_fast_add);

    auto sum = 0;
    for (int i = 0; i < rank_n(); i++) sum += i;
    if (!rank_me()) INFO("sum = ", sum, "\n");

    pr_3.fulfill().wait();
    if (rank_me() == 0) {
      assert(0 == fut_min1.wait());
      assert(rank_n() - 1 == fut_max1.wait());
      assert(sum == fut_sum1.wait());
    }

    pr.fulfill().wait();

    if (rank_me() == 0) {
      // DBG("fut_min=", fut_min.wait(), " fut_max=", fut_max.wait(), "\n");
      assert(fut_min.wait() == 0 && "min in 0");
      assert(fut_max.wait() == rank_n() - 1 && "max is last rank");
      // DBG("sum=", sum, " fut_sum=", fut_sum.wait(), "\n");
      assert(fut_sum.wait() == sum && "sum is correct");
      // DBG("fut_min_d=", fut_min_d.wait(), " fut_max_d=", fut_max_d.wait(), " fut_sum_d=", fut_sum_d.wait(), "\n");
      ASSERT_FLOAT(fut_min_d.wait(), 0.0, "min is 0.0");
      ASSERT_FLOAT(fut_max_d.wait(), (rank_n() - 1) * 1.0 / (double)rank_n(), "max is last rank");
      ASSERT_FLOAT(fut_sum_d.wait(), sum / (double)rank_n(), "sum is correct");
      ASSERT_FLOAT(fut_max_d2.wait(), (rank_n() - 1) * 1.0 / (double)rank_n(), "max is last rank");
    }
    barrier();
    auto fut_all_results = pr_2.fulfill();
    fut_all_results.wait();
    INFO("fut_a_min=", fut_a_min.wait(), " fut_a_max=", fut_a_max.wait(), "\n");
    assert(fut_a_min.wait() == 0 && "min in 0");
    assert(fut_a_max.wait() == rank_n() - 1 && "max is last rank");
    // INFO("sum=", sum, " fut_sum=", fut_sum.wait(), "\n");
    assert(fut_a_sum.wait() == sum && "sum is correct");
    // INFO("fut_min_d=", fut_a_min_d.wait(), " fut_max_d=", fut_a_max_d.wait(), " fut_sum_d=", fut_a_sum_d.wait(), "\n");
    ASSERT_FLOAT(fut_a_min_d.wait(), 0.0, "min in 0.0");
    ASSERT_FLOAT(fut_a_max_d.wait(), (rank_n() - 1.0) / (double)rank_n(), "max is last rank");
    ASSERT_FLOAT(fut_a_sum_d.wait(), sum / (double)rank_n(), "sum is correct");

    auto lme = local_team().rank_me();
    auto ln = local_team().rank_n();
    auto my_node = rank_me() / ln;
    INFO("lme=", lme, " ln=", ln, " my_node=", my_node, "\n");
    PromiseReduce pr3(local_team());
    auto tgt_root = ln / 2;
    auto fut_r = pr3.reduce_one(rank_me(), op_fast_max, tgt_root);
    auto fut_msm = pr3.msm_reduce_one(rank_me());
    auto fut_msm2 = pr3.msm_reduce_one(rank_me(), rank_me() != tgt_root, tgt_root);  // all active but 1
    auto fut_pr3 = pr3.fulfill();
    if (lme == 0) {
      auto msm = fut_msm.wait();
      assert(msm.my == rank_me());
      assert(msm.min == rank_me());
      assert(msm.max == rank_me() + ln - 1);
      assert(msm.active_ranks == ln);
    }
    if (lme == tgt_root) {
      auto msm = fut_msm2.wait();
      assert(msm.my == rank_me());
      assert(msm.min == rank_me() - lme);
      assert(msm.max == rank_me() - lme + ln - 1);
      assert(msm.active_ranks == ln - 1 && "All active but 1");
    }

    PromiseReduce pr4(local_team());
    auto fut_pr4_1 = pr4.reduce_one(rank_me(), op_fast_max, tgt_root);
    auto fut_pr4_2 = pr4.reduce_all(rank_me(), op_fast_max);
    upcxx::promise<int> prom_me(1);
    auto fut_pr4_3 = pr4.fut_reduce_all(prom_me.get_future(), op_fast_max);
    prom_me.fulfill_result(rank_me());

    PromiseReduce pr5;
    std::vector<upcxx::future<int>> futs;
    for (int i = 0; i < 20000; i++) {
      futs.push_back(pr5.reduce_all(rank_me() + i, op_fast_max));
    }
    pr5.fulfill().wait();

    auto fut_pr4 = pr4.fulfill();

    fut_pr3.wait();
    if (lme == tgt_root) {
      auto r = fut_r.wait();
      INFO("got=", r, " exp=", (my_node + 1) * ln - 1, "\n");
      assert(r == (my_node + 1) * ln - 1 && "world rank of last rank in my local team");
    }

    fut_pr4.wait();
    auto pr4_1 = fut_pr4_1.wait();
    auto pr4_2 = fut_pr4_2.wait();
    auto pr4_3 = fut_pr4_3.wait();
    if (lme == tgt_root) assert(pr4_1 == (my_node + 1) * ln - 1 && "target root got world rank of last rank in my local team");
    assert(pr4_2 == (my_node + 1) * ln - 1 && "all ranks got world rank of last rank in my local team");
    assert(pr4_3 == rank_n() - 1 && "all ranks got last rank");

    int i = 0;
    for (auto &fut : futs) {
      assert(i++ + rank_n() - 1 == fut.wait());
    }
  }

  upcxx_utils::close_dbg();
  barrier();
  return 0;
}

int test_promise_collectives(int argc, char **argv) {
  test_promise_barrier(argc, argv);
  test_promise_broadcast<int16_t>(argc, argv);
  test_promise_broadcast<uint32_t>(argc, argv);
  test_promise_broadcast<uint64_t>(argc, argv);
  test_promise_reduce(argc, argv);
  return 0;
}
