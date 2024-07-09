#include <iostream>
#include <upcxx/upcxx.hpp>

#include "upcxx_utils/log.hpp"
#include "upcxx_utils/split_rank.hpp"
#include "upcxx_utils/version.h"

using namespace upcxx;
using namespace upcxx_utils;

template <typename ForEachRank, typename TestFunc>
void test_foreach_(const upcxx::team &t, string label, TestFunc test_ordering) {
  barrier();
  dist_object<int> dist_int(t);
  barrier();

  future<> fut_chain;

  {
    DBG(label, " team obj\n");
    ForEachRank fr(t);
    barrier();
    assert(*dist_int == 0);
    barrier();

    fut_chain = make_future();
    for (auto rank = fr.begin(); rank != fr.end(); rank++) {
      auto fut = rpc(
          t, *rank, [](dist_object<int> &dist_int) { (*dist_int)++; }, dist_int);
      fut_chain = when_all(fut_chain, fut);
    }

    for (auto rank : fr) {
      auto fut = rpc(
          t, rank, [](dist_object<int> &dist_int) { (*dist_int)++; }, dist_int);
      fut_chain = when_all(fut_chain, fut);
    }
    fut_chain.wait();
    barrier();
    assert(*dist_int == t.rank_n() * 2);
    *dist_int = 0;
    barrier();
  }

  DBG(label, " team noobj\n");
  barrier();
  assert(*dist_int == 0);
  barrier();

  fut_chain = make_future();
  for (auto rank : ForEachRank(t)) {
    auto fut = rpc(
        t, rank, [](dist_object<int> &dist_int) { (*dist_int)++; }, dist_int);
    fut_chain = when_all(fut_chain, fut);
  }
  fut_chain.wait();
  barrier();
  assert(*dist_int == t.rank_n());
  barrier();
  *dist_int = 0;
  barrier();

  dist_object<vector<intrank_t>> dist_vect(t, t.rank_n() * 2);
  for (auto i = 0; i < dist_vect->size(); i++) (*dist_vect)[i] = t.rank_n();
  barrier();

  fut_chain = make_future();
  intrank_t i = 0;
  for (auto rank : ForEachRank(t)) {
    DBG("sending to rank=", rank, " i=", i, "\n");
    auto fut = rpc(
                   t, rank,
                   [](dist_object<vector<intrank_t>> &dist_vect, intrank_t pos, intrank_t source_rank) {
                     const auto &t = dist_vect.team();
                     DBG("Received pos=", pos, " from ", source_rank,
                         " should still be init=", (*dist_vect)[source_rank + t.rank_n()], "\n");
                     assert((*dist_vect)[source_rank + t.rank_n()] == t.rank_n() && "dist_vect not set yet");
                     (*dist_vect)[source_rank + t.rank_n()] = pos;
                     return make_future(t.rank_me(), pos);
                   },
                   dist_vect, i, t.rank_me())
                   .then([&t, i, &dist_vect, rank](intrank_t r, intrank_t pos) {
                     DBG("Returned pos=", pos, " back from ", r, "\n");
                     assert(r == rank);
                     assert(pos == i);
                     assert((*dist_vect)[r] == t.rank_n());
                     (*dist_vect)[r] = pos;
                   });
    fut_chain = when_all(fut_chain, fut);
    i++;
  }
  fut_chain.wait();
  barrier();
  auto me = t.rank_me();
  auto n = t.rank_n();
  auto prev = (me + n - 1) % n;
  auto next = (me + 1) % n;
  auto last = (me + n - 1) % n;
  DBG("order[0]=", (*dist_vect)[0], "\n");
  DBG("order[n-1=", n - 1, "]=", (*dist_vect)[n - 1], "\n");
  DBG("order[me=", me, "]=", (*dist_vect)[me], "\n");
  DBG("order[prev=", prev, "]=", (*dist_vect)[prev], "\n");
  DBG("order[next=", next, "]=", (*dist_vect)[next], "\n");
  DBG("order[last=", last, "]=", (*dist_vect)[last], "\n");
  test_ordering(*dist_vect);
  barrier();
}

void unique_test(vector<intrank_t> &order, const upcxx::team &t, bool is_balanced = true) {
  auto n = t.rank_n();
  for (int i = 0; i < n; i++) {
    DBG("send o[rank=", i, "]=", order[i], "\n");
    assert(order[i] < n);
    assert(order[i] >= 0);
  }
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      DBG_VERBOSE("i=", i, " j=", j, " o[i]=", order[i], " o[j]=", order[j], "\n");
      assert(order[i] != order[j] && "i != j send");
    }
  }
  for (int i = n; i < n * 2; i++) {
    DBG("recv o[i=", i, "(rank=", i - n, ")]=", order[i], "\n");
    assert(order[i] < n);
    assert(order[i] >= 0);
  }
  for (int i = n; i < n * 2; i++) {
    for (int j = i + 1; j < 2 * n; j++) {
      DBG_VERBOSE("i=", i, " j=", j, " o[i]=", order[i], " o[j]=", order[j], "\n");
      if (is_balanced) assert(order[i] != order[j] && "i != j recv");
    }
  }
}

void test_foreach(const upcxx::team &t = world()) {
  barrier(t);
  auto me = t.rank_me();
  auto n = t.rank_n();
  auto lme = local_team().rank_me();
  auto ln = local_team().rank_n();
  auto nodes = t.rank_n() / local_team().rank_n();
  DBG("Testing default\n");
  auto by_node_test = [n, me, lme, ln, nodes, &t](vector<intrank_t> order) {
    unique_test(order, t);
    assert(order[me] == 0 && "I am my first");
    if (nodes > 1) {
      DBG_VERBOSE("order[(me + ln) % n =", (me + ln) % n, "]=", order[(me + ln) % n], " == 1\n");
      assert(order[(me + ln) % n] == 1 && "next node is second");
      DBG_VERBOSE("order[(me + n - 1 + n - (lme == 0 ? 0 : ln)) % n=", (me + n - 1 + n - (lme == 0 ? 0 : ln)) % n,
                  "]=", order[(me + n - 1 + n - (lme == 0 ? 0 : ln)) % n], "\n");
      assert(order[(me + n - 1 + n - (lme == 0 ? 0 : ln)) % n] == n - 1 && "prev node prev lrank is last");
    } else if (n > 1) {
      assert(order[(me + 1) % n] == 1 && "next lrank is second");
      assert(order[(me + n - 1) % n] == n - 1 && "prev lrank is last");
    }
  };

  test_foreach_<foreach_rank>(t, "default", by_node_test);  // by node
  DBG("Testing by rank\n");
  test_foreach_<foreach_rank_by_rank>(t, "by rank", [n, me, &t](vector<intrank_t> order) {
    unique_test(order, t);
    assert(order[me] == 0 && "I am first");
    if (n > 1) {
      assert(order[(me + 1) % n] == 1 && "next ranks is second");
      assert(order[(me + n - 1) % n] == n - 1 && "prev_rank is last");
    }
  });
  DBG("Testing by node\n");
  test_foreach_<foreach_rank_by_node>(t, "by node", by_node_test);
  DBG("Testing by random");
  test_foreach_<foreach_rank_by_random>(t, "by random", [&t](vector<intrank_t> order) { unique_test(order, t, false); });
  DBG("Testing by random full");
  test_foreach_<foreach_rank_by_random_full>(t, "by random_full", [&t](vector<intrank_t> order) { unique_test(order, t, false); });
  barrier(t);
}

int test_split_rank(int argc, char **argv) {
  if (!upcxx::rank_me()) std::cout << "Found upcxx_utils version " << UPCXX_UTILS_VERSION << std::endl;

  open_dbg("test_split_rank");

  int nodes = rank_n() / local_team().rank_n();
  int threads = local_team().rank_n();

#ifdef DEBUG
  intrank_t split_node_rank_n = split_rank::get_split_node_rank_n();
  if (split_rank::split_local_team().rank_n() != split_node_rank_n)
    DIE("Invalid split local in debug: ", split_rank::split_local_team().rank_n(), " != ", split_node_rank_n,
        " split_by=", split_node_rank_n, "\n");
  if (split_rank::split_local_team().rank_me() != local_team().rank_me() % (split_node_rank_n))
    DIE("Invalid split local in debug split.rank_me = ", split_rank::split_local_team().rank_me(),
        " local.rank_me() = ", local_team().rank_me(), " local_team.rank_n() = ", local_team().rank_n(), "\n");
  nodes = rank_n() / split_node_rank_n;
  threads = split_node_rank_n;
#else
  if (split_rank::split_local_team().rank_n() != local_team().rank_n()) DIE("Invalid split local in release\n");
  if (split_rank::split_local_team().rank_me() != local_team().rank_me() % local_team().rank_n())
    DIE("Invalid split local in release split.rank_me = ", split_rank::split_local_team().rank_me(),
        " local.rank_me() = ", local_team().rank_me(), " local_team.rank_n() = ", local_team().rank_n(), "\n");
#endif

  {
    upcxx_utils::split_team mysplit;
    if (mysplit.node_n() != nodes) DIE(mysplit.node_n(), " != ", nodes, "\n");
    if (mysplit.thread_n() != threads) DIE("\n");
    if (mysplit.full_n() != rank_n()) DIE("\n");

    if (mysplit.full_from_thread(mysplit.thread_me()) != mysplit.full_me()) DIE("\n");
    if (mysplit.full_from_node(mysplit.node_me()) != mysplit.full_me()) DIE("\n");
    if (mysplit.thread_from_full(mysplit.full_me()) != mysplit.thread_me()) DIE("\n");
    if (mysplit.node_from_full(mysplit.full_me()) != mysplit.node_me()) DIE("\n");

    for (intrank_t t = 0; t < mysplit.thread_n(); t++) {
      auto w = mysplit.world_from_thread(t);
      if (!mysplit.thread_team_contains_world(w)) DIE("\n");
      auto f = mysplit.full_from_thread(t);
      auto f2 = mysplit.full_me() - mysplit.thread_me() + t;
      if (f != f2) DIE("\n");

      if (t == mysplit.thread_me()) {
        if (f != mysplit.full_me()) DIE("\n");
      }
    }
    for (intrank_t n = 0; n < mysplit.node_n(); n++) {
      auto w = mysplit.world_from_node(n);
      auto f = mysplit.full_from_node(n);
      if (n == mysplit.node_me()) {
        if (!mysplit.thread_team_contains_world(w)) DIE("\n");
        if (f != mysplit.full_me()) DIE("\n");
      } else {
        if (mysplit.thread_team_contains_world(w)) DIE("\n");
        if (f >= mysplit.full_me() - mysplit.thread_me() && f < mysplit.full_me() + mysplit.thread_n() - mysplit.thread_me())
          DIE("\n");
      }
    }
    for (intrank_t f = 0; f < mysplit.full_n(); f++) {
      auto n = mysplit.node_from_full(f);
      auto t = mysplit.thread_from_full(f);
      if (n == mysplit.node_me()) {
        if (!mysplit.thread_team_contains_full(f)) DIE("\n");
      } else {
        if (mysplit.thread_team_contains_full(f)) DIE("\n");
      }
    }

    DBG("testing world\n");
    test_foreach(world());
    DBG("testing local_team\n");
    test_foreach(local_team());

    // reverse team  -- just ensure all ranks are hit once and only once
    upcxx::team reverse_team = world().split(0, rank_n() - rank_me());
    DBG("testing reverse_team\n");

    test_foreach_<foreach_rank_by_rank>(reverse_team, "reverse by rank",
                                        [&reverse_team](vector<intrank_t> order) { unique_test(order, reverse_team, false); });
    test_foreach_<foreach_rank_by_node>(reverse_team, "reverse by node",
                                        [&reverse_team](vector<intrank_t> order) { unique_test(order, reverse_team, false); });
    test_foreach_<foreach_rank_by_random>(reverse_team, "reverse by random_full",
                                          [&reverse_team](vector<intrank_t> order) { unique_test(order, reverse_team, false); });
    test_foreach_<foreach_rank_by_random_full>(reverse_team, "reverse by random_full", [&reverse_team](vector<intrank_t> order) {
      unique_test(order, reverse_team, false);
    });

    barrier(reverse_team);
    reverse_team.destroy();
  }

  barrier();
  close_dbg();
  return 0;
}
