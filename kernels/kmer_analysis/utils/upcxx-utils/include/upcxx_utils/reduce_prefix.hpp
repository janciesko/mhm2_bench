#pragma once

/*
 * File:   reduce_prefix.hpp
 * Author: regan
 *
 * Created on June 19, 2020, 9:18 PM
 */

#include <cassert>
#include <functional>
#include <memory>
#include <tuple>
#include <upcxx/upcxx.hpp>
#include <utility>
#include <vector>

using std::make_shared;
using std::make_tuple;
using std::pair;
using std::shared_ptr;
using std::tie;
using std::tuple;
using std::vector;

#include "upcxx_utils/log.hpp"
#include "upcxx_utils/split_rank.hpp"
#include "upcxx_utils/version.h"

using upcxx::dist_object;
using upcxx::intrank_t;
using upcxx::local_team;
using upcxx::local_team_contains;
using upcxx::make_future;
using upcxx::make_view;
using upcxx::progress;
using upcxx::rank_me;
using upcxx::rank_n;
using upcxx::rpc;
using upcxx::rpc_ff;
using upcxx::view;
using upcxx::when_all;
using upcxx::world;

namespace upcxx_utils {

#define UPCXX_UTILS_REDUCE_PREFIX_PIPELINE_BYTES (16 * ONE_MB)
#define UPCXX_UTILS_REDUCE_PREFIX_BINARY_TREE_BYTES (2 * ONE_MB)

// special trivial case of rank_n() == 1

template <typename T, typename BinaryOp>
upcxx::future<> reduce_prefix_trivial(const T *src, T *dst, size_t count, const BinaryOp &op, const upcxx::team &team,
                                      bool return_final_to_first = false) {
  DBG("count=", count, "\n");

  for (size_t i = 0; i < count; i++) {
    dst[i] = src[i];
  }
  upcxx::future<> fut = make_future();
  return fut.then([count]() { DBG_VERBOSE("Completed reduce_prefix_trivial, count=", count, "\n"); });
}

// returns the prefix reduction count elements in src to count elements in dst
// optionally returns the last rank's values to the first rank instead of a copy of src to the first rank
// count must be single valued across the team
// Least communication, highest latency
// O(N) messages, O(N) latency

template <typename T, typename BinaryOp>
upcxx::future<> reduce_prefix_ring(const T *src, T *dst, size_t count, const BinaryOp &op, const upcxx::team &team = world(),
                                   bool return_final_to_first = false, intrank_t skip_regular_local_ranks = 0) {
  if (team.from_world(rank_me(), rank_n()) == rank_n())
    throw std::runtime_error("reduce_prefix called outside of given team");  // not of this team
  DBG_VERBOSE("src=", src, ", dst=", dst, ", count=", count, ", team.rank_n()=", team.rank_n(),
              ", return_final_to_first=", return_final_to_first, ", skip_regular_local_ranks=", skip_regular_local_ranks, "\n");
  using ShPromise = shared_ptr<upcxx::promise<>>;
  using Data = std::tuple<const T *, T *, size_t, const BinaryOp &, ShPromise>;
  using DistData = dist_object<Data>;
  using ShDistData = shared_ptr<DistData>;

  if (team.rank_n() == 1 || team.rank_n() == skip_regular_local_ranks)
    return reduce_prefix_trivial(src, dst, count, op, team, return_final_to_first);

  ShPromise my_prom = make_shared<upcxx::promise<>>();
  upcxx::future<> ret_fut = my_prom->get_future();

  if (team.rank_me() == 0) {
    // first is special case of copy without op
    for (size_t i = 0; i < count; i++) dst[i] = src[i];

    if (return_final_to_first) {
      // make ready for the rpc but do not fulfill my_prom yet
      ret_fut = make_future();
    } else {
      // make ready
      my_prom->fulfill_anonymous(1);
    }
    assert(team.rank_n() > 1);
  }

  // create a distributed object holding the data and promise
  ShDistData sh_dist_data = make_shared<DistData>(make_tuple(src, dst, count, std::cref(op), my_prom), team);

  if (skip_regular_local_ranks != 0) {
    assert(team.rank_n() % skip_regular_local_ranks == 0);
    if (local_team().rank_me() % skip_regular_local_ranks != 0) {
      DBG_VERBOSE("Skipping this node skip=", skip_regular_local_ranks, "\n");
      // this rank does not participate
      return make_future();
    }
  }

  intrank_t next_rank = team.rank_me() + (skip_regular_local_ranks == 0 ? 1 : skip_regular_local_ranks);
  if (return_final_to_first || next_rank != team.rank_n()) {
    // send rpc to the next rank when my prefix is ready
    DBG_VERBOSE("Sending to next_rank=", next_rank, "\n");
    ret_fut = ret_fut.then([next_rank, dst, count, sh_dist_data, &team]() -> upcxx::future<> {
      DBG_VERBOSE("mydst is ready:", dst, ", next_rank=", next_rank, "\n");
      rpc_ff(
          team, next_rank % team.rank_n(),
          [](DistData &dist_data, view<T> prev_prefix, bool just_copy) {
            DBG_VERBOSE("Receiving: count=", prev_prefix.size(), ", just_copy=", just_copy, "\n");
            auto &[mysrc, mydst, mycount, op, myprom] = *dist_data;
            assert(mycount == prev_prefix.size());
            if (just_copy) {
              for (size_t i = 0; i < mycount; i++) {
                mydst[i] = prev_prefix[i];
              }
            } else {
              for (size_t i = 0; i < mycount; i++) {
                mydst[i] = op(prev_prefix[i], mysrc[i]);
              }
            }
            myprom->fulfill_anonymous(1);  // mydst is ready
          },
          *sh_dist_data, make_view(dst, dst + count), next_rank == team.rank_n());
      return make_future();
    });
  } else {
    ret_fut = ret_fut.then([sh_dist_data]() {
      // keep the scope of the DistData object until ready
    });
  }

  if (team.rank_me() == 0 && return_final_to_first) {
    // wait on my_prom to be fulfilled by an rpc from the last rank of the team
    ret_fut = my_prom->get_future().then([sh_dist_data]() {
      // keep the scope of the DistData object until ready
    });
  }

  return ret_fut.then([sh_dist_data]() {
    // keep scope of DistData until complete
    // when promise is fulfilled, there will be no more incoming rpcs
    DBG_VERBOSE("Completed reduce_prefix_ring.\n");
  });
};

// high duplex communication, highest performance and lowest latency for small to mid-sized data
// O(Nlog(N)) messages, O(log(N)) latency

template <typename T, typename BinaryOp>
upcxx::future<> reduce_prefix_binomial(const T *src, T *dst, size_t count, const BinaryOp &op, const upcxx::team &team = world(),
                                       bool return_final_to_first = false, intrank_t skip_regular_local_ranks = 0) {
  using LevelPromises = vector<upcxx::promise<>>;  // two promises per level (receive and send)
  using ShLevelPromises = shared_ptr<LevelPromises>;
  using Data = std::tuple<const T *, T *, size_t, const BinaryOp &, ShLevelPromises>;
  using DistData = dist_object<Data>;
  using ShDistData = shared_ptr<DistData>;
  DBG("count=", count, ", teamsize=", team.rank_n(), " return_final_to_first=", return_final_to_first,
      ", skip_regular_local_ranks=", skip_regular_local_ranks, "\n");

  if (team.rank_n() == 1 || team.rank_n() == skip_regular_local_ranks)
    return reduce_prefix_trivial(src, dst, count, op, team, return_final_to_first);

  // if return_final_to_first, add a rank to the algorithm and wrap the communication
  // height is the ceil of log2(num_ranks)
  // level ranges from 0 <= level < height
  // establish log2 height of team and this rank
  int height = 1;
  intrank_t num_ranks = team.rank_n() + (return_final_to_first ? 1 : 0);
  intrank_t tmp =
      (skip_regular_local_ranks == 0 ? team.rank_n() : team.rank_n() / skip_regular_local_ranks) + (return_final_to_first ? 1 : 0);
  tmp--;
  while (tmp >>= 1) {
    height++;
  }

  // first copy src to dst for all ranks, before dist_object is created
  for (size_t i = 0; i < count; i++) {
    dst[i] = src[i];
  }

  ShLevelPromises sh_my_promises = make_shared<LevelPromises>(height * 2);
  ShDistData sh_dist_data = make_shared<DistData>(make_tuple(src, dst, count, std::cref(op), sh_my_promises), team);

  if (skip_regular_local_ranks != 0) {
    assert(team.rank_n() % skip_regular_local_ranks == 0);
    if (local_team().rank_me() % skip_regular_local_ranks != 0) {
      DBG_VERBOSE("Skipping this node skip=", skip_regular_local_ranks, "\n");
      // this rank does not participate
      return make_future();
    }
  }

  DBG_VERBOSE("DistData = ", sh_dist_data->id(), " dst=", dst, "\n");
  LevelPromises &my_promises = *sh_my_promises;

  // establish binomial dependency rpc chains
  // at each level, a given rank
  // 1) first sends to higher rank (optionally)
  // 2) then receives from lower rank (optionally)
  // 2a) then applies the operation for new prefix
  intrank_t my_recv_rank = (team.rank_me() == 0 && return_final_to_first) ? team.rank_n() : team.rank_me();
  intrank_t step = skip_regular_local_ranks == 0 ? 1 : skip_regular_local_ranks;
  upcxx::future<> level_fut = make_future();
  for (int level = 0; level < height; level++) {
    DBG_VERBOSE("setup level=", level, ", height=", height, "\n");

    upcxx::promise<> &received_prom = my_promises[level * 2];
    upcxx::promise<> &sent_prom = my_promises[level * 2 + 1];

    intrank_t prev_rank = step <= my_recv_rank ? my_recv_rank - step : num_ranks;
    intrank_t next_rank = team.rank_me() + step;

    if (next_rank < num_ranks) {
      // will send this level
      DBG_VERBOSE("Will send rpc to ", next_rank, ", level=", level, "\n");
      level_fut = level_fut.then([level, next_rank, num_ranks, sh_dist_data, src, dst, count, &team, &sent_prom]() {
        DBG_VERBOSE("Sending rpc to ", next_rank, ", level=", level, "\n");
        const T *copy = team.rank_me() == 0 ? src : dst;  // rank0 sends its src at every level, others send dst
        rpc_ff(
            team, next_rank % team.rank_n(),
            [](DistData &dist_data, int level, view<T> prev_prefix) {
              DBG_VERBOSE("Received rpc count=", prev_prefix.size(), " level=", level, ", DistData=", dist_data.id(), "\n");
              auto &[_mysrc, _mydst, _mycount, _op, _sh_level_promises] = *dist_data;
              // needed for clang/cray to capture in lambda below
              auto &mysrc = _mysrc;
              auto &mydst = _mydst;
              auto &mycount = _mycount;
              auto &op = _op;
              auto &sh_level_promises = _sh_level_promises;
              const upcxx::team &team = dist_data.team();
              upcxx::promise<> &my_received_prom = (*sh_level_promises)[level * 2];
              upcxx::promise<> &my_sent_prom = (*sh_level_promises)[level * 2 + 1];
              assert(mycount == prev_prefix.size());

              // wait on send for this level to complete before receiving and applying partial prefix op
              upcxx::future<> wait_fut = my_sent_prom.get_future();

              auto ret = wait_fut.then([&dist_data, &op, &team, &my_received_prom, prev_prefix, level, mycount, mydst]() {
                DBG_VERBOSE("Applying received prefix level=", level, ", DistData=", dist_data.id(), ", mydst=", mydst,
                            ", prev_prefix=", prev_prefix.size(), "\n");
                assert(mycount == prev_prefix.size());
                if (level == 0 && team.rank_me() == 0) {
                  // special case of copy only
                  for (int i = 0; i < mycount; i++) {
                    mydst[i] = prev_prefix[i];
                  }
                } else {
                  for (int i = 0; i < mycount; i++) {
                    mydst[i] = op(prev_prefix[i], mydst[i]);
                  }
                }
                DBG_VERBOSE("Fulfilling receive level=", level, "\n");
                my_received_prom.fulfill_anonymous(1);
              });

              // extend lifetime of view
              return ret;
            },
            *sh_dist_data, level, make_view(copy, copy + count));
        DBG_VERBOSE("Fulfilling sent level=", level, "\n");
        sent_prom.fulfill_anonymous(1);
      });
    } else {
      // will not send this level
      DBG_VERBOSE("Not sending this level\n");
      level_fut = level_fut.then([&sent_prom, level]() {
        DBG_VERBOSE("Fulfilling (empty) sent level=", level, "\n");
        sent_prom.fulfill_anonymous(1);
      });
    }

    if (prev_rank < num_ranks) {
      DBG_VERBOSE("Expecting rpc from ", prev_rank, ", level=", level, "\n");
      level_fut = when_all(level_fut, received_prom.get_future());
    } else {
      // will not receive on this level, so fulfill after previous level
      DBG_VERBOSE("Will not receive\n");
      level_fut = level_fut.then([&received_prom, level]() {
        DBG_VERBOSE("Fulfilling (empty) recv level=", level, "\n");
        received_prom.fulfill_anonymous(1);
      });
    }

    // next level waits until this level receive has completed
    level_fut = when_all(level_fut, received_prom.get_future());
    step <<= 1;
  }

  return level_fut.then([sh_dist_data]() {
    // keep scope of DistData until complete
    // when all promises have been fulfilled, no more rpcs will be incoming
    DBG_VERBOSE("Completed reduce_prefix_binomial.\n");
  });
}

struct in_order_binary_tree_node {
  intrank_t me, n, parent, left, right, root;
  int height, my_level, my_stride;

  in_order_binary_tree_node(intrank_t _me, intrank_t _n, intrank_t skip_regular_local_ranks = 0);

  bool leftmost() const;

  bool rightmost() const;

 protected:
  void reset(intrank_t _me, intrank_t _n);
  void init(intrank_t skip_regular_local_ranks);
  void init();
};

struct binary_tree_steps {
  upcxx::promise<> ready_for_up;  // for pipelined mode this rank has started up phase
  // step 1up receive left child
  // dst = op(partial_left, me)
  upcxx::promise<> dst_is_partial_left_me;  // (ll ... j)
  // step 2up receive right child
  upcxx::promise<> scratch_is_partial_right;  // (j+1 ... rr)
  // step 3up send to parent
  upcxx::promise<> scratch_is_partial_to_parent;  // i.e. left_me_right (ll ... rr)
  upcxx::promise<> sent_partial_to_parent;        // scratch & dst can be modified again
  // step 4down receive from parent
  upcxx::promise<> scratch_is_partial_from_parent;  // i.e. (0 ... ll-1)
  // step 5down send partial_from_parent to left child
  upcxx::promise<> sent_left_child;
  // step 6 down
  // apply dst = op( partial_from_parent, dst )
  // step 7 overloads meaning for return_final_to_first on first & last nodes
  upcxx::promise<> sent_right_child;
  upcxx::promise<> dst_is_ready;

  upcxx::future<> get_future() const;
  // up phase is done

  bool up_ready() const;

  upcxx::future<> get_up_future() const;
  // down phase is done

  bool down_ready() const;

  upcxx::future<> get_down_future() const;
};

using StepPromises = binary_tree_steps;  // named promises for each step
using ShSteplPromises = shared_ptr<StepPromises>;

template <typename T, typename BinaryOp>
using Data =
    std::tuple<const T *, T *, size_t, const BinaryOp &, ShSteplPromises, shared_ptr<vector<T>>, in_order_binary_tree_node, bool>;

template <typename T, typename BinaryOp>
using DistData = dist_object<Data<T, BinaryOp>>;

template <typename T, typename BinaryOp>
using ShDistData = shared_ptr<DistData<T, BinaryOp>>;

template <typename T, typename BinaryOp>
upcxx::future<> allocate_scratch(Data<T, BinaryOp> &data) {
  auto &[src, dst, count, op, sh_proms, sh_scratch, my_node, return_final_to_first] = data;
  binary_tree_steps &proms = *sh_proms;
  if (!sh_scratch->empty()) return make_future();
  return proms.ready_for_up.get_future().then([sh_scratch = sh_scratch, count = count]() { sh_scratch->resize(count); });
}
template <typename T, typename BinaryOp>
upcxx::future<> reduce_prefix_binary_tree_up(ShDistData<T, BinaryOp> sh_dist_data) {
  auto &[src, dst, count, op, sh_proms, sh_scratch, my_node, return_final_to_first] = *(*sh_dist_data);
  binary_tree_steps &proms = *sh_proms;
  const upcxx::team &team = sh_dist_data->team();

  // started up phase so can receive data in pipelined mode
  proms.ready_for_up.fulfill_anonymous(1);

  upcxx::future<> step = make_future();
  if (my_node.me == my_node.n) {
    // created dist_object, now return
    return step;
  }

  // step 1up
  // dst == partial_left_me
  if (my_node.left < my_node.me) {
    DBG_VERBOSE("Expecting from left child ", my_node.left, "\n");
    // incoming rpc will calculate partial_left_me
  } else {
    proms.dst_is_partial_left_me.fulfill_anonymous(1);
  }
  step = when_all(step, proms.dst_is_partial_left_me.get_future());

  // step 2up
  // cratch == partial_right
  if (my_node.right < my_node.n && !my_node.rightmost()) {
    DBG_VERBOSE("Expecting from right child ", my_node.right, "\n");
    // incoming rpc will populate scratch and fulfill prom1
  } else {
    proms.scratch_is_partial_right.fulfill_anonymous(1);
  }
  step = when_all(step, proms.scratch_is_partial_right.get_future());

  // step 3up
  // calculate both subtrees and send to parent
  // prom2 - partial_right has been sent and scratch is now partial_to_parent
  //
  if (my_node.parent < my_node.n && !my_node.rightmost()) {
    DBG_VERBOSE("Sending to a parent ", my_node.parent, "\n");
    if (my_node.right < my_node.n) {
      step = when_all(step, allocate_scratch(*(*sh_dist_data)));
    }
    step = step.then(
        [sh_dist_data, src = src, dst = dst, count = count, sh_scratch = sh_scratch, my_node = my_node, &proms, &team, &op = op]() {
          DBG_VERBOSE("Calculating partial_to_parent\n");
          // calculate partial_to_parent
          // scratch has partial_to_parent from right (j+1 ... rr) if there is a right
          // if there is a left child, dst already has applied from left (ll ... j)
          // calculate partial_to_parent as (ll ... rr) in scratch.
          assert(proms.scratch_is_partial_right.get_future().is_ready());

          if (my_node.right < my_node.n) {
            assert(!sh_scratch->empty());
          }
          T *partial_right = sh_scratch->data();
          T *partial_left_right = sh_scratch->data();

          assert(proms.dst_is_partial_left_me.get_future().is_ready());
          const T *partial_left_me = my_node.left < my_node.me ? dst : src;
          const T *send_to_parent = partial_left_me;

          if (my_node.right < my_node.n) {
            // right child
            assert(sh_scratch->size() == count);
            for (int i = 0; i < count; i++) {
              partial_left_right[i] = op(partial_left_me[i], partial_right[i]);
            }
            send_to_parent = partial_left_right;
          }
          proms.scratch_is_partial_to_parent.fulfill_anonymous(1);

          // scratch now has (ll ... rr)
          upcxx::future<> send_to_parents = make_future();  // use receipt from rpc to ensure pipelined ordering
          if (my_node.parent < my_node.me) {
            // send to left parent (step 2 up)
            DBG_VERBOSE("Sending to left parent ", my_node.parent, "\n");
            send_to_parents = rpc(
                team, my_node.parent,
                [](DistData<T, BinaryOp> &dist_data, view<T> partial_right) {
                  auto &[src, dst, count, op, sh_proms, sh_scratch, my_node, rf2f] = *dist_data;
                  auto fut_scratch = allocate_scratch(*dist_data);
                  return fut_scratch.then([&dist_data, partial_right]() {
                    auto &[src, dst, count, op, sh_proms, sh_scratch, my_node, rf2f] = *dist_data;

                    assert(sh_scratch->size() == count);
                    T *scratch = sh_scratch->data();
                    DBG_VERBOSE("Receiving from right child\n");
                    assert(partial_right.size() == count);
                    for (int i = 0; i < count; i++) {
                      scratch[i] = partial_right[i];
                    }
                    // step 2up from right complete scratch = (j+1 ... rr)
                    sh_proms->scratch_is_partial_right.fulfill_anonymous(1);
                  });
                },
                *sh_dist_data, make_view(send_to_parent, send_to_parent + count));
          } else {
            assert(my_node.parent < my_node.n);
            // send to right parent (step 1 up)
            DBG_VERBOSE("Sending to right parent ", my_node.parent, "\n");
            send_to_parents = rpc(
                team, my_node.parent,
                [](DistData<T, BinaryOp> &dist_data, view<T> partial_left) {
                  auto &[src, dst, count, op, sh_proms, sh_scratch, my_node, rf2f] = *dist_data;
                  DBG_VERBOSE("Receiving from left child ", "\n");
                  assert(count == partial_left.size());
                  for (int i = 0; i < count; i++) {
                    dst[i] = op(partial_left[i], src[i]);
                  }
                  // step 1up from left complete dst = (ll ... j)
                  sh_proms->dst_is_partial_left_me.fulfill_anonymous(1);
                },
                *sh_dist_data, make_view(send_to_parent, send_to_parent + count));
          }
          // step 3 up send to parent complete
          return send_to_parents.then([&proms]() { proms.sent_partial_to_parent.fulfill_anonymous(1); });
        });
  } else {
    proms.scratch_is_partial_to_parent.fulfill_anonymous(1);
    proms.sent_partial_to_parent.fulfill_anonymous(1);
  }
  step = when_all(step, proms.scratch_is_partial_to_parent.get_future(), proms.sent_partial_to_parent.get_future());
  return step;
}

template <typename T, typename BinaryOp>
upcxx::future<> reduce_prefix_binary_tree_down(ShDistData<T, BinaryOp> sh_dist_data) {
  auto &[src, dst, count, op, sh_proms, sh_scratch, my_node, return_final_to_first] = *(*sh_dist_data);
  binary_tree_steps &proms = *sh_proms;
  const upcxx::team &team = sh_dist_data->team();

  upcxx::future<> step = make_future();
  if (my_node.me == my_node.n) {
    // created dist_object, now return
    return step;
  }

  // check that upstage is completed
  assert(proms.ready_for_up.get_future().is_ready());
  assert(proms.dst_is_partial_left_me.get_future().is_ready());
  assert(proms.scratch_is_partial_right.get_future().is_ready());
  assert(proms.scratch_is_partial_to_parent.get_future().is_ready());
  assert(proms.sent_partial_to_parent.get_future().is_ready());

  // step 4 down
  // receive from parent
  // prom 3 scratch has partial_from_parent
  // prom 4 dst has applied partial from parent and is final
  if (my_node.parent < my_node.me && !my_node.leftmost()) {
    DBG_VERBOSE("Expecting partial from left parent ", my_node.parent, "\n");
  } else if (my_node.parent < my_node.n && my_node.parent > my_node.me && !my_node.leftmost()) {
    DBG_VERBOSE("Expecting partial from right parent ", my_node.parent, "\n");
  } else {
    proms.scratch_is_partial_from_parent.fulfill_anonymous(1);
  }
  step = when_all(step, proms.scratch_is_partial_from_parent.get_future());

  // step 5 & 6 down
  // send partial_from_parent left and final to right
  vector<intrank_t> send_children;
  if (my_node.left < my_node.n && !my_node.leftmost()) {
    DBG_VERBOSE("Sending partial from parent to left child ", my_node.left, "\n");
    send_children.push_back(my_node.left);
  } else {
    proms.sent_left_child.fulfill_anonymous(1);
  }
  if (my_node.right < my_node.n) {
    DBG_VERBOSE("Sending partial from parent to right child ", my_node.right, "\n");
    send_children.push_back(my_node.right);
  } else {
    proms.sent_right_child.fulfill_anonymous(1);
  }

  upcxx::future<> rpcs_sent = step;
  for (auto child : send_children) {
    if (child < my_node.me) {
      rpcs_sent = when_all(rpcs_sent, allocate_scratch(*(*sh_dist_data)));
    }
    rpcs_sent = rpcs_sent.then(
        [sh_dist_data, dst = dst, count = count, sh_scratch = sh_scratch, child, my_node = my_node, &proms, &team]() {
          assert(proms.up_ready());
          assert(proms.scratch_is_partial_from_parent.get_future().is_ready());
          const T *send_data;
          if (child < my_node.me) {
            // relay just a copy from my parent (0 ... ll-1)
            assert(sh_scratch->size() == count);
            send_data = sh_scratch->data();
          } else {
            // send (0 ... j)
            assert(child > my_node.me);
            send_data = dst;
          }
          assert(child < my_node.n);
          DBG_VERBOSE("Sending to child ", child, "\n");
          auto rpc_completed = rpc(
              team, child,
              [](DistData<T, BinaryOp> &dist_data, view<T> partial_from_parent) {
                auto &[_src, _dst, _count, _op, _sh_proms, _sh_scratch, _my_node, rf2f] = *dist_data;
                // needed for cray/clang compiler to caputure in lambda below
                auto &src = _src;
                auto &dst = _dst;
                auto &count = _count;
                auto &op = _op;
                auto &sh_proms = _sh_proms;
                auto &sh_scratch = _sh_scratch;
                auto &my_node = _my_node;
                DBG_VERBOSE("Receiving partial_from_parent.\n");
                assert(partial_from_parent.size() == count);
                upcxx::future<> ready_fut = when_all(sh_proms->get_up_future(), allocate_scratch(*dist_data));
                upcxx::future<> fut = ready_fut.then([src, dst, count, &op, sh_proms, sh_scratch, my_node, partial_from_parent]() {
                  DBG_VERBOSE("Saving partial from parent to scratch, and applying to dst\n");
                  assert(sh_proms->up_ready());
                  assert(sh_scratch->size() == count);
                  T *scratch = sh_scratch->data();  // second range for down phase no need to wait
                  const T *partial_left_me = my_node.left < my_node.me ? dst : src;
                  for (int i = 0; i < count; i++) {
                    scratch[i] = partial_from_parent[i];
                    dst[i] = op(scratch[i], partial_left_me[i]);  // final result
                  }
                  // scratch has new partial_from_parent (0 ... ll-1)
                  sh_proms->scratch_is_partial_from_parent.fulfill_anonymous(1);
                });
                return fut;
              },
              *sh_dist_data, make_view(send_data, send_data + count));
          if (child < my_node.me) {
            proms.sent_left_child.fulfill_anonymous(1);
          } else {
            proms.sent_right_child.fulfill_anonymous(1);
          }
          return rpc_completed;
        });
  }

  step = when_all(step, rpcs_sent, proms.get_down_future())
             .then([sh_dist_data, src = src, dst = dst, count = count, my_node = my_node,
                    return_final_to_first = return_final_to_first, &proms, &team]() -> upcxx::future<> {
               assert(proms.up_ready());
               assert(proms.down_ready());
               // keep dist_object in scope
               upcxx::future<> ret = proms.dst_is_ready.get_future();
               if (my_node.me == 0) {
                 if (!return_final_to_first) {
                   DBG_VERBOSE("Restoring src to first\n");
                   for (int i = 0; i < count; i++) {
                     dst[i] = src[i];
                   }
                   proms.dst_is_ready.fulfill_anonymous(1);
                 } else {
                   ret = ret.then([sh_dist_data]() {
                     // keep dist_object in scope
                     DBG_VERBOSE("Completed. Received final_from_last.\n");
                   });
                 }
               } else {
                 proms.dst_is_ready.fulfill_anonymous(1);
               }
               upcxx::future<> rpc_returned = make_future();
               if (return_final_to_first && my_node.me == my_node.n - 1) {
                 DBG_VERBOSE("Sending final to first\n");
                 rpc_returned = rpc(
                     team, 0,
                     [](DistData<T, BinaryOp> &dist_data, view<T> final_from_last) {
                       DBG_VERBOSE("Receiving final from last\n");
                       auto &[src, dst, count, op, sh_proms, sh_scratch, my_node, rf2f] = *dist_data;
                       assert(count == final_from_last.size());
                       // no need to block, dst is never read from rank 0
                       for (int i = 0; i < count; i++) {
                         dst[i] = final_from_last[i];
                       }
                       sh_proms->dst_is_ready.fulfill_anonymous(1);
                     },
                     *sh_dist_data, make_view(dst, dst + count));
               }

               DBG_VERBOSE("Completed reduce_prefix_binary_tree.\n");
               return when_all(ret, rpc_returned).then([sh_dist_data]() {
                 auto &[src, dst, count, op, sh_proms, sh_scratch, my_node, rf2f] = *(*sh_dist_data);
                 vector<T>().swap(*sh_scratch);  // clear memory
                 DBG_VERBOSE("Released memory for reduce_prefix_binary_tree.\n");
               });
             });

  return step;
}

template <typename T, typename BinaryOp>
upcxx::future<> reduce_prefix_binary_tree(const T *src, T *dst, size_t count, const BinaryOp &op, const upcxx::team &team = world(),
                                          bool return_final_to_first = false, intrank_t skip_regular_local_ranks = 0,
                                          size_t chunk_count = 0) {
  // For binary tree with root = (N-1) / 2, height log2(N) + 1
  // for node j, with l and r possible subtrees, and parent k
  // ll is left most entry in left subtree, rr is right most entry
  // These steps happen in this order, some steps may be omitted
  //   1up) from left child: if l < j:
  //     receive lower left: partial_left (ll ... j-1)
  //     apply op (ll ... j-1) & (j) = partial_left_me (ll ... j)
  //   2up) from right child: if j < r:
  //     receive lower right: partial_right (j+1 ... rr)
  //   3up) if k < n send to parent
  //     partial_to_parent = (ll ... rr) = partial_left_me & partial_right
  //   4down) if k < n receive parent
  //     receive partial_from_parent (0 ... ll-1)
  //   5down) if l < j:
  //     send left partial_from_parent
  //   6down)
  //     final = partial_from_parent & partial_left_me (0 ... j)
  //   7down) if j < r:
  //     send right final

  DBG("count=", count, ", teamsize=", team.rank_n(), " return_final_to_first=", return_final_to_first,
      ", skip_regular_local_ranks=", skip_regular_local_ranks, ", chunk_count=", chunk_count, "\n");

  in_order_binary_tree_node my_node(team.rank_me(), team.rank_n(), skip_regular_local_ranks);

  if (my_node.n == 1 && my_node.me != my_node.n) return reduce_prefix_trivial(src, dst, count, op, team, return_final_to_first);

  if (count > 0 && chunk_count == 0) {
    chunk_count = count;  // i.e. a single pipelined chunks
  }

  // pipeline batches
  // at most 1 chunk in up phase and 1 chunk in down phase at a time per rank
  // iteratively call reduce_prreduce_prefix_binary_tree up & down

  // create and allocate all dist objects at the same time here
  size_t num_chunks = (count + chunk_count - 1) / chunk_count;
  std::vector<ShDistData<T, BinaryOp>> dist_vector;
  dist_vector.reserve(num_chunks);
  size_t offset = 0;
  for (auto i = 0; i < num_chunks; i++) {
    auto this_count = std::min(count - offset, chunk_count);

    // allocation of scratch data is always present on the dist_object
    //   it is allocated when first needed, cleared when last needed
    // scratch is the same in the up and down phases and must not be shared between them

    ShDistData<T, BinaryOp> sh_dist_data = make_shared<DistData<T, BinaryOp>>(
        make_tuple(src + offset, dst + offset, this_count, std::cref(op), make_shared<StepPromises>(), make_shared<vector<T>>(),
                   my_node, return_final_to_first));
    dist_vector.push_back(sh_dist_data);
    offset += this_count;
  }

  // now start all the pipelined calls
  // care is taken such that at most 1 up and 1 down phase are active at a time (on any given rank)
  // this limits the total memory overhead to 2*chunk_size*sizeof(T) + num_chunks * small(100 bytes maybe)
  upcxx::future<> all_up = make_future();
  upcxx::future<> all_down = make_future();
  for (auto i = 0; i < num_chunks; i++) {
    auto sh_dist_data = dist_vector[i];
    all_up = all_up.then([sh_dist_data]() -> upcxx::future<> { return reduce_prefix_binary_tree_up(sh_dist_data); });
    all_down = when_all(all_up, all_down).then([sh_dist_data]() -> upcxx::future<> {
      return reduce_prefix_binary_tree_down(sh_dist_data);
    });
  }
  return when_all(all_up, all_down);
};

//
// reduce on the local_team first, then across nodes
//

template <typename T, typename BinaryOp>
upcxx::future<> reduce_prefix_auto_choice_local(const T *src, T *dst, size_t count, const BinaryOp &op,
                                                const upcxx::team &team = world(), bool return_final_to_first = false,
                                                const upcxx::team &lteam = local_team()) {
  DBG_VERBOSE("reduce_prefix_auto_choice_local count=", count, ", lteam.rank_n()=", lteam.rank_n(), "\n");
  assert(team.rank_n() % lteam.rank_n() == 0);
  assert(team.rank_me() % lteam.rank_n() == lteam.rank_me());
  intrank_t local_n = lteam.rank_n();
  const size_t binary_tree_threshold = UPCXX_UTILS_REDUCE_PREFIX_BINARY_TREE_BYTES;  // 2MB

  // reduce over the lteam first, then the remaining with 1 rank per node participating

  // create a dist_object to hold the nodePrefix
  using NodePrefix = tuple<shared_ptr<vector<T>>, upcxx::promise<>, upcxx::promise<>>;
  using NodePrefixDO = dist_object<NodePrefix>;

  // allocate memory for the node totals and create the dist_object
  shared_ptr<NodePrefixDO> sh_node_prefix =
      make_shared<NodePrefixDO>(make_tuple(make_shared<vector<T>>(count), upcxx::promise<>(), upcxx::promise<>()), team);
  auto &[_sh_scratch, _prom_skip_node_reduce, _prom_node_prefix_shift] = **sh_node_prefix;
  // needed for cray/clang in lambda capture below
  auto &sh_scratch = _sh_scratch;
  auto &prom_skip_node_reduce = _prom_skip_node_reduce;
  auto &prom_node_prefix_shift = _prom_node_prefix_shift;

  // reduce over the lteam i.e. local_team
  // return final to first so local rank0 has sum for the node at dst
  if (sizeof(T) * count < binary_tree_threshold) {
    reduce_prefix_binomial(src, dst, count, op, lteam, true, 0).wait();
    DBG_VERBOSE("Skip reduction binomial\n");
    reduce_prefix_binomial(dst, sh_scratch->data(), count, op, team, false, local_n).wait();
  } else {
    reduce_prefix_binary_tree(src, dst, count, op, lteam, true, 0).wait();
    DBG_VERBOSE("Skip reduction binary_tree\n");
    reduce_prefix_binary_tree(dst, sh_scratch->data(), count, op, team, false, local_n).wait();
  }

  DBG_VERBOSE("Reduced locally, skip lteam ranks, reducing just lteam rank 0\n");

  upcxx::future<> fut_shift;
  if (lteam.rank_me() == 0) {
    intrank_t shift_leader = team.rank_me() + lteam.rank_n();
    if (shift_leader >= team.rank_n()) {
      assert(shift_leader == team.rank_n());
      shift_leader = 0;  // wrap
    }

    DBG_VERBOSE("Shifting skip node reductions by 1 node to shift_leader=", shift_leader, " lteam.rank_me()=", lteam.rank_me(),
                "\n");
    rpc_ff(
        team, shift_leader,
        [](NodePrefixDO &node_prefix, view<T> vals) -> upcxx::future<> {
          DBG_VERBOSE("Shifted node_prefix. lrank=", local_team().rank_me(), "\n");
          auto &[_sh_scratch, _prom_skip_node_reduce, _prom_node_prefix_shift] = *node_prefix;
          // needed for cray/clan
          auto &sh_scratch = _sh_scratch;
          auto &prom_skip_node_reduce = _prom_skip_node_reduce;
          auto &prom_node_prefix_shift = _prom_node_prefix_shift;
          // wait for scratch to send before receiving with shift values
          auto fut_copy = prom_skip_node_reduce.get_future().then([vals, sh_scratch, &prom_node_prefix_shift] {
            DBG_VERBOSE("Receiving shift to scratch\n");
            T *copy = sh_scratch->data();
            for (int i = 0; i < vals.size(); i++) {
              copy[i] = vals[i];
            }
            // scratch is shifted
            prom_node_prefix_shift.fulfill_anonymous(1);
          });

          // extend lifetime of view
          return fut_copy.then([]() { DBG_VERBOSE("Copied for broadcast\n"); });
        },
        *sh_node_prefix, make_view(sh_scratch->data(), sh_scratch->data() + count));

    // scratch is ready to receive shift
    prom_skip_node_reduce.fulfill_anonymous(1);
    // wait for scratch to become shift
    prom_node_prefix_shift.get_future().wait();

  } else {
    DBG_VERBOSE("Not reducing across team, accepting broadcast to scratch\n");
    prom_skip_node_reduce.fulfill_anonymous(1);
    prom_node_prefix_shift.fulfill_anonymous(1);
  }

  prom_skip_node_reduce.get_future().wait();   // should be noop
  prom_node_prefix_shift.get_future().wait();  // should be noop
  DBG_VERBOSE("Broadcasting with lteam\n");
  fut_shift = broadcast(sh_scratch->data(), count, 0, lteam).then([sh_node_prefix, &prom_node_prefix_shift]() {
    DBG_VERBOSE("Broadcast is complete, shifted node prefix is ready\n");
  });

  DBG_VERBOSE("Done with nodes prefix and shift\n");

  upcxx::future<> fut_final_prefix =
      fut_shift.then([sh_scratch, src, dst, count, local_n, return_final_to_first, &team, &lteam, &op]() {
        const T *node_prefix = sh_scratch->data();
        DBG_VERBOSE("Last adjustments prefix node and local results\n");
        // fix the first rank of lteam dst, since it was calculated with return_final_to_first
        if (team.rank_me() == 0 && return_final_to_first) {
          // special case for rank0 and return_final_to_first
          DBG_VERBOSE("Copying final to dst\n");
          for (int i = 0; i < count; i++) {
            dst[i] = node_prefix[i];
          }
        } else if (lteam.rank_me() == 0) {
          // restore the src on the first ranks of the local teams
          DBG_VERBOSE("Restoring src to dst\n");
          for (int i = 0; i < count; i++) {
            dst[i] = src[i];
          }
        }
        if (team.rank_me() < local_n) {
          DBG_VERBOSE("First node is done\n");
          // first node is done
        } else {
          // apply op to node and local prefixes for final values
          DBG_VERBOSE("adding node to local prefix\n");
          for (int i = 0; i < count; i++) {
            dst[i] = op(node_prefix[i], dst[i]);
          }
        }
        return;
      });

  return fut_final_prefix;
};

//
// the multi-element simplified API auto-selecting the proper algorithm for the size and scale of the prefix reduction
//

template <typename T, typename BinaryOp>
upcxx::future<> reduce_prefix_auto_choice(const T *src, T *dst, size_t count, const BinaryOp &op, const upcxx::team &team = world(),
                                          bool return_final_to_first = false, int skip_regular_local_ranks = -1) {
  DBG("count=", count, " return_final_to_first=", return_final_to_first, " skip_regular_local_ranks=", skip_regular_local_ranks,
      "\n");
  // logic to select algorithm

  // first check the size of the data to be reduced
  const size_t pipeline_threshold = UPCXX_UTILS_REDUCE_PREFIX_PIPELINE_BYTES;        // 16MB
  const size_t binary_tree_threshold = UPCXX_UTILS_REDUCE_PREFIX_BINARY_TREE_BYTES;  // 2MB
  if (count > 0 && sizeof(T) * count >= pipeline_threshold) {
    // pipeline batches of at most 1 block in flight at a time
    size_t num_blocks = (sizeof(T) * count + pipeline_threshold - 1) / pipeline_threshold;
    size_t count_per_block = (count + num_blocks + 1) / num_blocks;
    assert(sizeof(T) * count_per_block < pipeline_threshold);
    DBG("Pipeline in batches of ", count_per_block, " elements ", get_size_str(sizeof(T) * count_per_block), "\n");
    // no local team optimization with the pipelined workflow
    upcxx::future<> fut = reduce_prefix_binary_tree(src, dst, count, op, team, return_final_to_first, 0, count_per_block);
    return fut;
  }

  //
  // detect if splitting the reduction across the local_team() first might improve performance
  //
  upcxx::future<intrank_t> fut_local_n = make_future(skip_regular_local_ranks <= 0 ? 0 : skip_regular_local_ranks);
  if (skip_regular_local_ranks == -1) {
    if (local_team().rank_n() >= 4) {
      if (team.id() == local_team().id() || team.rank_n() == local_team().rank_n()) {
        // no test
      } else if (team.id() == world().id()) {
        // this is definitely regular with local team and worth the extra step
        fut_local_n = make_future(local_team().rank_n());
      } else if (sizeof(T) * count > 4096 && team.rank_n() % local_team().rank_n() == 0 &&
                 team.rank_n() > local_team().rank_n() * 16) {
        // large reduction and possibly regular.
        // This is worth a reduction test for whether the team is regular with respect to local
        fut_local_n = regular_local_team_n(team);
      }
    }
  }

  auto local_n = fut_local_n.wait();

  DBG_VERBOSE("reduce_prefix_auto_choice count=", count, ", local_n=", local_n, "\n");
  upcxx::future<> fut_final_prefix;
  if (local_n == local_team().rank_n()) {
    fut_final_prefix = reduce_prefix_auto_choice_local(src, dst, count, op, team, return_final_to_first);
  } else {
    // no local_team() optimization.  just do the full reduction
    if (count > 0 && sizeof(T) * count >= binary_tree_threshold) {
      fut_final_prefix = reduce_prefix_binary_tree(src, dst, count, op, team, return_final_to_first, 0);
    } else {
      fut_final_prefix = reduce_prefix_binomial(src, dst, count, op, team, return_final_to_first, 0);
    }
  }
  return fut_final_prefix;
}

template <typename T, typename BinaryOp>
upcxx::future<> reduce_prefix(const T *src, T *dst, size_t count, const BinaryOp &op, const upcxx::team &team = world(),
                              bool return_final_to_first = false) {
  DBG("count=", count, ", return_final_to_first=", return_final_to_first, "\n");
  return reduce_prefix_auto_choice(src, dst, count, op, team, return_final_to_first);
}

///
// the one element simplified APIs
///

#define REDUCE_PREFIX_ONE_ELEMENT(ALGORITHM)                                                                          \
  template <typename T, typename BinaryOp>                                                                            \
  upcxx::future<T> reduce_prefix_one_##ALGORITHM(const T elem, const BinaryOp &op, const upcxx::team &team = world(), \
                                                 bool return_final_to_first = false) {                                \
    T *new_vals = new T[2];                                                                                           \
    new_vals[0] = elem;                                                                                               \
    upcxx::future<> fut = reduce_prefix_##ALGORITHM(new_vals, new_vals + 1, 1, op, team, return_final_to_first);      \
    upcxx::future<T> ret_fut = fut.then([new_vals]() -> T {                                                           \
      T ret = new_vals[1];                                                                                            \
      delete[] new_vals;                                                                                              \
      return ret;                                                                                                     \
    });                                                                                                               \
    return ret_fut;                                                                                                   \
  }

REDUCE_PREFIX_ONE_ELEMENT(trivial);
REDUCE_PREFIX_ONE_ELEMENT(ring);
REDUCE_PREFIX_ONE_ELEMENT(binomial);
REDUCE_PREFIX_ONE_ELEMENT(binary_tree);
REDUCE_PREFIX_ONE_ELEMENT(auto_choice);

template <typename T, typename BinaryOp>
upcxx::future<T> reduce_prefix(const T elem, const BinaryOp &op, const upcxx::team &team = world(),
                               bool return_final_to_first = false) {
  DBG("return_final_to_first=", return_final_to_first, "\n");
  upcxx::future<T> fut = reduce_prefix_one_auto_choice(elem, op, team, return_final_to_first);
  return fut;
}

// Reduce compile time by making extern templates of common types
// template instantiations each happen in src/CMakeLists via reduce_prefix-extern-template.in.cpp

#define COMMA ,
#define MACRO_REDUCE_PREFIX(TYPE, OP, MODIFIER)                                                                                  \
  MODIFIER upcxx::future<> reduce_prefix_trivial<TYPE, OP>(const TYPE *, TYPE *, size_t, const OP &, const upcxx::team &, bool); \
  MODIFIER upcxx::future<> reduce_prefix_ring<TYPE, OP>(const TYPE *, TYPE *, size_t, const OP &, const upcxx::team &, bool,     \
                                                        intrank_t);                                                              \
  MODIFIER upcxx::future<> reduce_prefix_binomial<TYPE, OP>(const TYPE *, TYPE *, size_t, const OP &, const upcxx::team &, bool, \
                                                            intrank_t);                                                          \
  MODIFIER upcxx::future<> allocate_scratch<TYPE, OP>(Data<TYPE, OP> &);                                                         \
  MODIFIER upcxx::future<> reduce_prefix_binary_tree_up<TYPE, OP>(ShDistData<TYPE, OP>);                                         \
  MODIFIER upcxx::future<> reduce_prefix_binary_tree_down<TYPE, OP>(ShDistData<TYPE, OP>);                                       \
  MODIFIER upcxx::future<> reduce_prefix_binary_tree<TYPE, OP>(const TYPE *, TYPE *, size_t, const OP &, const upcxx::team &,    \
                                                               bool, intrank_t, size_t);                                         \
  MODIFIER upcxx::future<> reduce_prefix_auto_choice_local<TYPE, OP>(const TYPE *, TYPE *, size_t, const OP &,                   \
                                                                     const upcxx::team &, bool, const upcxx::team &);            \
  MODIFIER upcxx::future<> reduce_prefix_auto_choice<TYPE, OP>(const TYPE *, TYPE *, size_t, const OP &, const upcxx::team &,    \
                                                               bool, int);                                                       \
  MODIFIER upcxx::future<> reduce_prefix<TYPE, OP>(const TYPE *, TYPE *, size_t, const OP &, const upcxx::team &, bool);         \
  MODIFIER upcxx::future<TYPE> reduce_prefix<TYPE, OP>(const TYPE, const OP &, const upcxx::team &, bool);

/** Could not get these to work
MODIFIER class Data<TYPE,OP>; \
MODIFIER class DistData<TYPE,OP>; \
MODIFIER class ShDistData<TYPE,OP>; \
*/

MACRO_REDUCE_PREFIX(float, upcxx::detail::op_wrap<upcxx::detail::opfn_min_not_max<true> COMMA true>, extern template);
MACRO_REDUCE_PREFIX(float, upcxx::detail::op_wrap<upcxx::detail::opfn_min_not_max<false> COMMA true>, extern template);
MACRO_REDUCE_PREFIX(float, upcxx::detail::op_wrap<upcxx::detail::opfn_add COMMA true>, extern template);

MACRO_REDUCE_PREFIX(double, upcxx::detail::op_wrap<upcxx::detail::opfn_min_not_max<true> COMMA true>, extern template);
MACRO_REDUCE_PREFIX(double, upcxx::detail::op_wrap<upcxx::detail::opfn_min_not_max<false> COMMA true>, extern template);
MACRO_REDUCE_PREFIX(double, upcxx::detail::op_wrap<upcxx::detail::opfn_add COMMA true>, extern template);

MACRO_REDUCE_PREFIX(int, upcxx::detail::op_wrap<upcxx::detail::opfn_min_not_max<true> COMMA true>, extern template);
MACRO_REDUCE_PREFIX(int, upcxx::detail::op_wrap<upcxx::detail::opfn_min_not_max<false> COMMA true>, extern template);
MACRO_REDUCE_PREFIX(int, upcxx::detail::op_wrap<upcxx::detail::opfn_add COMMA true>, extern template);

MACRO_REDUCE_PREFIX(uint64_t, upcxx::detail::op_wrap<upcxx::detail::opfn_min_not_max<true> COMMA true>, extern template);
MACRO_REDUCE_PREFIX(uint64_t, upcxx::detail::op_wrap<upcxx::detail::opfn_min_not_max<false> COMMA true>, extern template);
MACRO_REDUCE_PREFIX(uint64_t, upcxx::detail::op_wrap<upcxx::detail::opfn_add COMMA true>, extern template);

MACRO_REDUCE_PREFIX(int64_t, upcxx::detail::op_wrap<upcxx::detail::opfn_min_not_max<true> COMMA true>, extern template);
MACRO_REDUCE_PREFIX(int64_t, upcxx::detail::op_wrap<upcxx::detail::opfn_min_not_max<false> COMMA true>, extern template);
MACRO_REDUCE_PREFIX(int64_t, upcxx::detail::op_wrap<upcxx::detail::opfn_add COMMA true>, extern template);
};  // namespace upcxx_utils
