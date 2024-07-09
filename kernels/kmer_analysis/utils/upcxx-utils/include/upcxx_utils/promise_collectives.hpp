#pragma once
#include <functional>
#include <typeinfo>
#include <vector>

#include "upcxx/upcxx.hpp"
#include "upcxx_utils/min_sum_max.hpp"
#include "upcxx_utils/timers.hpp"

using std::function;
using std::pair;
using std::shared_ptr;
using std::string;
using std::vector;

using upcxx::dist_object;
using upcxx::intrank_t;
using upcxx::make_future;
using upcxx::op_fast_add;
using upcxx::op_fast_max;
using upcxx::op_fast_min;
using upcxx::promise;
using upcxx::team;
using upcxx::when_all;
using upcxx::world;

#ifndef MAX_PROMISE_REDUCTIONS
#ifdef DEBUG
#define MAX_PROMISE_REDUCTIONS (1 << 3)  // 8
#else
#define MAX_PROMISE_REDUCTIONS (1 << 12)  // 4096
#endif
#endif

// #define LOG_PROMISES LOG
#define LOG_PROMISES(...)

namespace upcxx_utils {

int roundup_log2(uint64_t n);

class PromiseBarrier {
  // a two step barrier with delayed execution
  //
  // construct in the master thread (with strict ordering of collectives on team)
  // fulfill() and get_future() methods are safe to call within the restricted context
  // of a progress callback
  //
  // implements the Dissemination Algorithm
  //   detailed in "Two algorithms for Barrier Synchronization"
  //   Manber et al, 1998
  // O(logN) latency
  // O(NlogN) total small messages
  //
  // Usage:
  // PromiseBarrier prom_barrier(team);
  // ...
  // prom_barrier.fulfill();  // may call within progess() callback
  // ...
  // prom_barrier.get_future().then(...); // may call within progress() callback
  //
  struct DisseminationWorkflow;
  using DistDisseminationWorkflow = dist_object<DisseminationWorkflow>;
  struct DisseminationWorkflow {
    static void init_workflow(DistDisseminationWorkflow &dist_dissem);

    DisseminationWorkflow();
    upcxx::future<> get_future() const;

    vector<upcxx::promise<>> level_proms;  // one for each level instance
    upcxx::promise<> initiated_prom;       // to signal this rank start
    upcxx::future<> done_future;
  };

  const upcxx::team &tm;
  DistDisseminationWorkflow dist_workflow;
  bool moved = false;

 public:
  PromiseBarrier(const upcxx::team &tm = upcxx::world());
  PromiseBarrier(PromiseBarrier &&move);
  PromiseBarrier(const PromiseBarrier &copy) = delete;
  PromiseBarrier &operator=(PromiseBarrier &&move);
  PromiseBarrier &operator=(const PromiseBarrier &copy) = delete;
  ~PromiseBarrier();
  void fulfill() const;
  upcxx::future<> get_future() const;
};  // class PromiseBarrier

template <typename T>
struct BroadcastWorkflow {
  static_assert(std::is_trivial<T>::value, "T must be trivial");
  using Data = std::pair<T *, size_t>;
  using SharedData = upcxx::global_ptr<T>;

  BroadcastWorkflow(const upcxx::team &tm, int root)
      : tm(tm)
      , root(root) {
    assert(tm.rank_n() > 0);
    assert(root < tm.rank_n());
    assert(root >= 0);
    auto levels = roundup_log2(tm.rank_n());
    level_proms.resize(levels);
    DBG_VERBOSE("Constructed BroadcastWorkflow levels=", levels, " tm=", tm.rank_n(), "\n");
  }
  ~BroadcastWorkflow() { DBG_VERBOSE("Deconstruct BroadcastWorkflow\n"); }

  int get_num_levels() const { return level_proms.size(); }
  int get_rotated_rank() const { return (tm.rank_me() + tm.rank_n() - root) % tm.rank_n(); }
  int get_step(int level) const {
    int num_levels = level_proms.size();
    assert(level >= 0 && level < num_levels);
    int step = 1 << (num_levels - level - 1);
    return step;
  }
  bool is_receiving(int level) const {
    int rrank = get_rotated_rank();
    int step = get_step(level);
    return (rrank % step == 0 && rrank % (step << 1) != 0);
  }
  int next_rank(int level) const {
    int rrank = get_rotated_rank();
    int step = get_step(level);
    int next = rrank + step;
    if (next < tm.rank_n() && rrank % (step << 1) == 0) {
      return (next + root) % tm.rank_n();
    }
    return -1;
  }
  upcxx::future<> get_future() {
    upcxx::future<> fut = local_prom.get_future().then([](const auto &ignore) {});
    for (auto &prom : level_proms) fut = when_all(fut, prom.get_future());
    return when_all(fut, workflow_prom.get_future(), local_ready_prom.get_future());
  }

  const int root;
  const upcxx::team &tm;
  vector<upcxx::promise<>> level_proms;  // one for each level
  promise<Data> local_prom;
  promise<> local_ready_prom;
  promise<SharedData> shared_prom;
  promise<> workflow_prom;
};  // BroadcastWorkflow

template <typename T>
class PromiseBroadcast {
  // a two step broadcast with delayed execution

  // construct in the master thread (with strict ordering of collectives on team)
  // fulfill() and get_future() methods are safe to call within the restricted context
  // of a progress callback

  static_assert(std::is_trivial<T>::value, "T must be trivial");

 public:
  using BWF = BroadcastWorkflow<T>;
  using DistBroadcastWorkflow = dist_object<BWF>;
  using Data = typename BWF::Data;
  using SharedData = typename BWF::SharedData;

  PromiseBroadcast(int root = 0, const upcxx::team &tm = upcxx::world())
      : sh_dist_workflow(make_shared<DistBroadcastWorkflow>(tm, tm, root)) {
    assert(upcxx::master_persona().active_with_caller() && "Called from master persona");
    assert(!upcxx::in_progress() && "Not called within the restricted context");
  }
  PromiseBroadcast(const PromiseBroadcast &copy) = delete;
  PromiseBroadcast(PromiseBroadcast &&move) = delete;
  PromiseBroadcast &operator=(const PromiseBroadcast &copy) = delete;
  PromiseBroadcast &operator=(PromiseBroadcast &&move) = delete;

  void fulfill(T *data, size_t count) {
    assert(upcxx::master_persona().active_with_caller() && "Called from master persona");
    AsyncTimer timer("PromiseBroadcast::fulfill on " + std::to_string(count) + " elements " + get_size_str(sizeof(T) * count));
    timer.start();
    auto &dbw = *sh_dist_workflow;
    upcxx::team &tm = dbw.team();
    dbw->local_prom.fulfill_result(Data(data, count));

    int rrank = dbw->get_rotated_rank();
    bool need_shared = rrank % 2 == 0;
    DBG("PromiseBroadcast fulfill: dbw=", dbw.id(), " count=", count, " rrank=", rrank, " need_shared=", need_shared, "\n");
    auto fut_shared_data = dbw->shared_prom.get_future();
    upcxx::future<> fut_local_to_shared = make_future();
    if (need_shared) {
      fut_local_to_shared = when_all(dbw->local_ready_prom.get_future(), dbw->local_prom.get_future())
                                .then([&dbw, fut_shared_data](const Data &data) {
                                  if (!fut_shared_data.is_ready()) {
                                    DBG_VERBOSE("dbw=", dbw.id(), " Allocate and copy local to shared\n");
                                    SharedData my_shared_data = upcxx::new_array<T>(data.second);
                                    memcpy(my_shared_data.local(), data.first, data.second * sizeof(T));
                                    dbw->shared_prom.fulfill_result(my_shared_data);
                                    assert(fut_shared_data.is_ready());
                                  } else {
                                    DBG_VERBOSE("dbw=", dbw.id(), " shared ready already: ", fut_shared_data.wait(), "\n");
                                    assert(fut_shared_data.wait().is_local());
                                  }
                                });
    }

    auto num_levels = dbw->get_num_levels();
    upcxx::future<> fut_workflow = make_future();
    for (int level = 0; level < num_levels; level++) {
      int step = dbw->get_step(level);
      int next = dbw->next_rank(level);
      bool is_receiving = dbw->is_receiving(level);
      auto &level_prom = dbw->level_proms[level];
      auto fut_wf_ready = when_all(fut_workflow, fut_shared_data);
      DBG_VERBOSE("dbw=", dbw.id(), " level=", level, " step=", step, " next=", next, "\n");
      auto wf_lambda = [&dbw, &tm, data, count, next, level](SharedData shared_data) {
        upcxx::future<> fut_rpc = make_future();
        if (next >= 0) {  // sending to next rank
          DBG_VERBOSE("dbw=", dbw.id(), " Sending my shared_data global ptr to next=", next, "\n");
          assert(next <= tm.rank_n());
          fut_rpc = upcxx::rpc(
                        next,
                        [](DistBroadcastWorkflow &dbw, SharedData shared_data) {
                          DBG_VERBOSE("dbw=", dbw.id(), " Received rpc to rget from ", shared_data, "\n");
                          // rget from remote directly into my data
                          return dbw->local_prom.get_future().then([&dbw, shared_data](Data d) {
                            upcxx::future<> fut_data_copy;
                            if (shared_data.is_local()) {
                              // local bypass
                              DBG_VERBOSE("dbw=", dbw.id(), " local bypass shared_data=", shared_data, "\n");
                              memcpy(d.first, shared_data.local(), d.second * sizeof(T));
                              dbw->shared_prom.fulfill_result(shared_data);     // reuse shared data from local team rank
                              dbw->local_ready_prom.fulfill_anonymous(1);       // should be noop via fut_local_to_shared
                              fut_data_copy = dbw->workflow_prom.get_future();  // hold rpc response until my workflow is complete
                            } else {
                              fut_data_copy = upcxx::rget(shared_data, d.first, d.second);
                              // fulfill local_copy
                              fut_data_copy.then([&dbw]() {
                                dbw->local_ready_prom.fulfill_anonymous(1);  // allocate and copy to shared via fut_local_to_shared
                                DBG_VERBOSE("dbw=", dbw.id(), " rget copied to local\n");
                              });
                            }
                            // notify source rank that the rget is complete
                            return fut_data_copy.then([&dbw, shared_data]() {
                              DBG_VERBOSE("dbw=", dbw.id(), " completed rget/copy from ", shared_data, "\n");
                            });
                          });
                        },
                        dbw, shared_data)
                        .then([&dbw, next]() { DBG_VERBOSE("dbw=", dbw.id(), " Received RPC response from next=", next, "\n"); });
        }
        return fut_rpc.then([&dbw, level]() { DBG_VERBOSE("dbw=", dbw.id(), " Completed wf_lambda level=", level, "\n"); });
      };

      fut_workflow = fut_wf_ready.then(wf_lambda).then([&dbw, &level_prom, level]() {
        level_prom.fulfill_anonymous(1);
        DBG_VERBOSE("dbw=", dbw.id(), " level=", level, " done\n");
      });
    }

    if (rrank == 0) {
      // initiate the broadcast
      Data d(data, count);
      dbw->local_ready_prom.fulfill_anonymous(1);
    }
    fut_workflow = when_all(fut_workflow, fut_local_to_shared);
    when_all(fut_workflow, fut_shared_data).then([&dbw, sh_dist_workflow = this->sh_dist_workflow, timer](SharedData shared_data) {
      assert(&dbw == sh_dist_workflow.get());
      if (shared_data && shared_data.where() == rank_me()) upcxx::delete_array(shared_data);
      dbw->workflow_prom.fulfill_anonymous(1);
      timer.stop();
      DBG_VERBOSE("dbw=", dbw.id(), " Workflow done in ", timer.get_elapsed(), "\n");
    });
    if (!upcxx::in_progress()) upcxx::progress();
  }
  void fulfill(T &data) { fulfill(&data, 1); }

  upcxx::future<> get_future() const {
    auto &dbw = *sh_dist_workflow;
    return when_all(dbw->get_future()).then([sh_dist_workflow = this->sh_dist_workflow]() {
      DBG_VERBOSE(sh_dist_workflow->id(), " Done.\n");
    });
  }

 private:
  shared_ptr<DistBroadcastWorkflow> sh_dist_workflow;
};  // PromiseBroadcast

// PromiseReduce functor

struct op_PromiseReduce {
  using T = int64_t;
  using Func = std::function<T(T, T)>;
  using Funcd = std::function<double(double, double)>;
  using BothFunc = std::pair<Func, Funcd>;
  using FuncBool = std::pair<BothFunc, bool>;
  using Funcs = vector<FuncBool>;
  Funcs &_ops;

  // get_op() rotates through the operations each time operator() is called
  // FuncBool &get_op() const { return _ops[_rrobin++ % _ops.size()]; }
  op_PromiseReduce(Funcs &ops);
  op_PromiseReduce() = delete;
  op_PromiseReduce(const op_PromiseReduce &copy);
  static const double &T2double(const T &x);
  static const T &double2T(const double &x);

  // Each operation will be applied to an array where each operation could be different
  template <typename Ta, typename Tb>
  Ta operator()(Ta &__a, Tb &&__b) const {
    T *_a = const_cast<T *>(__a.data());
    T *_b = const_cast<T *>(__b.data());
    // DBG_VERBOSE("Operating on ", _ops.size(), " elements at ", (void*) _a, " ", (void*) _b, " this=", (void*) this, "\n");
    for (int i = 0; i < _ops.size(); i++) {
      T &a = _a[i];
      T &b = _b[i];
      FuncBool &op = _ops[i];
      // DBG_VERBOSE("Operating ", typeid(op.first).name(), " ", op.second ? " double " : " long ", " on a=", a, " b=", b, "\n");
      if (op.second) {
        const double &a_d = T2double(a);
        const double &b_d = T2double(b);
        const double a_d2 = op.first.second(a_d, b_d);
        // DBG_VERBOSE("a_d=", a_d, " b_d=", b_d, " a_d2=", a_d2, "\n");
        a = double2T(a_d2);
      } else {
        a = op.first.first(a, std::forward<T>(b));
      }
      // DBG_VERBOSE("Result a=", a, "\n");
    }
    return static_cast<Ta &&>(__a);
  };
};  // struct op_PromiseReduce

class PromiseReduce {
  // PromiseReduce
  //
  // a class to consolidate and delay a series of arbitrary reductions.
  // All values are converted to uint64_t (doubles are first bit-smashed into an int64_t and un-smashed when operating on and
  // returned)
  //   bit smashing is facilitated by op_PromiseReduce::T2double and op_PromiseReduce::double2T
  //   The aggregate return contains the double bit-smashed into int64_t, but the individual return type is the original type that
  //   reduce_* was called on.
  // Any series of binary operators is allowed
  //   ranks may initiate reduce from within the restricted contex
  //   **BUT** the order of reduce_one and reduce_all calls must be consistant across ranks
  // if any of the ops are reduce_all then all the values are broadcasted after the reduction.
  // or if all the ops are reduce_one and the root is not rank0 for a given operation, then a rpc is issued to get the value to the
  // correct rank
  //
  // fulfill can be called any number of times, and trigger a reduction for all pending reduce_one and reduce_all operations
  // fulfill can also not be called at all and will trigger all reductions before the PromiseReduce instance is destroyed
  //
  // Usage:
  //  PromiseReduce pr(local_team());  // create a PromiseReduce working on the local_team
  //  auto fut1 = pr.reduce_one(rank_me(), op_fast_max, 0); // max to rank 0
  //  auto fut2 = pr.reduce_one(rank_me(), op_fast_add, local_team().rank_n()-1); // sum to last rank on node
  //  auto fut3 = pr.reduce_all(rank_me(), op_fast_add); // sum to all ranks
  //  auto fut4 = pr.reduce_all(1.0/rank_me(), op_fast_add); // sum double to all ranks
  //  auto fut_4_reductions = pr.fulfill();
  //  assert(fut_4_redutions.wait().size() == 4);
  //  auto [ int1, int2, int3, double4 ] = fut_4_reductions.wait();
  //  assert(fut1.wait() == int1);
  //  assert(fut2.wait() == int2);
  //  assert(fut3.wait() == int3);
  //  assert(fabs(fut4.wait() - op_PromiseReduce::T2double(double4)) < 0.001);
  //  pr.reduce_all(rank_me(), op_fast_add);
  //  auto fut1_reduction = pr.fulfill(); // instance can be used to initate more reductions
  //  assert(fut_reduction.wait().size() == 1);

 public:
  using T = op_PromiseReduce::T;
  using Funcs = op_PromiseReduce::Funcs;
  using Promises = vector<shared_ptr<promise<T>>>;
  using Vals = vector<T>;

 protected:
  Vals _vals;
  const upcxx::team &tm;
  op_PromiseReduce::Funcs _ops;
  vector<int> _roots;
  Promises _proms;
  upcxx::future<> _vals_ready;

 public:
  PromiseReduce(const team &_team = world());

  ~PromiseReduce();

  static size_t &get_global_count() {
    static size_t _ = 0;
    return _;
  }

  static size_t &get_fulfilled_count() {
    static size_t _ = 0;
    return _;
  }

  const upcxx::team &get_team() const { return tm; }

  template <typename ValType, typename Op>
  upcxx::future<ValType> reduce_one(ValType orig_val, Op &op, int root = 0) {
    assert(!upcxx::in_progress() && "Not called within the restricted context");
    T val;
    bool is_float = false;
    if (std::is_floating_point<ValType>::value) {
      // val = (T)(orig_val * TRANSFORM_FLOAT);
      double conv_val = orig_val;  // upscale floats to doubles
      val = op_PromiseReduce::double2T(conv_val);
      is_float = true;
    } else
      val = orig_val;

    LOG_PROMISES("Added reduce_", (root < 0 ? "all" : "one"), " op #", _vals.size(), " ", typeid(op).name(), " ", (void *)&op,
                 "on ", typeid(ValType).name(), " val=", val, " orig_val=", orig_val, ". global_count=", get_global_count(),
                 " fulfilled_count=", get_fulfilled_count(), " this=", (void *)this, "\n");
    get_global_count()++;
    auto sh_prom = make_shared<promise<T>>();
    _vals.reserve(MAX_PROMISE_REDUCTIONS);
    _proms.reserve(MAX_PROMISE_REDUCTIONS);
    _ops.reserve(MAX_PROMISE_REDUCTIONS);
    _roots.reserve(MAX_PROMISE_REDUCTIONS);
    _vals.push_back(val);
    _proms.push_back(sh_prom);
    _ops.push_back({{op, op}, is_float});
    _roots.push_back(root);
    auto fut_ret = sh_prom->get_future().then([&self = *this, sh_prom, is_float, sz = _vals.size(), root](const T &result) {
      ValType conv_result = result;
      if (std::is_floating_point<ValType>::value) {
        assert(is_float);
        double double_result = op_PromiseReduce::T2double(result);
        conv_result = double_result;  // downscale doubles to floats if needed
        DBG_VERBOSE("Converted reduce_", (root < 0 ? "all" : "one"), " op #", sz - 1, " from ", result, " back to ", conv_result,
                    "\n");
      } else {
        assert(!is_float);
      }
      DBG_VERBOSE("Returning ", conv_result, " sh_prom:", (void *)sh_prom.get(), "\n");
      return conv_result;
    });
    DBG_VERBOSE("promise on ", (void *)this, " pending: ", _vals.size(), " global_count=", get_global_count(),
        " fulfilled_count=", get_fulfilled_count(), "\n");
    upcxx::progress();
    return fut_ret;
  };

  template <typename ValType, typename Op>
  upcxx::future<> reduce_one(ValType *in_vals, ValType *out_vals, int count, Op &op, int root = 0) {
    upcxx::future<> chain_fut = make_future();
    for (int i = 0; i < count; count++) {
      auto fut_val = reduce_one(in_vals[i], op, root).then([&out = out_vals[i]](const ValType &val) { return out = val; });
      chain_fut = when_all(chain_fut, fut_val.then([](const ValType &ignore) {}));
    }
    return chain_fut;
  };

  template <typename ValType, typename Op>
  upcxx::future<ValType> reduce_all(ValType orig_val, Op &op) {
    return reduce_one(orig_val, op, -1);
  };

  template <typename ValType, typename Op>
  upcxx::future<> reduce_all(ValType *in_vals, ValType *out_vals, int count, Op &op) {
    return reduce_one(in_vals, out_vals, count, op, -1);
  };

  template <typename ValType, typename Op>
  upcxx::future<ValType> fut_reduce_one(upcxx::future<ValType> fut_orig_val, Op &op, int root = 0) {
    assert(!upcxx::in_progress() && "Not called within the restricted context");
    _vals_ready = when_all(_vals_ready, fut_orig_val.then([](auto ignored) {}));
    auto fut_ret = reduce_one((ValType)0, op, root);
    auto fut_val_set = fut_orig_val.then([&val = _vals.back()](auto new_val) {
      // Capture of vector element reference is safe because reduce_one reserves the vector
      // and fulfill swaps the vector
      val = new_val;
    });
    _vals_ready = when_all(_vals_ready, fut_val_set);
    return fut_ret;
  }

  template <typename ValType, typename Op>
  upcxx::future<ValType> fut_reduce_all(upcxx::future<ValType> fut_orig_val, Op &op) {
    return fut_reduce_one(fut_orig_val, op, -1);
  }

  template <typename ValType>
  upcxx::future<MinSumMax<ValType>> msm_reduce_one(ValType orig_val, bool is_active, int root = 0) {
    assert(!upcxx::in_progress());
    assert(root == -1 || root < get_team().rank_n());
    auto fut_my = make_future(orig_val);
    auto fut_min = reduce_one(orig_val, op_fast_min, root);
    auto fut_sum = reduce_one(orig_val, op_fast_add, root);
    auto fut_max = reduce_one(orig_val, op_fast_max, root);
    auto fut_active = reduce_one((int)(is_active ? 1 : 0), op_fast_add, root);
    return when_all(fut_my, fut_min, fut_sum, fut_max, fut_active)
        .then([this, is_active, root](const ValType &my, const ValType &min, const ValType &sum, const ValType &max,
                                      int active_ranks) {
          DBG_VERBOSE("my=", my, " min=", min, " max=", max, " active_ranks=", active_ranks, " is_active=", is_active,
                      " root=", root, "\n");
          MinSumMax<ValType> msm;
          msm.my = my;
          msm.min = min;
          msm.sum = sum;
          msm.max = max;
          assert(root != this->get_team().rank_me() || (active_ranks >= 0 && active_ranks <= this->get_team().rank_n()));
          msm.active_ranks = active_ranks;
          msm.apply_avg();
          return msm;
        });
  };

  template <typename ValType>
  upcxx::future<MinSumMax<ValType>> msm_reduce_one(ValType orig_val, int root = 0) {
    return msm_reduce_one(orig_val, true, root);
  };

  template <typename ValType>
  upcxx::future<MinSumMax<ValType>> msm_reduce_all(ValType orig_val, bool is_active = true) {
    return msm_reduce_one(orig_val, is_active, -1);
  };

  template <typename T>
  upcxx::future<> msm_reduce_one(const MinSumMax<T> *msm_in, MinSumMax<T> *msm_out, int count, intrank_t root = 0) {
    upcxx::future<> chain_fut = make_future();
    for (int i = 0; i < count; i++) {
      auto &out = msm_out[i];
      auto &in = msm_in[i];
      auto fut_msm =
          msm_reduce_one(msm_in[i].my, msm_in[i].active_ranks >= 1, root).then([&in, &out](const MinSumMax<T> &msm) { out = msm; });
      chain_fut = when_all(chain_fut, fut_msm);
    }
    return chain_fut;
  };

  template <typename T>
  upcxx::future<> msm_reduce_all(const MinSumMax<T> *msm_in, MinSumMax<T> *msm_out, int count) {
    return msm_reduce_one(msm_in, msm_out, count, -1);
  }

  template <typename ValType>
  upcxx::future<MinSumMax<ValType>> fut_msm_reduce_one(upcxx::future<ValType> fut_orig_val, upcxx::future<bool> fut_is_active,
                                                       int root = 0) {
    assert(!upcxx::in_progress() && "Not called within the restricted context");
    _vals_ready = when_all(_vals_ready, fut_orig_val.then([](auto ignored) {}), fut_is_active.then([](auto ignored) {}));
    auto fut_ret = msm_reduce_one((ValType)0, true, root);
    auto fut_val_set =
        when_all(fut_orig_val, fut_is_active).then([&_vals = this->_vals, sz = this->_vals.size()](auto new_val, bool is_active) {
          // Capture of vector element reference is safe because reduce_one reserves the vector
          // and fulfill swaps the vector
          assert(sz >= 4);
          // set min sum & max
          for (int i = sz - 4; i < sz - 1; i++) {
            _vals[i] = new_val;
          }
          _vals[sz - 1] = is_active;
        });
    _vals_ready = when_all(_vals_ready, fut_val_set);
    return fut_ret;
  }

  template <typename ValType>
  upcxx::future<MinSumMax<ValType>> fut_msm_reduce_all(upcxx::future<ValType> fut_orig_val, upcxx::future<bool> fut_is_active) {
    return fut_msm_reduce_one(fut_orig_val, fut_is_active, -1);
  }

  upcxx::future<> fulfill();
  upcxx::future<> bulk_reduce(shared_ptr<Vals> sh_vals, shared_ptr<Vals> sh_results, shared_ptr<Funcs> sh_ops,
                              bool requires_broadcast);

};  // class PromiseReduce

};  // namespace upcxx_utils
