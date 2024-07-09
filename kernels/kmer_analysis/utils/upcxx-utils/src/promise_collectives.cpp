// two_step_barrier.cpp
#include "upcxx_utils/promise_collectives.hpp"

#include <cassert>
#include <typeinfo>

#include "upcxx_utils/log.hpp"
#include "upcxx_utils/timers.hpp"

using upcxx::future;
using upcxx::make_future;
using upcxx::when_all;

namespace upcxx_utils {

int roundup_log2(uint64_t n) {
  // DBG("n=", n, "\n");
  if (n == 0) return -1;
#define S(k)                     \
  if (n >= (UINT64_C(1) << k)) { \
    i += k;                      \
    n >>= k;                     \
  }

  n--;
  // DBG(" n=", n, "\n");
  n |= n >> 1;
  // DBG(" n=", n, "\n");
  n |= n >> 2;
  // DBG(" n=", n, "\n");
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  n++;
  // DBG(" n=", n, "\n");

  int i = 0;
  S(32);
  S(16);
  S(8);
  S(4);
  S(2);
  S(1);

  // DBG(" n=", n, " i=", i, "\n");
  return i;
#undef S
}

PromiseBarrier::DisseminationWorkflow::DisseminationWorkflow()
    : level_proms()  // empty
    , initiated_prom()
    , done_future()  // never ready
{}

void PromiseBarrier::DisseminationWorkflow::init_workflow(DistDisseminationWorkflow &dist_dissem) {
  DBG_VERBOSE("\n");
  assert(upcxx::master_persona().active_with_caller());

  DisseminationWorkflow &dissem = *dist_dissem;
  assert(dissem.level_proms.size() == 0);

  const upcxx::team &tm = dist_dissem.team();
  intrank_t me = tm.rank_me(), n = tm.rank_n();
  int levels = roundup_log2(n);
  dissem.level_proms.resize(levels);

  intrank_t mask = 0x01;
  future<> fut_chain = dissem.initiated_prom.get_future();
  for (int level = 0; level < levels; level++) {
    intrank_t send_to = (me + mask) % n;

    fut_chain = fut_chain.then([&dist_dissem, &tm, send_to, level]() {
      rpc_ff(
          tm, send_to,
          [](DistDisseminationWorkflow &dist_dissem, int level) {
            auto &prom = dist_dissem->level_proms[level];
            prom.fulfill_anonymous(1);
          },
          dist_dissem, level);
    });

    auto &prom = dissem.level_proms[level];
    fut_chain = when_all(fut_chain, prom.get_future());
    mask <<= 1;
  }
  dissem.done_future = fut_chain.then([&dissem]() {
    vector<upcxx::promise<>> empty{};
    dissem.level_proms.swap(empty);  // cleanup memory
  });
}

upcxx::future<> PromiseBarrier::DisseminationWorkflow::get_future() const { return done_future; }

PromiseBarrier::PromiseBarrier(const upcxx::team &tm)
    : tm(tm)
    , dist_workflow(tm) {
  DBG_VERBOSE("tm.n=", tm.rank_n(), " this=", this, "\n");
  assert(upcxx::master_persona().active_with_caller() && "Called from master persona");
  assert(!upcxx::in_progress() && "Not called within the restricted context");
  DisseminationWorkflow::init_workflow(dist_workflow);
  progress();
}

PromiseBarrier::PromiseBarrier(PromiseBarrier &&mv)
    : tm(mv.tm)
    , dist_workflow(std::move(mv.dist_workflow)) {
  DBG_VERBOSE("moved mv=", &mv, " to this=", this, "\n");
  mv.moved = true;
}

PromiseBarrier &upcxx_utils::PromiseBarrier::operator=(PromiseBarrier &&mv) {
  PromiseBarrier newme(std::move(mv));
  std::swap(*this, newme);
  DBG_VERBOSE("Swapped newme=", &newme, " to this=", this, "\n");
  return *this;
}

PromiseBarrier::~PromiseBarrier() {
  DBG_VERBOSE("Destroy this=", this, " move=", moved, "\n");
  if (moved) return;  // invalidated
  assert(upcxx::master_persona().active_with_caller() && "Called from master persona");
  assert(dist_workflow->initiated_prom.get_future().is_ready());
  get_future().wait();
}

void PromiseBarrier::fulfill() const {
  DBG_VERBOSE("fulfill this=", this, "\n");
  assert(upcxx::master_persona().active_with_caller());
  assert(!dist_workflow->initiated_prom.get_future().is_ready());
  dist_workflow->initiated_prom.fulfill_anonymous(1);
}

upcxx::future<> PromiseBarrier::get_future() const {
  assert(upcxx::master_persona().active_with_caller());
  return dist_workflow->get_future();
}

// PromiseReduce class and helper functor struct

op_PromiseReduce::op_PromiseReduce(op_PromiseReduce::Funcs &ops)
    : _ops(ops) {
  assert(!upcxx::in_progress());
}
op_PromiseReduce::op_PromiseReduce(const op_PromiseReduce &copy)
    : _ops(copy._ops) {}

const double &op_PromiseReduce::T2double(const op_PromiseReduce::T &x) {
  // Just mash the int64_t bits into a double
  assert(sizeof(T) == sizeof(double));
  const void *t_ptr = &x;
  const double *d_ptr = reinterpret_cast<const double *>(t_ptr);
  const double &d = *d_ptr;
  // DBG_VERBOSE("T2double x=", x, " d=", d, "\n");
  return d;
}

const op_PromiseReduce::T &op_PromiseReduce::double2T(const double &x) {
  // Just mash the double bits into a int64_t
  assert(sizeof(T) == sizeof(double));
  const void *d_ptr = &x;
  const T *t_ptr = reinterpret_cast<const T *>(d_ptr);
  const T &t = *t_ptr;
  // DBG_VERBOSE("double2T x=", x, " t=", t, "\n");
  return t;
}

PromiseReduce::PromiseReduce(const team &_team)
    : _vals{}
    , tm(_team)
    , _ops{}
    , _roots{}
    , _proms{}
    , _vals_ready(make_future()) {
  assert(upcxx::master_persona().active_with_caller() && "Called from master persona");
  assert(!upcxx::in_progress() && "Not called within the restricted context");
}

PromiseReduce::~PromiseReduce() {
  if (upcxx::initialized()) {
    assert(upcxx::master_persona().active_with_caller() && "Called from master persona");
    assert(!upcxx::in_progress() && "Not called within the restricted context");
    if (!_vals.empty()) {
      WARN("PromiseReduce destructor found pending reductions\n");
      fulfill();
    }
  } else {
    if (!_vals.empty()) DIE("Pending ", _vals.size(), " PromiseReducitons in ", (void *)this, "\n");
    assert(_vals.empty() && "All PromiseReduce options are completed");
  }
}

static uint32_t crc32c(uint32_t crc, const unsigned char *buf, size_t len) {
  int k;

  crc = ~crc;
  while (len--) {
    crc ^= *buf++;
    for (k = 0; k < 8; k++) crc = crc & 1 ? (crc >> 1) ^ 0xedb88320 : crc >> 1;
  }
  return ~crc;
}

upcxx::future<> PromiseReduce::fulfill() {
  assert(!upcxx::in_progress() && "Not called within the restricted context");
  assert(upcxx::master_persona().active_with_caller() && "Called from master persona");

  AsyncTimer fulfill_t(*this, "fulfill reductions of " + std::to_string(_vals.size()) + " at global_count=" +
                                  std::to_string(get_global_count()) + " fulfilled_count=" + std::to_string(get_fulfilled_count()));
  fulfill_t.start();

  auto sz = _vals.size();

#if DEBUG
  auto all_max_sz = upcxx::reduce_all(sz, op_fast_max).wait();
  if (all_max_sz != sz || sz != _vals.size()) {
    DIE("Not all ranks have the same size for PromiseReduce ops all_max_sz=", all_max_sz, " my=", sz, " size()=", _vals.size());
  }
  static auto lambda_crc = [](uint32_t crc, const unsigned char *buf, size_t len) {
    int k;
    crc = ~crc;
    while (len--) {
      crc ^= *buf++;
      for (k = 0; k < 8; k++) crc = crc & 1 ? (crc >> 1) ^ 0xedb88320 : crc >> 1;
    }
    return ~crc;
  };
  int64_t op_checksum = lambda_crc(0L, nullptr, 0);
  for (auto &op : _ops) {
    auto hash = typeid(op).hash_code();
    op_checksum = lambda_crc(op_checksum, (const unsigned char *)&hash, sizeof(hash));
  }
  auto max_checksum = upcxx::reduce_all(op_checksum, op_fast_max).wait();
  if (max_checksum != op_checksum) {
    DIE("Not all ranks have the same operation checksum max=", max_checksum, " my=", op_checksum);
  }
#endif

  assert(_vals.size() == sz);
  assert(_proms.size() == sz);
  assert(_ops.size() == sz);
  assert(_roots.size() == sz);

  if (_vals.empty()) {
    DBG("No pending reductions for ", (void *)this, "\n");
    return make_future();
  }

  shared_ptr<Promises> sh_proms = make_shared<Promises>();
  sh_proms->swap(_proms);

  future<> fut_chain = make_future();
  size_t offset = 0;
  while (offset < sz) {
    auto batch_size = sz - offset;
    if (batch_size > MAX_PROMISE_REDUCTIONS) batch_size = MAX_PROMISE_REDUCTIONS;

    // copy ops, vals
    // index promises & roots
    // allocate results
    shared_ptr<Funcs> sh_ops = make_shared<Funcs>();
    sh_ops->reserve(MAX_PROMISE_REDUCTIONS);
    sh_ops->insert(sh_ops->end(), _ops.begin() + offset, _ops.begin() + offset + batch_size);

    shared_ptr<Vals> sh_vals = make_shared<Vals>();
    sh_vals->reserve(MAX_PROMISE_REDUCTIONS);
    sh_vals->insert(sh_vals->end(), _vals.begin() + offset, _vals.begin() + offset + batch_size);

    shared_ptr<Vals> sh_results = make_shared<Vals>();
    sh_results->reserve(MAX_PROMISE_REDUCTIONS);
    sh_results->insert(sh_results->end(), batch_size, std::numeric_limits<T>::max());

    assert(sh_vals->size() == batch_size);
    assert(sh_ops->size() == batch_size);

    // determine the operations with non-zero roots and reduce_all
    // that have different needs for syncing
    int count_reduce_all = 0, count_not_root = 0;
    for (int i = 0; i < batch_size; i++) {
      auto &root = _roots[i + offset];
      if (root < 0)
        count_reduce_all++;
      else if (root > 0)
        count_not_root++;
    }

    get_fulfilled_count() += batch_size;
    LOG_PROMISES("Fulfilling ", batch_size, " reducts, ", count_reduce_all, " reduce_all and ", batch_size - count_reduce_all,
                 " reduce_one with ", count_not_root,
                 " on non-zero-root combined reduction. new fulfilled_count=", get_fulfilled_count(), " this=", (void *)this, "\n");

    assert(sh_vals->size() == sh_results->size());
    assert(sh_vals->size() == batch_size);
    assert(sh_results->size() == batch_size);

    if (!_vals_ready.is_ready()) {
      LOG("Waiting for future (promised) vals to become ready\n");
      _vals_ready.wait();
    }

    // bulk reduce by batch
    bool requires_broadcast = count_reduce_all > 0 || count_not_root > 0;

    auto fut_bulk_reduce = bulk_reduce(sh_vals, sh_results, sh_ops, requires_broadcast);

    auto fut_ret = fut_bulk_reduce.then([&_team = get_team(), sh_results, sh_proms, offset, batch_size]() {
      assert(sh_results->size() == batch_size);
      assert(sh_proms->size() >= offset + batch_size);
      for (int i = 0; i < batch_size; i++) {
        promise<T> &prom = *((*sh_proms)[i + offset]);
        prom.fulfill_result((*sh_results)[i]);
      }
    });

    offset += batch_size;
    fut_chain = when_all(fut_chain, fut_ret);
    progress();
  }
  fut_chain = fut_chain.then([fulfill_t, sz]() { fulfill_t.stop(); });
  _ops.clear();
  _vals.clear();
  _roots.clear();
  _proms.clear();

  Timings::set_pending(fut_chain);
  return fut_chain;
}

future<> PromiseReduce::bulk_reduce(shared_ptr<Vals> sh_vals, shared_ptr<Vals> sh_results, shared_ptr<Funcs> sh_ops,
                                    bool requires_broadcast) {
  assert(!upcxx::in_progress() && "Not called within the restricted context");
  assert(upcxx::master_persona().active_with_caller() && "Called from master persona");

  auto sz = sh_vals->size();
  AsyncTimer timer("PromiseReduce::bulk_reduce on " + std::to_string(sz));
  timer.start();

  assert(sz > 0);
  assert(sz <= MAX_PROMISE_REDUCTIONS);

  assert(sz == sh_results->size());
  assert(sz == sh_ops->size());

  int power = roundup_log2(sz);
  auto max_sz = 1 << power;
  assert(max_sz <= MAX_PROMISE_REDUCTIONS);
  DBG("bulk_reduce using power=", power, " sz=", sz, " max_sz=", max_sz, "\n");

  // make sure src and dest variables in the reduction have the maximum memory allocated
  sh_vals->reserve(max_sz);
  sh_results->reserve(max_sz);

  auto sh_my_op = make_shared<op_PromiseReduce>(*sh_ops);

  future<> fut_reduce;

  switch (power) {
    case (0):
      using T0 = std::array<T, 1>;
      assert(sz * sizeof(T) <= sizeof(T0));
      fut_reduce = upcxx::reduce_one<T0>((T0 *)sh_vals->data(), (T0 *)sh_results->data(), 1, *sh_my_op, 0, get_team());
      break;
    case (1):
      using T1 = std::array<T, 2>;
      assert(sz * sizeof(T) <= sizeof(T1));
      fut_reduce = upcxx::reduce_one<T1>((T1 *)sh_vals->data(), (T1 *)sh_results->data(), 1, *sh_my_op, 0, get_team());
      break;
    case (2):
      using T2 = std::array<T, 4>;
      assert(sz * sizeof(T) <= sizeof(T2));
      fut_reduce = upcxx::reduce_one<T2>((T2 *)sh_vals->data(), (T2 *)sh_results->data(), 1, *sh_my_op, 0, get_team());
      break;
    case (3):
      using T3 = std::array<T, 8>;
      assert(sz * sizeof(T) <= sizeof(T3));
      fut_reduce = upcxx::reduce_one<T3>((T3 *)sh_vals->data(), (T3 *)sh_results->data(), 1, *sh_my_op, 0, get_team());
      break;
    case (4):
      using T4 = std::array<T, 16>;
      assert(sz * sizeof(T) <= sizeof(T4));
      fut_reduce = upcxx::reduce_one<T4>((T4 *)sh_vals->data(), (T4 *)sh_results->data(), 1, *sh_my_op, 0, get_team());
      break;
    case (5):
      using T5 = std::array<T, 32>;
      assert(sz * sizeof(T) <= sizeof(T5));
      fut_reduce = upcxx::reduce_one<T5>((T5 *)sh_vals->data(), (T5 *)sh_results->data(), 1, *sh_my_op, 0, get_team());
      break;
    case (6):
      using T6 = std::array<T, 64>;
      assert(sz * sizeof(T) <= sizeof(T6));
      fut_reduce = upcxx::reduce_one<T6>((T6 *)sh_vals->data(), (T6 *)sh_results->data(), 1, *sh_my_op, 0, get_team());
      break;
    case (7):
      using T7 = std::array<T, 128>;
      assert(sz * sizeof(T) <= sizeof(T7));
      fut_reduce = upcxx::reduce_one<T7>((T7 *)sh_vals->data(), (T7 *)sh_results->data(), 1, *sh_my_op, 0, get_team());
      break;
    case (8):
      using T8 = std::array<T, 256>;
      assert(sz * sizeof(T) <= sizeof(T8));
      fut_reduce = upcxx::reduce_one<T8>((T8 *)sh_vals->data(), (T8 *)sh_results->data(), 1, *sh_my_op, 0, get_team());
      break;
    case (9):
      using T9 = std::array<T, 512>;
      assert(sz * sizeof(T) <= sizeof(T9));
      fut_reduce = upcxx::reduce_one<T9>((T9 *)sh_vals->data(), (T9 *)sh_results->data(), 1, *sh_my_op, 0, get_team());
      break;
    case (10):
      using T10 = std::array<T, 1024>;
      assert(sz * sizeof(T) <= sizeof(T10));
      fut_reduce = upcxx::reduce_one<T10>((T10 *)sh_vals->data(), (T10 *)sh_results->data(), 1, *sh_my_op, 0, get_team());
      break;
    case (11):
      using T11 = std::array<T, 2048>;
      assert(sz * sizeof(T) <= sizeof(T11));
      fut_reduce = upcxx::reduce_one<T11>((T11 *)sh_vals->data(), (T11 *)sh_results->data(), 1, *sh_my_op, 0, get_team());
      break;
    case (12):
      using T12 = std::array<T, 4096>;
      assert(sz * sizeof(T) <= sizeof(T12));
      fut_reduce = upcxx::reduce_one<T12>((T12 *)sh_vals->data(), (T12 *)sh_results->data(), 1, *sh_my_op, 0, get_team());
      break;
    case (13):
      using T13 = std::array<T, 8192>;
      assert(sz * sizeof(T) <= sizeof(T13));
      fut_reduce = upcxx::reduce_one<T13>((T13 *)sh_vals->data(), (T13 *)sh_results->data(), 1, *sh_my_op, 0, get_team());
      break;
    case (14):
      using T14 = std::array<T, 16384>;
      assert(sz * sizeof(T) <= sizeof(T14));
      fut_reduce = upcxx::reduce_one<T14>((T14 *)sh_vals->data(), (T14 *)sh_results->data(), 1, *sh_my_op, 0, get_team());
      break;
    case (15):
      using T15 = std::array<T, 32768>;
      assert(sz * sizeof(T) <= sizeof(T15));
      fut_reduce = upcxx::reduce_one<T15>((T15 *)sh_vals->data(), (T15 *)sh_results->data(), 1, *sh_my_op, 0, get_team());
      break;
    case (16):
      using T16 = std::array<T, 65536>;
      assert(sz * sizeof(T) <= sizeof(T16));
      fut_reduce = upcxx::reduce_one<T16>((T16 *)sh_vals->data(), (T16 *)sh_results->data(), 1, *sh_my_op, 0, get_team());
      break;
    default: DIE("Cannot handle this yet power=", power, " sz=", sz); break;
  }

  future<> fut_results = fut_reduce.then([sh_my_op, sz, sh_vals, sh_results]() {
    assert(sh_vals->size() == sz);
    assert(sh_results->size() == sz);
    DBG("Finished reduction\n");
    return;
  });

  if (requires_broadcast) {
    // broadcast to all ranks as some are all reducations or to root!=0
    auto sh_promise_broadcast = make_shared<PromiseBroadcast<T>>(0, get_team());
    fut_results = fut_results.then([sh_promise_broadcast, sh_results, sz]() {
      DBG("Broadcasting results sz=", sz, "\n");
      sh_promise_broadcast->fulfill(sh_results->data(), sz);
      return sh_promise_broadcast->get_future();
    });
  }

  return fut_results.then([sh_my_op, sh_vals, sh_results, sh_ops, timer]() { timer.stop(); });  // keep shared ptrs in scope
}

};  // namespace upcxx_utils
