#pragma once

// gather.hpp
#include <memory>
#include <upcxx/upcxx.hpp>
#include <upcxx_utils/log.hpp>
#include <upcxx_utils/promise_collectives.hpp>
#include <vector>

using upcxx::dist_object;
using upcxx::intrank_t;
using upcxx::make_future;
using upcxx::rpc_ff;
using upcxx::when_all;

namespace upcxx_utils {
// implement gather and all gather

// dest_buf on root rank must be sufficiently large to hold send_count * tm.rank_n() elements of T
template <typename T>
upcxx::future<> linear_gather(const T* send_buf, size_t send_count, T* dest_buf, intrank_t root,
                              const upcxx::team& tm = upcxx::world(), intrank_t skip_ranks = 1) {
  DBG("linear_gather send_count=", send_count, " root=", root, " n=", tm.rank_n(), " skip=", skip_ranks, "\n");
  assert(skip_ranks > 0 && "skip ranks is not 0");
  // if (skip_ranks > 1 && root != 0) DIE("Can not both skip and have a non-zero root (yet)");
  assert((skip_ranks == 1 || root < skip_ranks) && "root is less than skip");
  using Pair = std::pair<upcxx::promise<>, T*>;
  using DO = upcxx::dist_object<Pair>;
  using ShDO = std::shared_ptr<DO>;

  if (tm.rank_me() == root) {
    // skip extra serialization and copies for root
    std::copy(send_buf, send_buf + send_count, dest_buf);
  }

  // short circuit the trivial case
  if (tm.rank_n() == 1) return upcxx::make_future();

  ShDO sh_dist_prom = std::make_shared<DO>(tm, Pair(tm.rank_n() / skip_ranks, dest_buf));
  DO& dist_prom = *sh_dist_prom;

  if (tm.rank_me() == root) {
    dist_prom->first.fulfill_anonymous(1);
  } else if (skip_ranks == 1 || tm.rank_me() % skip_ranks == root) {
    auto rrank_me = (tm.rank_me() + tm.rank_n() - root) % tm.rank_n();  // rotated rank for the ordering of the gathered data
    rrank_me /= skip_ranks;                                             // adjusted for any possible skip_ranks (>= 1)
    assert(tm.rank_me() != root && "not the root");
    rpc_ff(
        tm, root,
        [](DO& dist_prom, intrank_t from_rrank, const upcxx::view<T>& payload, int64_t send_count, int root) {
          auto& tm = dist_prom.team();
          assert(from_rrank != 0 && "rroot did not send an rpc to this rroot");
          assert((from_rrank + root) % tm.rank_n() != tm.rank_me());
          assert(tm.rank_me() == root);
          DBG_VERBOSE("writing payload=", payload.size(), " from rrank=", from_rrank, ": ", *payload.begin(), "\n");
          assert(send_count == payload.size() && "same send_count across all ranks");
          T* dest = (dist_prom->second) + from_rrank * payload.size();
          std::copy(payload.begin(), payload.end(), dest);
          dist_prom->first.fulfill_anonymous(1);
        },
        dist_prom, rrank_me, upcxx::make_view(send_buf, send_buf + send_count), send_count, root);
    DBG_VERBOSE("Fired rpc_ff to ", root, " as rotated ", rrank_me, "\n");
  }

  if (tm.rank_me() != root) dist_prom->first.fulfill_anonymous(tm.rank_n() / skip_ranks);
  DBG_VERBOSE("Done with linear gather prep\n");
  return dist_prom->first.get_future().then([sh_dist_prom]() { DBG_VERBOSE("Completed linear gather\n"); });
};

template <typename T>
upcxx::future<std::vector<T>> linear_gather(const T* send_buf, size_t send_count, intrank_t root,
                                            const upcxx::team& tm = upcxx::world(), intrank_t skip_ranks = 1) {
  bool am_root = root == tm.rank_me();
  auto sh_dest = std::make_shared<std::vector<T>>(am_root ? tm.rank_n() * send_count : 0);
  auto fut = linear_gather(send_buf, send_count, sh_dest->data(), root, tm, skip_ranks);
  return fut.then([sh_dest]() {
    std::vector result(std::move(*sh_dest));
    return result;
  });
}

template <typename T>
upcxx::future<std::vector<T>> linear_gather(const T& send_buf, intrank_t root, const upcxx::team& tm = upcxx::world(),
                                            intrank_t skip_ranks = 1) {
  return linear_gather(&send_buf, 1, root, tm, skip_ranks);
}

// dest_buf on root rank must be sufficiently large to hold send_count * tm.rank_n() elements of T
template <typename T>
upcxx::future<> binomial_gather(const T* send_buf, size_t send_count, T* dest_buf, intrank_t root,
                                const upcxx::team& tm = upcxx::world()) {
  intrank_t rrank = (tm.rank_me() + tm.rank_n() - root) % tm.rank_n();  // rotated rank_me -- ordering of data will be rotated too!
  DBG("binomial_gather(send_count=", send_count, " root=", root, " n=", tm.rank_n(), " dest_buf=", (void*)dest_buf, "\n");
  const T* orig_dest_buf = dest_buf;
  // short circuit the trivial case
  if (tm.rank_n() == 1) return linear_gather(send_buf, send_count, dest_buf, root, tm);

  // 7 rank example, 3 levels
  // level 0, offset 1: 1->0, 3->2, 5->4
  // level 1, offset 2: (3+)2->0,  6->4
  // level 2, offset 4: (6+5+)4->0

  // 13 example, 4 levels
  // level 0, offset 1: 1->0, 3->2, 5->4, 7->6, 9->8, 11->10, 12
  // level 1, offset 2: (3+)2->0, (7+)6->4, (11+)10->8,
  // level 2, offset 4: (7+6+5+)4->0, 12->8
  // level 3, offset 8: (12+11+10+9+)8->0

  // each rank first receives up to [0-num_levels) messages
  // one receive buffer is reused and appended in each level
  // then sends the appended message once
  // rotated_rank 0 only receives and it writes directly to dest_buf

  intrank_t num_levels = 0;
  intrank_t x = 1;
  size_t my_max_recv_size = send_count;  // buffer for own
  while (x < tm.rank_n()) {
    num_levels++;
    if (rrank % (x << 1) == 0 && rrank + x < tm.rank_n()) my_max_recv_size += send_count * x;
    x <<= 1;
  }
  if (rrank == 0) {
    assert(tm.rank_me() == root);
    my_max_recv_size = 0;  // no buffer needed, all received data will be directly written to dest_buf
  } else {
    dest_buf = nullptr;  // prevent other ranks from accidentally using dest_buf
  }

  using Buffer = vector<T>;
  using ShBuffer = std::shared_ptr<Buffer>;
  struct LevelWorkflow {
    LevelWorkflow()
        : data(nullptr)
        , prom_buffer()
        , prom_buffer_filled(1) {}
    T* data;                               // if nullptr, copy to the prom_buffer when ready, otherwise copy data directly here
    upcxx::promise<ShBuffer> prom_buffer;  // buffer is allocated
    upcxx::promise<> prom_buffer_filled;   // buffer is filled
  };
  struct DW {
    DW(intrank_t root, intrank_t num_levels)
        : level_workflows(num_levels)
        , root(root) {}
    std::vector<LevelWorkflow> level_workflows;
    intrank_t root;
    LevelWorkflow& level(intrank_t l) { return level_workflows[l]; }
    intrank_t num_levels() const { return level_workflows.size(); }
  };

  DBG_VERBOSE("send_count=", send_count, " root=", root, " n=", tm.rank_n(), " num_levels=", num_levels,
              " my_max_recv_size=", my_max_recv_size, "\n");
  using DWs = dist_object<DW>;
  auto sh_dist_workflows = std::make_shared<DWs>(tm, root, num_levels);
  DWs& dist_workflows = *sh_dist_workflows;
  upcxx::future<> all_done = make_future();

  LevelWorkflow scratch;
  size_t send_size = 0;
  bool first_recv = true;
  bool have_received = false;
  bool have_sent = false;
  intrank_t offset = 1;
  for (intrank_t level = 0; level < num_levels; level++) {
    bool is_sending = rrank % (offset << 1) == offset && offset <= rrank;
    bool is_receiving = rrank % (offset << 1) == 0 && rrank + offset < tm.rank_n();

    LevelWorkflow& workflow = dist_workflows->level(level);
    ShBuffer tmp;
    if (is_receiving) {
      DBG_VERBOSE("is receiving level=", level, "\n");
      tmp = std::make_shared<Buffer>();
      assert(!is_sending && "receiving is not also sending");
      size_t recv_size = send_count;
      if (rrank == 0) {
        assert(tm.rank_me() == root);
        assert(dest_buf);
        workflow.data = dest_buf + (first_recv ? send_count : 0);
        DBG_VERBOSE("Direct Receiving copy to ", (void*)workflow.data, " level=", level,
                    " offset=", (workflow.data - orig_dest_buf), " (up to)bytes=", get_size_str(offset * send_count * sizeof(T)),
                    "\n");
        // no tmp.reserve, since dest is going to real dest, and empty buffer is okay
      } else {
        // only reserve the size
        recv_size = my_max_recv_size;
        assert(workflow.data == nullptr);
      }
      if (first_recv) {
        // special first receive which bypasses extra copies and rpcs to self

        // copy my data first (root to actual dest, others to buffer)
        tmp->reserve(recv_size);
        if (rrank == 0) {
          assert(tm.rank_me() == root);
          assert(workflow.data == dest_buf + send_count);
          DBG_VERBOSE("Direct copy (first my send) to dest_buf=", (void*)dest_buf, "\n");
          std::copy(send_buf, send_buf + send_count, dest_buf);
          dest_buf += send_count;
        } else {
          assert(workflow.data == nullptr);
          tmp->insert(tmp->end(), send_buf, send_buf + send_count);
        }
      }
      if (first_recv || rrank == 0) {
        // signal the receive buffer is allocated and ready for this receive level
        assert(tmp);
        tmp->reserve(recv_size);
        workflow.prom_buffer.fulfill_result(std::move(tmp));
      }

      workflow.prom_buffer.get_future().then([level](ShBuffer shbuf) {
        DBG_VERBOSE("receive buffer for level=", level, " is ready to accept data size=", shbuf->size(), " cap=", shbuf->capacity(),
                    "\n");
      });
      workflow.prom_buffer_filled.get_future().then(
          [level]() { DBG_VERBOSE("receive buffer for level=", level, " has been filled\n"); });

      // not done until has finished receiving
      all_done = when_all(all_done, workflow.prom_buffer_filled.get_future()).then([level, first_recv, have_received]() {
        DBG_VERBOSE("Received level=", level, " first=", first_recv, " have_recv=", have_received, "\n");
      });
      first_recv = false;
      have_received = true;
    }
    if (is_sending) {
      DBG_VERBOSE("is sending level=", level, "\n");
      assert(!is_receiving && "sending is not also receiving");
      assert(!workflow.prom_buffer_filled.get_future().is_ready() && "sending buffer has not been filled before workflow is prepared");
      if (!have_received) {
        assert(send_buf);
        assert(send_count != 0);
        assert(workflow.data == nullptr);
        workflow.data = const_cast<T*>(send_buf);
        tmp = std::make_shared<Buffer>(send_count);
        workflow.prom_buffer.fulfill_result(std::move(tmp));  // dummy signal
      } else {
        // both promises will have been fulfilled by receiving rpcs
      }
      auto fut_send_ready = workflow.prom_buffer.get_future().then([level](ShBuffer sending_buf) {
        DBG_VERBOSE("send buffer ready at level=", level, " size=", sending_buf->size(), " cap=", sending_buf->capacity(), "\n");
      });
      workflow.prom_buffer_filled.get_future().then([level]() { DBG_VERBOSE("send buffer filled at level=", level, "\n"); });
      auto fut_buf_filled = when_all(workflow.prom_buffer.get_future(), fut_send_ready);

      auto fut = fut_buf_filled.then([rrank, root, level, offset, send_count, have_received, &tm, &dist_workflows,
                                      &workflow](ShBuffer sending_buf) {
        // sending
        auto sending_size = sending_buf->size();
        T* sending_data = (workflow.data == nullptr) ? sending_buf->data() : workflow.data;
        intrank_t dest_rrank = (rrank + tm.rank_n() - offset) % tm.rank_n();
        intrank_t dest_rank = (dest_rrank + root) % tm.rank_n();
        rpc_ff(
            tm, dest_rank,
            [](DWs& dist_workflows, intrank_t level, const upcxx::view<T>& payload) {
              assert(level < dist_workflows->num_levels());
              LevelWorkflow& recv_workflow = dist_workflows->level(level);
              auto fut_buf = recv_workflow.prom_buffer.get_future();
              return fut_buf.then([&recv_workflow, payload, &dist_workflows, level](ShBuffer dest_buf) {
                assert(dest_buf);
                DBG_VERBOSE("receiving level=", level, " dest_buf->size()=", dest_buf->size(), " capacity=", dest_buf->capacity(),
                            " payload=", payload.size(), " workflow.data=", (void*)recv_workflow.data, "\n");
                if (recv_workflow.data == nullptr) {
                  auto tot_size = dest_buf->size() + payload.size();
                  assert(dest_buf->capacity() >= tot_size && "dest_buf has already been reserved");
                  // append payload
                  dest_buf->insert(dest_buf->end(), payload.begin(), payload.end());
                } else {
                  // direct copy payload for case of rroot
                  assert(recv_workflow.data);
                  DBG_VERBOSE("Direct copy to ", (void*)recv_workflow.data, "\n");
                  std::copy(payload.begin(), payload.end(), recv_workflow.data);
                }
                // this buffer for this level is now filled
                DBG_VERBOSE("fulfilling this buffer at level=", level, " is filled\n");
                recv_workflow.prom_buffer_filled.fulfill_anonymous(1);
              });
            },
            dist_workflows, level, upcxx::make_view(sending_data, sending_data + sending_size));

        DBG_VERBOSE("Sent rpc to dest_rrank=", dest_rrank, ", dest_rank=", dest_rank, " rrank=", rrank, " level=", level,
                    " count=", sending_size, " ", get_size_str(sending_size * sizeof(T)), "\n");
        // make buffer available for next level (may never be needed but steps workflow to completion)
        assert(!workflow.prom_buffer_filled.get_future().is_ready() && "sending buffer that was just used has not been filled yet");
        workflow.prom_buffer_filled.fulfill_anonymous(1);
      });
      if (!have_received)
        assert(fut.is_ready() &&
               "first sending level always immediately executes ensuring send_buf is ready for reuse on exit of binomial_gather");
      have_sent = true;
      // not done until I have sent my message
      all_done = when_all(all_done, fut).then([level, rrank]() {
        DBG_VERBOSE("Sent and Executed rpc_ff level=", level, " and buffer is filled for reuse\n");
      });
    }
    if (!is_sending & !is_receiving) {
      // some ranks on some levels have nothing to do but make the buffer available
      DBG_VERBOSE("idle level=", level, "\n");
      assert(!workflow.prom_buffer_filled.get_future().is_ready());
      workflow.prom_buffer_filled.fulfill_anonymous(1);
      if (!have_received && !have_sent) workflow.prom_buffer.fulfill_result(ShBuffer());
    }

    // move the buffer to the next level when ready
    auto fut = when_all(workflow.prom_buffer.get_future(), workflow.prom_buffer_filled.get_future())
                   .then([level, &dist_workflows](ShBuffer sh_buf) {
                     if (!sh_buf) {
                       DBG_VERBOSE("Ignoring empty shbuf level=", level, "\n");
                     } else if (level + 1 < dist_workflows->num_levels()) {
                       LevelWorkflow& next = dist_workflows->level(level + 1);
                       DBG_VERBOSE("Setting buffer for next level: ", level + 1, " size=", sh_buf->size(),
                                   " cap=", sh_buf->capacity(), "\n");
                       if (!next.prom_buffer.get_future().is_ready()) {
                         // fulfill buffer for next level on this rank for it to send or recv
                         next.prom_buffer.fulfill_result(std::move(sh_buf));
                       } else {
                         assert(dist_workflows.team().rank_me() == dist_workflows->root &&
                                "only rroot will already have an already ready buffer at the next level");
                       }
                     }
                   });
    all_done = when_all(all_done, fut);

    if (rrank == 0) dest_buf += offset * send_count;  // next level starts at a different output position
    offset <<= 1;                                     // double offset
  }
  return all_done.then([sh_dist_workflows]() {
    DBG_VERBOSE(
        "Completed binomial_gather\n"); /* preserve lifetime of dist_workflows until all levels for this rank are completed */
  });
}

template <typename T>
upcxx::future<std::vector<T>> binomial_gather(const T* send_buf, size_t send_count, intrank_t root,
                                              const upcxx::team& tm = upcxx::world()) {
  bool am_root = root == tm.rank_me();
  auto sh_dest = std::make_shared<std::vector<T>>(am_root ? tm.rank_n() * send_count : 0);
  auto fut = binomial_gather(send_buf, send_count, sh_dest->data(), root, tm);
  return fut.then([sh_dest, root, send_count, &tm]() -> std::vector<T> {
    assert(sh_dest);
    std::vector result(std::move(*sh_dest));
    if (tm.rank_me() == root)
      assert(result.size() == tm.rank_n() * send_count);
    else
      assert(result.empty());
    return std::move(result);
  });
}

template <typename T>
upcxx::future<std::vector<T>> binomial_gather(const T& send_buf, intrank_t root, const upcxx::team& tm = upcxx::world()) {
  return binomial_gather(&send_buf, 1, root, tm);
}

// generic gather method that choose the best algorithm

template <typename T>
upcxx::future<> two_stage_gather(const T* send_buf, size_t send_count, T* dest_buf, intrank_t root, const upcxx::team& tm1,
                                 const upcxx::team& tm2) {
  // first binomial_gather with buffers on the local team then linear_gather these large messages to root

  auto local_n = tm2.rank_n();
  auto local_root = root % local_n;
  assert(root < local_n);
  assert(tm1.rank_n() % tm2.rank_n() == 0);
  std::vector<T> buffer;
  if (tm1.rank_me() % local_n == local_root) buffer.resize(send_count * local_n);
  auto fut = binomial_gather(send_buf, send_count, buffer.data(), local_root, tm2);
  if (tm1.rank_me() % local_n == local_root) fut.wait();
  auto fut2 = linear_gather(buffer.data(), send_count * local_n, dest_buf, root, tm1, local_n);
  return when_all(fut, fut2);
}

template <typename T>
upcxx::future<std::vector<T>> two_stage_gather(const T* send_buf, size_t send_count, intrank_t root, const upcxx::team& tm1,
                                               const upcxx::team& tm2) {
  bool am_root = root == tm1.rank_me();
  auto sh_dest = std::make_shared<std::vector<T>>(am_root ? tm1.rank_n() * send_count : 0);
  auto fut = two_stage_gather(send_buf, send_count, sh_dest->data(), root, tm1, tm2);
  return fut.then([sh_dest]() {
    std::vector result(std::move(*sh_dest));
    return result;
  });
}

template <typename T>
upcxx::future<std::vector<T>> two_stage_gather(const T& send_buf, intrank_t root, const upcxx::team& tm1, const upcxx::team& tm2) {
  return two_stage_gather(&send_buf, 1, root, tm1, tm2);
}

// dest_buf on root rank must be sufficiently large to hold send_count * tm.rank_n() elements of T
template <typename T>
upcxx::future<> gather(const T* send_buf, size_t send_count, T* dest_buf, intrank_t root, const upcxx::team& tm = upcxx::world()) {
  size_t send_bytes = sizeof(T) * send_count;
  intrank_t local_n = upcxx::local_team().rank_n();
  if (send_bytes >= ONE_KB && tm.rank_n() >= 4096 && tm.id() == upcxx::world().id() && tm.id() != upcxx::local_team().id() &&
      root < local_n) {
    return two_stage_gather(send_buf, send_count, dest_buf, root, tm, local_team());
  } else if (send_bytes > 16 * ONE_KB) {
    // excessive copies and buffering on some ranks, so just perform linear messages
    return linear_gather(send_buf, send_count, dest_buf, root, tm);
  } else {
    return binomial_gather(send_buf, send_count, dest_buf, root, tm);
  }
};

template <typename T>
upcxx::future<std::vector<T>> gather(const T* send_buf, size_t send_count, intrank_t root, const upcxx::team& tm = upcxx::world()) {
  bool am_root = root == tm.rank_me();
  auto sh_dest = std::make_shared<std::vector<T>>(am_root ? tm.rank_n() * send_count : 0);
  auto fut = gather(send_buf, send_count, sh_dest->data(), root, tm);
  return fut.then([sh_dest]() {
    std::vector result(std::move(*sh_dest));
    return result;
  });
}

template <typename T>
upcxx::future<std::vector<T>> gather(const T& send_buf, intrank_t root, const upcxx::team& tm = upcxx::world()) {
  return gather(&send_buf, 1, root, tm);
}

// dest_buf on every rank must be sufficiently large to hold send_count * tm.rank_n() elements of T
template <typename T>
upcxx::future<> inefficient_all_gather(T* send_buf, size_t send_count, T* dest_buf, const upcxx::team& tm = upcxx::world()) {
  gather(send_buf, send_count, dest_buf, 0, tm).wait();
  return upcxx::broadcast(dest_buf, send_count * tm.rank_n(), 0, tm);
}

// generic all_gather method that choose the best algorithm

// dest_buf on every rank must be sufficiently large to hold send_count * tm.rank_n() elements of T
template <typename T>
upcxx::future<> all_gather(T* send_buf, size_t send_count, T* dest_buf, const upcxx::team& tm = upcxx::world()) {
  return inefficient_all_gather(send_buf, send_count, dest_buf, tm);
}

};  // namespace upcxx_utils
