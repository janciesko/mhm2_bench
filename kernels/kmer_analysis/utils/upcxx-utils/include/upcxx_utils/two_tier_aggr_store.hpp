#pragma once

#include <algorithm>
#include <sstream>
#include <upcxx/upcxx.hpp>
#include <utility>
#include <cassert>

#include "log.hpp"
#include "split_rank.hpp"
#include "timers.hpp"
#include "version.h"

using std::list;
using std::make_shared;
using std::ostream;
using std::ostringstream;
using std::pair;
using std::shared_ptr;
using std::string;
using std::vector;

using upcxx::barrier;
using upcxx::dist_object;
using upcxx::global_ptr;
using upcxx::intrank_t;
using upcxx::make_future;
using upcxx::make_view;
using upcxx::op_fast_add;
using upcxx::op_fast_max;
using upcxx::progress;
using upcxx::rank_me;
using upcxx::rank_n;
using upcxx::reduce_all;
using upcxx::reduce_one;
using upcxx::rget;
using upcxx::rpc;
using upcxx::to_future;
using upcxx::view;
using upcxx::when_all;

// this class aggregates updates into local buffers and then periodically does an rpc to dispatch them

#ifdef DEBUG
#define DEBUG_MINIMAL_STORE
#endif

#ifndef MAX_RPCS_IN_FLIGHT
#define MAX_RPCS_IN_FLIGHT 4096
#endif

#ifndef MIN_INFLIGHT_BYTES
#define MIN_INFLIGHT_BYTES (1024L * 1024L) /* always use at least 1MB in flight */
#endif

namespace upcxx_utils {

template <typename T>
class BlockDispatcher {
  // does not allocate or deallocate memory, just handles pointer to it
  // available are backed blocks that are empty and ready to be consumed
  // promises is a strictly FIFO queue of blocks that will be fulfilled by a backed block
  //  pushing / poping from this queue is a signal for asynchronos processing at a later time when resources permit
  // all methods are non-blocking, returning futures
  // push() will fulfill outstanding promises before returning a block to the available heap
  // only pop() can grow promises beyond a fixed size
  // all blocks must be returned to available, with no promises before clear() can be called
 public:
  using ptr_t = T;
  using block_t = pair<ptr_t, size_t>;
  using future_block_t = upcxx::future<block_t>;
  using promise_block_t = ActiveInstantiationTimer<upcxx::promise<block_t> >;

  using blocks_t = vector<block_t>;
  using reservation_t = shared_ptr<blocks_t>;
  using promise_reservation_t = ActiveInstantiationTimer<upcxx::promise<reservation_t> >;

  using promise_blocks_t = list<promise_block_t>;
  using promise_reservations_t = list<promise_reservation_t>;

 private:
  block_t backing;
  size_t count_per_block, num_blocks, reservation_size;
  blocks_t available, reservation;
  promise_blocks_t promised_blocks;  // may grow indefinitely, but elements are small
  ActiveCountTimer promise_block_count_timer;
  size_t promised_blocks_count;
  promise_reservations_t promised_reservations;  // may grow indefinitely, but elements are small
  ActiveCountTimer promise_reservation_count_timer;
  size_t promised_reservations_count;

 protected:
  // drains the current reservation and returns the contents
  reservation_t claim_reservation(bool require_full = true) {
    // no need to get a lock as all methods already have a lock
    if (require_full && reservation.size() != reservation_size)
      DIE("claim_reservation is not full but this is required. ", reservation.size(), "\n");
    reservation_t reserved(new blocks_t());
    reserved->reserve(reservation_size);
    reserved->swap(reservation);  // claim and empty the current reservation
    assert(reservation.size() == 0);
    // repopulate the reservation from the available set
    while (!available.empty() && reservation.size() < reservation_size) {
      reservation.push_back(available.back());
      available.pop_back();
    }
    assert(reserved);
    return reserved;
  }

 public:
  string description;
  BlockDispatcher(const string description)
      : backing({})
      , count_per_block(0)
      , num_blocks(0)
      , reservation_size(0)
      , available()
      , reservation()
      , promised_blocks()
      , promise_block_count_timer(description + "-promised_blocks")
      , promised_blocks_count(0)
      , promised_reservations()
      , promise_reservation_count_timer(description + "-promised_reservations")
      , promised_reservations_count(0)
      , description(description) {}
  BlockDispatcher(const BlockDispatcher &) = delete;
  BlockDispatcher(BlockDispatcher &&) = default;
  virtual ~BlockDispatcher() { clear(); }

  // a valid dispatcher has an allocation and blocks
  bool valid() const {
    bool is_valid = (backing.first && backing.second > 0 && num_blocks > 0 && count_per_block > 1 &&
                     num_blocks * count_per_block <= backing.second);
    return is_valid;
  }

  // checks if a block is backed by the allocation
  bool is_backed(block_t &block) const {
    return valid() && count_per_block > 0 &&
           (block.first && (backing.first <= block.first) && (backing.first + backing.second >= block.first + count_per_block));
  }

  // accepts a large block that will be divided into regular num blocks of count elements for dispatching
  // all blocks will be put in the available heap and have a zero count
  void set(block_t allocation, size_t num, size_t count, size_t thread_offset = 0, size_t reservation_count = 0) {
    if (valid()) DIE("set called on an already valid dispatcher.  clear() MUST be called first\n");
    if (!available.empty()) DIE("set called with non-empty available heap\n");
    if (!promised_blocks.empty()) DIE("set called with non-empty promised_blocks queue\n");
    if (!promised_reservations.empty()) DIE("set called with non-empty promised_reservations queue\n");
    if (num != 0 && (!allocation.first || allocation.second <= 0)) DIE("set called with null backing\n");
    if (num * (count + thread_offset) != allocation.second)
      DIE("set called with an incorrectly sized allocated_backing: ", allocation.second, ", blocks ", num, ", count per ", count,
          " and ", thread_offset, " thread_offset\n");
    if (reservation_count > num / 2) DIE("reservation_count=", reservation_count, " can not fit within num=", num, "\n");
    clear();
    num_blocks = num;
    count_per_block = count;
    reservation_size = reservation_count;

    backing = allocation;
    available.reserve(num_blocks);
    reservation.reserve(reservation_size);
    promised_blocks_count = 0;
    promised_reservations_count = 0;
    for (size_t i = 0; i < num_blocks; i++) {
      block_t tmp(backing.first + i * (count_per_block + thread_offset) + thread_offset, 0);
      assert(is_backed(tmp));
      push(tmp);
    }
    if (available.size() + reservation.size() != num_blocks)
      DIE("Invalid set() - available size is not num_blocks: ", available.size(), " available + ", reservation.size(),
          " reserved vs ", num_blocks, "\n");
    if (!promised_blocks.empty()) DIE("Invalid set() - promised_blocks should be empty: ", promised_blocks.size(), "\n");
    if (!promised_reservations.empty())
      DIE("Invalid set() - promised_blocks should be empty: ", promised_reservations.size(), "\n");
    assert(num_blocks == 0 || valid());
  }

  // clears this dispatcher. Aborts if promises queue is not empty or available is not full
  void clear() {
    if (!valid()) {
      assert(available.empty());
      assert(reservation.empty());
      assert(promised_blocks.empty());
      assert(promised_reservations.empty());
      assert(num_blocks == 0);
      assert(promised_blocks_count == 0);
      assert(promised_reservations_count == 0);
    }

    if (num_blocks) {
      promise_block_count_timer.print_reduce_timings();
      if (reservation_size) {
        promise_reservation_count_timer.print_reduce_timings();
      }
    }

    if (!promised_blocks.empty()) DIE("clear() called with entries in the promised_blocks queue\n");
    if (!promised_reservations.empty()) DIE("clear() called with entries in the promised_reservations queue\n");
    if (reservation.size() + available.size() != num_blocks)
      DIE("clear() called witnout all blocks returned to available: ", available.size(), " + reserved ", reservation.size(),
          " expected ", num_blocks, "\n");

    backing = {};
    available.resize(0);
    reservation.resize(0);
    num_blocks = 0;
    count_per_block = 0;
    promised_blocks_count = 0;
    promised_reservations_count = 0;
    assert(!valid());
    barrier();
  }

  inline size_t get_count_per_block() const { return count_per_block; }
  inline size_t get_num_blocks() const { return num_blocks; }

 public:
  // returns a full reservation of global block that can be used immediately
  // non blocking
  upcxx::future<reservation_t> acquire_reservation() {
    assert(reservation_size > 0);
    if (reservation_size == 0) DIE("There is no reservation to acquire as reservation_size == 0\n");
    // lock against concurrent modification on all containers
    if (reservation.size() == reservation_size) {
      reservation_t reserved = claim_reservation();
      assert(reserved->front().first);
      assert(reserved->back().first);
      DBG("acquire_reservation: got immediately:", reserved.get(), " -- ", to_string(), "\n");
      return make_future(reserved);
    } else {
      // add a promise for a reservation
      DBG("acquire_reservation: issuing a promise -- ", to_string(), "\n");
      promise_reservation_t res(promise_reservation_count_timer);
      auto fut = res.get_future();
      promised_reservations.push_back(std::move(res));
      promised_reservations_count++;
      assert(promised_reservations_count == promised_reservations.size());
      return res;
    }
  }

  reservation_t acquire_partial_reservation() {
    if (reservation_size == 0) DIE("There is no reservation to acquire as reservation_size == 0\n");
    reservation_t res;
    res = claim_reservation(false);
    assert(res);
    DBG("acquire_partial_reservation got one with ", res->size(), " -- ", to_string(), "\n");
    return res;
  }

  void release_reservation(reservation_t reserved) {
    assert(reserved);
    DBG("Release_reservation:", reserved.get(), ", size=", reserved->size(), "\n");
    for (auto block : *reserved) {
      // DBG("Pushing back reserved block=", block.first, " reservation:", reserved.get(), "\n");
      assert(block.first);
      push(block);
      assert(!block.first);  // invalidated
    }
    reserved->clear();
  }

  // if the reservation is not full, push to it
  // otherwise if there is a promised_block, fulfill that promise
  // otherwise push the block into the available heap
  // assigns a zero count, and invalidates block so it can not be reused
  void push(block_t &block) {
    assert(block.first);
    if (!valid()) DIE("push called on invalid BlockDispatcher!\n");
    assert(is_backed(block));
    if (!is_backed(block))
      DIE("push called on foreign block(", block.first, " ", block.second, "): ", backing.first, " ", backing.second, "\n");
    assert(promised_blocks_count == promised_blocks.size());
    block.second = 0;  // reset the count

    // a reservation may already be ready so fulfill apply before promised_block fulfillment
    try_fulfill_promised_reservation();

    if (reservation.size() < reservation_size) {
      // put into the reservation
      reservation.push_back(block);
    } else if (!promised_blocks.empty()) {
      // deliver this block to the first promised_blocks
      block_t promised_block = block;  // copy before invalidation below
      DBG("push fulfilling promised_block: ", promised_block.first, "\n");
      assert(!promised_blocks.empty());
      auto promise_for_block = std::move(promised_blocks.front());
      promised_blocks.pop_front();
      promised_blocks_count--;
      promise_for_block.fulfill_result(promised_block);
    } else {
      // put back on available heap
      available.push_back(block);
    }
    block = {};  // invalidate it

    if (available.size() + reservation.size() > num_blocks)
      DIE("push added too many blocks: ", available.size(), " + ", reservation.size(), " vs ", num_blocks, "\n");

    // a reservation may now also be ready
    try_fulfill_promised_reservation();
  }

  void try_fulfill_promised_reservation() {
    // check for any outstanding promised_reservations and fulfill if possible
    if (reservation_size > 0 && !promised_reservations.empty() && reservation.size() == reservation_size) {
      // fulfill this promised reservation
      auto promised_reservation = claim_reservation();
      assert(promised_reservation);
      assert(promised_reservation->front().first);
      assert(promised_reservation->back().first);
      assert(promised_reservation->size() == reservation_size);
      auto promise_for_reservation = std::move(promised_reservations.front());
      promised_reservations.pop_front();
      promised_reservations_count--;
      assert(promised_reservations_count == promised_reservations.size());
      DBG("fulfilling promised_reservation: ", promised_reservation.get(), "\n");
      promise_for_reservation.fulfill_result(promised_reservation);
    }
  }

  // returns a future block.
  // if one is available it may be immediately ready
  // otherwise a promised_block is made and tracked
  // if available.empty, create a promised_block
  future_block_t pop(bool from_reservation = false) {
    if (!valid()) DIE("pop_available called on invalid BlockDispatcher!\n");
    future_block_t future_block;
    // lock against concurrent modification on all containers
    if (available.empty() && from_reservation && reservation.size() > 0) {
      block_t block = reservation.back();
      reservation.pop_back();
      assert(is_backed(block));
      assert(block.second == 0);
      future_block = to_future(block);
    } else if (available.empty()) {
      // add a new promised_block and return its future
      promise_block_t prom(promise_block_count_timer);
      future_block = prom.get_future();
      promised_blocks.push_back(std::move(prom));
      promised_blocks_count++;
      // DBG("pop got promised_blocks\n");
    } else {
      block_t block = available.back();
      available.pop_back();
      assert(is_backed(block));
      assert(block.second == 0);
      future_block = to_future(block);
      // DBG("pop got available\n");
    }
    return future_block;
  }

  // const status accessors

  inline size_t reserved_size() const { return reservation.size(); }

  inline size_t available_size() const { return available.size(); }

  inline bool available_empty() const { return available.empty(); }

  // true if both the promises queues are empty
  bool empty() const {
    // DBG("empty():  promised reservations=", promised_reservations_count, " promised_blocks_count=", promised_blocks_count, "
    // available=", available.size() , " reserved=", reservation.size(), " num_blocks=", num_blocks, "\n");
    return promised_reservations.empty() && promised_blocks.empty() && available_size() + reservation.size() == num_blocks;
  }

  // true if the promises queue has entries
  inline size_t promises_size() const { return promised_reservations_count + promised_blocks_count; }

  inline bool promises_empty() const { return promised_reservations.empty() && promised_blocks.empty(); }

  // to_string for debug output
  string to_string() const {
    ostringstream os;
    os << description << "-";
    os << "BlockDispatcher(backing=" << backing.first;
    os << ",count=" << count_per_block << ",num=" << num_blocks;
    os << ",avail=" << available.size();
    os << ",promised_reservations=" << promised_reservations_count;
    os << ",promised_blocks=" << promised_blocks_count << ")";
    assert(promised_reservations_count == promised_reservations.size());
    assert(promised_blocks_count == promised_blocks.size());
    return os.str();
  }
};

class TrackRPCs {
 public:
  using future_ack_t = upcxx::future<>;
  using rpc_acks_t = list<future_ack_t>;

  TrackRPCs(const string description_)
      : rpcs_in_flight()
      , sent_rpcs(0)
      , returned_rpcs(0)
      , rpc_timer()
      , rpc_inner_timer()
      , rpc_relay_timer()
      , description(description_)
      , t_prog(description_) {}
  TrackRPCs(const TrackRPCs &) = delete;
  TrackRPCs(TrackRPCs &&) = default;
  virtual ~TrackRPCs() { clear(); }

  bool empty() const;

  // frees memory.  Can only be called when all futures have completed
  void clear();

  // track an rpc acknowledgment
  void push(future_ack_t fut);

  // tests all rpcs and returns the remaining count
  // if ready, wait on it and remove, otherwise count it
  size_t pop_finished();

  size_t count_pending();

  void flush(size_t max_pending = 0);

  string to_string() const;

  inline ActiveCountTimer &get_rpc_timer() { return rpc_timer; }
  inline ActiveCountTimer &get_rpc_inner_timer() { return rpc_inner_timer; }
  inline ActiveCountTimer &get_rpc_relay_timer() { return rpc_relay_timer; }

 protected:
  rpc_acks_t rpcs_in_flight;
  size_t sent_rpcs, returned_rpcs;
  ActiveCountTimer rpc_timer, rpc_inner_timer, rpc_relay_timer;
  string description;
  ProgressTimer t_prog;
};

template <typename T>
class FixedMemoryRPC {
  // consists of a global pointer dispatcher, and an acknowledgement queue for RPC calls
  // The global pointer dispatcher is for receiving rgets from remote global shared memory
  // The global pointer dispatcher is also for accumulating a block locally and then sending the blocks remotely

 public:
  // for global data on this node
  using global_block_dispatch_t = BlockDispatcher<global_ptr<T> >;
  using global_block_t = typename global_block_dispatch_t::block_t;
  using future_global_block_t = typename global_block_dispatch_t::future_block_t;
  using global_store_t = typename global_block_dispatch_t::blocks_t;
  using future_src_dest_block_t = upcxx::future<global_block_t, global_block_t>;
  using global_reservation_t = typename global_block_dispatch_t::reservation_t;
  using inst_timer_t = GenericInstantiationTimer;

 private:
  // backing and dispatchers
  global_block_t global_backing;  // for dest stores and sourcing rgets
  global_block_dispatch_t global_dispatcher;

  global_store_t dest_stores;  // a block of data for aggregation to each destination
  size_t thread_offset;
  ActiveCountTimer rput_timer, rget_timer, rget_wait_timer;
  ProgressTimer t_prog;
  string description;

 public:
  FixedMemoryRPC(const string description)
      : global_backing({})
      , global_dispatcher(description + string("-global-dispatcher"))
      , dest_stores()
      , thread_offset(0)
      , rput_timer()
      , rget_timer()
      , rget_wait_timer()
      , t_prog(description)
      , description(description) {}
  FixedMemoryRPC(const FixedMemoryRPC &) = delete;
  FixedMemoryRPC(FixedMemoryRPC &&) = default;
  virtual ~FixedMemoryRPC() { clear(); }

  bool valid() const {
    bool is_valid = (global_dispatcher.get_count_per_block() == global_dispatcher.get_count_per_block()) &&
                    ((global_dispatcher.get_count_per_block() == 1 && !global_backing.first) ||
                     (global_dispatcher.get_count_per_block() > 1 && global_backing.first && global_backing.second > 0));

    return is_valid;
  }

  inline void my_progress() { t_prog.progress(); }

  // we may have no intra dispatchers if there is 1 thread per node
  // we may have no inter dispatchers if there is just 1 node
  void set_dest_stores(size_t num_stores) {
    DBG("FixedMemoryRPC - ", description, "::set_dest_stores(num_stores=", num_stores,
        ") global_blocks=", global_dispatcher.get_num_blocks(), " count=", global_dispatcher.get_count_per_block(),
        " avail=", global_dispatcher.available_size(), "\n");
    assert(valid());
    if (global_dispatcher.get_count_per_block() > 1) {
      // require split_rank::split_local_team().rank_n() - 1 available blocks at this point (no blocking!)
      if (global_dispatcher.available_size() < num_stores) {
        DIE("There are an insufficient number of available blocks to populate the dest stores: available_blocks=",
            global_dispatcher.available_size(), ", num_stores=", num_stores, "\n");
      }
      dest_stores.reserve(num_stores);
      for (int i = 0; i < num_stores; i++) {
        auto fut = global_dispatcher.pop();
        if (!fut.is_ready()) {
          DIE("Detected a global block that is not ready! i=", i, " available_size=", global_dispatcher.available_size(), "\n");
        }
        dest_stores.push_back(fut.wait());
        assert(dest_stores.back().first);
      }
    }
    assert(valid());
    barrier();  // required so that no other global_dispatcher.pop() happends before dest_stores are filled
  }

  void clear_dest_stores() {
    DBG("FixedMemoryRPC - ", description, " clear_dest_stores:", dest_stores.size(), "\n");
    if (global_dispatcher.get_num_blocks() == 0) {
      assert(dest_stores.empty());
    } else {
      assert(valid());
      for (auto s : dest_stores) {
        if (s.second > 0) DIE("Can not clear_dest_stores if they are not empty!\n");
        global_dispatcher.push(s);
      }
      dest_stores.resize(0);
      assert(valid());
    }
  }

  size_t count_empty_dest_stores() const {
    size_t empty = 0;
    for (auto s : dest_stores) {
      if (s.second == 0) empty++;
    }
    return empty;
  }

  // only true both the local and global dispatcher are empty themselves (*this be invalid)
  // and the dispatchers have a full available heap
  bool empty() const {
    bool is_empty = global_dispatcher.empty() &&
                    global_dispatcher.available_size() + global_dispatcher.reserved_size() + count_empty_dest_stores() ==
                        global_dispatcher.get_num_blocks();
    // DBG("FixedMemoryRPC::empty(): ", (is_empty?"True":"False"), ", global.empty()=", (global_dispatcher.empty()?"True":"False"),
    // "\n");
    return is_empty;
  }

  // allocates the blocks and sets the dispatchers
  void set_fixed_mem(size_t num_global_blocks, size_t count_per_block, size_t num_stores, bool includes_thread_offset = false,
                     size_t num_reserved_blocks = 0) {
    global_dispatcher.clear();
    clear();

    if (num_global_blocks == 0) {
      assert(count_per_block == 1);
      SOUT("Using empty FixedMemoryRPC\n");
      return;
    }

    if (num_global_blocks <= num_stores + num_reserved_blocks) {
      DIE("Invalid set_fixed_mem num_global_blocks=", num_global_blocks, " num_stores=", num_stores,
          " num_reserved_blocks=", num_reserved_blocks, "\n");
    }

    size_t global_count = num_global_blocks * count_per_block;

    SOUT("Allocating ", description, " dispatchers: global_count=", global_count, " ", get_size_str(global_count * sizeof(T)),
         "\n");

    // allocate global memory for global dispatcher
    thread_offset = includes_thread_offset ? (sizeof(thread_num_t) * count_per_block + sizeof(T) - 1) / sizeof(T) : 0;

    // allocate thread_offset more elements and start the block that many into the actual allocation
    global_backing.second = global_count + (thread_offset * num_global_blocks);
    global_backing.first = upcxx::new_array<T>(global_backing.second);
    size_t global_num = global_backing.second / (count_per_block + thread_offset);
    assert(global_num == num_global_blocks);
    global_dispatcher.set(global_backing, global_num, count_per_block, thread_offset, num_reserved_blocks);

    SOUT("finished allocating ", description, " dispatchers\n");

    set_dest_stores(num_stores);

    size_t total_global_bytes = sizeof(T) * global_backing.second;
    size_t total_bytes = total_global_bytes;
    SOUT("Using ", num_global_blocks, " global of ", count_per_block, " elements (of ", get_size_str(sizeof(T)),
         ") aggregate & send (", get_size_str(total_global_bytes), ") per thread or ",
         get_size_str(total_bytes * upcxx::local_team().rank_n()), " per node in shared memory for dest and send buffers\n");
    assert(valid());
  }

  // frees memory.  Can only be called when all futures have completed
  void clear() {
    if (!valid()) {
      assert(!global_dispatcher.valid());
      assert(dest_stores.empty());
      return;
    }
    clear_dest_stores();
    t_prog.print_out();
    rget_timer.print_reduce_timings(description + "-rget");
    rget_wait_timer.print_reduce_timings(description + "-rget-wait");
    rput_timer.print_reduce_timings(description + "-rput");

    // deallocate global_dispatcher
    global_dispatcher.clear();
    upcxx::delete_array(global_backing.first);
    global_backing = {};
    assert(!valid());
    barrier();
  }

  inline bool has_dest_stores() const { return dest_stores.size() > 0; }

  global_block_t &dest_store(size_t store_idx) {
    if (store_idx >= dest_stores.size()) DBG("getting ", description, " dest_store store_idx=", store_idx, "\n");
    assert(store_idx < dest_stores.size());
    if (dest_stores.size() <= store_idx)
      DIE("There are no dest stores at the moment:", dest_stores.size(), " looking for ", store_idx, "\n");
    global_block_t &gblock = dest_stores[store_idx];
    return gblock;
  }

  // push a global block back to the dispatcher
  void push_global(global_block_t &gblock) {
    // DBG("FixedSize::push_global: ", gblock.first, "\n");
    if (!valid()) DIE("push called on an invalid FixedMemoryRPC!\n");
    assert(gblock.first);
    assert(gblock.first.where() == rank_me());
    global_dispatcher.push(gblock);
    assert(!gblock.first);  // is invalidated
  }

  // pops a future global block from the dispatcher
  future_global_block_t pop_global(bool from_reservation = false) {
    // DBG("pop_global\n");
    if (!valid()) DIE("pop_global called on invalid FixedMemoryRPC!\n");
    return global_dispatcher.pop(from_reservation);
  }

  inline upcxx::future<global_reservation_t> acquire_reservation() { return global_dispatcher.acquire_reservation(); }

  inline global_reservation_t acquire_partial_reservation() { return global_dispatcher.acquire_partial_reservation(); }

  inline void release_reservation(global_reservation_t reserved) { global_dispatcher.release_reservation(reserved); }

  inline bool has_promises() const { return !global_dispatcher.promises_empty(); }

  inline size_t global_available_size() const { return global_dispatcher.available_size(); }

  inline size_t global_reserved_size() const { return global_dispatcher.reserved_size(); }

  size_t _prep_xfer(global_block_t &src, global_block_t &dest) {
    if (src.second == 0) DIE(__func__, " Invalid state - src block is EMPTY\n");
    if (dest.second > 0) DIE(__func__, " Invalid state - dest is not empty: ", dest.second, "\n");
    assert(global_dispatcher.get_count_per_block() >= src.second);
    // dest will have src's size
    dest.second = src.second;
    size_t send_offset = 0;
    if (thread_offset > 0) {
      // blocks start inside of the allocation so the thread_num decends and the element ascends from the pointer
      send_offset = (dest.second * sizeof(thread_num_t) + sizeof(T) - 1) / sizeof(T);
    }
    assert(send_offset <= thread_offset);
    return send_offset;
  }

  // starts an rput of local source to remote dest
  future_src_dest_block_t rput_block(global_block_t &src, global_block_t &dest) {
    DBG("rput_block( src ", src.first, ", dest: ", dest.first, ", ", src.second, ")\n");
    size_t send_offset = _prep_xfer(src, dest);
    auto rput_t = make_shared<inst_timer_t>(rput_timer);

    // perform the rpet
    assert(src.first.is_local());
    assert(dest.second == src.second);
    auto rput_fut = rput(src.first.local() - send_offset, dest.first - send_offset, dest.second + send_offset);
    src.second = 0;  // signal it is drained
    auto fut_return = when_all(make_future(src, dest), rput_fut);

    // prevent reuse
    src = {};
    dest = {};

    return fut_return.then([rput_t, send_offset](global_block_t src, global_block_t dest) {
      size_t count = dest.second + send_offset;
      DBG("rput completed ", dest.second, " elements with ", send_offset, " extra (", get_size_str(count * sizeof(T)),
          ") src=", src.first, " dest=", dest.first, " in ", rput_t->get_elapsed_since_start(), " s, ",
          get_size_str(count * sizeof(T) / rput_t->get_elapsed_since_start()), " / s\n");
      return make_future(src, dest);
    });
  }

  // starts an rget of the global block copied to the local block
  // creates a future of the same global and local blocks once the rget has completed
  // invalidates both inputs: src and dest
  // non-blocking
  future_src_dest_block_t rget_block(global_block_t &src, global_block_t &dest) {
    DBG("rget_block( src ", src.first, ", dest: ", dest.first, ", ", src.second, ")\n");
    size_t send_offset = _prep_xfer(src, dest);
    auto rget_t = make_shared<inst_timer_t>(rget_timer);

    // perform the rget
    assert(dest.first.is_local());
    assert(dest.second == src.second);
    auto rget_fut = rget(src.first - send_offset, dest.first.local() - send_offset, dest.second + send_offset);
    src.second = 0;  // signal it is drained
    auto fut_return = when_all(make_future(src, dest), rget_fut);

    // prevent reuse
    dest = {};
    src = {};

    return fut_return.then([rget_t, send_offset](global_block_t src, global_block_t dest) {
      size_t count = dest.second + send_offset;
      DBG("rget completed ", dest.second, " elements with ", send_offset, " extra (", get_size_str(count * sizeof(T)),
          ") src=", src.first, " dest=", dest.first, " in ", rget_t->get_elapsed_since_start(), " s, ",
          get_size_str(count * sizeof(T) / rget_t->get_elapsed_since_start()), " / s\n");
      return make_future(src, dest);
    });
  }

  // static rget_block pops a new block for dest
  future_src_dest_block_t rget_block(global_block_t &gblock) {
    // get a future block_t
    auto rget_wait_t = make_shared<inst_timer_t>(rget_wait_timer);
    auto fut_loc = pop_global(true);  // allow extraction from reservation
    auto fut_both = when_all(to_future(gblock), fut_loc);
    // rget the block
    auto fut_blocks = fut_both
                          .then([rget_wait_t](global_block_t src, global_block_t dest) {
                            // just stop the timer
                            return make_future(src, dest);
                          })
                          .then([this](global_block_t src, global_block_t dest) { return this->rget_block(src, dest); });
    return fut_blocks;
  }

  inline size_t get_count_per_block() const {
    assert(global_dispatcher.get_count_per_block() == global_dispatcher.get_count_per_block());
    return global_dispatcher.get_count_per_block();
  }

  inline size_t get_thread_offset() const { return thread_offset; }

  string to_string() const {
    ostringstream os;
    os << description << "-";
    os << "FixedMemoryRPC(";
    os << "thread_offset=" << thread_offset;
    os << ",global_back=" << global_backing.first << "," << global_backing.second;
    os << ",global_dispatch=" << global_dispatcher.to_string();
    os << ")";
    return os.str();
  }
};

template <typename FuncDistObj, typename T>
class TwoTierAggrStore {
 private:
  // T for intra node RPCs
  using intra_fixed_memory_rpc_t = FixedMemoryRPC<T>;
  using intra_fixed_memory_t = dist_object<intra_fixed_memory_rpc_t>;
  using intra_global_ptr_t = global_ptr<T>;
  using intra_global_block_t = typename intra_fixed_memory_rpc_t::global_block_t;
  using intra_future_global_block_t = typename intra_fixed_memory_rpc_t::future_global_block_t;
  using intra_reservation_t = typename intra_fixed_memory_rpc_t::global_reservation_t;

  // For inter-node global,use a more compact array than a pair<Elem, thread_num_t>
  //  as the pair packs very inefficiently and sends a lot of zeros over the net
  //  #'s for thread-dest appending descending, E for element appending ascending from the pointer at the first E
  //  .....4321EEEE.....
  //  --------->-------- // start of element ptr  == T* (alloc + thread_offset)
  //  --------<--------- // start of thread_num ptr == ((thread_num_t*) (alloc + thread_offset)) - 1
  //  only sending the non-zero data in the middle over the wire
  // thread_offset represents the # of elements from the start of the allocation that the pointer will be at

  using inter_fixed_memory_rpc_t = FixedMemoryRPC<T>;
  using inter_fixed_memory_t = dist_object<inter_fixed_memory_rpc_t>;
  using inter_global_ptr_t = global_ptr<T>;
  using inter_global_block_t = typename inter_fixed_memory_rpc_t::global_block_t;
  using inter_future_global_block_t = typename inter_fixed_memory_rpc_t::future_global_block_t;
  using inst_timer_t = GenericInstantiationTimer;

  using track_rpcs_t = dist_object<TrackRPCs>;

  FuncDistObj &func;

  size_t max_store_size;      // the count of T per block (may be 0)
  size_t max_rpcs_in_flight;  // Limit for the number of rpcs in flight. This limit exists to prevent the dispatch buffers from
                              // growing indefinitely

  inter_fixed_memory_t inter_fixed_memory_store;
  intra_fixed_memory_t intra_fixed_memory_store;
  track_rpcs_t track_inter_rpcs, track_intra_rpcs;
  ProgressTimer t_prog;
  static IntermittentTimer &t_process_local() {
    static IntermittentTimer _(string("process_local()"));
    return _;
  }

  // private static methods

  // proceses a batch of data that is local (must be intra)
  static void process_local(T *elem, size_t count, FuncDistObj &func) {
    assert(elem);
    assert(count > 0);
    t_process_local().start();
    auto func_inst = *func;
    for (size_t i = 0; i < count; i++) {
      func_inst(elem[i]);
    }
    t_process_local().stop();
  }

  static void process_local(intra_global_block_t lblock, FuncDistObj &func) {
    assert(lblock.first);
    assert(lblock.first.is_local());
    process_local(lblock.first.local(), lblock.second, func);
  }

  // static my_partial_progress version does NOT call upcxx::progress()
  // just clears any ready rpcs
  static size_t my_partial_progress(track_rpcs_t &track_rpcs) {
    size_t pending_rpcs = track_rpcs->pop_finished();
    assert(pending_rpcs == track_rpcs->count_pending());
    return pending_rpcs;
  }

  bool my_progress_is_required;
  inline bool &my_progress_required() { return my_progress_is_required; }
  // performs upcxx::progress() and TwoTierAggrStore progress on rpc acknowledgments
  // returns the number of pending rpcs
  bool calc_my_progress_required() {
    my_progress_is_required = false;
    if (inter_fixed_memory_store->has_promises() || intra_fixed_memory_store->has_promises()) {
      // some promises exist
      my_progress_is_required = true;
    } else if (track_inter_rpcs->get_rpc_inner_timer().get_total_count() +
                   track_inter_rpcs->get_rpc_inner_timer().get_active_count() <
               track_inter_rpcs->get_rpc_timer().get_total_count() + track_inter_rpcs->get_rpc_timer().get_active_count()) {
      // inter inner rpcs (receiving) is less than rpcs (sending)
      my_progress_is_required = true;
    } else if (track_inter_rpcs->get_rpc_inner_timer().get_active_count() > 2 * split_rank::num_nodes()) {
      // there are more active inter rpcs requiring my progress than there are nodes.  Get them completed
      my_progress_is_required = true;
    } else if (track_intra_rpcs->get_rpc_inner_timer().get_active_count() > 2 * split_rank::num_threads()) {
      // there are more active intra rpcs requiring my progress than there are threads in a node.  Get them completed
      my_progress_is_required = true;
    }
    // DBG(__func__, ": ", my_progress_is_required, "\n");
    return my_progress_is_required;
  }

  size_t my_progress() {
    // DBG(__func__, " my_progress_is_required=", my_progress_is_required, " -- ", to_string(), "\n");
    t_prog.progress();
    calc_my_progress_required();
    return my_partial_progress(track_inter_rpcs) + my_partial_progress(track_intra_rpcs);
  }

  void wait_max_rpcs() {
    // limit pending RPCs still
    StallTimer is_stalled(description + string("-wait_max_rpcs"));
    while (my_progress() >= max_rpcs_in_flight) is_stalled.check();
  }

  // simply sends a single element via rpc, bypassing all blocks
  void send_rpc1(intrank_t target_rank, const T &elem) {
    auto fut = rpc(
        target_rank, [](T elem, FuncDistObj &func) { (*func)(elem); }, elem, func);
    track_inter_rpcs->push(fut);
  }

  // get the thread from a block with a thread_offset
  static inline thread_num_t &get_thread_from_block(T *block, int idx) {
    assert(idx >= 0);
    return *(((thread_num_t *)block) - 1 - idx);
  }

  // This function takes last element as pivot, places
  // the pivot element at its correct position in sorted
  // array, and places all smaller (smaller than pivot)
  // to left of pivot and all greater elements to right
  // of pivot
  static int block_quicksort_partition(T *block, int low, int high) {
    thread_num_t pivot = get_thread_from_block(block, high);  // pivot
    int i = (low - 1);                                        // Index of smaller element

    for (int j = low; j <= high - 1; j++) {
      // If current element is smaller than the pivot
      if (get_thread_from_block(block, j) < pivot) {
        i++;  // increment index of smaller element
        assert(i >= low);
        assert(j >= low);
        assert(i < high);
        assert(j < high);
        if (i != j) {
          assert(i < j);
          std::swap(block[i], block[j]);
          std::swap(get_thread_from_block(block, i), get_thread_from_block(block, j));
        }
        assert(get_thread_from_block(block, i) < pivot);
      }
    }
    assert(i + 1 >= low);
    if (i + 1 != high) {
      assert(i + 1 < high);
      std::swap(block[i + 1], block[high]);
      std::swap(get_thread_from_block(block, i + 1), get_thread_from_block(block, high));
    }
    return (i + 1);
  }

  // The main function that implements QuickSort
  // low --> Starting index,
  // high --> Ending index
  static void block_quicksort(T *block, int low, int high) {
    if (low < high) {
      /* pi is partitioning index, block[pi] is now
      at right place */
      int pi = block_quicksort_partition(block, low, high);

      // Separately sort elements before
      // partition and after partition
      block_quicksort(block, low, pi - 1);
      block_quicksort(block, pi + 1, high);
    }
  }

  // returns "virtual" intra blocks based on the underlying gblock for each local thread
  // some may be empty but there will be one entry for every local thread
  static vector<intra_global_block_t> inter_to_sorted_intra_blocks(inter_global_block_t &gblock, size_t start = 0) {
    assert(gblock.first);
    assert(gblock.first.is_local());
    assert(gblock.second > 0);
    assert(start <= gblock.second);
    DBG("Sorting ", gblock.second, " inter entries into intra blocks\n");

    T *block = gblock.first.local();
    if (gblock.second - start > 1) {  // no need to sort a single entry, right?
      block_quicksort(block, start, gblock.second - 1);
#ifdef DEBUG
      /* validate it was indeed sorted */
      int last_thread = -1;
      for (size_t idx = start; idx < gblock.second; idx++) {
        thread_num_t t = get_thread_from_block(gblock.first.local(), idx);
        assert(last_thread <= t);
        assert(t >= 0);
        assert(t < split_rank::num_threads());
        last_thread = t;
      }
#endif
    }
    vector<intra_global_block_t> intra_blocks;
    intra_blocks.resize(split_rank::num_threads(), {});
    int last_thread = -1;
    // find the partitions by thread in the sorted array
    // TODO there should be a faster way to do this in a long list
    for (size_t idx = start; idx < gblock.second; idx++) {
      thread_num_t &thread = get_thread_from_block(block, idx);
      assert(thread >= 0);
      assert(thread < split_rank::num_threads());
      if (last_thread > thread)
        DIE("inter_to_sorted_intra_blocks did not sort properly. idx=", idx, " gblock.second=", gblock.second,
            " last_thread=", (int)last_thread, " thread=", (int)thread, "\n");
      intra_global_block_t &gb = intra_blocks[thread];
      if (!gb.first) {
        assert(gb.second == 0);
        gb.first = gblock.first + idx;
      }
      gb.second++;
      last_thread = thread;
    }
    assert(gblock.first);             // did not modify
    assert(gblock.first.is_local());  // not modify
    assert(gblock.second > 0);
    return intra_blocks;
  }

  static upcxx::future<inter_global_block_t> inter_intra_inner_rpc_relay(inter_global_block_t lblock, track_rpcs_t &track_inter_rpcs,
                                                                  track_rpcs_t &track_intra_rpcs,
                                                                  intra_fixed_memory_t &intra_fixed_mem,
                                                                  inter_fixed_memory_t &inter_fixed_mem, FuncDistObj &func) {
    DBG("inter_relay processing inter RPC:", lblock.first, " ", lblock.second, "\n");
    assert(lblock.first);
    assert(lblock.first.is_local());
    assert(lblock.second > 0);
    inter_global_block_t lblock_consumed = lblock;
    lblock_consumed.second = 0;
    upcxx::future<intra_global_block_t> all_rpcs = make_future(lblock_consumed);
    intra_reservation_t reservation = intra_fixed_mem->acquire_partial_reservation();  // may be empty
    T *elem_ptr = lblock.first.local();
    thread_num_t *thread_ptr = ((thread_num_t *)elem_ptr) - 1;
    size_t res_sent = 0;
    size_t idx = 0;
    for (; idx < lblock.second; idx++) {
      if (reservation->empty()) break;             // must resort to plan B
      T &elem = elem_ptr[idx];                     // element increase up the stack
      thread_num_t &thread = thread_ptr[0 - idx];  // threads increase down the stack
      split_rank split = split_rank::from_thread(thread);
      bool sent_rpc = add_to_dest_store_intranode_nb(split, elem, track_intra_rpcs, intra_fixed_mem, reservation, func);
      if (sent_rpc) res_sent++;
    }
    if (res_sent) DBG("inter_relay res_sent=", res_sent, "\n");
    if (idx < lblock.second) {
      // plan B
      // sort the remaining entries by thread and send intra_rpcs directly using this set of virtual blocks
      auto inter_intra_rpc_timer = make_shared<inst_timer_t>(track_inter_rpcs->get_rpc_relay_timer());
      DBG("inter_relay sorting remaining ", lblock.second - idx, " entries for direct intra rpc\n");
      size_t direct_sent = 0;
      vector<intra_global_block_t> intra_blocks = inter_to_sorted_intra_blocks(lblock, idx);
      thread_num_t thread_idx = 0;
      for (intra_global_block_t intra_block : intra_blocks) {
        assert(thread_idx < split_rank::num_threads());
        if (intra_block.first && intra_block.second > 0) {
          size_t count = intra_block.second;
          auto fut_rpc = just_send_intra_rpc_nb(split_rank::get_rank_from_thread(thread_idx), intra_block, intra_fixed_mem,
                                                track_intra_rpcs, func)
                             .then([](intra_global_block_t ignored) {});
          all_rpcs = when_all(all_rpcs, fut_rpc);
          direct_sent++;
        }
        thread_idx++;
      }
      all_rpcs = all_rpcs.then([inter_intra_rpc_timer, direct_sent](inter_global_block_t lblock) {
        DBG("inter_relay All direct_sent=", direct_sent, " intra rpcs relayed from inter rpc took ",
            inter_intra_rpc_timer->get_elapsed_since_start(), " s count=", inter_intra_rpc_timer->get_total_count(),
            " active=", inter_intra_rpc_timer->get_active_count(), "\n");
        assert(lblock.first);
        assert(lblock.first.is_local());
        assert(lblock.second == 0);  // it is drained
        // stop timer, return the block
        return lblock;
      });
    }
    intra_fixed_mem->release_reservation(reservation);
    assert(reservation->empty());  // reservation was drained after release
    return all_rpcs;
  }

  // sends and rpc to the corresponding target rank on a different node with the same split_rank::split_local_team().rank_me()
  // consumes the gblock
  // returns the gblock once it is consumed
  void send_inter_rpc(intrank_t target_rank, inter_global_block_t &gblock) {
    assert(!gblock.first.is_null());
    assert(gblock.first.where() == rank_me());
    assert(gblock.second > 0);

    DBG("send_inter_rpc(", target_rank, ", gblock=", gblock.first, " size=", gblock.second, "\n");
    assert(inter_fixed_memory_store->valid());
    assert(intra_fixed_memory_store->valid());

    // time the round trip
    auto t_rpc = make_shared<inst_timer_t>(track_inter_rpcs->get_rpc_timer());

    auto fut = rpc(
        target_rank,
        [](inter_global_block_t gblock, track_rpcs_t &track_inter_rpcs, track_rpcs_t &track_intra_rpcs,
           intra_fixed_memory_t &intra_fixed_mem, inter_fixed_memory_t &inter_fixed_mem, FuncDistObj &func) {
          assert(gblock.first.where() != rank_me());  // no local data should be transmitted via RPC
          assert(split_rank::get_my_node() !=
                 split_rank::get_node_from_rank(gblock.first.where()));  // no inter-node transfer on the same node
          DBG("Executing process rpc inter node ", gblock.first, " ", gblock.second, "\n");
          auto t_inner_rpc = make_shared<inst_timer_t>(track_inter_rpcs->get_rpc_inner_timer());

          // future for both blocks after the rget completes
          auto fut_blocks = inter_fixed_mem->rget_block(gblock);

          // future for lblock after it is consumed
          auto fut_relay =
              fut_blocks
                  .then([&track_inter_rpcs, &track_intra_rpcs, &inter_fixed_mem, &intra_fixed_mem, &func](
                            inter_global_block_t gblock_IGNORED, inter_global_block_t lblock) {
                    assert(gblock_IGNORED.first);
                    assert(gblock_IGNORED.first.where() != rank_me());
                    assert(gblock_IGNORED.second == 0);  // is drained
                    return inter_intra_inner_rpc_relay(lblock, track_inter_rpcs, track_intra_rpcs, intra_fixed_mem, inter_fixed_mem,
                                                       func)
                        .then([&inter_fixed_mem](inter_global_block_t lblock) {
                          assert(lblock.first);
                          assert(lblock.first.is_local());
                          assert(lblock.second == 0);  // consumed
                          inter_fixed_mem->push_global(lblock);
                        });
                  })
                  .then([t_inner_rpc]() {
                    DBG("Completed inter rpc relay in ", t_inner_rpc->get_elapsed_since_start(),
                        " s count=", t_inner_rpc->get_total_count(), " active=", t_inner_rpc->get_active_count(), "\n");
                    // stop the timer
                  });
          // no need to wait on fut_relay -- cleanup of the timer and push back to inter_fixed_mem will eventually happen and be
          // verified

          // future for gblock after it is consumed -- possibly with a return package
          auto fut_return =
              fut_blocks
                  .then([&inter_fixed_mem](inter_global_block_t gblock,
                                           inter_global_block_t lblock_ignored) -> upcxx::future<inter_global_block_t> {
                    // optionally send this rank dest store back to the sending process
                    DBG("Returning inter rpc gblock=", gblock.first, "\n");
                    assert(gblock.first);
                    assert(gblock.second == 0);  // it is drained
                    assert(gblock.first.where() != rank_me());
                    assert(lblock_ignored.first);
                    assert(lblock_ignored.first.where() == rank_me());
                    node_num_t node = split_rank(gblock.first.where()).get_node();
                    DBG("to node=", node, "\n");
                    // check for another available lblock and enough in dest store to have efficient transfer speed
                    if (inter_fixed_mem->has_dest_stores() && inter_fixed_mem->global_reserved_size() > 0) {
                      inter_global_block_t &store_block = inter_fixed_mem->dest_store(node);
                      size_t store_size = store_block.second;
                      if (store_block.first && store_size * 2 > inter_fixed_mem->get_count_per_block() &&
                          store_size * sizeof(T) > 4 * ONE_KB) {
                        // send this store back to sender with this gblock
                        DBG("Sending store_size=", store_size, " back to ", gblock.first.where(), " store=", store_block.first,
                            " (swapped)\n");
                        assert(store_block.first);
                        assert(store_block.first.where() == rank_me());
                        auto lblock_fut = inter_fixed_mem->pop_global(true);
                        assert(lblock_fut.is_ready());
                        auto lblock = lblock_fut.result();
                        assert(lblock.first);
                        assert(lblock.first.where() == rank_me());
                        assert(lblock.second == 0);  // empty
                        // swap the store and free block
                        std::swap(store_block, lblock);
                        assert(lblock.second > 0);
                        auto fut_rput_blocks = inter_fixed_mem->rput_block(lblock, gblock);
                        return fut_rput_blocks.then([&inter_fixed_mem](inter_global_block_t lblock, inter_global_block_t gblock) {
                          DBG("rput finished lblock=", lblock.first, " ", lblock.second, " gblock=", gblock.first, " ",
                              gblock.second, "\n");
                          assert(lblock.first);
                          assert(lblock.first.where() == rank_me());
                          assert(lblock.second == 0);  // is drained
                          assert(gblock.first);
                          assert(gblock.first.where() != rank_me());
                          assert(gblock.second > 0);
                          // push lblock after the rput completes
                          inter_fixed_mem->push_global(lblock);
                          return gblock;
                        });
                      }
                    }
                    // did not end up sending, just return the empty gblock
                    DBG("Just returning gblock=", gblock.first, "\n");
                    assert(gblock.first);
                    assert(gblock.first.where() != rank_me());
                    assert(gblock.second == 0);  // it is drained
                    return make_future(gblock);
                  })
                  .then([t_inner_rpc](inter_global_block_t gblock) {
                    DBG("Returning inter rpc gblock=", gblock.first, " ", gblock.second, " in ",
                        t_inner_rpc->get_elapsed_since_start(), "\n");
                    return gblock;
                  });

          return fut_return;  // just the gblock
        },
        gblock, track_inter_rpcs, track_intra_rpcs, intra_fixed_memory_store, inter_fixed_memory_store, func);

    gblock = {};  // do not allow reuse of this global pointer until the return is ready and pushed back

    // handle returned global_block
    inter_fixed_memory_t &inter_fixed_mem = inter_fixed_memory_store;
    intra_fixed_memory_t &intra_fixed_mem = intra_fixed_memory_store;
    FuncDistObj &_func = func;
    track_rpcs_t &_track_inter_rpcs = track_inter_rpcs;
    track_rpcs_t &_track_intra_rpcs = track_intra_rpcs;
    auto fut_returned =
        fut.then([t_rpc, target_rank](inter_global_block_t gblock) {
             DBG("Got inter rpc ack from ", (int)target_rank, ": ", gblock.first, " in ", t_rpc->get_elapsed_since_start(), " s\n");
             // just stop the timer
             assert(gblock.first);
             assert(gblock.first.where() == rank_me());
             return make_future(gblock);
           })
            .then([&inter_fixed_mem, &_track_inter_rpcs, &_track_intra_rpcs, &intra_fixed_mem,
                   &_func](inter_global_block_t gblock) -> upcxx::future<inter_global_block_t> {
              assert(gblock.first);
              assert(gblock.first.where() == rank_me());
              auto fut_gblock = make_future(gblock);
              if (gblock.second > 0) {
                // there is data to be processed in this return ack, so relay it
                DBG("Processing inter rpc ack gblock=", gblock.first, " ", gblock.second, "\n");
                return inter_intra_inner_rpc_relay(gblock, _track_inter_rpcs, _track_intra_rpcs, intra_fixed_mem, inter_fixed_mem,
                                                   _func);
              } else {
                return to_future(gblock);
              }
            })
            .then([&inter_fixed_mem](inter_global_block_t gblock) {
              assert(gblock.first);
              assert(gblock.second == 0);  // it is drained
              inter_fixed_mem->push_global(gblock);
              assert(!gblock.first);  // is invalidated
            });

    // remember this rpc to wait on later (may not be necessary)
    track_inter_rpcs->push(fut_returned);
  }

  // sends an rpc to target_rank located on this node (intra).
  // consumes the gblock
  // returns the gblock one it is consumed
  static void send_intra_rpc(intrank_t target_rank, intra_global_block_t &gblock, track_rpcs_t &track_intra_rpcs,
                             intra_fixed_memory_t &intra_fixed_mem, FuncDistObj &func) {
    assert(!gblock.first.is_null());
    assert(gblock.first.where() == rank_me());
    assert(gblock.second > 0);

    DBG("send_intra_rpc(", target_rank, ", gblock=", gblock.first, " size=", gblock.second, "\n");
    assert(intra_fixed_mem->valid());
    my_partial_progress(track_intra_rpcs);

    auto fut_gblock = just_send_intra_rpc_nb(target_rank, gblock, intra_fixed_mem, track_intra_rpcs, func);

    // handle returned global_block
    auto fut_returned = fut_gblock.then([&intra_fixed_mem](intra_global_block_t gblock) {
      DBG("Returned acknowledged global ", gblock.first, "\n");
      assert(gblock.first);
      assert(gblock.first.where() == rank_me());
      intra_fixed_mem->push_global(gblock);
      assert(!gblock.first);  // is invalidated
    });

    // remember this rpc to wait on later
    track_intra_rpcs->push(fut_returned);
  }

  static upcxx::future<intra_global_block_t> just_send_intra_rpc_nb(intrank_t target_rank, intra_global_block_t &gblock,
                                                             intra_fixed_memory_t &intra_fixed_mem, track_rpcs_t &track_intra_rpcs,
                                                             FuncDistObj &func) {
    // time the round-trip
    auto t_rpc = make_shared<inst_timer_t>(track_intra_rpcs->get_rpc_timer());

    // This RPC just starts consuming the global block and makes progress on the remote rank
    // It returns a future global block when the remote rank has finished consuming it
    auto fut = rpc(
        target_rank,
        [](intra_global_block_t gblock, FuncDistObj &func, intra_fixed_memory_t &intra_fixed_mem, track_rpcs_t &track_intra_rpcs) {
          DBG("Executing process rpc intra node ", gblock.first, " ", gblock.second, ", intra_fixed_mem: ", &(*intra_fixed_mem),
              "\n");
          auto t_inner_rpc = make_shared<inst_timer_t>(track_intra_rpcs->get_rpc_inner_timer());
          upcxx::future<> finished;

          upcxx::future<intra_global_block_t> fut_gblock;
          if (gblock.first.is_local()) {
            // processes the data immediately
            process_local(gblock, func);
            fut_gblock = to_future(gblock);
            finished = make_future();
          } else {
            // copy the global data, then process it eventually
            DBG("initiating rget of non-local intra gblock:", gblock.first, "\n");
            auto fut_blocks = intra_fixed_mem->rget_block(gblock);
            fut_gblock = fut_blocks.then([](intra_global_block_t gblock, intra_global_block_t ignored) { return gblock; });
            finished = fut_blocks.then([&func, &intra_fixed_mem](intra_global_block_t ignored, intra_global_block_t lblock) {
              process_local(lblock, func);
              intra_fixed_mem->push_global(lblock);
            });
          }
          finished.then([t_inner_rpc]() {
            DBG("intra rpc finished in ", t_inner_rpc->get_elapsed_since_start(), " s\n");
            // stop the timer
          });
          return fut_gblock;  // return th global_block to sender for reuse
        },
        gblock, func, intra_fixed_mem, track_intra_rpcs);
    gblock = {};  // do not allow reuse of this global pointer until the return is ready and pushed back

    return fut.then([t_rpc](intra_global_block_t gblock) {
      DBG("intra rpc returned in ", t_rpc->get_elapsed_since_start(), " s\n");
      // stop the timer
      return gblock;
    });
  }

  inline void send_intra_rpc(intrank_t target_rank, intra_global_block_t &gblock) {
    send_intra_rpc(target_rank, gblock, track_intra_rpcs, intra_fixed_memory_store, func);
  }

  // operation on 1 element (i.e. no dest_store)
  // will block until sufficient global available blocks are available
  // and subject to the maximum rpcs in flight
  void update_remote1(intrank_t target_rank, const T &elem) {
    assert(max_store_size <= 1);
    // limit pending RPCs still
    wait_max_rpcs();
    update_remote1_nb(target_rank, elem);
  }

  // non blocking version (for use in future chains)
  inline void update_remote1_nb(intrank_t target_rank, const T &elem) { send_rpc1(target_rank, elem); }

  // operate on a vector of elements in the dest_stores
  // will block until sufficient global available blocks are available
  inline static void update_remote_intra(intrank_t target_rank, intra_global_block_t &gblock, track_rpcs_t &track_intra_rpcs,
                                         intra_fixed_memory_t &intra_fixed_mem, FuncDistObj &func) {
    intra_reservation_t empty_res;
    update_remote_intra_nb(target_rank, gblock, track_intra_rpcs, intra_fixed_mem, empty_res, func);
  }

  // and subject to the maximum rpcs in flight
  static void update_remote_intra_nb(intrank_t target_rank, intra_global_block_t &gblock, track_rpcs_t &track_intra_rpcs,
                                     intra_fixed_memory_t &intra_fixed_mem, intra_reservation_t &reservation, FuncDistObj &func) {
    DBG("update_remote_intra(target_rank=", target_rank, ", gblock=", gblock.first, ", size=", gblock.second, "\n");
    assert(gblock.first);
    assert(gblock.first.where() == rank_me());
    if (gblock.second == 0) {
      return; // noop to send an empty block
    }
    intra_global_block_t send_gblock = gblock;  // make a copy
    gblock = {};                                // invalidate it first

    // now get another gblock
    if (reservation && !reservation->empty()) {
      replace_intra_store_nb(gblock, intra_fixed_mem, reservation);
    } else {
      assert(!gblock.first);
      auto fut = replace_intra_store(gblock, intra_fixed_mem);
      if (!fut.is_ready()) DBG(__func__, " will wait\n");
      fut.wait();
    }
    assert(gblock.first);
    assert(gblock.first.where() == rank_me());

    // now send the copy after gblock is available again
    send_intra_rpc(target_rank, send_gblock, track_intra_rpcs, intra_fixed_mem, func);
    assert(!send_gblock.first);  // is invalidated
  }

  inline void update_remote_intra(intrank_t target_rank, intra_global_block_t &gblock) {
    intra_reservation_t empty_res;
    update_remote_intra_nb(target_rank, gblock, track_intra_rpcs, intra_fixed_memory_store, empty_res, func);
  }

  inline void update_remote_intra_nb(intrank_t target_rank, intra_global_block_t &gblock, intra_reservation_t &reservation) {
    update_remote_intra(target_rank, gblock, track_intra_rpcs, intra_fixed_memory_store, reservation, func);
  }

  static upcxx::future<> replace_inter_store(inter_global_block_t &gblock, inter_fixed_memory_t &inter_fixed_memory_store) {
    assert(!gblock.first);
    assert(gblock.second == 0);
    upcxx::future<inter_global_block_t> newblock = inter_fixed_memory_store->pop_global();
    return when_all(make_future(std::ref(gblock)), newblock)
        .then([&inter_fixed_memory_store](inter_global_block_t &gblock, inter_global_block_t newblock) {
          assert(newblock.first);
          assert(newblock.first.where() == rank_me());
          assert(newblock.second == 0);
          if (!gblock.first) {
            // won the race
            gblock = newblock;
          } else {
            // lost the race put the newblock back
            inter_fixed_memory_store->push_global(newblock);
          }
        });
  }

  static upcxx::future<> replace_intra_store(intra_global_block_t &gblock, intra_fixed_memory_t &intra_fixed_memory_store) {
    upcxx::future<intra_global_block_t> newblock = intra_fixed_memory_store->pop_global();
    return when_all(make_future(std::ref(gblock)), newblock)
        .then([&intra_fixed_memory_store](intra_global_block_t &gblock, intra_global_block_t newblock) {
          assert(newblock.first);
          assert(newblock.first.where() == rank_me());
          assert(newblock.second == 0);
          if (!gblock.first) {
            // won the race
            gblock = newblock;
            DBG("update_remote_intra: got an used new gblock from dispatcher:", gblock.first, "\n");
          } else {
            // lost the race put the newblock back
            intra_fixed_memory_store->push_global(newblock);
          }
        });
  }

  static void replace_intra_store_nb(intra_global_block_t &gblock, intra_fixed_memory_t &intra_fixed_mem,
                                     intra_reservation_t &reservation) {
    assert(reservation);
    if (!reservation) DIE("invalid call without reservation!\n");
    assert(!reservation->empty());
    if (reservation->empty()) DIE("Unexpected - the reservation is fully drained!\n");

    // replace the gblock with a reserved block
    assert(!gblock.first);
    gblock = reservation->back();
    reservation->pop_back();

    assert(gblock.first);
    assert(gblock.first.where() == rank_me());
    assert(gblock.second == 0);
    DBG("update_remote_intra: got new gblock from reservation:", gblock.first, "\n");
  }

  // operate on a vector of elements in the dest_stores
  // will block until sufficient global available blocks are available
  // and subject to the maximum rpcs in flight
  void update_remote_inter(intrank_t target_rank, inter_global_block_t &gblock) {
    assert(gblock.first);
    assert(gblock.second > 0);
    auto fut = update_remote_inter_nb(target_rank, gblock);
    DBG(__func__, " my_progress\n");
    if (!fut.is_ready()) {
      DBG(__func__, " still waiting on inter dest store\n");
    }
    fut.wait();
    assert(gblock.first);  // is valid again
  }

  upcxx::future<> update_remote_inter_nb(intrank_t target_rank, inter_global_block_t &gblock) {
    if (gblock.second == 0) DIE("Invalid call to update_remote_inter on an empty global block\n");
    assert(gblock.first);
    assert(gblock.first.where() == rank_me());
    inter_global_block_t sendBlock = gblock;  // copy
    size_t node = split_rank(target_rank).get_node();
    assert(inter_fixed_memory_store->dest_store(node) == sendBlock);
    gblock = {};  // invalidate it
    auto fut = replace_inter_store(gblock, inter_fixed_memory_store);
    send_inter_rpc(split_rank::get_rank_from_node(node), sendBlock);  // send to dedicated rank on remote node
    if (!fut.is_ready()) DBG("intra dest store is not immediately ready\n");
    return fut;
  }

  // returns true if an RPC was initiated
  inline static bool add_to_dest_store_intranode(split_rank split, const T &elem, track_rpcs_t &track_intra_rpcs,
                                                 intra_fixed_memory_t &intra_fixed_mem, FuncDistObj &func) {
    intra_reservation_t empty_res;
    return add_to_dest_store_intranode_nb(split, elem, track_intra_rpcs, intra_fixed_mem, empty_res, func);
  }
  // non-blocking version (with non-empty reservation)
  static bool add_to_dest_store_intranode_nb(split_rank split, const T &elem, track_rpcs_t &track_intra_rpcs,
                                             intra_fixed_memory_t &intra_fixed_mem, intra_reservation_t &reservation,
                                             FuncDistObj &func) {
    // intranode
    size_t max_store_size = intra_fixed_mem->get_count_per_block();
    intra_global_block_t &gblock = intra_fixed_mem->dest_store(split.get_thread());
    if (!gblock.first && reservation) {
      // This is a race between the master persona waiting for a free gblock and an intra node rpc executing while it waits
      // it is safe for this stack to swap in a new gblock from the reservation
      assert(!reservation->empty());
      gblock = reservation->back();
      reservation->pop_back();
      DBG("add_to_dest_store_intranode replaced empty gblock with one from my reservation\n");
    }
    assert(gblock.second < max_store_size);
    if (gblock.second >= max_store_size)
      DIE("Invalid state of gblock with ", gblock.second, " elements but max of ", max_store_size, "\n");
    assert(gblock.first);
    assert(gblock.first.where() == rank_me());
    if (gblock.first.where() != rank_me()) DIE("Invalid state of gblock not local to current rank: ", gblock.first, "\n");
    T *lptr = gblock.first.local();
    lptr[gblock.second++] = elem;
    if (gblock.second == max_store_size) {
      // DBG("add_to_dest_store_intranode found full for ", (int) split.get_thread(), "/", split.get_rank(), " reservation:",
      // reservation.get(), ", gblock=", gblock.first, ",", gblock.second, "\n");
      if (reservation && reservation->empty()) DIE("Invalid state for ", __func__, " reservation is present but empty\n");
      update_remote_intra_nb(split.get_rank(), gblock, track_intra_rpcs, intra_fixed_mem, reservation, func);
      assert(gblock.first);  // gblock is restored
      assert(gblock.first == intra_fixed_mem->dest_store(split.get_thread()).first);
      return true;
    }
    assert(gblock.first);  // gblock is still good
    assert(gblock.first.local());
    assert(gblock.first == intra_fixed_mem->dest_store(split.get_thread()).first);
    return false;
  }

  void _add_to_dest_store_internode_fast(split_rank split, const T &elem, inter_global_block_t &gblock) {
    assert(gblock.first);
    assert(gblock.first.where() == rank_me());
    assert(inter_fixed_memory_store->get_thread_offset() > 0);
    if (gblock.second >= max_store_size) DIE("Invalid call to add_to_dest_store_internode_fast\n");
    T *lptr = gblock.first.local();
    // element ascendes after the pointer
    lptr[gblock.second] = elem;
    // thread num decends before the pointer
    thread_num_t *t = ((thread_num_t *)lptr) - 1 - gblock.second;
    *t = split.get_thread();
    gblock.second++;
    assert(gblock.second <= max_store_size);
  }

  // adds an entry to the dest store
  // may send an rpc (returning true in that case)
  // may block if inter global blocks are unavailable
  bool add_to_dest_store_internode(split_rank split, const T &elem) {
    inter_global_block_t &gblock = inter_fixed_memory_store->dest_store(split.get_node());
    assert(gblock.first);
    assert(gblock.first == inter_fixed_memory_store->dest_store(split.get_node()).first);
    if (gblock.second >= max_store_size)
      DIE("Invalid state of gblock with ", gblock.second, " elements but max of ", max_store_size, "\n");
    assert(gblock.second < max_store_size);
    if (gblock.first.where() != rank_me()) DIE("Invalid state of gblock not local to current rank: ", gblock.first, "\n");

    bool did_send = false;
    if (gblock.second < max_store_size) {
      _add_to_dest_store_internode_fast(split, elem, gblock);
    }
    if (gblock.second == max_store_size) {
      update_remote_inter(split.get_rank(), gblock);
      did_send = true;
    }
    assert(gblock.second < max_store_size);
    assert(gblock.first);
    assert(gblock.first.where() == rank_me());
    assert(gblock.first == inter_fixed_memory_store->dest_store(split.get_node()).first);
    return did_send;
  }

  // returns true if an RPC was initiated
  // may block
  bool add_to_dest_store(intrank_t target_rank, const T &elem) {
    bool sent_rpc = false;
    if (max_store_size <= 1) {
      update_remote1(target_rank, elem);
      sent_rpc = true;
    } else {
      split_rank split(target_rank);
      if (split.is_local()) {
        // intranode
        sent_rpc = add_to_dest_store_intranode(split, elem, track_intra_rpcs, intra_fixed_memory_store, func);
      } else {
        // internode
        assert(split_rank::num_nodes() > 1);
        sent_rpc = add_to_dest_store_internode(split, elem);
      }
    }
    if (sent_rpc) {
      my_progress();  // progress anyway to kick off the rpc
    }
    return sent_rpc;
  }

 public:
  string description;

  TwoTierAggrStore(FuncDistObj &f, const string description)
      : func(f)
      , max_store_size(0)
      , max_rpcs_in_flight(MAX_RPCS_IN_FLIGHT)
      , intra_fixed_memory_store(func.team(), description + string("-intra-store"))
      , inter_fixed_memory_store(func.team(), description + string("-inter-store"))
      , track_inter_rpcs(func.team(), description + string("-track-inter-rpc"))
      , track_intra_rpcs(func.team(), description + string("-track-intra-rpc"))
      , t_prog(description + string("-TwoTierAggrStore"))
      , description(description)
      , my_progress_is_required(false) {}
  TwoTierAggrStore(const TwoTierAggrStore &) = delete;
  TwoTierAggrStore(TwoTierAggrStore &&) = default;
  virtual ~TwoTierAggrStore() { clear(); }

  string to_string() const {
    ostringstream os;
    os << description;
    os << "-TwoTierAggrStore";
    os << "inter_store=" << inter_fixed_memory_store->to_string() << ",";
    os << "intra_store=" << intra_fixed_memory_store->to_string() << ",";
    os << "track_inter_rpcs=" << track_inter_rpcs->to_string() << ",";
    os << "inter_rpc_t=" << track_inter_rpcs->get_rpc_timer().get_total_count() << "/"
       << track_inter_rpcs->get_rpc_inner_timer().get_total_count() << ",";
    os << "inter_intra_rpc_t=" << track_inter_rpcs->get_rpc_relay_timer().get_total_count() << ",";
    os << "track_intra_rpcs=" << track_intra_rpcs->to_string() << ",";
    os << "intra_rpc_t=" << track_intra_rpcs->get_rpc_timer().get_total_count() << "/"
       << track_intra_rpcs->get_rpc_inner_timer().get_total_count() << ",";
    os << ")";
    return os.str();
  }

  static void optimal_num_blocks_and_count_per(const size_t max_bytes, const size_t max_rpcs, size_t &num_intra_blocks,
                                               size_t &num_inter_blocks, size_t &count_per_block) {
    // a few constraints and priorities for optimization
    // required:
    //   num_blocks * sizeof(T) * count_per_block <= max_bytes
    //   min_rpcs_in_flight <= rpcs_in_flight <= max_rpcs_in_flight
    //   rpcs_in_flight == num_blocks - dest_store_size
    //
    // optimization compromises:
    //   count_per_block * sizeof(T) >= 8KB, optimally much larger 1MB
    //   dest_store_size == count_per_block == 1 ? 0 : rank_n()
    //   max_rpcs_in_flight == min(rank_n() * 10, 2048);
    //   min_rpcs_in_flight == rank_n() // possibly nodes (technically 1)
    //
    // furthermore if num_nodes == 1 there will be 0 internode blocks

    // start calcs with min limits
    size_t sz = sizeof(T) + sizeof(thread_num_t);
    size_t inter_dest_store_size = split_rank::num_nodes();
    size_t intra_dest_store_size = split_rank::num_threads();
    size_t res_size = split_rank::num_threads();
    size_t min_inter_rpcs_in_flight = 2 * inter_dest_store_size + 16;                   // every 2 * inter dest store + 16
    size_t min_reservations = 1 + split_rank::num_nodes() / split_rank::num_threads();  // 1 + nodes/(cores/node)
    size_t min_intra_rpcs_in_flight =
        intra_dest_store_size + min_reservations * res_size + 16;  // every dest store + a few reservations + 16

    if (split_rank::num_nodes() == 1) {
      sz = sizeof(T);
      inter_dest_store_size = 0;
      res_size = 0;
      min_inter_rpcs_in_flight = 0;
      min_intra_rpcs_in_flight = intra_dest_store_size + 16;
    }

    size_t num_blocks = 0;
    size_t min_rpcs_in_flight = min_intra_rpcs_in_flight + min_inter_rpcs_in_flight;
    size_t rpcs_in_flight = min_rpcs_in_flight * 8;
    if (min_rpcs_in_flight > 2 * max_rpcs) {
      rpcs_in_flight = min_rpcs_in_flight;  // min == max and it will exceed max_rpcs
    } else if (rpcs_in_flight > 2 * max_rpcs) {
      rpcs_in_flight = 2 * max_rpcs;  // reduce the starting max rpc
    }

    DBG("optimizing max_bytes=", get_size_str(max_bytes), " min_rpcs=", min_rpcs_in_flight, " inter=", inter_dest_store_size,
        " intra=", intra_dest_store_size, "\n");
    if (max_bytes >= 2 * sz * (min_rpcs_in_flight + inter_dest_store_size + intra_dest_store_size)) {
      // start with large blocks and max rpcs in flight
      // decrease rpcs_in_flight to 2* minimums
      // decrease block size to 16KB
      // decrease rpcs_in_flight to minimum
      // decrease block size further.
      size_t target_min_mem = 16 * ONE_KB - 64;  // still fast but below gets noticibly slower
      size_t mem_per_block = 128 * ONE_KB - 64;  // initial best case
      count_per_block = (mem_per_block + sz - 1) / sz;
      DBG("optimizing mem_per_block=", get_size_str(mem_per_block), " rpcs=", rpcs_in_flight, " count_per_block=", count_per_block,
          " block_size=", get_size_str(count_per_block * sz), "\n");
      do {
        num_blocks = rpcs_in_flight + inter_dest_store_size + intra_dest_store_size;
        if (sz * num_blocks * count_per_block < max_bytes) break;
        size_t try_mem = 3 * mem_per_block / 4;    // reduce to 75%
        size_t try_rpcs = 3 * rpcs_in_flight / 4;  // reduce to 75%
        if (try_rpcs > 2 * min_rpcs_in_flight) {   // first reduce in-flight to 2*minimum
          rpcs_in_flight = try_rpcs;
        } else if (try_mem > target_min_mem) {  // next reduce count to target minimum
          rpcs_in_flight = 2 * min_rpcs_in_flight;
          mem_per_block = try_mem;
        } else if (try_rpcs > min_rpcs_in_flight) {  // next reduce in-flight to minimum
          mem_per_block = target_min_mem;
          rpcs_in_flight = try_rpcs;
        } else {  // lastly reduce block size below the target_min_mem
          rpcs_in_flight = min_rpcs_in_flight;
          mem_per_block = (mem_per_block > ONE_KB) ? (3 * mem_per_block / 4) : (mem_per_block / 2);
        }
        count_per_block = (mem_per_block + sz - 1) / sz;
        DBG("optimizing mem_per_block=", get_size_str(mem_per_block), " rpcs=", rpcs_in_flight,
            " count_per_block=", count_per_block, "\n");
      } while (count_per_block > 1);
    } else {
      count_per_block = 1;
    }

    if (count_per_block <= 1) {
      // no allocation - just direct rpcs
      count_per_block = 1;
      inter_dest_store_size = 0;
      intra_dest_store_size = 0;
      num_blocks = 0;
      rpcs_in_flight = max_bytes / sz < max_rpcs ? max_bytes / sz : max_rpcs;
    } else {
      num_blocks = rpcs_in_flight + inter_dest_store_size + intra_dest_store_size;
    }

    // All of these must still be true
    assert(count_per_block * sizeof(T) * num_blocks <= max_bytes);
    assert(count_per_block >= 1);

    if (num_blocks > 1) {
      // calculate the inter and intra block counts
      double inter_fraction = .75;  // 75% of the extra blocks go to inter stores
      assert(min_inter_rpcs_in_flight + min_intra_rpcs_in_flight == min_rpcs_in_flight);
      assert(min_inter_rpcs_in_flight + min_intra_rpcs_in_flight <= rpcs_in_flight);

      if (split_rank::num_nodes() > 1) {
        num_inter_blocks =
            inter_dest_store_size + min_inter_rpcs_in_flight + (rpcs_in_flight - min_rpcs_in_flight) * inter_fraction;
      } else {
        num_inter_blocks = 0;
        inter_fraction = 0.0;
      }
      num_intra_blocks =
          intra_dest_store_size + min_intra_rpcs_in_flight + (rpcs_in_flight - min_rpcs_in_flight) * (1.0 - inter_fraction);
      assert(num_blocks >= num_inter_blocks + num_intra_blocks);

    } else {
      num_intra_blocks = num_inter_blocks = 0;
    }
    SOUT("Calculated optimal num and block size for ", split_rank::num_nodes(), " internode sets and ",
         (size_t)split_rank::num_threads(), " intranode ranks per node\n");
    SOUT("Found optimal TwoTierAggrStore of num_intra_blocks=", num_intra_blocks, " num_inter_blocks=", num_inter_blocks,
         " count_per_block=", count_per_block, " (", get_size_str(count_per_block * sz), " per block, ",
         get_size_str((num_inter_blocks + num_intra_blocks) * count_per_block * sz), ")\n");
  }

  void set_size(size_t max_store_bytes) {
    DBG("TwoTierAggrStore::set_size(", max_store_bytes, ")\n");

    size_t count_per_block = 0, num_intra_blocks = 0, num_inter_blocks = 0;
    optimal_num_blocks_and_count_per(max_store_bytes, max_rpcs_in_flight, num_intra_blocks, num_inter_blocks, count_per_block);
    assert(count_per_block > 0);

    if (count_per_block <= 1) {
      // no reason for delay and storage of 1 entry (i.e. small max mem at large scale), still uses max_rpcs_in_flight
      max_store_size = 0;
      num_intra_blocks = 0;
      num_inter_blocks = 0;
      count_per_block = 1;  // will send single rpcs
      if (max_store_bytes > 0) {
        // not intentionally disabled
        SWARN("Using no TwoTierAggrStore to aggregate messages because no configutation works with less than ",
              get_size_str(max_store_bytes), " at this scale\n");
      }
    } else {
      max_store_size = count_per_block;
    }

    size_t per_intra_rpc_bytes = count_per_block * sizeof(T);
    size_t per_inter_rpc_bytes = count_per_block * (sizeof(T) + sizeof(thread_num_t));
    size_t total_blocks = num_intra_blocks + num_inter_blocks;

    if (num_intra_blocks == 0) {
      // no dest stores will be used intra or inter
      assert(num_inter_blocks == 0);
      assert(max_store_size == 0);
      assert(count_per_block == 1);
      num_intra_blocks = 0;
      num_inter_blocks = 0;
    }

    node_num_t nodes = split_rank::num_nodes();
    // always have intra node

    SOUT("Establishing ", description, " intra dest stores\n");
    intra_fixed_memory_store->set_fixed_mem(num_intra_blocks, count_per_block, split_rank::num_threads(), false,
                                            nodes == 1 ? 0 : split_rank::num_threads());

    SOUT("Establishing ", description, " inter dest stores\n");
    if (nodes == 1) {
      // special case for single node with no internode rpcs needed
      if (rank_n() > 1) assert(num_inter_blocks == 0);
      inter_fixed_memory_store->set_fixed_mem(0, 1, 1, false, 0);
    } else {
      if (num_inter_blocks > 0)
        assert(num_inter_blocks >= split_rank::num_nodes() * 3);  // room for dest store, 1 reservation and 1 in flight
      inter_fixed_memory_store->set_fixed_mem(num_inter_blocks, count_per_block, split_rank::num_nodes(), true,
                                              split_rank::num_nodes());
    }

    SOUT("Using a ", description, " store of max ",
         get_size_str(num_intra_blocks * per_intra_rpc_bytes + num_inter_blocks * per_inter_rpc_bytes),
         " per target rank, giving max ", max_store_size, " of ", get_size_str(sizeof(T)), "/",
         get_size_str(sizeof(T) + sizeof(thread_num_t)), " entries per target rank (", get_size_str(per_intra_rpc_bytes), "/",
         get_size_str(per_inter_rpc_bytes), ", ", get_size_str(per_intra_rpc_bytes * num_intra_blocks), "/",
         get_size_str(per_inter_rpc_bytes * num_inter_blocks), ") and ", max_rpcs_in_flight, " rpcs in flight\n");
  }

  // true only if there are no element stored and no rpcs in flight
  inline bool empty() const {
    return track_intra_rpcs->empty() && intra_fixed_memory_store->empty() && track_inter_rpcs->empty() &&
           inter_fixed_memory_store->empty();
  }

  void clear() {
    DBG("TwoTierAggrStore::clear()\n");
    inter_fixed_memory_store->clear_dest_stores();
    intra_fixed_memory_store->clear_dest_stores();
    if (!empty()) DIE("clear() called on a non-empty TwoTierAggrStore!\n");
    track_inter_rpcs->clear();
    inter_fixed_memory_store->clear();
    track_intra_rpcs->clear();
    intra_fixed_memory_store->clear();
    t_prog.print_out();
    t_process_local().print_out();
    Timings::wait_pending();
    assert(intra_fixed_memory_store->empty());
    assert(inter_fixed_memory_store->empty());
    assert(!intra_fixed_memory_store->valid());
    assert(!inter_fixed_memory_store->valid());
    barrier();
  }

  bool update(intrank_t target_rank, const T &elem) {
    static size_t update_count = 0;
    bool ret = add_to_dest_store(target_rank, elem);
    update_count++;
    bool progress_is_required = my_progress_required();
    if (update_count % (progress_is_required ? 32 : 4096) == 0) {
      my_progress();
    }
    return ret;
  }

  void flush_inter_updates() {
    if (split_rank::num_nodes() == 1) {
      if (!inter_fixed_memory_store->empty()) DIE("flush_inter_updates called when there is only 1 node from split_rank!\n");
      return;
    }
    Timer timer(description + "-TwoTierAggrStore::flush_inter_updates");
    DBG("flushing inter updates...\n");

    // first flush inter node stores
    size_t num_inter_dest = split_rank::num_nodes() == 1 ? 0 : split_rank::num_nodes();
    if (num_inter_dest == 0) assert(inter_fixed_memory_store->empty());
    for (node_num_t _node = 0; _node < num_inter_dest; _node++) {
      node_num_t node = (_node + 1 + split_rank::get_my_node()) %
                        split_rank::num_nodes();  // rotate the flushes, starting with the next node in the job
      if (max_store_size > 0) {
        inter_global_block_t &gblock = inter_fixed_memory_store->dest_store(node);
        if (node == split_rank::get_my_node()) {
          assert(gblock.second == 0);
        }
        assert(gblock.first);
        assert(gblock.first.where() == rank_me());
        if (gblock.second > 0) {
          update_remote_inter(split_rank::get_rank_from_node(node), gblock);
        }
      }
    }
    my_progress();
    DBG("all my internode data send rpcs have been sent\n");
  }

  void flush_intra_updates() {
    Timer timer(description + "-TwoTierAggrStore::flush_intra_updates");
    DBG("flushing intra updates...\n");

    for (thread_num_t _thread = 0; _thread < split_rank::num_threads(); _thread++) {
      thread_num_t thread = (_thread + 1 + split_rank::get_my_thread()) %
                            split_rank::num_threads();  // rotate the flushes starting with the next thread
      if (max_store_size > 0) {
        intra_global_block_t &gblock = intra_fixed_memory_store->dest_store(thread);
        assert(gblock.first);
        assert(gblock.first.where() == rank_me());
        if (gblock.second > 0) {
          update_remote_intra(split_rank::get_rank_from_thread(thread), gblock);
          assert(gblock.first);
          assert(gblock.first.where() == rank_me());
        }
      }
    }
    DBG(__func__, " my_progress\n");
    my_progress();
    DBG("all my intranode data send rpcs have been sent\n");
  }

  static void flush_intra_updates_with_res(intra_reservation_t &reservation, track_rpcs_t &track_intra_rpcs,
                                           intra_fixed_memory_t &intra_fixed_mem, FuncDistObj &func) {
    assert(reservation);
    int count_flushed = 0;
    if (!reservation->empty()) {
      // flush the most full dest_stores first, stopping at 1/4 capacity
      vector<size_t> rank_counts;
      rank_counts.reserve(split_rank::num_threads());
      for (thread_num_t thread = 0; thread < split_rank::num_threads(); thread++) {
        size_t s = intra_fixed_mem->dest_store(thread).second;
        assert(s < (1ull << 32));
        if (s >= intra_fixed_mem->get_count_per_block() / 4) {
          s = (s << 32) | thread;  // combine count in high bits, thread in low bits
          rank_counts.push_back(s);
        }
      }
      if (!rank_counts.empty()) {
        // sort ascending by count, then thread
        std::sort(rank_counts.begin(), rank_counts.end());
      }
      while (!reservation->empty() && !rank_counts.empty()) {
        size_t r_c = rank_counts.back();
        rank_counts.pop_back();
        size_t _thread = r_c & 0xffffffff;
        thread_num_t thread = _thread;
        size_t count = (r_c >> 32) & 0xffffffff;
        intra_global_block_t &gblock = intra_fixed_mem->dest_store(thread);
        assert(gblock.first);
        assert(gblock.first.where() == rank_me());
        update_remote_intra_nb(split_rank::get_rank_from_thread(thread), gblock, track_intra_rpcs, intra_fixed_mem, reservation,
                               func);
        assert(gblock.first);
        assert(gblock.first.where() == rank_me());
        count_flushed++;
      }
    }
    DBG("flush_intra_updates with reservation flushed ", count_flushed, " intra stores\n");
  }

  void flush_updates() {
    BarrierTimer timer(description + "-TwoTierAggrStore::flush_updates", false);
    DBG("flushing updates...\n");

    flush_inter_updates();

    // pre-emptively flush intra_stores
    // create a (possibly small) reservation of intra blocks
    intra_reservation_t res = make_shared<vector<intra_global_block_t> >();
    // clear dest stores so global_dispatcher can be empty()
    inter_fixed_memory_store->clear_dest_stores();
    StallTimer is_inter_stalled(description + "-flush_updates-inter-store-empty");
    do {
      is_inter_stalled.check();
      my_progress();
      while (res->size() < split_rank::num_threads() &&
             intra_fixed_memory_store->global_available_size() > split_rank::num_threads()) {
        auto fut_gblock = intra_fixed_memory_store->pop_global();
        if (!fut_gblock.is_ready()) DIE("Invalid state - there were available blocks but just popped one not ready!\n");
        res->push_back(fut_gblock.result());
      }
      if (!res->empty()) {
        flush_intra_updates_with_res(res, track_intra_rpcs, intra_fixed_memory_store, func);
      }
    } while (!inter_fixed_memory_store->empty());
    // replace the temporary intra blocks within the reservation and destroy it
    intra_fixed_memory_store->release_reservation(res);
    assert(res->empty());
    res.reset();
    track_inter_rpcs->flush(0);

    DBG("all my data send rpcs returned and global blocks have returned too\n");

    {
      BarrierTimer timer2(description + "-TwoTierAggrStore::flush_updates after inter-node", split_rank::num_nodes() > 1);
      // now all threads have received all inter node rpcs.  Flush last internode RPCs that may not have been processed yet
      StallTimer is_inter_stalled_again(description + "-flush_updates-inter-store-empty-again");
      do {
        is_inter_stalled_again.check();
        my_progress();
      } while (!inter_fixed_memory_store->empty());
      assert(inter_fixed_memory_store->empty());  // should remain empty

      // now all threads have received all inter node rpcs.  Flush last intra node stores
      // last flush intra node stores
      flush_intra_updates();

      // clear dest stores so global_dispatcher can be empty()
      intra_fixed_memory_store->clear_dest_stores();
      StallTimer is_intra_stalled(description + "-flush_updates-intra-store-empty");
      do {
        is_intra_stalled.check();
        assert(inter_fixed_memory_store->empty());  // should remain empty
        my_progress();
      } while (!intra_fixed_memory_store->empty());
      assert(inter_fixed_memory_store->empty());  // should still be empty
      track_intra_rpcs->flush(0);
    }  // implicit barrier from BarrierTimer timer2
    DBG(__func__, " last my_progress\n");
    my_progress();

    assert(inter_fixed_memory_store->empty());
    assert(intra_fixed_memory_store->empty());
    DBG("Done with flush_updates\n");

    // restore dest_stores for next round
    if (intra_fixed_memory_store->get_count_per_block() > 1) {
      intra_fixed_memory_store->set_dest_stores(split_rank::num_threads());
    }
    // restore dest_stores for next round
    if (inter_fixed_memory_store->get_count_per_block() > 1 && split_rank::num_nodes() > 1) {
      inter_fixed_memory_store->set_dest_stores(split_rank::num_nodes());
    }
    // barrier at exit from BarrierTimer
  }
};

};  // namespace upcxx_utils
