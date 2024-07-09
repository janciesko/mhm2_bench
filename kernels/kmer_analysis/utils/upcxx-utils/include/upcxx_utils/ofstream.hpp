#pragma once

/*
 * File:   ofstream.hpp
 * Author: regan
 *
 * Created on June 19, 2020, 10:06 AM
 */

#include <cassert>
#include <exception>
#include <fstream>
#include <ios>
#include <memory>
#include <sstream>
#include <string>
#include <upcxx/upcxx.hpp>

#include "upcxx_utils/log.hpp"
#include "upcxx_utils/promise_collectives.hpp"
#include "upcxx_utils/thread_pool.hpp"
#include "upcxx_utils/timers.hpp"
#include "upcxx_utils/version.h"

using std::basic_string;
using std::ofstream;
using std::ostream;
using std::shared_ptr;
using std::string;
using std::stringstream;

using upcxx::atomic_domain;
using upcxx::dist_object;
using upcxx::global_ptr;
using upcxx::world;

namespace upcxx_utils {

#ifndef UPCXX_UTILS_FILE_BLOCK_SIZE
#define UPCXX_UTILS_FILE_BLOCK_SIZE (16ULL * 1024ULL * 1024ULL)  // 16MB
#endif

class dist_ofstream_handle {
 public:
  using Byte = string::value_type;
  using Buf = vector<Byte>;
  using Prom = upcxx::promise<>;
  using ShSS = shared_ptr<stringstream>;

  struct OptimizedBlockWrite {
    OptimizedBlockWrite(ShSS sh_ss, uint64_t offset, uint64_t block_size, uint64_t receive_bytes, uint64_t send_first_bytes,
                        const upcxx::team &tm)
        : sh_ss(sh_ss)
        , receive_buf(receive_bytes)
        , receive_prom(receive_bytes + 1)
        , all_done_barrier(tm)
        , offset(offset)
        , block_size(block_size)
        , send_first_bytes(send_first_bytes)
        , my_size(sh_ss->tellp() - sh_ss->tellg()) {}
    ShSS sh_ss;
    Buf receive_buf;
    Prom receive_prom;
    PromiseBarrier all_done_barrier;
    uint64_t offset, block_size, send_first_bytes, my_size;
    void receive_bytes(uint64_t bytes) {
      assert(receive_buf.empty());
      receive_buf.resize(bytes);
      receive_prom.require_anonymous(bytes);
    }
    void clear_receive_buf() { Buf().swap(receive_buf); }
    auto receive_bytes() const { return receive_buf.size(); }
    bool will_receive() const { return receive_bytes() > 0; }
    bool will_send() const { return send_first_bytes > 0; }
    auto last_byte() const { return offset + my_size; }
    auto start_block() const { return offset / block_size; }
    auto end_block() const { return last_byte() / block_size; }
    auto start_block_offset() const { return offset % block_size; }
    auto end_block_offset() const { return last_byte() % block_size; }
  };
  using ShOptimizedBlockWrite = shared_ptr<OptimizedBlockWrite>;

  struct OffsetSizeBuffer {
    uint64_t offset, size;
    ShOptimizedBlockWrite sh_obw;
    UPCXX_SERIALIZED_FIELDS(offset, size);

    OffsetSizeBuffer();
    OffsetSizeBuffer(uint64_t offset, uint64_t size, ShOptimizedBlockWrite sh_obw = nullptr);

    // comparison operator dist_obj to pos
    int operator()(const dist_object<OffsetSizeBuffer> &dist_osb, const uint64_t &pos) const;

    // helpers for debugging
    string to_string() const;
    operator string() const;
  };
  using DistOffsetSizeBuffer = dist_object<OffsetSizeBuffer>;
  using ShDistOffsetSizeBuffer = shared_ptr<DistOffsetSizeBuffer>;

  struct OffsetPrefix {
    uint64_t start, size;
  };
  struct OffsetPrefixes {
    OffsetPrefix global, my;
  };

  using AD = atomic_domain<uint64_t>;
  using ShAD = shared_ptr<AD>;

  struct dist_ofstream_handle_state {
    // This is necessary for concurrent calls to dist_ofstream -- never clean up AD
    static std::map<upcxx::team_id, ShAD> &get_ad_map();
    static void clear_ad_map();
    static AD &get_ad(const upcxx::team &team);

    dist_ofstream_handle_state(const string fname, const upcxx::team &myteam);
    ~dist_ofstream_handle_state();
    bool is_open() const { return fd > 0; }
    const upcxx::team &myteam;
    AD &ad;
    global_ptr<uint64_t> global_offset;
    uint64_t count_async, count_collective, count_bytes, wrote_bytes;
    IntermittentTimer io_t, network_latency_t;
    std::chrono::time_point<std::chrono::high_resolution_clock> open_time, open_complete_time;
    vector<MinSumMax<uint64_t> > msm_metrics;
    upcxx::future<> opening_ops, pending_io_ops, pending_net_ops;
    const string fname;
    uint64_t last_known_tellp;
    PromiseBarrier close_barrier;      // to order operations during close
    PromiseBarrier tear_down_barrier;  // to ensure all ranks recognize the final rename
#ifndef UPCXX_UTILS_IO_NO_THREAD
    upcxx_utils::ThreadPool serial_tp;
#endif
    int fd;
  };
  using ShState = shared_ptr<dist_ofstream_handle_state>;

  static vector<upcxx::atomic_op> &ad_ops() {
    static vector<upcxx::atomic_op> _ = {upcxx::atomic_op::fetch_add, upcxx::atomic_op::load, upcxx::atomic_op::store};
    return _;
  }

 protected:
  ShState sh_state;  // all of the actual data
  // helper references
  int &fd;
  const string &fname;
  const upcxx::team &myteam;
  AD &ad;
  global_ptr<uint64_t> &global_offset;  // for global file offset (i.e. async appending)
  uint64_t &count_async, &count_collective, &count_bytes, &wrote_bytes;
  IntermittentTimer &io_t, &network_latency_t;
  upcxx::future<> &opening_ops, &pending_io_ops, &pending_net_ops;
  bool is_closed;

  // helper method
  static void read_all(ShSS sh_ss, char *buf, uint64_t len);
  static OffsetPrefixes getOffsetPrefixes(ShState sh_state, uint64_t my_size);

  static ShDistOffsetSizeBuffer write_blocked_batch_collective_start(ShState sh_state, OffsetPrefixes offset_prefixes,
                                                                     uint64_t block_size, ShSS sh_ss);
  static upcxx::future<> write_blocked_batch_collective_finish(ShState sh_state, ShDistOffsetSizeBuffer sh_dist_osb);

  // returns the new file  position after writing
  static uint64_t write_block(ShState sh_state, const char *src, uint64_t len, uint64_t file_offset);
  static upcxx::future<uint64_t> write_block(ShState sh_state, ShSS sh_ss, uint64_t file_offset);

  static void open_file_sync(ShState sh_state, bool append = false);
  static upcxx::future<> open_file(ShState sh_state, bool append = false);

  static upcxx::future<bool> is_open_async(ShState sh_state);
  static bool is_open(ShState sh_state);

  static upcxx::future<> report_timings(ShState sh_state);

  static double close_file_sync(ShState sh_state);
  static void tear_down(ShState sh_state, double file_op_duration, double io_time, double net_time, double open_time,
                        bool did_open);

  static upcxx::future<> get_pending_ops(ShState sh_state);

 public:
  dist_ofstream_handle(const string fname, const upcxx::team &myteam, bool append = false);
  // collective

  ~dist_ofstream_handle();

  upcxx::future<> get_pending_ops() const;
  // void append_future_op(upcxx::future<> fut);

  // opens the file, if not already open
  upcxx::future<> open_file(bool append = false);

  upcxx::future<bool> is_open_async() const;
  bool is_open() const;

  string get_file_name() const;
  uint64_t get_last_known_tellp() const;

  // closes the file if not already closed
  upcxx::future<> close_file();
  upcxx::future<> report_timings() const;

  // append this batch to the file
  // contents may be out of rank order
  // atomically gets offset -- may lead to network imbalance
  // returns the new file position after writing
  // consumes and resets the shared_ptr<stringstream>
  upcxx::future<uint64_t> append_batch_async(ShSS sh_ss);

  // append this batch collectively
  // writes are ordered by rank
  // returns the new file position after writing
  // consumes and resets the shared_ptr<stringstream>
  upcxx::future<uint64_t> append_batch_collective(ShSS sh_ss, uint64_t block_size = 0);

  class _dist_ofstream_report {
   public:
    using DistOFSHandle = dist_object<dist_ofstream_handle>;
    using ShDistOFSHandle = shared_ptr<DistOFSHandle>;
    using FutShDistOFSHandle = upcxx::future<ShDistOFSHandle>;
    using ShFutShDistOFSHandle = shared_ptr<FutShDistOFSHandle>;
    _dist_ofstream_report()
        : sh_fut{} {
      DBG("Constructed empty ", (void *)this, " sh_fut=", sh_fut, "\n");
    };
    _dist_ofstream_report(FutShDistOFSHandle fut_handle)
        : sh_fut{} {
      sh_fut = make_shared<FutShDistOFSHandle>(fut_handle);
      DBG("Constructed ", (void *)this, " sh_fut=", sh_fut, "\n");
    }
    _dist_ofstream_report(const _dist_ofstream_report &copy) = delete;
    _dist_ofstream_report(_dist_ofstream_report &&move)
        : sh_fut(std::move(move.sh_fut)) {
      move.sh_fut.reset();
      DBG("Constructed move from=", (void *)&move, " ", (void *)this, " sh_fut=", sh_fut, "\n");
    }
    _dist_ofstream_report &operator=(_dist_ofstream_report &&move) {
      if (sh_fut) wait();
      sh_fut = move.sh_fut;
      move.sh_fut.reset();
      DBG("Move assigned from=", (void *)&move, " ", (void *)this, " sh_fut=", sh_fut, "\n");
      return *this;
    }
    void wait() {
      if (!sh_fut) return;
      assert(!upcxx::in_progress() && "Not called within the restricted context");
      assert(upcxx::master_persona().active_with_caller() && "Called from master persona while upcxx is still active");

      // wait and unpack OFS handle and state
      ShDistOFSHandle shdofsh = sh_fut->wait();
      sh_fut.reset();
      dist_ofstream_handle::report_timings((*shdofsh)->sh_state).wait();
    }
    virtual ~_dist_ofstream_report() {
      if (upcxx::initialized()) {
        DBG("Destroying sh_fut=", sh_fut, " ", (void *)this, "\n");
        wait();
      }
      assert(!sh_fut);
    }

   protected:
    ShFutShDistOFSHandle sh_fut;
  };
};
using dist_ofstream_report = dist_ofstream_handle::_dist_ofstream_report;

using std::streampos;
using std::streamsize;
class ProtectedOSS : public std::stringstream {
  // a full stringstream with a public ostringstream and protected istringstream
 protected:
  const std::stringstream &parent() const { return *((std::stringstream *)this); }
  std::stringstream &parent() { return *((std::stringstream *)this); }
  template <typename T>
  ProtectedOSS &operator>>(T &val) {
    parent().operator>>(val);
    return *this;
  };
  auto get() { return parent().get(); }
  ProtectedOSS &get(char &ch) {
    parent().get(ch);
    return *this;
  }
  auto peek() { return parent().peek(); }
  ProtectedOSS &unget() {
    parent().unget();
    return *this;
  }
  ProtectedOSS &putback(char ch) {
    parent().putback(ch);
    return *this;
  }
  ProtectedOSS &ignore(streamsize count = 1) {
    parent().ignore(count);
    return *this;
  }
  ProtectedOSS &read(char *s, streamsize count) {
    parent().read(s, count);
    return *this;
  }
  streamsize readsome(char *s, streamsize count) { return parent().readsome(s, count); }
  streamsize gcount() const { return parent().gcount(); }
  ProtectedOSS &getline(char *s, streamsize count) {
    parent().getline(s, count);
    return *this;
  }
  auto tellg() { return parent().tellg(); }
  ProtectedOSS &seekg(streampos pos) {
    parent().seekg(pos);
    return *this;
  }
  ProtectedOSS &seekg(streampos off, std::ios_base::seekdir dir) {
    parent().seekg(off, dir);
    return *this;
  }
};

class dist_ofstream : public ProtectedOSS {
  std::stringstream &ss;
  static vector<upcxx::future<> > all_files;  // FIXME needed for cleanup of AD
  using DistOFSHandle = dist_object<dist_ofstream_handle>;
  using ShDistOFSHandle = shared_ptr<DistOFSHandle>;
  ShDistOFSHandle sh_ofsh;
  DistOFSHandle &ofsh;
  uint64_t block_size;
  uint64_t bytes_written;
  upcxx::future<> close_fut;
  bool is_closed;

 protected:
  // collective
  upcxx::future<> flush_batch(bool async);

 public:
  static void sync_all_files();  // FIXME needed for cleanup of AD

  // collective / blocking
  dist_ofstream(const string ofname, bool append = false, uint64_t block_size = UPCXX_UTILS_FILE_BLOCK_SIZE);
  dist_ofstream(const upcxx::team &myteam, const string ofname, bool append = false,
                uint64_t block_size = UPCXX_UTILS_FILE_BLOCK_SIZE);

  // collective / blocking
  ~dist_ofstream();

  // public ostream methods
  template <typename Obj>
  dist_ofstream &operator<<(Obj o) {
    ss << o;
    return *this;
  };
  dist_ofstream &put(char c) {
    ss.put(c);
    return *this;
  }
  dist_ofstream &write(const char *s, std::streamsize n) {
    ss.write(s, n);
    return *this;
  }
  // overridden ostream methods
  std::streampos tellp();
  dist_ofstream &seekp(std::streampos pos);
  dist_ofstream &seekp(std::streamoff off, std::ios_base::seekdir way);

  // blocking & calls report_timings
  void close();

  // (somewhat) non-blocking
  upcxx::future<> close_async();

  // non-blocking.
  dist_ofstream_report close_and_report_timings(upcxx::future<> = make_future());
  upcxx::future<> report_timings();  // DEPRECATED

  // returns the current buffered size
  uint64_t size();

  // returns a copy of the current buffer (not anything already flushed)
  string str() const;

  // the last pos known to this rank
  uint64_t get_last_known_tellp() const;

  // async not blocking but with communication to rank0
  // write to filesystem by this rank
  upcxx::future<> flush_async();

  // blocking with communication to rank0
  // write to filesystem by this rank
  dist_ofstream &flush();

  // collective not blocking with communication between some ranks
  // optimal UPCXX_UTILS_FILE_BLOCK_SIZE sized writes from a subset of ranks
  upcxx::future<> flush_collective();
};

};  // namespace upcxx_utils
