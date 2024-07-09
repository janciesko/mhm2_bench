#pragma once

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <sys/resource.h>
#include <sys/time.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <ctime>
#include <type_traits>
#include <upcxx/upcxx.hpp>

#include "colors.h"
#include "log.hpp"
#include "min_sum_max.hpp"
#include "version.h"

using upcxx::barrier;
using upcxx::dist_object;
using upcxx::intrank_t;
using upcxx::make_future;
using upcxx::op_fast_add;
using upcxx::op_fast_max;
using upcxx::op_fast_min;
using upcxx::progress;
using upcxx::promise;
using upcxx::rank_me;
using upcxx::rank_n;
using upcxx::reduce_all;
using upcxx::reduce_one;
using upcxx::team;
using upcxx::to_future;
using upcxx::when_all;
using upcxx::world;

using std::make_shared;
using std::ostringstream;
using std::shared_ptr;
using std::string;
using std::stringstream;
using timepoint_t = std::chrono::time_point<std::chrono::high_resolution_clock>;
using std::chrono::seconds;
using duration_seconds = std::chrono::duration<double>;

namespace upcxx_utils {

class PromiseReduce;
class Timings;
using ShTimings = shared_ptr<Timings>;

class Timings {
  std::chrono::time_point<std::chrono::high_resolution_clock> t;
  std::chrono::time_point<std::chrono::high_resolution_clock> before, after;
  static upcxx::future<> &get_last_pending();

 public:
  double before_elapsed, after_elapsed, reduction_elapsed;
  size_t my_count, my_instance;
  MinSumMax<double> before_msm, after_msm, count_msm, instance_msm;

  Timings();

  // a singleton which can be used throughout the program
  static PromiseReduce &get_promise_reduce();

  static PromiseReduce *get_promise_reduce(const upcxx::team &tm);  // returns nullptr if tm != world

  static bool pr_matches(const PromiseReduce *pr_ptr, const upcxx::team &tm);

  static upcxx::future<> get_pending();

  // return the previously pending future
  static upcxx::future<> set_pending(upcxx::future<> fut);

  static void wait_pending();

  string to_string(bool print_count = false, bool print_label = false) const;

  static void set_before(Timings &timings, size_t count, double elapsed, size_t instances = 0);

  // timings must remain in scope until the returned future is is_ready()
  static upcxx::future<> set_after(
      const upcxx::team &team, Timings &timings,
      std::chrono::time_point<std::chrono::high_resolution_clock> after = std::chrono::high_resolution_clock::now(),
      PromiseReduce *pr_ptr = nullptr);

  static upcxx::future<> set_after(
      Timings &timings,
      std::chrono::time_point<std::chrono::high_resolution_clock> after = std::chrono::high_resolution_clock::now(),
      PromiseReduce *pr_ptr = nullptr);

  static upcxx::future<> set_after(Timings &timings, PromiseReduce &pr);

  // barrier and reduction
  static Timings barrier(const upcxx::team &team, size_t count, double elapsed, size_t instances = 0,
                         PromiseReduce *pr_ptr = nullptr);
  static Timings barrier(PromiseReduce &pr, size_t count, double elapsed, size_t instances = 0);
  static Timings barrier(size_t count, double elapsed, size_t instances = 0, PromiseReduce *pr_ptr = nullptr);

  static void print_barrier_timings(const upcxx::team &team, string label, PromiseReduce *pr_ptr = nullptr);
  static void print_barrier_timings(PromiseReduce &pr, string label);
  static void print_barrier_timings(string label) { print_barrier_timings(upcxx::world(), label); }

  // no barrier but a future reduction is started
  static upcxx::future<ShTimings> reduce(const upcxx::team &team, size_t count, double elapsed, size_t instances = 0,
                                         PromiseReduce *pr_ptr = nullptr);
  static upcxx::future<ShTimings> reduce(PromiseReduce &pr, size_t count, double elapsed, size_t instances = 0);
  static upcxx::future<ShTimings> reduce(size_t count, double elapsed, size_t instances = 0, PromiseReduce *pr_ptr = nullptr);

  static void print_reduce_timings(const upcxx::team &team, string label, PromiseReduce *pr_ptr = nullptr);

  static void print_reduce_timings(string label, PromiseReduce *pr_ptr = nullptr);
};  // class Timings

class BaseTimer {
  // Just times between start & stop, does not print a thing
  // does not time construction / destruction

 private:
  static size_t &instance_count();

 protected:
  PromiseReduce *pr_ptr;
  timepoint_t t, first_t, last_t;
  double t_elapsed;
  size_t count;
  string name;

  static void increment_instance();
  static void decrement_instance();
  static size_t get_instance_count();

  bool pr_good() const;
  bool pr_good(const upcxx::team &tm) const;
  bool pr_matches(const upcxx::team &tm) const;

 public:
  BaseTimer(PromiseReduce *_pr_ptr = upcxx::initialized() ? &Timings::get_promise_reduce() : nullptr);
  BaseTimer(const string &_name, PromiseReduce *_pr_ptr = upcxx::initialized() ? &Timings::get_promise_reduce() : nullptr);
  BaseTimer(const BaseTimer &copy) = default;
  BaseTimer(BaseTimer &&move) = default;
  BaseTimer &operator=(const BaseTimer &copy) = default;
  BaseTimer &operator=(BaseTimer &&move) = default;

  BaseTimer &operator+=(const BaseTimer &other) {
    assert(other.t == timepoint_t() && "Other timer is stopped");
    t_elapsed += other.t_elapsed;
    count += other.count;
    return *this;
  }
  BaseTimer &operator+=(const double t) {
    t_elapsed += t;
    count++;
    return *this;
  }

  virtual ~BaseTimer();

  void clear();

  void start();

  void stop();

  double get_elapsed() const;

  double get_elapsed_since_start(timepoint_t x = now()) const;

  timepoint_t get_first() const { return first_t; }

  timepoint_t get_last() const { return last_t; }

  size_t get_count() const;

  const string &get_name() const;

  void done() const;
  upcxx::future<MinSumMax<double>> done_all_async(const upcxx::team &tm = upcxx::world()) const;
  void done_all(const upcxx::team &tm = upcxx::world()) const;

  string get_final() const;
  // TODO option to ignore 0 count contributions to reduction

  static upcxx::future<ShTimings> reduce_timings(const upcxx::team &team, size_t my_count, double my_elapsed,
                                                 size_t my_instances = 0, PromiseReduce *pr_ptr = nullptr);

  static upcxx::future<ShTimings> reduce_timings(size_t my_count, double my_elapsed, size_t my_instances = 0,
                                                 PromiseReduce *pr_ptr = nullptr);
  upcxx::future<ShTimings> reduce_timings(const upcxx::team &team, size_t my_instances = 0) const;
  upcxx::future<ShTimings> reduce_timings(size_t my_instances = 0) const;

  static Timings barrier_timings(const upcxx::team &team, size_t my_count, double my_elapsed, size_t my_instances = 0,
                                 PromiseReduce *pr_ptr = nullptr);

  static Timings barrier_timings(size_t my_count, double my_elapsed, size_t my_instances = 0, PromiseReduce *pr_ptr = nullptr);

  Timings barrier_timings(const upcxx::team &team, size_t my_instances = 0) const;
  Timings barrier_timings(size_t my_instances = 0) const;

  static upcxx::future<MinSumMax<double>> reduce_timepoint(const upcxx::team &team = upcxx::world(), timepoint_t my_now = now(),
                                                           PromiseReduce *pr_ptr = nullptr);

  // member functionmust be called after start() and before stop()
  upcxx::future<MinSumMax<double>> reduce_start(const upcxx::team &team = upcxx::world()) {
    return reduce_timepoint(team, t, pr_matches(team) ? pr_ptr : nullptr);
  }

  static timepoint_t now();

  static string now_str();

  static upcxx::future<MinSumMax<double>> reduce_now(PromiseReduce &pr);
  static upcxx::future<MinSumMax<double>> reduce_now(const upcxx::team &team = upcxx::world()) {
    return reduce_timepoint(team, now());
  }

  // returns the indent nesting depending on how many nested BaseTimers are active
  static string get_indent(int indent = -1);
};  // class BaseTimer

class StallTimer : public BaseTimer {
  // prints a Warning if called too many times or for too long. use in a while loop that could be indefinite
  double warn_seconds, max_seconds;
  int64_t warn_count, max_count;

 public:
  StallTimer(const string _name, double _warn_seconds = 16.0, int64_t _warn_count = -1, double _max_seconds = 130.0,
             int64_t _max_count = -1, PromiseReduce *_pr_ptr = upcxx::initialized() ? &Timings::get_promise_reduce() : nullptr);
  virtual ~StallTimer();
  void check();
  void reset(string append_description);
};  // class StallTimer

struct Rusage : rusage {
  int who;
  Rusage(int who = RUSAGE_SELF);  // or RUSAGE_CHILDREN or RUSAGE_THREAD
  double get_utime() const;
  double get_stime() const;
  double get_cpu_time() const;
  double get_utime(const Rusage &other) const;
  double get_stime(const Rusage &other) const;
  double get_cpu_time(const Rusage &other) const;
  string get_rusage_str(double elapsed = 0.0) const;
};

class RusageTimer : public BaseTimer {
 public:
  Rusage rusage;
  RusageTimer(int who = RUSAGE_SELF, PromiseReduce *_pr_ptr = upcxx::initialized() ? &Timings::get_promise_reduce() : nullptr);
  RusageTimer(const string &_name, int who = RUSAGE_SELF,
              PromiseReduce *_pr_ptr = upcxx::initialized() ? &Timings::get_promise_reduce() : nullptr);
  RusageTimer(const RusageTimer &copy) = default;
  RusageTimer(RusageTimer &&move) = default;
  RusageTimer &operator=(const RusageTimer &copy) = default;
  RusageTimer &operator=(RusageTimer &&move) = default;
  string get_rusage_str() const;
};

class IntermittentTimer : public RusageTimer {
  // prints a summary on destruction
  double t_interval;
  string interval_label;
  void start_interval();
  void stop_interval();

 public:
  IntermittentTimer(const string name, string interval_label = "", int who = RUSAGE_SELF,
                    PromiseReduce *_pr_ptr = upcxx::initialized() ? &Timings::get_promise_reduce() : nullptr);

  virtual ~IntermittentTimer();

  void start() {
    BaseTimer::start();
    start_interval();
  }

  void stop() {
    stop_interval();
    BaseTimer::stop();
  }

  inline double get_interval() const { return t_interval; }

  void clear();

  inline void inc_elapsed(double secs) { t_elapsed += secs; }

  void print_out(const upcxx::team &tm = upcxx::world());
};  // class IntermittentTimer

class ProgressTimer : public BaseTimer {
 private:
  size_t calls;

 public:
  ProgressTimer(const string &name, PromiseReduce *_pr_ptr = upcxx::initialized() ? &Timings::get_promise_reduce() : nullptr);

  virtual ~ProgressTimer();

  void clear();

  void progress(size_t run_every = 1);

  void discharge(size_t run_every = 1);

  template <typename Future>
  void wait_for_ready(Future &fut) {
    while (!fut.is_ready()) this->progress();
  }

  void print_out(const upcxx::team &tm = upcxx::world());
};  // class ProgressTimer

class Timer : public RusageTimer {
  // times between construction and destruction
  // reduced load balance calcs on destruction
  const upcxx::team &tm;
  bool exited;
  bool logged;
  void init();

 public:
  Timer(PromiseReduce &pr, const string &name, bool exit_reduction = true);
  Timer(const upcxx::team &tm, const string &name, bool exit_reduction = true);
  Timer(const string &name, bool exit_reduction = true);
  Timer(const Timer &copy) = delete;
  Timer(Timer &&move);
  Timer &operator=(const Timer &copy) = delete;
  Timer &operator=(Timer &&move);

  upcxx::future<> initiate_entrance_reduction();
  upcxx::future<> initiate_exit_reduction();
  virtual ~Timer();
};

class BarrierTimer : public RusageTimer {
  // barrier AND reduced load balance calcs on destruction
  const upcxx::team &_team;
  bool exit_barrier, exited;
  Rusage start_rusage;

 public:
  BarrierTimer(PromiseReduce &pr, const string name, bool entrance_barrier = true, bool exit_barrier = true);
  BarrierTimer(const upcxx::team &team, const string name, bool entrance_barrier = true, bool exit_barrier = true);
  BarrierTimer(const string name, bool entrance_barrier = true, bool exit_barrier = true);
  upcxx::future<> initiate_exit_barrier();
  virtual ~BarrierTimer();

 protected:
  upcxx::future<> init(bool entrance_barrier);
};  // class Timer

class _AsyncTimer : public BaseTimer {
 public:
  _AsyncTimer(PromiseReduce &pr, const string &name);
  _AsyncTimer(const upcxx::team &tm, const string &name);
  _AsyncTimer(const _AsyncTimer &copy) = delete;
  _AsyncTimer(_AsyncTimer &&move) = delete;
  void start();
  void stop();
  void report(const string label, MinSumMax<double> msm, bool verbose = true);
  upcxx::future<> initiate_construct_reduction(bool verbose = false);
  upcxx::future<> initiate_start_reduction(bool verbose = false);
  upcxx::future<> initiate_stop_reduction(bool verbose = false);
  const upcxx::team &tm;
  timepoint_t construct_t;
  timepoint_t start_t;
};  // class _AsyncTimer

class AsyncTimer {
  // meant to be lambda captured and/or stopped within progress callbacks and returned as a future
  // times the delay between construction and start() and the time between start() and stop()
  // optionally, after stop(), can reduce the construction delay, start delay and start to stop duration across the team
  shared_ptr<_AsyncTimer> timer;

 public:
  AsyncTimer(PromiseReduce &pr, const string &name);
  AsyncTimer(const upcxx::team &tm, const string &name);
  AsyncTimer(const string &name);
  void start() const;
  void stop() const;
  double get_elapsed() const;
  // these methods may not be called within a progress() callback!
  upcxx::future<> initiate_construct_reduction(bool verbose = false);
  upcxx::future<> initiate_start_reduction(bool verbose = false);
  upcxx::future<> initiate_stop_reduction(bool verbose = false);
  operator const _AsyncTimer &() const { return *timer; }
};  // class AsyncTimer

class ActiveCountTimer {
 protected:
  double total_elapsed;
  size_t total_count;
  size_t active_count;
  size_t max_active;
  string name;
  upcxx::future<> my_fut;

 public:
  ActiveCountTimer(const string _name = "");
  ~ActiveCountTimer();

  void clear();

  timepoint_t begin();

  void end(timepoint_t t);

  inline double get_total_elapsed() const { return total_elapsed; }
  inline size_t get_total_count() const { return total_count; }
  inline size_t get_active_count() const { return active_count; }
  inline size_t get_max_active_count() const { return max_active; }

  void print_barrier_timings(const upcxx::team &team, string label = "");
  void print_barrier_timings(string label = "") { print_barrier_timings(upcxx::world(), label); }

  void print_reduce_timings(const upcxx::team &team, string label = "");
  void print_reduce_timings(string label = "") { print_reduce_timings(upcxx::world(), label); }

  void print_timings(Timings &timings, string label = "");
};  // class ActiveCountTimer

template <typename Base>
class ActiveInstantiationTimerBase : public Base {
 protected:
  ActiveCountTimer &_act;
  timepoint_t _t;

 public:
  ActiveInstantiationTimerBase(ActiveCountTimer &act)
      : Base()
      , _act(act)
      , _t() {
    _t = act.begin();
  }

  ActiveInstantiationTimerBase(ActiveInstantiationTimerBase &&move)
      : Base(std::move(static_cast<Base &>(move)))
      , _act(move._act)
      , _t(move._t) {
    if (this != &move) move._t = {};  // delete timer for moved instance
  }

  virtual ~ActiveInstantiationTimerBase() {
    if (_t != timepoint_t()) _act.end(_t);
  }

  double get_elapsed_since_start() const {
    duration_seconds interval = BaseTimer::now() - this->_t;
    return interval.count();
  }

  // move but not copy
  ActiveInstantiationTimerBase(const ActiveInstantiationTimerBase &copy) = delete;
};  // template class ActiveInstantiationTimerBase

// to be used in inheritence to time all the instances of a class (like the duration of promises)
// to be used with an external ActiveContTimer
template <typename Base>
class ActiveInstantiationTimer : public ActiveInstantiationTimerBase<Base> {
 public:
  ActiveInstantiationTimer(ActiveCountTimer &act)
      : ActiveInstantiationTimerBase<Base>(act) {}
  ActiveInstantiationTimer(const ActiveInstantiationTimer &copy) = delete;
  ActiveInstantiationTimer(ActiveInstantiationTimer &&move)
      : ActiveInstantiationTimerBase<Base>(std::move(static_cast<ActiveInstantiationTimerBase<Base> &>(move))) {}
  virtual ~ActiveInstantiationTimer() {}

  void print_barrier_timings(string label = "") { this->_act.print_barrier_timings(label); }
  void print_barrier_timings(const upcxx::team &team, string label = "") { this->_act.print_barrier_timings(team, label); }
  void print_reduce_timings(string label = "") { this->_act.print_reduce_timings(label); }
  void print_reduce_timings(const upcxx::team &team, string label = "") { this->_act.print_reduce_timings(team, label); }
  void print_timings(Timings &timings, string label = "") { this->_act.print_timings(timings, label); }
  double get_total_elapsed() const { return this->_act.get_total_elapsed(); }
  size_t get_total_count() const { return this->_act.get_total_count(); }
  size_t get_active_count() const { return this->_act.get_active_count(); }
  size_t get_max_active_count() const { return this->_act.get_max_active_count(); }
  void clear() { return this->_act.clear(); }
};  // template class ActiveInstantiationTimer

// to be used in inheritence to time all the instances of a class (like the duration of promises)
// hold a static (by templated-class) specific ActiveCountTimer an external ActiveContTimer
// e.g. template <A,...> class my_timed_class : public my_class<A,...>, public InstantiationTimer<A,...> {};
// then when all instances have been destryed, call my_timed_class::print_barrier_timings();
template <typename Base, typename... DistinguishingArgs>
class InstantiationTimer : public ActiveInstantiationTimerBase<Base> {
 protected:
  static ActiveCountTimer &get_ACT() {
    static ActiveCountTimer _act = ActiveCountTimer();
    return _act;
  }

 public:
  InstantiationTimer()
      : ActiveInstantiationTimerBase<Base>(get_ACT()) {}
  // move but not copy this timer
  InstantiationTimer(const InstantiationTimer &copy) = delete;
  InstantiationTimer(InstantiationTimer &&move)
      : ActiveInstantiationTimerBase<Base>(std::move(static_cast<ActiveInstantiationTimerBase<Base> &>(move))) {}

  virtual ~InstantiationTimer() {}

  static void print_barrier_timings(string label = "") { get_ACT().print_barrier_timings(label); }
  static void print_barrier_timings(const upcxx::team &team, string label = "") { get_ACT().print_barrier_timings(team, label); }
  static void print_reduce_timings(string label) { get_ACT().print_reduce_timings(label); }
  static void print_reduce_timings(const upcxx::team &team, string label) { get_ACT().print_reduce_timings(team, label); }
  static void print_timings(Timings &timings, string label = "") { get_ACT().print_timings(timings, label); }
  static size_t get_total_count() { return get_ACT().get_total_count(); }
  static size_t get_active_count() { return get_ACT().get_active_count(); }
  static void clear() { get_ACT().clear(); }
};  // template class InstantiationTimer

//
// speed up compile with standard implementations of the Instantiation Timers
//

struct _upcxx_utils_dummy {};

typedef ActiveInstantiationTimer<_upcxx_utils_dummy> GenericInstantiationTimer;

typedef InstantiationTimer<_upcxx_utils_dummy> SingletonInstantiationTimer;

#ifndef _TIMERS_CPP

// use extern templates (implemented in timers.cpp) to speed up compile
extern template class ActiveInstantiationTimer<_upcxx_utils_dummy>;
extern template class InstantiationTimer<_upcxx_utils_dummy>;

#endif

};  // namespace upcxx_utils
