#include <cassert>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <upcxx/upcxx.hpp>
#include <vector>

#define _TIMERS_CPP
#include "upcxx_utils/log.hpp"
#include "upcxx_utils/promise_collectives.hpp"
#include "upcxx_utils/timers.hpp"

using upcxx::future;

namespace upcxx_utils {

// Reduce compile time by making templates instantiations of common types
// these are each constructed in CMakeLists.txt and timers-extern-template.in.cpp
// extern templates declarations all happen in timers.hpp

/*
 * This is now handled by CMakeLists.txt
 *
   MACRO_MIN_SUM_MAX(float,    template);
   MACRO_MIN_SUM_MAX(double,   template);
   MACRO_MIN_SUM_MAX(int64_t,  template);
   MACRO_MIN_SUM_MAX(uint64_t, template);
   MACRO_MIN_SUM_MAX(int,      template);

 */

//
// Timings
//

PromiseReduce &Timings::get_promise_reduce() {
  static PromiseReduce _(world());
  return _;
}

PromiseReduce *Timings::get_promise_reduce(const upcxx::team &tm) {
  PromiseReduce &pr = get_promise_reduce();
  return pr_matches(&pr, tm) ? &pr : nullptr;
}

future<> &Timings::get_last_pending() {
  static future<> _ = make_future();
  return _;
}

Timings::Timings()
    : t()
    , before_elapsed(0.0)
    , after_elapsed(0.0)
    , reduction_elapsed(0.0)
    , my_count(0)
    , my_instance(0) {}

bool Timings::pr_matches(const PromiseReduce *pr_ptr, const upcxx::team &tm) {
  // either no PromiseReduce or it matches the team
  return pr_ptr == nullptr || pr_ptr->get_team().id() == tm.id();
}

future<> Timings::get_pending() { return get_last_pending(); }

future<> Timings::set_pending(future<> fut) {
  DBG_VERBOSE(__func__, "\n");
  future<> old = get_last_pending();
  get_last_pending() = when_all(get_last_pending(), fut);
  return old;
}

void Timings::wait_pending() {
  assert(!upcxx::in_progress());
  DBG(__func__, "\n");
  if (upcxx::initialized()) {
    assert(!upcxx::in_progress());
    DBG("Fulfilling pending PromiseReduce operations\n");
    auto fut_pr = get_promise_reduce().fulfill();
    DBG("Waiting for pending\n");
    auto fut = when_all(fut_pr, get_last_pending());
    upcxx::progress();
    wait_wrapper(fut);
    assert(get_last_pending().is_ready());
    get_last_pending() = make_future();
    DBG("Fully waited\n");
  }
}

string Timings::to_string(bool print_count, bool print_label) const {
  ostringstream os;
  if (print_label) os << "(min/my/avg/max, bal) ";
  os << std::setprecision(2) << std::fixed;
  // print the timing metrics
  auto &before_max = before_msm.max;
  auto &before_min = before_msm.min;
  auto &before_sum = before_msm.sum;
  if (before_max > 0.0) {
    double bal = (before_max > 0.0 ? before_sum / rank_n() / before_max : 1.0);
    if (before_max > 10.0 && bal < .9) os << KLRED;  // highlight large imbalances
    os << before_min << "/" << before_elapsed << "/" << before_sum / rank_n() << "/" << before_max << " s, " << bal;
    if (before_max > 1.0 && bal < .9) os << KLCYAN;
  } else {
    os << "0/0/0/0 s, 1.00";
  }

  os << std::setprecision(1) << std::fixed;

  auto &after_max = after_msm.max;
  auto &after_min = after_msm.min;
  auto &after_sum = after_msm.sum;
  // print the timings around a barrier if they are significant
  if (after_max >= 0.1) {
    os << (after_max > 1.0 ? KLRED : "") << " barrier " << after_min << "/" << after_elapsed << "/" << after_sum / rank_n() << "/"
       << after_max << " s, " << (after_max > 0.0 ? after_sum / rank_n() / after_max : 0.0) << (after_max > 1.0 ? KLCYAN : "");
  } else if (after_max > 0.0) {
    os << std::setprecision(2) << std::fixed;
    os << " barrier " << after_max << " s";
    os << std::setprecision(1) << std::fixed;
  }

  auto &count_max = count_msm.max;
  auto &count_min = count_msm.min;
  auto &count_sum = count_msm.sum;
  // print the max_count if it is more than 1 or more than 0 if asked to print the count
  if (count_max > (print_count ? 0.0 : 1.00001))
    os << " count " << count_min << "/" << my_count << "/" << count_sum / rank_n() << "/" << count_max << ", "
       << (count_max > 0.0 ? count_sum / rank_n() / count_max : 0.0);

  auto &instance_max = instance_msm.max;
  auto &instance_min = instance_msm.min;
  auto &instance_sum = instance_msm.sum;
  // print the instances if it is both non-zero and not 1 per rank
  if (instance_sum > 0 && ((int)(instance_sum + 0.01)) != rank_n() && ((int)(instance_sum + 0.99)) != rank_n())
    os << " inst " << instance_min << "/" << my_instance << "/" << instance_sum / rank_n() << "/" << instance_max << ", "
       << (instance_max > 0.0 ? instance_sum / rank_n() / instance_max : 0.0);
  // print the reduction timings if they are significant
  if (reduction_elapsed > 0.05)
    os << (reduction_elapsed > .5 ? KLRED : "") << " reduct " << reduction_elapsed << (reduction_elapsed > .5 ? KLCYAN : "");
  return os.str();
}

void Timings::set_before(Timings &timings, size_t count, double elapsed, size_t instances) {
  DBG_VERBOSE("set_before: my_count=", count, " my_elapsed=", elapsed, " instances=", instances, "\n");
  timings.before = std::chrono::high_resolution_clock::now();

  timings.my_count = count;
  timings.count_msm.reset(timings.my_count);

  timings.before_elapsed = elapsed;
  timings.before_msm.reset(elapsed);

  timings.my_instance = instances;
  timings.instance_msm.reset(instances);
}

// timings must remain in scope until the returened future is is_ready()
future<> Timings::set_after(const upcxx::team &team, Timings &timings,
                            std::chrono::time_point<std::chrono::high_resolution_clock> t_after, PromiseReduce *pr_ptr) {
  assert(!upcxx::in_progress());
  assert(pr_matches(pr_ptr, team) && "No PromiseReduce or team matches");
  timings.after = t_after;
  duration_seconds interval = timings.after - timings.before;
  timings.after_elapsed = interval.count();
  timings.after_msm.reset(timings.after_elapsed);
  DBG_VERBOSE("set_after: ", interval.count(), "\n");

  // time the reductions
  timings.t = t_after;

  assert(&timings.instance_msm == &timings.before_msm + 3);  // memory is in order
  auto fut_msms = make_future();
  if (pr_ptr) {
    auto &before_msm = timings.before_msm;
    auto &after_msm = timings.after_msm;
    auto &count_msm = timings.count_msm;
    auto &instance_msm = timings.instance_msm;
    auto fut_before = pr_ptr->msm_reduce_all(before_msm.my).then([&out = before_msm](const MinSumMax<double> &msm) { out = msm; });
    auto fut_after = pr_ptr->msm_reduce_all(after_msm.my).then([&out = after_msm](const MinSumMax<double> &msm) { out = msm; });
    auto fut_count = pr_ptr->msm_reduce_all(count_msm.my).then([&out = count_msm](const MinSumMax<double> &msm) { out = msm; });
    auto fut_instance =
        pr_ptr->msm_reduce_all(instance_msm.my).then([&out = instance_msm](const MinSumMax<double> &msm) { out = msm; });
    fut_msms = when_all(fut_before, fut_after, fut_count, fut_instance);
  } else {
    DBG("NO PRF for set_after\n");
    fut_msms = min_sum_max_reduce_all(&timings.before_msm, &timings.before_msm, 4, team);
  }
  auto ret = fut_msms.then([&timings]() {
    duration_seconds interval = std::chrono::high_resolution_clock::now() - timings.t;
    timings.reduction_elapsed = interval.count();
  });

  set_pending(ret);
  return ret;
}

upcxx::future<> Timings::set_after(Timings &timings, std::chrono::time_point<std::chrono::high_resolution_clock> after,
                                   PromiseReduce *pr_ptr) {
  return set_after(pr_ptr ? pr_ptr->get_team() : upcxx::world(), timings, after, pr_ptr);
}

upcxx::future<> Timings::set_after(Timings &timings, PromiseReduce &pr) {
  // INFO("set_after pr_ptr=", &pr, "\n");
  return set_after(pr.get_team(), timings, std::chrono::high_resolution_clock::now(), &pr);
}

// barrier and reduction
Timings Timings::barrier(const upcxx::team &team, size_t count, double elapsed, size_t instances, PromiseReduce *pr_ptr) {
  assert(!upcxx::in_progress());
  assert(pr_matches(pr_ptr, team) && "No PromiseReduce or teams match");
  Timings timings;
  set_before(timings, count, elapsed, instances);
  upcxx::progress();
  barrier_wrapper(team, false);  // thread friendly
  if (!pr_ptr) DBG("NO PRF for barrier\n");
  auto fut = pr_ptr ? set_after(timings, *pr_ptr) : set_after(team, timings);
  wait_pending();
  wait_wrapper(fut);  // wait for the reductions to complete also
  return timings;
}

Timings Timings::barrier(size_t count, double elapsed, size_t instances, PromiseReduce *pr_ptr) {
  return barrier(pr_ptr ? pr_ptr->get_team() : upcxx::world(), count, elapsed, instances, pr_ptr);
}

Timings Timings::barrier(PromiseReduce &pr, size_t count, double elapsed, size_t instances) {
  return barrier(pr.get_team(), count, elapsed, instances, &pr);
}
void Timings::print_barrier_timings(const upcxx::team &team, string label, PromiseReduce *pr_ptr) {
  assert(!upcxx::in_progress());
  assert(pr_matches(pr_ptr, team) && "No PromiseReduce or same team");
  if (!pr_ptr || !pr_matches(pr_ptr, team)) DBG("NO PRF for print_barrier_timings ", label, "\n");
  Timings timings = barrier(team, 0, 0, 0, pr_ptr);
  wait_pending();
  SLOG_VERBOSE(KLCYAN, "Timing ", label, ":", timings.to_string(), !pr_ptr ? " (NO PRF)" : "", KNORM, "\n");
}

void Timings::print_barrier_timings(PromiseReduce &pr, string label) { print_barrier_timings(pr.get_team(), label, &pr); }

// no barrier but a future reduction is started
future<ShTimings> Timings::reduce(const upcxx::team &team, size_t count, double elapsed, size_t instances, PromiseReduce *pr_ptr) {
  assert(pr_ptr || !upcxx::in_progress());
  DBG("Timings::reduce(", count, ", ", elapsed, ", ", instances, ")\n");
  assert(pr_matches(pr_ptr, team) && "No PromiseReduce or teams match");
  auto sh_timings = make_shared<Timings>();
  set_before(*sh_timings, count, elapsed, instances);
  if (!pr_ptr) DBG("NO PRF for reduce\n");
  auto future_reduction =
      set_after(team, *sh_timings, sh_timings->before, pr_ptr);  // after == before, so no barrier info will be output
  return when_all(make_future(sh_timings), future_reduction, get_pending());
}

upcxx::future<ShTimings> Timings::reduce(size_t count, double elapsed, size_t instances, PromiseReduce *pr_ptr) {
  return reduce(pr_ptr ? pr_ptr->get_team() : upcxx::world(), count, elapsed, instances, pr_ptr);
}

upcxx::future<ShTimings> Timings::reduce(PromiseReduce &pr, size_t count, double elapsed, size_t instances) {
  return reduce(pr.get_team(), count, elapsed, instances, &pr);
}

void Timings::print_reduce_timings(const upcxx::team &team, string label, PromiseReduce *pr_ptr) {
  assert(pr_matches(pr_ptr, team) && "No PromiseReduce or teams match");
  if (!pr_ptr || !pr_matches(pr_ptr, team)) DBG("NO PRF for print_reduce_timings ", label, "\n");
  future<ShTimings> fut_timings = reduce(team, 0, 0, 0, pr_ptr);
  auto fut = when_all(fut_timings, get_pending()).then([label = std::move(label), pr_ptr](ShTimings shptr_timings) {
    SLOG_VERBOSE(KLCYAN, "Timing ", label, ": ", shptr_timings->to_string(), KNORM, !pr_ptr ? " (NO PRF)" : "", "\n");
  });
  set_pending(fut);
}

void Timings::print_reduce_timings(string label, PromiseReduce *pr_ptr) {
  print_reduce_timings(pr_ptr ? pr_ptr->get_team() : upcxx::world(), label, pr_ptr);
}

//
// BaseTimer
//

size_t &BaseTimer::instance_count() {
  static size_t _ = 0;
  return _;
}

void BaseTimer::increment_instance() { ++instance_count(); }
void BaseTimer::decrement_instance() { instance_count()--; }
size_t BaseTimer::get_instance_count() { return instance_count(); }

BaseTimer::BaseTimer(PromiseReduce *_pr_ptr)
    : pr_ptr(_pr_ptr)
    , t()
    , first_t(now())
    , last_t(first_t)
    , name()
    , t_elapsed(0.0)
    , count(0) {}

BaseTimer::BaseTimer(const string &_name, PromiseReduce *_pr_ptr)
    : pr_ptr(_pr_ptr)
    , t()
    , first_t(now())
    , last_t(first_t)
    , name(_name)
    , t_elapsed(0.0)
    , count(0) {}

BaseTimer::~BaseTimer() {}

bool BaseTimer::pr_good() const { return (pr_ptr != nullptr); }

bool BaseTimer::pr_good(const upcxx::team &tm) const { return pr_good() && Timings::pr_matches(pr_ptr, tm); }

bool BaseTimer::pr_matches(const upcxx::team &tm) const { return Timings::pr_matches(pr_ptr, tm); }

void BaseTimer::clear() {
  t = timepoint_t();
  first_t = last_t = now();
  t_elapsed = 0.0;
  count = 0;
}

void BaseTimer::start() {
  assert(t == timepoint_t());
  t = now();
  if (count == 0) first_t = t;
}

void BaseTimer::stop() {
  last_t = now();
  double elapsed = get_elapsed_since_start(last_t);
  t = timepoint_t();  // reset to 0
  t_elapsed += elapsed;
  count++;
}

double BaseTimer::get_elapsed() const { return t_elapsed; }

double BaseTimer::get_elapsed_since_start(timepoint_t then) const {
  assert(t != timepoint_t());
  assert(then != timepoint_t());
  duration_seconds interval = then - t;
  return interval.count();
}

size_t BaseTimer::get_count() const { return count; }

const string &BaseTimer::get_name() const { return name; }

void BaseTimer::done() const {
  assert(t == timepoint_t());
  SLOG_VERBOSE(KLCYAN, "Timing ", name, ": ", std::setprecision(2), std::fixed, t_elapsed, " s ", KNORM, "\n");
  DBG(name, " took ", std::setprecision(2), std::fixed, t_elapsed, " s ", "\n");
}

future<MinSumMax<double>> BaseTimer::done_all_async(const upcxx::team &tm) const {
  assert(pr_matches(tm));
  assert(pr_ptr || !upcxx::in_progress());
  assert(t == timepoint_t());
  if (!pr_good(tm) || !pr_ptr) DBG("NO PRF for done_all_async: ", name, "\n");
  auto msm_elapsed_fut = pr_good(tm) ? pr_ptr->msm_reduce_one(t_elapsed, count != 0, 0) :
                                       upcxx_utils::min_sum_max_reduce_one(t_elapsed, count != 0, 0, tm);
  auto msm_count_fut =
      pr_good(tm) ? pr_ptr->msm_reduce_one(count, count != 0, 0) : upcxx_utils::min_sum_max_reduce_one(count, count != 0, 0, tm);
  auto msm_first_fut = reduce_timepoint(tm, get_first(), pr_ptr);
  auto msm_last_fut = reduce_timepoint(tm, get_last(), pr_ptr);
  DBG(name, " took ", t_elapsed, " \n");
  auto name_copy = name;
  auto msm_fut =
      when_all(msm_elapsed_fut, msm_count_fut, msm_first_fut, msm_last_fut, Timings::get_pending())
          .then([name_copy, pr_ptr = this->pr_ptr](const MinSumMax<double> &msm, const MinSumMax<size_t> &msm_ct,
                                                   const MinSumMax<double> &msm_first, const MinSumMax<double> &msm_last) {
            string msm_string;
            if (msm_ct.min != msm_ct.max || msm_ct.max > 1) msm_string = string(" counts: ") + msm_ct.to_string(false);
            msm_string += string(" first: ") + msm_first.to_string(false);
            msm_string += string(" last: ") + msm_last.to_string(false);
            SLOG_VERBOSE(KLCYAN, "Timing ", name_copy, ": ", msm.to_string(), msm_string, !pr_ptr ? " (NO PRF)" : "", KNORM, "\n");
            return msm;
          });
  Timings::set_pending(msm_fut.then([](const MinSumMax<double> &ignored) {}));
  return msm_fut;
}

void BaseTimer::done_all(const upcxx::team &tm) const {
  assert(!upcxx::in_progress());
  auto fut = done_all_async(tm);
  if (pr_ptr) wait_wrapper(pr_ptr->fulfill());
  wait_wrapper(fut.then([](const auto &ignored) {}));
}

string BaseTimer::get_final() const {
  ostringstream os;
  os << name << ": " << std::setprecision(2) << std::fixed << t_elapsed << " s";
  if (count > 1) os << " " << count << " count";
  return os.str();
}

static auto msm_to_secs(const upcxx::team &team) {
  return [&team](MinSumMax<double> msm) {
    duration_seconds interval;
    if (team.rank_me()) return msm;
    // translate to seconds since the first rank entered
    msm.my = msm.my - msm.min;
    msm.max = msm.max - msm.min;
    msm.sum = msm.sum - msm.min * team.rank_n();
    msm.min = 0.0;
    msm.apply_avg(team);
    return msm;
  };
}

future<MinSumMax<double>> BaseTimer::reduce_timepoint(const upcxx::team &tm, timepoint_t timepoint, PromiseReduce *pr_ptr) {
  static const int64_t epoch_adjustment = 1640995200LL;  // seconds since 1/1/2022 reduce the max
  assert(Timings::pr_matches(pr_ptr, tm) || !upcxx::in_progress());
  duration_seconds secs = timepoint.time_since_epoch();
  DBG_VERBOSE("reduce_timepoint ", secs.count(), " since epoch\n");
  if (!pr_ptr) DBG("NO PRF for reduce timepoint\n");
  future<MinSumMax<double>> fut_msm = pr_ptr ? pr_ptr->msm_reduce_one(secs.count() - epoch_adjustment, 0) :
                                               min_sum_max_reduce_one<double>(secs.count() - epoch_adjustment, 0, tm);
  return fut_msm.then(msm_to_secs(tm));
}

future<ShTimings> BaseTimer::reduce_timings(const upcxx::team &team, size_t my_count, double my_elapsed, size_t my_instances,
                                            PromiseReduce *pr_ptr) {
  if (!pr_ptr || !Timings::pr_matches(pr_ptr, team)) DBG("NO PRF for reduce_timings\n");
  return Timings::reduce(team, my_count, my_elapsed, my_instances, pr_ptr);
}

future<ShTimings> BaseTimer::reduce_timings(const upcxx::team &team, size_t my_instances) const {
  return reduce_timings(team, count, t_elapsed, my_instances, Timings::pr_matches(pr_ptr, team) ? pr_ptr : nullptr);
}

upcxx::future<ShTimings> BaseTimer::reduce_timings(size_t my_count, double my_elapsed, size_t my_instances, PromiseReduce *pr_ptr) {
  return reduce_timings(pr_ptr ? pr_ptr->get_team() : upcxx::world(), my_count, my_elapsed, my_instances, pr_ptr);
}

upcxx::future<ShTimings> BaseTimer::reduce_timings(size_t my_instances) const {
  return reduce_timings(pr_ptr ? pr_ptr->get_team() : upcxx::world(), my_instances);
}
Timings BaseTimer::barrier_timings(const upcxx::team &team, size_t my_count, double my_elapsed, size_t my_instances,
                                   PromiseReduce *pr_ptr) {
  return Timings::barrier(team, my_count, my_elapsed, my_instances, Timings::pr_matches(pr_ptr, team) ? pr_ptr : nullptr);
}

Timings BaseTimer::barrier_timings(const upcxx::team &team, size_t my_instances) const {
  return barrier_timings(team, count, t_elapsed, my_instances, Timings::pr_matches(pr_ptr, team) ? pr_ptr : nullptr);
}
Timings BaseTimer::barrier_timings(size_t my_count, double my_elapsed, size_t my_instances, PromiseReduce *pr_ptr) {
  return barrier_timings(pr_ptr ? pr_ptr->get_team() : upcxx::world(), my_count, my_elapsed, my_instances, pr_ptr);
}
Timings BaseTimer::barrier_timings(size_t my_instances) const {
  return barrier_timings(pr_ptr ? pr_ptr->get_team() : upcxx::world(), my_instances);
}

timepoint_t BaseTimer::now() { return std::chrono::high_resolution_clock::now(); }

upcxx::future<MinSumMax<double>> BaseTimer::reduce_now(PromiseReduce &pr) { return reduce_timepoint(pr.get_team(), now(), &pr); }

string BaseTimer::now_str() {
  std::time_t result = std::time(nullptr);
  char buffer[100];
  struct tm tmp;
  size_t sz = strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", localtime_r(&result, &tmp));
  return string(sz > 0 ? buffer : "BAD TIME");
}

//
// StallTimer
//

StallTimer::StallTimer(const string _name, double _warn_seconds, int64_t _warn_count, double _max_seconds, int64_t _max_count,
                       PromiseReduce *_pr_ptr)
    : BaseTimer(_name, pr_ptr)
    , warn_seconds(_warn_count)
    , max_seconds(_max_seconds)
    , warn_count(_warn_count)
    , max_count(_max_count) {
  start();
}

StallTimer::~StallTimer() { stop(); }

void StallTimer::check() {
  stop();
  bool print = false, fatal = false;
  if ((warn_seconds > 0.0 && t_elapsed >= warn_seconds) || (warn_count > 0 && count > warn_count)) {
    print = true;
  } else if ((max_seconds > 0.0 && t_elapsed > max_seconds) || (max_count > 0 && max_count > count)) {
    print = true;
    fatal = true;
  }
  if (print) {
    if (fatal) {
      DIE("StallTimer - ", name, " on ", rank_me(), " stalled for ", t_elapsed, " s and ", count, " iterations");
    } else {
      WARN("StallTimer - ", name, " on ", rank_me(), " stalled for ", t_elapsed, " s and ", count, " iterations\n");
    }
    warn_seconds *= 2.0;
    warn_count *= 2;
  }
  start();
}

void StallTimer::reset(string append_description) {
  if (t == timepoint_t()) t = now();
  this->t_elapsed = 0;
  this->count = 0;
  name += append_description;
}

//
// IntermittentTimer
//

IntermittentTimer::IntermittentTimer(const string _name, string _interval_label, int who, PromiseReduce *_pr_ptr)
    : RusageTimer(_name, who, _pr_ptr)
    , t_interval(0.0)
    , interval_label(_interval_label) {}

IntermittentTimer::~IntermittentTimer() {}

void IntermittentTimer::clear() {
  ((BaseTimer *)this)->clear();
  t_interval = 0.0;
  interval_label = "";
}

void IntermittentTimer::start_interval() { t_interval = get_elapsed_since_start(); }

void IntermittentTimer::stop_interval() {
  t_interval = get_elapsed_since_start() - t_interval;
  if (!interval_label.empty()) {
    ostringstream oss;
    oss << KBLUE << std::left << std::setw(40) << interval_label << std::setprecision(2) << std::fixed << t_interval << " s"
        << KNORM << "\n";
    SLOG(oss.str());
  }
}

void IntermittentTimer::print_out(const upcxx::team &tm) {
  future<ShTimings> fut_shptr_timings = reduce_timings(tm);
  auto fut = when_all(Timings::get_pending(), fut_shptr_timings)
                 .then([&name = this->name, &count = this->count, pr_ptr = this->pr_ptr](ShTimings shptr_timings) {
                   if (shptr_timings->count_msm.max > 0.0)
                     SLOG_VERBOSE(KLCYAN, "Timing ", name, ": ", count, " intervals, ", shptr_timings->to_string(true),
                                  !pr_ptr ? " (NO PRF)" : "", KNORM, "\n");
                 });
  Timings::set_pending(fut);
  clear();
}

//
// ProgressTimer
//

ProgressTimer::ProgressTimer(const string &_name, PromiseReduce *_pr_ptr)
    : BaseTimer(_name, _pr_ptr)
    , calls(0) {}

ProgressTimer::~ProgressTimer() {}

void ProgressTimer::clear() {
  ((BaseTimer *)this)->clear();
  calls = 0;
}

void ProgressTimer::progress(size_t run_every) {
  if (run_every > 1 && ++calls % run_every != 0) return;
  start();
  upcxx::progress();
  stop();
  // DBG("ProgressTimer(", name, ") - ", t_elapsed, "\n");
}

void ProgressTimer::discharge(size_t run_every) {
  if (run_every != 1 && ++calls % run_every != 0) return;
  start();
  upcxx::discharge();
  upcxx::progress();
  stop();
  // DBG("ProgressTimer(", name, ").discharge() - ", t_elapsed, "\n");
}

void ProgressTimer::print_out(const upcxx::team &tm) {
  future<ShTimings> fut_shptr_timings = reduce_timings(tm);
  auto fut =
      when_all(Timings::get_pending(), fut_shptr_timings)
          .then([&name = this->name, pr_ptr = this->pr_ptr](ShTimings shptr_timings) {
            if (shptr_timings->count_msm.max > 0.0)
              SLOG_VERBOSE(KLCYAN, "Timing ", name, ": ", shptr_timings->to_string(true), !pr_ptr ? " (NO PRF)" : "", KNORM, "\n");
          });
  Timings::set_pending(fut);
  clear();
}

//
// Timer
//
Timer::Timer(PromiseReduce &_pr, const string &_name, bool exit_reduction)
    : RusageTimer(_name, RUSAGE_SELF, &_pr)
    , tm(_pr.get_team())
    , exited(exit_reduction)
    , logged(false) {
  init();
}
Timer::Timer(const upcxx::team &tm, const string &_name, bool exit_reduction)
    : RusageTimer(_name, RUSAGE_SELF, Timings::get_promise_reduce(tm))
    , tm(tm)
    , exited(exit_reduction)
    , logged(false) {
  init();
}
Timer::Timer(const string &_name, bool exit_reduction)
    : RusageTimer(_name, RUSAGE_SELF, &Timings::get_promise_reduce())
    , tm(upcxx::world())
    , exited(exit_reduction)
    , logged(false) {
  init();
}
Timer::Timer(Timer &&move)
    : RusageTimer(std::move((RusageTimer &&) move))
    , tm(move.tm)
    , exited(move.exited) {
  move.exited = true;
  move.logged = true;
}
void Timer::init() {
  increment_instance();
  auto fut = when_all(Timings::get_pending(), make_future(now_str())).then([name = this->name](const string &ignored) {});
  Timings::set_pending(fut);
  start();
}
Timer &Timer::operator=(Timer &&move) {
  Timer mv(std::move(move));
  std::swap(*this, mv);
  return *this;
}

Timer::~Timer() {
  if (!exited)
    initiate_exit_reduction();
  else if (!logged) {
    stop();
    LOG(KLCYAN, "Timing ", name, ":", get_elapsed(), KNORM, "\n");
  }
}

future<> Timer::initiate_entrance_reduction() {
  assert(pr_matches(tm));
  DBG_VERBOSE("Tracking entrance of ", name, "\n");
  auto fut_msm = reduce_timepoint(tm, now(), pr_ptr);

  auto fut =
      when_all(Timings::get_pending(), fut_msm).then([name = this->name, pr_ptr = this->pr_ptr](const MinSumMax<double> &msm) {
        DBG_VERBOSE("got reduction: ", msm.to_string(), "\n");
        SLOG_VERBOSE(KLCYAN, "Timing (entrance) ", name, ":", msm.to_string(), !pr_ptr ? " (NO PRF)" : "", KNORM, "\n");
      });
  Timings::set_pending(fut);
  return fut;
}

future<> Timer::initiate_exit_reduction() {
  stop();
  assert(pr_matches(tm));
  future<ShTimings> fut_shptr_timings = reduce_timings(tm, count, t_elapsed, 0, pr_ptr);
  auto fut =
      when_all(Timings::get_pending(), fut_shptr_timings).then([name = this->name, pr_ptr = this->pr_ptr](ShTimings shptr_timings) {
        SLOG_VERBOSE(KLCYAN, "Timing ", name, " exit: ", shptr_timings->to_string(), !pr_ptr ? " (NO PRF)" : "", KNORM, "\n");
      });
  Timings::set_pending(fut);
  decrement_instance();
  exited = true;
  logged = true;
  return fut;
}

//
// Rusage
//
Rusage::Rusage(int _who)
    : who(_who) {
  int ret = getrusage(who, this);
  if (ret != 0) DIE("Could not getrusage(who=", who, ") ret=", ret, " error: ", strerror(errno), "\n");
}
double Rusage::get_utime() const { return this->ru_utime.tv_sec + this->ru_utime.tv_usec / 1000000.; }
double Rusage::get_stime() const { return this->ru_stime.tv_sec + this->ru_stime.tv_usec / 1000000.; }
double Rusage::get_cpu_time() const { return get_utime() + get_stime(); }
double Rusage::get_utime(const Rusage &other) const { return get_utime() - other.get_utime(); }
double Rusage::get_stime(const Rusage &other) const { return get_stime() - other.get_stime(); }
double Rusage::get_cpu_time(const Rusage &other) const { return get_utime(other) + get_stime(other); }
string Rusage::get_rusage_str(double elapsed) const {
  Rusage end_rusage(who);
  double utime = end_rusage.get_utime(*this), stime = end_rusage.get_stime(*this);
  stringstream ss;
  _logger_recurse(ss, "utime=", utime, "s sys=", stime, "s tot=", utime + stime, "s. ",
                  elapsed > 0.0001 ? ((elapsed != 0.0 ? (utime + stime) / elapsed : 1) * 100.0) : 100, "% of ", elapsed, "s");
  return ss.str();
}

RusageTimer::RusageTimer(int who, PromiseReduce *_pr_ptr)
    : BaseTimer(_pr_ptr)
    , rusage(who) {}
RusageTimer::RusageTimer(const string &_name, int who, PromiseReduce *_pr_ptr)
    : BaseTimer(_name, _pr_ptr)
    , rusage(who) {}
string RusageTimer::get_rusage_str() const { return rusage.get_rusage_str(get_elapsed()); }
//
// BarrierTimer
//
BarrierTimer::BarrierTimer(PromiseReduce &pr, const string _name, bool _entrance_barrier, bool _exit_barrier)
    : RusageTimer(_name, RUSAGE_SELF, &pr)
    , _team(pr.get_team())
    , exit_barrier(_exit_barrier)
    , exited(false)
    , start_rusage() {
  init(_entrance_barrier);
}
BarrierTimer::BarrierTimer(const upcxx::team &tm, const string _name, bool _entrance_barrier, bool _exit_barrier)
    : RusageTimer(_name, RUSAGE_SELF, Timings::get_promise_reduce(tm))
    , _team(tm)
    , exit_barrier(_exit_barrier)
    , exited(false)
    , start_rusage() {
  init(_entrance_barrier);
}
BarrierTimer::BarrierTimer(const string _name, bool _entrance_barrier, bool _exit_barrier)
    : RusageTimer(_name, RUSAGE_SELF, &Timings::get_promise_reduce())
    , _team(upcxx::world())
    , exit_barrier(_exit_barrier)
    , exited(false)
    , start_rusage() {
  init(_entrance_barrier);
}

future<> BarrierTimer::init(bool _entrance_barrier) {
  assert(!upcxx::in_progress());
  increment_instance();
  if (!_entrance_barrier && !exit_barrier) SLOG_VERBOSE("Why are we using a BarrierTimer without any barriers???\n");
  future<> fut;
  DBG("Entering BarrierTimer ", name, "\n");
  if (_entrance_barrier) {
    auto timings = barrier_timings(_team);
    Timings::wait_pending();
    SLOG_VERBOSE(KLCYAN, "Timing (entrance barrier) ", name, ": ", timings.to_string(), KNORM, "\n");
  } else {
    fut = when_all(Timings::get_pending(), make_future(now_str())).then([&name = this->name](string now) {});
    Timings::set_pending(fut);
  }
  start();
  return fut;
}

BarrierTimer::~BarrierTimer() {
  if (!exited) wait_wrapper(initiate_exit_barrier());
}

future<> BarrierTimer::initiate_exit_barrier() {
  assert(!upcxx::in_progress());
  stop();
  LOG(name, " ", get_rusage_str(), "\n");
  future<> fut;
  DBG("Exiting BarrierTimer ", name, "\n");
  if (exit_barrier) {
    fut = make_future();
    auto timings = barrier_timings(_team);
    Timings::wait_pending();
    SLOG_VERBOSE(KLCYAN, "Timing ", name, ": ", timings.to_string(), KNORM, "\n");
  } else {
    future<ShTimings> fut_shptr_timings = reduce_timings(_team);
    fut = when_all(Timings::get_pending(), fut_shptr_timings).then([name = this->name](ShTimings shptr_timings) {
      SLOG_VERBOSE(KLCYAN, "Timing ", name, ": ", shptr_timings->to_string(), KNORM, "\n");
    });
    Timings::set_pending(fut);
  }
  decrement_instance();
  exited = true;
  return fut;
}

//
// AsyncTimer
//

_AsyncTimer::_AsyncTimer(PromiseReduce &pr, const string &name)
    : BaseTimer(name, &pr)
    , tm(pr.get_team())
    , construct_t(BaseTimer::now())
    , start_t{} {}
_AsyncTimer::_AsyncTimer(const upcxx::team &tm, const string &name)
    : BaseTimer(name, Timings::get_promise_reduce(tm))
    , tm(tm)
    , construct_t(BaseTimer::now())
    , start_t{} {}
void _AsyncTimer::start() {
  start_t = now();
  ((BaseTimer *)this)->start();
}
void _AsyncTimer::stop() { ((BaseTimer *)this)->stop(); }
void _AsyncTimer::report(const string label, MinSumMax<double> msm, bool verbose) {
  if (verbose)
    SLOG_VERBOSE(KLCYAN, "Timing ", name, " ", label, ":", msm.to_string(), !pr_ptr ? " (NO PRF)" : "", KNORM, "\n");
  else
    LOG("Timing ", name, " ", label, ":", msm.to_string(), !pr_ptr ? " (NO PRF)" : "", "\n");
}

future<> _AsyncTimer::initiate_construct_reduction(bool verbose) {
  assert(!upcxx::in_progress());
  assert(pr_matches(tm));
  auto fut_msm = BaseTimer::reduce_timepoint(tm, construct_t, pr_ptr);
  auto fut = when_all(Timings::get_pending(), fut_msm).then([this, verbose](const MinSumMax<double> &msm) {
    this->report("construct", msm, verbose);
  });
  Timings::set_pending(fut);
  return fut;
}
future<> _AsyncTimer::initiate_start_reduction(bool verbose) {
  assert(!upcxx::in_progress());
  assert(pr_matches(tm));
  auto fut_msm = BaseTimer::reduce_timepoint(tm, start_t, pr_ptr);
  auto fut = when_all(Timings::get_pending(), fut_msm).then([this, verbose](const MinSumMax<double> &msm) {
    this->report("start", msm, verbose);
  });
  Timings::set_pending(fut);
  return fut;
}
future<> _AsyncTimer::initiate_stop_reduction(bool verbose) {
  assert(!upcxx::in_progress());
  assert(pr_matches(tm));
  auto fut_msm = Timings::reduce(tm, 1, get_elapsed(), 1, pr_ptr);
  auto fut = when_all(Timings::get_pending(), fut_msm).then([this, verbose](ShTimings sh_timings) {
    this->report("stop", sh_timings->before_elapsed, verbose);
  });
  Timings::set_pending(fut);
  return fut;
}

AsyncTimer::AsyncTimer(PromiseReduce &pr, const string &name)
    : timer(make_shared<_AsyncTimer>(pr, name)) {}
AsyncTimer::AsyncTimer(const upcxx::team &tm, const string &name)
    : timer(make_shared<_AsyncTimer>(tm, name)) {}
AsyncTimer::AsyncTimer(const string &name)
    : timer(make_shared<_AsyncTimer>(upcxx::world(), name)) {}
void AsyncTimer::start() const { timer->start(); }
void AsyncTimer::stop() const {
  timer->stop();
  LOG(timer->get_name(), " completed in ", timer->get_elapsed(), " s\n");
}
double AsyncTimer::get_elapsed() const { return timer->get_elapsed(); }
future<> AsyncTimer::initiate_construct_reduction(bool verbose) {
  return timer->initiate_construct_reduction(verbose).then([timer = this->timer]() {
    // keep timer alive
  });
}
future<> AsyncTimer::initiate_start_reduction(bool verbose) {
  return timer->initiate_start_reduction(verbose).then([timer = this->timer]() {
    // keep timer alive
  });
}
future<> AsyncTimer::initiate_stop_reduction(bool verbose) {
  return timer->initiate_stop_reduction(verbose).then([timer = this->timer]() {
    // keep timer alive
  });
}

//
// ActiveCountTimer
//

ActiveCountTimer::ActiveCountTimer(const string _name)
    : total_elapsed(0.0)
    , total_count(0)
    , active_count(0)
    , max_active(0)
    , name(_name)
    , my_fut(make_future()) {}

ActiveCountTimer::~ActiveCountTimer() {
  if (upcxx::initialized()) {
    Timings::wait_pending();
    wait_wrapper(my_fut);  // keep alive until all futures have finished
  }
}

void ActiveCountTimer::clear() {
  total_elapsed = 0.0;
  total_count = 0;
  active_count = 0;
  max_active = 0;
}

timepoint_t ActiveCountTimer::begin() {
  active_count++;
  if (max_active < active_count) max_active = active_count;
  return BaseTimer::now();
}

void ActiveCountTimer::end(timepoint_t t) {
  duration_seconds interval = BaseTimer::now() - t;
  active_count--;
  total_count++;
  total_elapsed += interval.count();
}

void ActiveCountTimer::print_barrier_timings(const upcxx::team &team, string label) {
  assert(!upcxx::in_progress());
  Timings timings = BaseTimer::barrier_timings(team, total_count, total_elapsed, max_active);
  Timings::wait_pending();
  clear();
  print_timings(timings, label);
}

void ActiveCountTimer::print_reduce_timings(const upcxx::team &team, string label) {
  label = name + label;
  auto fut_timings = BaseTimer::reduce_timings(team, total_count, total_elapsed, max_active);
  auto _this = this;
  auto fut_clear = fut_timings.then([_this](ShTimings ignored) { _this->clear(); });
  auto fut = when_all(Timings::get_pending(), fut_timings, fut_clear).then([_this, label](ShTimings shptr_timings) {
    _this->print_timings(*shptr_timings, label);
  });
  my_fut = when_all(fut_clear, my_fut, fut);  // keep this in scope until clear has been called...
  Timings::set_pending(my_fut);
}

void ActiveCountTimer::print_timings(Timings &timings, string label) {
  label = name + label;
  DBG_VERBOSE(__func__, " label=", label, "\n");
  if (active_count > 0)
    SWARN("print_timings on ActiveCountTimer '", label, "' called while ", active_count, " (max ", max_active,
          ") are still active\n");
  if (timings.count_msm.max > 0.0) {
    SLOG_VERBOSE(KLCYAN, "Timing instances of ", label, ": ",
                 (timings.count_msm.max > 0.0 ? timings.to_string(true) : string("(none)")), KNORM, "\n");
  }
}

ActiveCountTimer _GenericActiveCountTimer("_upcxx_dummy");
GenericInstantiationTimer _GenericInstantiationTimer(_GenericActiveCountTimer);
template class ActiveInstantiationTimer<_upcxx_utils_dummy>;

SingletonInstantiationTimer _SingletonInstantiationTimer();
template class InstantiationTimer<_upcxx_utils_dummy>;

};  // namespace upcxx_utils
