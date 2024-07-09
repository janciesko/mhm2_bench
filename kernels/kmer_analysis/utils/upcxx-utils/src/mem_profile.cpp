#include "upcxx_utils/mem_profile.hpp"

#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <upcxx/upcxx.hpp>
#include <utility>

#include "upcxx_utils/log.hpp"
#include "upcxx_utils/promise_collectives.hpp"
#include "upcxx_utils/timers.hpp"
#include "upcxx_utils/version.h"

using namespace std;
using std::pair;

#ifndef TICKS_PER_ACTIVITY_CHECK
#define TICKS_PER_ACTIVITY_CHECK 10
#endif

namespace upcxx_utils {

std::ofstream _memlog;

void open_memlog(string name) {
  assert(!_memlog.is_open());
  // only 1 process per node opens
  if (upcxx::local_team().rank_me()) return;

  if (!get_rank_path(name, upcxx::rank_me())) DIE("Could not get rank_path for: ", name, "\n");

  bool old_file = file_exists(name);
  DBG("Opening ", name, " old_file=", old_file, "\n");
  _memlog.open(name, std::ofstream::out | std::ofstream::app);
  if (!_memlog.is_open()) DIE("Could not open: ", name, "\n");
  LOG_MEM("Open");
}

void close_memlog() {
  if (_memlog.is_open()) {
    if (upcxx::initialized()) LOG_MEM("Close");
    _memlog.flush();
    _memlog.close();
  }
  assert(!_memlog.is_open());
}

pair<double, string> get_free_mem(const vector<string> &capture_fields) {
  // comma separated fields to find and return concatenated
  string buf;
  stringstream ss;
  ifstream f("/proc/meminfo");
  double mem_free = 0;
  while (!f.eof()) {
    getline(f, buf);
    if (buf.find("MemFree") == 0 || buf.find("Buffers") == 0 || buf.find("Cached") == 0) {
      stringstream fields;
      string units;
      string name;
      double mem;
      fields << buf;
      fields >> name >> mem >> units;
      if (units[0] == 'k') mem *= 1024;
      mem_free += mem;
    }
    for (const auto &field : capture_fields) {
      if (buf.find(field) != string::npos) {
        ss << buf << "\t";
        break;
      }
    }
  }
  pair<double, string> ret(mem_free, ss.str());
  return ret;
}

double get_free_mem(bool local_barrier) {
  vector<string> no_fields;
  double ret = 0;
  if (local_barrier) barrier(local_team());
  if (!local_barrier || local_team().rank_me() == 0) ret = get_free_mem(no_fields).first;
  if (local_barrier) {
    ret = upcxx::broadcast(ret, 0, local_team()).wait();
    auto &max = __max_free_mem_per_rank();
    auto ret_per = ret / local_team().rank_n();
    if (ret_per > max) max = ret_per;
    barrier(local_team());
  }
  return ret;
}

double &__max_free_mem_per_rank() {
  static double _ = 0;
  return _;
}
double get_max_free_mem_per_rank() { return __max_free_mem_per_rank(); }

string get_self_stat(void) {
  std::stringstream buffer;
  buffer << "Rank" << upcxx::world().rank_me() << ": ";
  std::ifstream i;
  i.open("/proc/self/statm");
  buffer << i.rdbuf();
  i.close();
  i.open("/proc/self/stat");
  buffer << i.rdbuf();
  i.close();
#ifdef UPCXX_UTILS_VERBOSE_MEMORY
  if (upcxx::local_team().rank_me() == 0) {
    i.open("/proc/self/status");
    buffer << i.rdbuf();
    i.close();
    i.open("/proc/meminfo");
    buffer << i.rdbuf();
    i.close();
  }
#endif
  return buffer.str();
}

#define IN_NODE_TEAM() (!(upcxx::rank_me() % upcxx::local_team().rank_n()))

void MovingAverageStdDev::Update(const double &x) {
  if (n_++ == 0) {
    min_ = x;
    max_ = x;
  } else {
    if (min_ > x) min_ = x;
    if (max_ < x) max_ = x;
  }
  double delta = x - mean_;
  mean_ += delta / n_;
  double delta2 = x - mean_;
  m2_ += delta * delta2;
}

double MovingAverageStdDev::Variance() const {
  if (n_ < 2) {
    return 0;
  } else {
    return m2_ / (n_ - 1);
  }
}

#include <cmath>
double MovingAverageStdDev::StdDev() const { return std::sqrt(Variance()); }

TrackActivity::TrackActivity() { reset(); }

void TrackActivity::reset() {
  last_check = upcxx_utils::BaseTimer::now();
  stats = {};
  outliers.clear();
  max_timepoints = {};
}

double TrackActivity::get_time_since_last_check(timepoint_t t) const {
  duration_seconds interval = t - last_check;
  return interval.count();
}

bool TrackActivity::check(bool record) {
  timepoint_t now = upcxx_utils::BaseTimer::now();
  bool is_outlier = false;
  double s = get_time_since_last_check(now);
  auto newstats = stats;
  newstats.Update(s);
  if (newstats.Count() > 1 && s > 4 * newstats.StdDev() + newstats.Mean() * 1.5)  // 4 sigmas + 50% above the (new) average
    is_outlier = true;

  if (record) {
    timepoints t = {last_check, now};
    if (stats.Count() == 0 || (t.second - t.first) > (max_timepoints.second - max_timepoints.first)) {
      max_timepoints = t;
      DBG("New worst Activity of ", s, "\n");
    }
    stats = newstats;
    if (is_outlier) outliers.push_back({last_check, now});

    last_check = now;
  }
  return is_outlier;
}

size_t TrackActivity::get_count() const { return stats.Count(); }

double TrackActivity::get_avg() const { return stats.Mean(); }

double TrackActivity::get_stddev() const { return stats.StdDev(); }

double TrackActivity::get_max() const { return stats.Max(); }

double TrackActivity::get_min() const { return stats.Min(); }

TrackActivity::timepoints TrackActivity::get_max_times() const { return max_timepoints; }

std::vector<TrackActivity::timepoints> &TrackActivity::get_outliers() { return outliers; }

#ifdef UPCXX_UTILS_NO_THREADS

void MemoryTrackerThread::send_activity_check() {}
void MemoryTrackerThread::start() {}
void MemoryTrackerThread::stop() {}

#else  // yes UPCXX_UTILS_THREADS

void MemoryTrackerThread::send_activity_check() {
  if (!upcxx::initialized()) return;
  if (!upcxx::master_persona().active_with_caller()) {
    upcxx::master_persona().lpc_ff([&self = *this]() { self.send_activity_check(); });
    return;
  }
  // local rank 0 sends checks to the previous node round-robin
  // other ranks send local rpcs to local rank 0
  bool is_world = upcxx::local_team().rank_me() == 0;
  int tgt = is_world ? (upcxx::rank_me() + upcxx::rank_n() - upcxx::local_team().rank_n()) % upcxx::rank_n() : 0;
  rpc_ff(
      is_world ? upcxx::world() : upcxx::local_team(), tgt,
      [](DistActivities &da, int from_rank) {
        static int count_checks = 0;
        auto &activity = (*da)[from_rank % upcxx::local_team().rank_n()];
        if (activity.check()) {
          auto &times = activity.outliers.back();
          duration_seconds interval = times.second - times.first;
          LOG("Abnormally long activity check-in from ", from_rank, " of ", interval.count(), "s avg=", activity.get_avg(), "s +- ",
              activity.get_stddev(), " count=", activity.get_count(), "\n");
        } else
          DBG_VERBOSE2("Got good check-in from ", from_rank, "\n");
        if (++count_checks % upcxx::local_team().rank_n() == 0) {
          // check all ranks for unresponsiveness
          int rank = 0;
          for (auto &act : *da) {
            if (act.check(false)) {
              LOG("Abnormally long since last activity check-in from ",
                  (upcxx::rank_me() + (rank == 0 ? upcxx::local_team().rank_n() : rank)) % upcxx::rank_n(), " of ",
                  act.get_time_since_last_check(), "s avg=", act.get_avg(), "s +- ", act.get_stddev(), " count=", act.get_count(),
                  "\n");
            } else
              DBG_VERBOSE2("Getting regular check-ins from ", from_rank, "\n");
            rank++;
          }
        }
      },
      dist_activities, rank_me());
}

void MemoryTrackerThread::start() {
  fin = false;
  barrier();
  if (IN_NODE_TEAM()) {
    start_free_mem = get_free_mem();
    dist_activities->resize(upcxx::local_team().rank_n());
    for (auto &a : *dist_activities) a.reset();
  }
  barrier(local_team());
  min_free_mem = start_free_mem = upcxx::broadcast(start_free_mem, 0, local_team()).wait();
  auto &pr = Timings::get_promise_reduce();
  auto msm_fut = pr.msm_reduce_one(start_free_mem, 0);

  auto thread_log_flush = [&] {
    long last_flush_tick = 0;
    auto me = upcxx::local_team().rank_me();
    auto n = upcxx::local_team().rank_n();
    while (!fin) {
      std::this_thread::sleep_for(std::chrono::milliseconds(sample_ms));
      // flush per_rank logs every minute stagger by rank
      if ((ticks + me) % n == 0 && (ticks - last_flush_tick) * sample_ms > UPCXX_UTILS_LOG_FLUSH_INTERVAL_MS) {
        if (!me) DBGLOG("Flushing logs with ticks=", ticks, " approx_run_time=", ticks * sample_ms / 1000., " s\n");
        flush_logger();
        last_flush_tick = ticks;
      }
      ticks++;
      if (ticks % TICKS_PER_ACTIVITY_CHECK == 0) {
        send_activity_check();
      }
    }
  };

  auto thread_lambda = [&] {
    ofstream _tracker_file;
    ofstream &tracker_file = ::upcxx_utils::_memlog.is_open() ? ::upcxx_utils::_memlog : _tracker_file;
    if (!tracker_file.is_open() && !tracker_filename.empty() && IN_NODE_TEAM()) {
      get_rank_path(tracker_filename, upcxx::rank_me());
      tracker_file.open(tracker_filename, ios_base::out | ios_base::app);
      if (!tracker_file.is_open() || !tracker_file.good()) DIE("Could not open tracker file:", tracker_filename);
      opened = true;
    }

    double prev_free_mem = 0;
    LOG_MEM_OS(tracker_file, "MemTracker start");
    long last_flush_tick = 0;
    while (!fin) {
      std::this_thread::sleep_for(std::chrono::milliseconds(sample_ms));
      double free_mem = get_free_mem();
      // only report memory if it changed sufficiently - otherwise this produces a great deal of
      // gumpf in the logs
      if (fabs(free_mem - prev_free_mem) > ONE_GB) {
        DBGLOG("MemoryTrackerThread free_mem=", get_size_str(free_mem), "\n");
        prev_free_mem = free_mem;
        flush_logger();
        last_flush_tick = ticks;
      }
      if (free_mem < min_free_mem) min_free_mem = free_mem;
      if (tracker_file.is_open()) {
        LOG_MEM_OS(tracker_file, "MemTracker");
      }
      // flush this root log per node at least every 30 seconds
      if ((ticks - last_flush_tick) * sample_ms > UPCXX_UTILS_LOG_FLUSH_INTERVAL_MS / 2) {
        DBGLOG("Flushing logs with ticks=", ticks, " approx_run_time=", ticks * sample_ms / 1000.,
               " s free_mem=", get_size_str(free_mem), "\n");
        flush_logger();
        if (tracker_file.is_open()) tracker_file.flush();
        last_flush_tick = ticks;
      }
      ticks++;
      if (ticks % TICKS_PER_ACTIVITY_CHECK == 0) {
        send_activity_check();
      }
    }
    LOG_MEM_OS(tracker_file, "MemTracker end");

    if (tracker_file.is_open()) tracker_file.flush();
    if (opened) {
      tracker_file.close();
      opened = false;
    }
  };

  if (IN_NODE_TEAM()) {
    t = new std::thread(thread_lambda);
  } else {
    t = new std::thread(thread_log_flush);
  }

  barrier(local_team());
  double delta_mem;
  if (IN_NODE_TEAM()) delta_mem = start_free_mem - get_free_mem();
  barrier(local_team());
  delta_mem = upcxx::broadcast(delta_mem, 0, local_team()).wait();
  auto msm_fut2 = pr.msm_reduce_one(delta_mem, 0);
  // Finish all pending reductions
  pr.fulfill().wait();
  auto msm = msm_fut.wait();
  auto msm2 = msm_fut2.wait();

  int num_nodes = upcxx::rank_n() / upcxx::local_team().rank_n();
  SLOG("Initial free memory across all ", num_nodes, " nodes: ", get_size_str(msm.sum / upcxx::local_team().rank_n()), " (",
       get_size_str((double)msm.avg), " avg, ", get_size_str(msm.min), " min, ", get_size_str(msm.max), " max)\n");
  SLOG_VERBOSE("Change in free memory after reduction and thread construction ",
               get_size_str(msm2.sum / upcxx::local_team().rank_n()), " (", get_size_str((double)msm2.avg), " avg, ",
               get_size_str(msm2.min), " min, ", get_size_str(msm2.max), " max)\n");
  barrier();
}

void MemoryTrackerThread::stop() {
  if (t) {
    fin = true;
    t->join();
    delete t;
  }
  t = nullptr;
  barrier(local_team());
  double peak_mem;
  if (IN_NODE_TEAM()) peak_mem = start_free_mem - min_free_mem;
  barrier(local_team());
  peak_mem = upcxx::broadcast(peak_mem, 0, local_team()).wait();
  auto &pr = Timings::get_promise_reduce();
  auto msm_fut = pr.msm_reduce_one(peak_mem, 0);

  // summarize TrackActivity stats
  double sum_stddev = 0.0;
  double max_stddev = 0.0;
  double sum_delay = 0.0;
  double max_delay = 0.0;
  auto r = 0;
  for (TrackActivity &act : *dist_activities) {
    auto worst_ts = act.get_max_times();
    duration_seconds interval = worst_ts.second - worst_ts.first;
    double worst_s = interval.count();
    duration_seconds since = upcxx_utils::BaseTimer::now() - worst_ts.first;
    // Add the worst_s number of seconds to now
    auto new_duration = std::chrono::system_clock::now().time_since_epoch() -
                        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::duration<double>(since.count() + worst_s));
    // Convert the new duration to a time_t value
    auto new_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::time_point(new_duration));
    // Convert the time_t value to a struct tm
    struct tm tmp;
    struct tm *new_tm = localtime_r(&new_time_t, &tmp);

    LOG("Activity rank=", rank_me() + r == 0 ? (upcxx::rank_me() + upcxx::local_team().rank_n()) % upcxx::rank_n() : r,
        " count=", act.get_count(), " avg=", act.get_avg(), " +- ", act.get_stddev(), " min=", act.get_min(),
        " max=", act.get_max(), " responding ", since.count(), " s ago at ", std::put_time(new_tm, "%Y-%m-%d %H:%M:%S"), "\n");
    auto delay = act.get_avg();
    auto max = act.get_max();
    sum_delay += delay;
    if (max_delay < max) max_delay = max;
    auto stddev = act.get_stddev();
    sum_stddev += stddev;
    if (max_stddev < stddev) max_stddev = stddev;
    r++;
  }
  auto local_n = upcxx::local_team().rank_n();
  auto local_me = upcxx::local_team().rank_me();
  auto avg_delay = sum_delay / local_n;
  auto avg_stddev = sum_stddev / local_n;
  if (!local_me)
    LOG("Activity delay avg=", avg_delay, " max=", max_delay, " bal=", avg_delay > 0 ? max_delay / avg_delay : 0.0,
        " avg stddev=", avg_stddev, " max stddev=", max_stddev, " bal=", avg_stddev > 0 ? max_stddev / avg_stddev : 0.0, "\n");

  auto sum_delay_fut = pr.reduce_one(sum_delay, upcxx::op_fast_add);
  auto max_delay_fut = pr.reduce_one(max_delay, upcxx::op_fast_max);
  auto sum_stddev_fut = pr.reduce_one(sum_stddev, upcxx::op_fast_add);
  auto max_stddev_fut = pr.reduce_one(max_stddev, upcxx::op_fast_max);
  // Finish all pending reductions
  pr.fulfill().wait();
  avg_delay = sum_delay_fut.wait() / upcxx::rank_n();
  max_delay = max_delay_fut.wait();
  avg_stddev = sum_stddev_fut.wait() / upcxx::rank_n();
  max_stddev = max_stddev_fut.wait();
  SLOG("Activity delay overall avg=", avg_delay, " max=", max_delay, " bal=", avg_delay > 0 ? max_delay / avg_delay : 0.0,
       " avg stddev=", avg_stddev, " max stddev=", max_stddev, " bal=", avg_stddev > 0 ? max_stddev / avg_stddev : 0.0, "\n");

  auto msm = msm_fut.wait();
  int num_nodes = upcxx::rank_n() / upcxx::local_team().rank_n();
  SLOG("Peak memory used across all ", num_nodes, " nodes: ", get_size_str(msm.sum / upcxx::local_team().rank_n()), " (",
       get_size_str((double)msm.avg), " avg, ", get_size_str(msm.min), " min, ", get_size_str(msm.max), " max)\n");
  upcxx::barrier();
}

#endif  // yes UPCXX_UTILS_THREADS

};  // namespace upcxx_utils
