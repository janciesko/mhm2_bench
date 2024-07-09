#include "upcxx_utils/progress_bar.hpp"

#include <sys/stat.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <upcxx/upcxx.hpp>

#include "upcxx_utils/colors.h"
#include "upcxx_utils/log.hpp"
#include "upcxx_utils/mem_profile.hpp"
#include "upcxx_utils/promise_collectives.hpp"
#include "upcxx_utils/timers.hpp"

using upcxx::future;

namespace upcxx_utils {

ProgressBar::ProgressBar(int64_t total, string prefix, int pwidth, int width, char complete, char incomplete)
    : total_ticks{total}
    , prefix_str{prefix}
    , prefix_width{pwidth}
    , bar_width{width}
    , complete_char{complete}
    , incomplete_char{incomplete}
    , is_done{false} {
  upcxx::progress();
  if (upcxx::rank_me() != RANK_FOR_PROGRESS) return;
  ten_perc = total / 10;
  if (ten_perc == 0) ten_perc = 1;
  if (ProgressBar::SHOW_PROGRESS) SOUT(KLGREEN, "* ", prefix_str, "... ", KNORM, "\n");
  prev_time = start_time;
  LOG("ProgressBar ", prefix_str, " ", total, "\n");
}

ProgressBar::ProgressBar(std::ifstream *infile, string prefix, bool one_file_per_rank, int pwidth, int width, char complete,
                         char incomplete)
    : infile{infile}
    , prefix_str{prefix}
    , prefix_width{pwidth}
    , bar_width{width}
    , complete_char{complete}
    , incomplete_char{incomplete} {
  upcxx::progress();
  if (upcxx::rank_me() != RANK_FOR_PROGRESS) return;
  auto num_ranks = one_file_per_rank ? 1 : upcxx::rank_n();
  infile->seekg(0, infile->end);
  total_ticks = infile->tellg() / num_ranks;
  infile->seekg(0);
  ten_perc = total_ticks / 10;
  if (ten_perc == 0) ten_perc = 1;
  ticks = 0;
  prev_ticks = ticks;
  if (ProgressBar::SHOW_PROGRESS) {
    std::ostringstream oss;
    oss << KLGREEN << std::setw(prefix_width) << std::left << prefix_str << " " << std::flush << std::endl;
    SOUT(oss.str());
  }
  LOG("ProgressBar ", prefix_str, "\n");
}

ProgressBar::ProgressBar(const string &fname, istream *infile_, string prefix_, int pwidth, int width, char complete,
                         char incomplete)
    : infile{infile_}
    , total_ticks{0}
    , prefix_str{prefix_}
    , prefix_width{pwidth}
    , bar_width{width}
    , complete_char{complete}
    , incomplete_char{incomplete}
    , is_done{false} {
  upcxx::progress();
  if (upcxx::rank_me() != RANK_FOR_PROGRESS) return;
  int64_t sz = get_file_size(fname);
  if (sz < 0) WARN("Could not read the file size for: ", fname);
  total_ticks = sz;
  ten_perc = total_ticks / 10;
  if (ten_perc == 0) ten_perc = 1;
  ticks = 0;
  prev_ticks = ticks;
  if (ProgressBar::SHOW_PROGRESS)
    SOUT(KLGREEN, "* ", prefix_str, " (", fname.substr(fname.find_last_of("/\\") + 1), " ", get_size_str(sz), ")...", KNORM, "\n");
  LOG("ProgressBar ", prefix_str, " (", fname.substr(fname.find_last_of("/\\") + 1), " ", get_size_str(sz), "\n");
  prev_time = start_time;
}

ProgressBar::~ProgressBar() {
  if (!is_done) done();
}

void ProgressBar::display(bool is_last) {
  if (upcxx::rank_me() != RANK_FOR_PROGRESS) return;
  if (total_ticks == 0) return;
  float progress = (float)ticks / total_ticks;
  std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
  auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
  auto time_delta = std::chrono::duration_cast<std::chrono::milliseconds>(now - prev_time).count();
  prev_time = now;
  if (ProgressBar::SHOW_PROGRESS) {
    std::cout << std::setprecision(2) << std::fixed;
    std::cout << KLGREEN << "  " << int(progress * 100.0) << "% " << (float(time_elapsed) / 1000.0) << "s "
              << (float(time_delta) / 1000.0) << "s " << get_size_str(get_free_mem()) << KNORM << std::endl;
  }
}

future<> ProgressBar::set_done(bool wait_pending) {
  assert(!upcxx::in_progress());
  auto &pr = Timings::get_promise_reduce();
  is_done = true;
  std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
  double time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() / 1000.;
  auto msm_fut = pr.msm_reduce_one(time_elapsed);
  LOG("ProgressBar ", prefix_str, " done in ", time_elapsed, " s.\n");
  // print once all have completed, but do not create an implicit barrier
  auto fut_report = msm_fut.then([prefix_str = this->prefix_str](const auto &msm) {
    if (upcxx::rank_me() == RANK_FOR_PROGRESS) {
      double av_time = msm.sum / upcxx::rank_n();
      stringstream ss;
      ss << std::setprecision(2) << std::fixed;
      ss << KLGREEN << "  min " << msm.min << " Average " << av_time << " max " << msm.max << " (balance "
         << (msm.max == 0.0 ? 1.0 : (av_time / msm.max)) << ")" << KNORM << std::endl;
      if (ProgressBar::SHOW_PROGRESS) SOUT(ss.str());
      LOG("ProgressBar ", prefix_str, " done. ", msm.to_string(), "\n");
    }
  });
  Timings::set_pending(fut_report);
  if (wait_pending) Timings::wait_pending();
  return fut_report;
}

void ProgressBar::done() { set_done(true).wait(); }

bool ProgressBar::update(int64_t new_ticks) {
  if (!SHOW_PROGRESS) return false;
  if (total_ticks == 0) return false;
  if (new_ticks != -1)
    ticks = new_ticks;
  else if (infile)
    ticks = infile->tellg();
  else
    ticks++;
  if (ticks - prev_ticks > ten_perc) {
    display();
    prev_ticks = ticks;
    return true;
  }
  return false;
}

// set static variables
bool ProgressBar::SHOW_PROGRESS = false;
int ProgressBar::RANK_FOR_PROGRESS = 0;

};  // namespace upcxx_utils
