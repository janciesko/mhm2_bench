#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <upcxx/upcxx.hpp>

#include "version.h"

using std::istream;
using std::string;

namespace upcxx_utils {

class ProgressBar {
 private:
  int64_t ticks = 0;
  int64_t prev_ticks = 0;
  int64_t ten_perc = 0;
  int64_t total_ticks = 0;
  const int64_t bar_width;
  const int64_t prefix_width;
  const string prefix_str = "";
  const char complete_char = '=';
  const char incomplete_char = ' ';
  const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point prev_time;
  std::istream *infile = nullptr;
  bool is_done;

 public:
  static bool SHOW_PROGRESS;
  static int PERC_STEPS;
  static int MIN_SECS;
  static int RANK_FOR_PROGRESS;

  ProgressBar(int64_t total, string prefix = "", int pwidth = 20, int width = 50, char complete = '=', char incomplete = ' ');

  ProgressBar(std::ifstream *infile, string prefix = "", bool one_file_per_rank = false, int pwidth = 20, int width = 50,
              char complete = '=', char incomplete = ' ');

  ProgressBar(const string &fname, istream *infile, string prefix = "", int pwidth = 20, int width = 50, char complete = '=',
              char incomplete = ' ');

  virtual ~ProgressBar();

  void display(bool is_last = false);

  upcxx::future<> set_done(bool wait_pending = false);

  void done();

  bool update(int64_t new_ticks = -1);
};

};  // namespace upcxx_utils
