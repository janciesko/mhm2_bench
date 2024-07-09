#include <upcxx/upcxx.hpp>
#ifndef UPCXX_UTILS_NO_THREADS
#include <thread>
#endif

#include "upcxx_utils/log.hpp"

std::string &left_trim(std::string &str) {
  auto it = std::find_if(str.begin(), str.end(), [](char ch) { return !std::isspace<char>(ch, std::locale()); });
  str.erase(str.begin(), it);
  return str;
}

string get_proc_pin() {
  ifstream f("/proc/self/status");
  string line;
  string prefix = "Cpus_allowed_list:";
  while (getline(f, line)) {
    if (line.substr(0, prefix.length()) == prefix) {
      DBG(line, "\n");
      line = line.substr(prefix.length(), line.length() - prefix.length());
      return left_trim(line);
    }
  }
  return "";
}

int mt_log() {
#ifndef UPCXX_UTILS_NO_THREADS
  SOUT("Starting mt_log\n");
  upcxx::barrier();
  const int num_threads = 10;
  std::thread threads[num_threads]{};
  OUT("Rank ", upcxx::rank_me(), " starting all threads\n");
  for (int i = 0; i < num_threads; i++) {
    threads[i] = std::thread([&, i] { OUT("From thread ", i, " rank ", upcxx::rank_me(), " of ", upcxx::rank_n(), "\n"); });
  }
  upcxx::progress();
  OUT("Rank ", upcxx::rank_me(), " started all threads\n");
  for (int i = 0; i < num_threads; i++) {
    threads[i].join();
  }
  OUT("Rank ", upcxx::rank_me(), " finished all threads\n");
  upcxx::barrier();
  upcxx::discharge();
  upcxx::barrier();
  SOUT("Finished mt_log\n");
#endif
  return 1;
}

int test_log(int argc, char **argv) {
  upcxx_utils::open_dbg("test_log");

  OUT("Success: ", upcxx::rank_me(), " of ", upcxx::rank_n(), " pin:", get_proc_pin(), "\n");
  upcxx::barrier();

  mt_log();

  SOUT("Done\n");

  upcxx_utils::close_dbg();

  return 0;
}
