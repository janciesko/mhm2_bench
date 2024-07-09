// log.cpp

#include "upcxx_utils/log.hpp"

#include "upcxx_utils/log.h"

#ifdef __GNUC__
#include <execinfo.h>
#endif

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <string>
#ifndef UPCXX_UTILS_NO_THREADS
#include <thread>
#endif
#include <upcxx/upcxx.hpp>

#include "upcxx_utils/thread_pool.hpp"
#include "upcxx_utils/timers.hpp"

using upcxx::rank_me;
using upcxx::rank_n;

using std::cerr;
using std::cout;
using std::ifstream;
using std::ofstream;
using std::ostream;
using std::ostringstream;
using std::string;
using std::stringstream;
using std::to_string;

namespace upcxx_utils {

// the log files
ofstream _logstream;
ofstream _dbgstream;
std::list<std::shared_ptr<ofstream>> _dbgstream_list;
bool _verbose = true;
std::atomic<long> thread_messages(0), thread_logged(0);

void init_logger(string name, bool verbose, bool own_path) {
  assert(!_logstream.is_open());
  assert((!upcxx::initialized() || upcxx::master_persona().active_with_caller()) &&
         "init_logger must be called in the primary context");
  if (own_path)
    if (!get_rank_path(name, upcxx::initialized() ? upcxx::rank_me() : 0)) DIE("Could not get rank_path for: ", name, "\n");

  _verbose = verbose;
  bool old_file = file_exists(name);
  DBG("Opening ", name, " old_file=", old_file, " ", world_rank_me(), " of ", world_rank_n(), " local ", local_rank_me(), " of ",
      local_rank_n(), "\n");
  _logstream.open(name, std::ofstream::out | std::ofstream::app);
  if (!_logstream.is_open()) DIE("Could not open: ", name, "\n");
}

void flush_logger() {
  if (upcxx::initialized() && !upcxx::master_persona().active_with_caller()) {
    upcxx::master_persona().lpc_ff([]() { flush_logger(); });
    return;
  }

  if (upcxx::initialized() && !upcxx::in_progress() && _logstream.is_open()) {
    // quiesse for any outstanding lpc_ff messages or flushes
    auto ct = 0;
    while (thread_logged.load() != thread_messages.load()) {
      if (++ct % 10000 == 0) {
        WARN("Stopped waiting for all ", thread_messages.load(), " thread messages to be accounted:", thread_logged.load(), "\n");
        break;
      }
#ifndef UPCXX_UTILS_NO_THREADS
      std::this_thread::yield();
#endif
      upcxx::progress();
    }
  }

  if (_logstream.is_open()) _logstream.flush();
  if (_dbgstream.is_open()) _dbgstream.flush();
}

void close_logger() {
  assert((!upcxx::initialized() || upcxx::master_persona().active_with_caller()) &&
         "close_logger must be called in the primary context");
  if (_logstream.is_open()) {
    flush_logger();
    _logstream.close();
  }
  assert(!_logstream.is_open());
}

// for C interface
void init_logger_cxx(const char *_name, int verbose, int own_path) {
  string name(_name);
  return init_logger(name, verbose != 0, own_path != 0);
}
void flush_logger_cxx() { flush_logger(); }
void close_logger_cxx() { close_logger(); }

int world_rank_me() {
  static int _ = -1;
  if (_ == -1) {
    if (!upcxx::initialized()) return 0;  // allow before initialized, but do not cache
    _ = upcxx::world().rank_me();
  }
  return _;
}
int world_rank_n() {
  static int _ = -1;
  if (_ == -1) {
    if (!upcxx::initialized()) return 1;  // allow before initialized, but do not cache
    _ = upcxx::world().rank_n();
  }
  return _;
}

int local_rank_me() {
  static int _ = -1;
  if (_ == -1) {
    if (!upcxx::initialized()) return 0;  // allow before initialized, but do not cache
    _ = upcxx::local_team().rank_me();
  }
  return _;
}

int local_rank_n() {
  static int _ = -1;
  if (_ == -1) {
    if (!upcxx::initialized()) return 1;  // allow before initialized, but do not cache
    _ = upcxx::local_team().rank_n();
  }
  return _;
}

void open_dbg_cxx(const char *_name) {
  string name(_name);
  open_dbg(_name);
}
void open_dbg(string name) {
  assert((!upcxx::initialized() || upcxx::master_persona().active_with_caller()) &&
         "open_dbg must be called in the primary context");
  if (_dbgstream.is_open()) {
    auto tmp = std::make_shared<ofstream>();
    _dbgstream_list.push_back(tmp);
    _dbgstream.swap(*_dbgstream_list.back());
  }
  time_t curr_t = std::time(nullptr);
  string dbg_fname = name + "-" + to_string(curr_t) + ".dbg";  // never in cached_io
  get_rank_path(dbg_fname, rank_me());
  _dbgstream.open(dbg_fname);
  SOUT("Opened debug log file:", dbg_fname, "\n");
  char hnbuf[64];
  gethostname(hnbuf, sizeof(hnbuf) - 1);
  DBG("Opened debug log: ", dbg_fname, " using ", UPCXX_UTILS_VERSION_DATE, " on branch ", UPCXX_UTILS_BRANCH, " on ", hnbuf,
      " pid=", getpid(), " world=", upcxx::world().rank_me(), "of", upcxx::world().rank_n(),
      " local=", upcxx::local_team().rank_me(), "of", upcxx::local_team().rank_n(), "\n");
}

int close_dbg() {
  assert((!upcxx::initialized() || upcxx::master_persona().active_with_caller()) &&
         "close_dbg must be called in the primary context");
  if (_dbgstream.is_open()) {
    DBG("Closing this debug log.\n");
    _dbgstream.flush();
    _dbgstream.close();
  }
  assert(!_dbgstream.is_open());
  if (!_dbgstream_list.empty()) {
    _dbgstream.swap(*_dbgstream_list.back());
    _dbgstream_list.pop_back();
    return 1;
  }
  return 0;
}

// for C interface
void open_dbg_dxx(const char *_name) {
  string name(_name);
  return open_dbg(name);
}
void close_dbg_cxx() { close_dbg(); }

int log_try_catch_main_cxx(int argc, char **argv, int (*main_pfunc)(int, char **)) {
  std::function<int(int, char **)> func = main_pfunc;
  return log_try_catch_function_wrapper(func, argc, argv);
}

ostream &_logger_write(ostream &os, string str) {
  if (upcxx::initialized() && !upcxx::master_persona().active_with_caller()) {
    // This is a different thread and requires a lpc on the master persona
    // in order to write to file handles without rare races and crashes
    long messageid = thread_messages.fetch_add(1);
#ifndef UPCXX_UTILS_NO_THREADS
    std::ostringstream oss;
    oss << "T" << std::this_thread::get_id() << "(";
#if __GLIBC_MINOR__ >= 30
    oss << gettid() << ";";
#endif
    oss << getpid() << ":" << messageid << ") " << str;
    str = oss.str();
#endif
    upcxx::master_persona().lpc_ff([&os, str]() {
      assert(upcxx::initialized && upcxx::master_persona().active_with_caller());
      _logger_write(os, str);
      thread_logged++;
    });
    return os;
  }
  bool is_screen = false;
  if (os.rdbuf() == std::cout.rdbuf() || os.rdbuf() == std::cerr.rdbuf()) {
    // print raw to stdout/err
    os << str;
    is_screen = true;
  }
#ifdef CONFIG_USE_COLORS
  if ((is_screen && (_dbgstream.is_open() || _logstream.is_open())) || !is_screen) {
    // strip off colors for log file
    for (auto c : COLORS) {
      find_and_replace(str, c, "");
    }
  }
#endif
  static string timestamp_prefix;
  if (timestamp_prefix.empty()) {
    timestamp_prefix = _logger_timestamp().substr(0, 8);
  }
  if (str.compare(0, 8, timestamp_prefix) != 0) {
    // prepend the timestamp to the following log file(s)
    str = _logger_timestamp() + str;
  }
  if (is_screen) {
    // echo to the log / debug file(s) -- will be noop if they are not open
    if (_logstream.is_open()) {
      _logstream << str;
    }
    if (_dbgstream.is_open()) {
      _dbgstream << str;
      _dbgstream.flush();
    }
  } else {
    // is file
    os << str;
    if (_dbgstream.is_open() && os.rdbuf() == _logstream.rdbuf()) {
      // echo log to debug too
      _dbgstream << str;
      _dbgstream.flush();
    }
  }
  return os;
}

string _logger_timestamp() {
  std::time_t result = std::time(nullptr);
  char buffer[64];
  buffer[0] = '\0';
  struct tm tmp;
  size_t sz = strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S ", localtime_r(&result, &tmp));
  return string(sz > 0 ? buffer : "BAD TIME ");
}

void net_log(std::ofstream &ofs, const std::string &msg) {
  // relay a message up the ranks until an open log file is found
  if (ofs.is_open()) {
    ofs << msg;
  } else {
    upcxx::intrank_t uprank = local_rank_me() == 0 ? 0 : world_rank_me() - local_rank_me();
    if (world_rank_me() != uprank) {
      upcxx::rpc_ff(
          uprank,
          [](const std::string msg) {
            // call recursively
            upcxx_utils::net_log(LOG_OR_DBG_STREAM, msg);
          },
          msg);
    }
  }
}

// LoggedError methods
LoggedError::LoggedError(const char *msg)
    : runtime_error(msg) {
  flush_logs("\n", this->what(), "\n\n");
  log_stacktrace();
}

LoggedError::LoggedError(const exception &e)
    : runtime_error(e.what()) {
  flush_logs("\n", this->what(), "\n\n");
  log_stacktrace();
}

void LoggedError::log_exception(const exception &e, int level) {
  flush_logs("Exception: level=", level, "\t", e.what(), "\n");
  try {
    std::rethrow_if_nested(e);
    log_stacktrace();
  } catch (const exception &e) {
    log_exception(e, level + 1);
  } catch (...) {
  }
}

void LoggedError::flush_logs() {
  if (_logstream.is_open()) _logstream.flush();
  if (_dbgstream.is_open()) _dbgstream.flush();
}

void LoggedError::log_stacktrace() {
#ifdef __GNUC__
  const int stack_size = 60;
  void *buffers[stack_size];
  int nptrs = backtrace(buffers, stack_size);
  if (nptrs <= 0 && _logstream.is_open()) _logstream << "Empty stack trace\n";
  char **strings = backtrace_symbols(buffers, nptrs);
  if (strings == NULL) return;
  ostringstream os;
  for (int i = 0; i < nptrs; i++) {
    os << strings[i] << "\n";
  }
  flush_logs("StackTrace:\n", os.str(), "\n");
  free(strings);
#else
  if (_logstream.is_open()) _logstream << "No stack trace available\n";
  if (_dbgstream.is_open()) _dbgstream << "No stack trace available\n";
#endif
}

//
// file path methods
//

bool file_exists(const string &fname) {
  ifstream ifs(fname, std::ios_base::binary);
  return ifs.is_open();
}

void check_file_exists(const string &filename) {
  auto fnames = split(filename, ',');
  for (auto fname : fnames) {
    ifstream ifs(fname);
    if (!ifs.is_open()) SDIE("File ", fname, " cannot be accessed: ", strerror(errno), "\n");
  }
  upcxx_utils::ThreadPool::barrier();
}

// returns 1 when it created the directory, 0 otherwise, -1 if there is an error
int check_dir(const char *path) {
  if (0 != access(path, F_OK)) {
    if (ENOENT == errno) {
      // does not exist
      // note: we make the directory to be world writable, so others can delete it later if we
      // crash to avoid cluttering up memory
      mode_t oldumask = umask(0000);
      if (0 != mkdir(path, 0777) && 0 != access(path, F_OK)) {
        umask(oldumask);
        fprintf(stderr, "Could not create the (missing) directory: %s (%s)", path, strerror(errno));
        return -1;
      }
      umask(oldumask);
    }
    if (ENOTDIR == errno) {
      // not a directory
      fprintf(stderr, "Expected %s was a directory!", path);
      return -1;
    }
  } else {
    return 0;
  }
  return 1;
}

// replaces the given path with a rank based path, inserting a rank-based directory
// example:  get_rank_path("path/to/file_output_data.txt", rank) -> "path/to/per_rank/<rankdir>/<rank>/file_output_data.txt"
// of if rank == -1, "path/to/per_rank/file_output_data.txt"
bool get_rank_path(string &fname, int rank) {
  char buf[MAX_FILE_PATH];
  strcpy(buf, fname.c_str());
  int pathlen = strlen(buf);
  char newPath[MAX_FILE_PATH * 2 + 50];
  char *lastslash = strrchr(buf, '/');
  int checkDirs = 0;
  int thisDir;
  char *lastdir = NULL;

  if (pathlen + 25 >= MAX_FILE_PATH) {
    WARN("File path is too long (max: ", MAX_FILE_PATH, "): ", buf, "\n");
    return false;
  }
  if (lastslash) {
    *lastslash = '\0';
  }
  if (rank < 0) {
    if (lastslash) {
      snprintf(newPath, MAX_FILE_PATH * 2 + 50, "%s/per_rank/%s", buf, lastslash + 1);
      checkDirs = 1;
    } else {
      snprintf(newPath, MAX_FILE_PATH * 2 + 50, "per_rank/%s", buf);
      checkDirs = 1;
    }
  } else {
    if (lastslash) {
      snprintf(newPath, MAX_FILE_PATH * 2 + 50, "%s/per_rank/%08d/%08d/%s", buf, rank / MAX_RANKS_PER_DIR, rank, lastslash + 1);
      checkDirs = 3;
    } else {
      snprintf(newPath, MAX_FILE_PATH * 2 + 50, "per_rank/%08d/%08d/%s", rank / MAX_RANKS_PER_DIR, rank, buf);
      checkDirs = 3;
    }
  }
  strcpy(buf, newPath);
  while (checkDirs > 0) {
    strcpy(newPath, buf);
    thisDir = checkDirs;
    while (thisDir--) {
      lastdir = strrchr(newPath, '/');
      if (!lastdir) {
        WARN("What is happening here?!?!\n");
        return false;
      }
      *lastdir = '\0';
    }
    check_dir(newPath);
    checkDirs--;
  }
  fname = buf;
  return true;
}

std::vector<string> find_rank_files(string &fname_list, const string &ext, bool cached_io, const string local_tmp_dir) {
  std::vector<string> full_fnames;
  auto fnames = split(fname_list, ',');
  for (auto fname : fnames) {
    if (cached_io) fname = local_tmp_dir + "/" + fname;
    // first check for gzip file
    fname += ext;
    get_rank_path(fname, upcxx::rank_me());
    string gz_fname = fname + ".gz";
    struct stat stbuf;
    if (stat(gz_fname.c_str(), &stbuf) == 0) {
      // gzip file exists
      SOUT("Found compressed file '", gz_fname, "'\n");
      fname = gz_fname;
    } else {
      // no gz file - look for plain file
      if (stat(fname.c_str(), &stbuf) != 0)
        SDIE("File '", fname, "' cannot be accessed (either .gz or not): ", strerror(errno), "\n");
    }
    full_fnames.push_back(fname);
  }
  return full_fnames;
}

string remove_file_ext(const string &fname) {
  size_t lastdot = fname.find_last_of(".");
  if (lastdot == std::string::npos) return fname;
  return fname.substr(0, lastdot);
}

string get_basename(const string &fname) {
  size_t i = fname.find_last_of('/');
  if (i != string::npos) return fname.substr(i + 1);
  return fname;
}

int64_t get_file_size(string fname) {
  struct stat s;
  if (stat(fname.c_str(), &s) != 0) return -1;
  return s.st_size;
}

int64_t get_file_size(int fd) {
  struct stat s;
  if (fstat(fd, &s) != 0) return -1;
  return s.st_size;
}

int64_t get_file_size(FILE *f) {
  fseek(f, 0, SEEK_END);
  auto size = ftell(f);
  fseek(f, 0, SEEK_SET);
  return size;
}

int64_t get_file_size(ifstream &inf) {
  auto pos = inf.tellg();
  inf.seekg(0, std::ios_base::end);
  auto len = inf.tellg();
  inf.seekg(pos);
  return len;
}

//
// formatting methods
//
string get_size_str(int64_t sz) {
  int64_t absz = llabs(sz);
  double dsize = sz;
  ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  if (absz >= ONE_EB)
    oss << (dsize / ONE_EB) << "EB";
  else if (absz >= ONE_TB)
    oss << (dsize / ONE_TB) << "TB";
  else if (absz >= ONE_GB)
    oss << (dsize / ONE_GB) << "GB";
  else if (absz >= ONE_MB)
    oss << (dsize / ONE_MB) << "MB";
  else if (absz >= ONE_KB)
    oss << (dsize / ONE_KB) << "KB";
  else
    oss << absz << "B";
  return oss.str();
}

string get_float_str(double fraction, int precision) {
  std::stringstream ss;
  ss << std::setprecision(precision) << std::fixed << fraction;
  return ss.str();
}

string perc_str(int64_t num, int64_t tot) {
  ostringstream os;
  os.precision(2);
  os << std::fixed;
  os << num << " (" << (tot == 0 ? 0.0 : (100.0 * num / tot)) << "%)";
  return os.str();
}

string get_current_time(bool fname_fmt) {
  auto t = std::time(nullptr);
  std::ostringstream os;
  struct tm tmp;
  if (!fname_fmt)
    os << std::put_time(localtime_r(&t, &tmp), "%D %T");
  else
    os << std::put_time(localtime_r(&t, &tmp), "%y%m%d%H%M%S");
  return os.str();
}

vector<string> split(const string &s, char delim) {
  std::vector<string> elems;
  std::stringstream ss(s);
  string token;
  while (std::getline(ss, token, delim)) elems.push_back(token);
  return elems;
}

void find_and_replace(std::string &subject, const std::string &search, const std::string &replace) {
  size_t pos = 0;
  while ((pos = subject.find(search, pos)) != std::string::npos) {
    subject.replace(pos, search.length(), replace);
    pos += replace.length();
  }
}

std::string_view substr_view(const std::string &s, size_t from, size_t len) {
  if (from >= s.size()) return {};
  return std::string_view(s.data() + from, std::min(s.size() - from, len));
}

void replace_spaces(string &s) {
  for (int i = 0; i < s.size(); i++)
    if (s[i] == ' ') s[i] = '_';
}

string tail(const string &s, int n) { return s.substr(s.size() - n); }

string head(const string &s, int n) { return s.substr(0, n); }

const upcxx::team &barrier_wrapper(const upcxx::team &team, bool wait_pending) {
  if (wait_pending && team.id() == upcxx::world().id()) Timings::wait_pending();
  upcxx_utils::ThreadPool::barrier(team);
  return team;
}

void wait_wrapper(upcxx::future<> fut) { upcxx_utils::ThreadPool::wait(fut); }

};  // namespace upcxx_utils
