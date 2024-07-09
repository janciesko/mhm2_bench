#pragma once

#include <cassert>
#include <climits>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <upcxx/upcxx.hpp>
#include <vector>

#include "colors.h"
#include "version.h"
#include "log.h"

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define __FILEFUNC__ (__FILENAME__ + string(":") + __func__)

#define ONE_B (1LL)
#define ONE_KB (1024LL)
#define ONE_MB (ONE_KB * 1024LL)
#define ONE_GB (ONE_MB * 1024LL)
#define ONE_TB (ONE_GB * 1024LL)
#define ONE_EB (ONE_TB * 1024LL)

#define CLOCK_NOW std::chrono::high_resolution_clock::now

using upcxx::local_team;
using upcxx::rank_me;
using upcxx::rank_n;

using std::cerr;
using std::cout;
using std::exception;
using std::ifstream;
using std::ofstream;
using std::ostream;
using std::ostringstream;
using std::runtime_error;
using std::string;
using std::stringstream;
using std::vector;

namespace upcxx_utils {

// the log files as externs
extern ofstream _logstream;
extern ofstream _dbgstream;
extern bool _verbose;

// public methods to open loggers

// see log.h for "C" exposed init_logger_cxx, flush_logger_cxx and close_logger_cxx header functions
// if own_path==true, it will be in its own per_rank subdir (like open_dbg below)
void init_logger(string name, bool verbose, bool own_path = true);
void flush_logger();
void close_logger();

// void init_logger_cxx(const char *, int verbose, int own_path); // see log.h
// void flush_logger_cxx(); // see log.h
// void close_logger_cxx(); // see log.h

// dbg loggers are meant to be open and closed by each module
// every rank opens its own file, and name will append a '-timestamp.dbg'
// and place it in the per_rank directory so name can be the same across all ranks
// only one dbg log is active at a time
// if a dbg log is already open, it will pause and stack on another open_dbg and will be restored on close_dbg
void open_dbg(string name);
int close_dbg();
// void open_dbg_cxx(const char *name); // see log.h
// void close_dbg_cxx(); // see log.h

// method to write to a stream and/or logs
ostream &_logger_write(ostream &os, string str);

string _logger_timestamp();

// std::cout for node 0, log for others...
#define INFO_OR_LOG_STREAM                                      \
  (upcxx_utils::world_rank_me() < upcxx_utils::local_rank_n() ? \
       std::cout :                                              \
       (upcxx_utils::_logstream.is_open() ? upcxx_utils::_logstream : upcxx_utils::_dbgstream))
#define LOG_OR_DBG_STREAM (upcxx_utils::_logstream.is_open() ? upcxx_utils::_logstream : upcxx_utils::_dbgstream)
#define DBG_OR_LOG_STREAM (upcxx_utils::_dbgstream.is_open() ? upcxx_utils::_dbgstream : upcxx_utils::_logstream)

#define LOG_LINE_LABEL "[", upcxx_utils::world_rank_me(), "] <", __FILENAME__, ":", __LINE__, "> "
#define LOG_LINE_TS_LABEL upcxx_utils::_logger_timestamp(), LOG_LINE_LABEL

// last in list is a noop
inline void _logger_recurse(ostream &os) {}

// log the next item in a list
template <typename T, typename... Params>
inline void _logger_recurse(ostream &os, const T &first, const Params &...params) {
  os << first;
  _logger_recurse(os, params...);
}

class LoggedError : public runtime_error {
 public:
  LoggedError(const char *msg);
  LoggedError(const exception &e);

  template <typename T>
  LoggedError(const T &what_arg)
      : runtime_error(what_arg) {
    flush_logs("\n", this->what(), "\n\n");
    log_stacktrace();
  };

  static void flush_logs();
  template <typename... Params>
  static void flush_logs(const Params &...params) {
    if (_logstream.is_open()) {
      upcxx_utils::_logger_recurse(_logstream, params...);
    }
    if (_dbgstream.is_open()) {
      upcxx_utils::_logger_recurse(_dbgstream, params...);
    }
    flush_logs();
  }

  static void log_exception(const exception &e, int level = 0);
  static void log_stacktrace();
};

// initial log line

template <typename... Params>
void logger(ostream &stream, bool fail, bool serial, bool flush, const Params &...params) {
  if (!stream.good()) return;
  ofstream *ofstream_ptr = dynamic_cast<ofstream *>(&stream);
  if (ofstream_ptr != nullptr) {
    if (!ofstream_ptr->is_open()) return;
  }
  if (serial && upcxx_utils::world_rank_me()) return;

  ostringstream os;
  _logger_recurse(os, params...);  // recurse through remaining parameters
  string outstr = os.str();
  // don't need to write on fail because this will be thrown
  if (!fail) {
    _logger_write(stream, outstr);
    if (flush) stream.flush();
  } else if (LOG_OR_DBG_STREAM) {
    // fail! Write the error to the log and/or dbg stream(s) and flush
    _logger_write(LOG_OR_DBG_STREAM, outstr);
    if (_logstream.is_open()) _logstream.flush();
    if (_dbgstream.is_open()) _dbgstream.flush();
  }
  if (fail) {
    std::cerr << outstr << std::flush;
    if (stream.rdbuf() != std::cout.rdbuf() && stream.rdbuf() != std::cerr.rdbuf()) _logger_write(stream, outstr).flush();
    ::upcxx_utils::LoggedError::flush_logs();
    std::abort();  // throw std::runtime_error(outstr);
    // do not throw exceptions -- does not work properly within progress() throw std::runtime_error(outstr);
  }
}

#define LOG_THROW_EXCEPTION(...)                                       \
  do {                                                                 \
    std::ostringstream os;                                             \
    ::upcxx_utils::_logger_recurse(os, LOG_LINE_LABEL, ##__VA_ARGS__); \
    ::upcxx_utils::LoggedError::flush_logs();                          \
    std::throw_with_nested(upcxx_utils::LoggedError(os.str()));        \
  } while (0)

#define LOG_TRY_CATCH(...)                                                                            \
  try {                                                                                               \
    ::upcxx_utils::LoggedError::flush_logs(LOG_LINE_TS_LABEL, "LOG_TRY_CATCH: ", #__VA_ARGS__, "\n"); \
    __VA_ARGS__;                                                                                      \
    ::upcxx_utils::LoggedError::flush_logs();                                                         \
  } catch (const exception &e) {                                                                      \
    LOG_THROW_EXCEPTION(e.what());                                                                    \
  } catch (const char *msg) {                                                                         \
    LOG_THROW_EXCEPTION(msg);                                                                         \
  } catch (...) {                                                                                     \
    LOG_THROW_EXCEPTION("UNKNOWN ERROR");                                                             \
  }

template <typename Ret, typename... Args>
Ret log_try_catch_function_wrapper(std::function<Ret(Args...)> wrapped_func, const Args &...args) {
  LOG_TRY_CATCH(return wrapped_func(args...););
}

// relay a message up the ranks until an open log file is found
void net_log(ofstream &ofs, const std::string &msg);

};  // namespace upcxx_utils

// rank0 to stdout and log/dbg if open
#define SOUT(...)                                                     \
  do {                                                                \
    upcxx_utils::logger(std::cout, false, true, true, ##__VA_ARGS__); \
  } while (0)

// any to stdout (take care) and log/dbg if open
#define OUT(...)                                                       \
  do {                                                                 \
    upcxx_utils::logger(std::cout, false, false, true, ##__VA_ARGS__); \
  } while (0)

// any with timestamp to stdout (take care)
#define INFO(...)                                                                         \
  do {                                                                                    \
    upcxx_utils::logger(std::cout, false, false, true, LOG_LINE_TS_LABEL, ##__VA_ARGS__); \
  } while (0)

// any to logfile (if open)
#define LOG(...)                                                                                                               \
  do {                                                                                                                         \
    upcxx_utils::logger(LOG_OR_DBG_STREAM, false, false, upcxx_utils::local_rank_me() == 0, LOG_LINE_TS_LABEL, ##__VA_ARGS__); \
  } while (0)

// any rank to its logfile (if open, if not not open, relay to a lower rank with an open logfile)
// always evaluated
#define NET_LOG(...)                                                      \
  do {                                                                    \
    ostringstream _oss;                                                   \
    upcxx_utils::_logger_recurse(_oss, LOG_LINE_TS_LABEL, ##__VA_ARGS__); \
    upcxx_utils::net_log(LOG_OR_DBG_STREAM, _oss.str());                  \
  } while (0)

// any to stdout (if first node) to log otherwise
#define INFO_OR_LOG(...)                                                                                                        \
  do {                                                                                                                          \
    upcxx_utils::logger(INFO_OR_LOG_STREAM, false, false, upcxx_utils::local_rank_me() == 0, LOG_LINE_TS_LABEL, ##__VA_ARGS__); \
  } while (0)

// any to dbgfile if open or logfile if open
#define DBGLOG(...)                                                                                \
  do {                                                                                             \
    upcxx_utils::logger(DBG_OR_LOG_STREAM, false, false, false, LOG_LINE_TS_LABEL, ##__VA_ARGS__); \
  } while (0)

// rank0 to log (if open)
#define SLOG(...)                                                     \
  do {                                                                \
    upcxx_utils::logger(std::cout, false, true, true, ##__VA_ARGS__); \
  } while (0)

// rank0 to logfile and if _verbose also to stdout
#define SLOG_VERBOSE(...)                                                                                         \
  do {                                                                                                            \
    upcxx_utils::logger(upcxx_utils::_verbose ? std::cout : LOG_OR_DBG_STREAM, false, true, true, ##__VA_ARGS__); \
  } while (0)

// extra new lines around errors and warnings for readability and do not color the arguments as it can lead to terminal color leaks
#define WARN(...)                                                                                                         \
  do {                                                                                                                    \
    upcxx_utils::logger(std::cerr, false, false, true, KRED, LOG_LINE_TS_LABEL, "WARNING: ", KNORM, ##__VA_ARGS__, "\n"); \
  } while (0)

// warn but only from rank 0
#define SWARN(...)                                                                                                       \
  do {                                                                                                                   \
    upcxx_utils::logger(std::cerr, false, true, true, KRED, LOG_LINE_TS_LABEL, "WARNING: ", KNORM, ##__VA_ARGS__, "\n"); \
  } while (0)

#define DIE(...)                                                                                                     \
  do {                                                                                                               \
    upcxx_utils::logger(std::cerr, true, false, true, KLRED, LOG_LINE_LABEL, "ERROR: ", KNORM, ##__VA_ARGS__, "\n"); \
  } while (0)

// die but only from rank0
#define SDIE(...)                                                                                                   \
  do {                                                                                                              \
    upcxx_utils::logger(std::cerr, true, true, true, KLRED, LOG_LINE_LABEL, "ERROR: ", KNORM, ##__VA_ARGS__, "\n"); \
  } while (0)

#if defined(DEBUG) && !defined(NO_DBG_LOGS)
// any rank writes to its dbg log file, if available
#define DBG(...)                                                                                                           \
  do {                                                                                                                     \
    if (upcxx_utils::_dbgstream.is_open()) {                                                                               \
      upcxx_utils::logger(upcxx_utils::_dbgstream, false, false, true, LOG_LINE_TS_LABEL, __func__, " - ", ##__VA_ARGS__); \
    }                                                                                                                      \
  } while (0)
#define DBG_CONT(...)                                                                  \
  do {                                                                                 \
    if (upcxx_utils::_dbgstream.is_open()) {                                           \
      upcxx_utils::logger(upcxx_utils::_dbgstream, false, false, true, ##__VA_ARGS__); \
    }                                                                                  \
  } while (0)
#else
#define DBG(...)      /* noop */
#define DBG_CONT(...) /* noop */
#endif

// #define DBG_VERBOSE_LOGS // define to enable extremely verbose debug messages
// #define DBG_VERBOSE_LOGS2 // define to enable even more extremely verbose debug messages

#ifndef DBG_VERBOSE
#ifdef DBG_VERBOSE_LOGS
#define DBG_VERBOSE DBG  // DBG alias to enable extremely verbose debug messages
#else
#define DBG_VERBOSE(...)  // noop to disable extremely verbose debug messages
#endif
#endif

#ifdef DBG_VERBOSE_LOGS2
#define DBG_VERBOSE2 DBG_VERBOSE
#else
#define DBG_VERBOSE2(...)
#endif

//
// file path methods
//
#ifndef MAX_FILE_PATH
#define MAX_FILE_PATH PATH_MAX
#endif

#define MAX_RANKS_PER_DIR 1000

namespace upcxx_utils {

bool file_exists(const string &filename);

void check_file_exists(const string &filename);

// returns 1 when it created the directory, 0 otherwise, -1 if there is an error
int check_dir(const char *path);

// replaces the given path with a rank based path, inserting a rank-based directory
// example:  get_rank_path("path/to/file_output_data.txt", rank) -> "path/to/per_rank/<rankdir>/<rank>/file_output_data.txt"
// of if rank == -1, "path/to/per_rank/file_output_data.txt"
bool get_rank_path(string &fname, int rank);

std::vector<string> find_rank_files(string &fname_list, const string &ext, bool cached_io, const string local_tmp_dir = "/dev/shm");

string remove_file_ext(const string &fname);

string get_basename(const string &fname);

int64_t get_file_size(string fname);

int64_t get_file_size(int fd);

int64_t get_file_size(FILE *f);

int64_t get_file_size(ifstream &inf);

//
// formatting methods
//

string get_size_str(int64_t sz);

string get_float_str(double fraction, int precision = 3);

string perc_str(int64_t num, int64_t tot);

string get_current_time(bool fname_fmt = false);

vector<string> split(const string &s, char delim);

void find_and_replace(std::string &subject, const std::string &search, const std::string &replace);

std::string_view substr_view(const std::string &s, size_t from, size_t len = string::npos);

void replace_spaces(string &s);

string tail(const string &s, int n);

string head(const string &s, int n);

// a thread friendly barrier that fully discharges and does not require thread_pool.hpp to be loaded
const upcxx::team &barrier_wrapper(const upcxx::team &team = upcxx::world(), bool wait_pending = true);

// a thread friendly wait that does not require thread_pool.hpp to be loaded
void wait_wrapper(upcxx::future<> fut);

};  // namespace upcxx_utils
