#pragma once

#ifdef __cplusplus
extern "C" {
namespace upcxx_utils {
#endif

// logger is opened once per application, closed automatically
// typically only rank 0 writes to this, but any rank can open.
void init_logger_cxx(const char *name, int verbose, int own_path);
void flush_logger_cxx();
void close_logger_cxx();

void open_dbg_cxx(const char *name);
void close_dbg_cxx();

int world_rank_me();
int world_rank_n();
int local_rank_me();
int local_rank_n();

int log_try_catch_main_cxx(int argc, char **argv, int (*main_pfunc)(int, char **));

#ifdef __cplusplus
};  // namespace upcxx_utils
}  // extern "C"
#endif
