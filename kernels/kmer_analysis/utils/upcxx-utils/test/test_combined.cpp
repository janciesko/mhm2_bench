/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <cassert>
#include <upcxx/upcxx.hpp>

int test_version(int argc, char **argv);
int test_thread_pool(int argc, char **argv);
int test_allocators(int argc, char **argv);
int test_binary_search(int argc, char **argv);
int test_flat_aggr_store(int argc, char **argv);
int test_gather(int argc, char **argv);
int test_limit_outstanding(int argc, char **argv);
int test_log(int argc, char **argv);
int test_ofstream(int argc, char **argv);
int test_progress_bar(int argc, char **argv);
int test_reduce_prefix(int argc, char **argv);
int test_shared_global_ptr(int argc, char **argv);
int test_split_rank(int argc, char **argv);
int test_three_tier_aggr_store(int argc, char **argv);
int test_timers(int argc, char **argv);
int test_upcxx_utils(int argc, char **argv);

void clear_all() {
  assert(!upcxx::progress_required());
  upcxx::barrier();
  // while(upcxx::progress_required()) upcxx::progress();
}
int test_combined(int argc, char **argv) {
  clear_all();

  int ret;

  ret = test_version(argc, argv);
  assert(ret == 0);
  clear_all();
  ret = test_thread_pool(argc, argv);
  assert(ret == 0);
  clear_all();
  ret = test_allocators(argc, argv);
  assert(ret == 0);
  clear_all();
  ret = test_binary_search(argc, argv);
  assert(ret == 0);
  clear_all();
  ret = test_flat_aggr_store(argc, argv);
  assert(ret == 0);
  clear_all();
  ret = test_limit_outstanding(argc, argv);
  assert(ret == 0);
  clear_all();
  ret = test_log(argc, argv);
  assert(ret == 0);
  clear_all();
  ret = test_ofstream(argc, argv);
  assert(ret == 0);
  clear_all();
  ret = test_progress_bar(argc, argv);
  assert(ret == 0);
  clear_all();
  ret = test_reduce_prefix(argc, argv);
  assert(ret == 0);
  clear_all();
  ret = test_shared_global_ptr(argc, argv);
  assert(ret == 0);
  clear_all();
  ret = test_split_rank(argc, argv);
  assert(ret == 0);
  clear_all();
  ret = test_three_tier_aggr_store(argc, argv);
  assert(ret == 0);
  clear_all();
  ret = test_timers(argc, argv);
  assert(ret == 0);
  clear_all();
  ret = test_upcxx_utils(argc, argv);
  assert(ret == 0);
  clear_all();

  return 0;
}
