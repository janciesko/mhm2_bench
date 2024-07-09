#include <unistd.h>

#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <upcxx/upcxx.hpp>

#include "upcxx_utils/log.hpp"
#include "upcxx_utils/min_sum_max.hpp"
#include "upcxx_utils/ofstream.hpp"
#include "upcxx_utils/split_rank.hpp"
#include "upcxx_utils/thread_pool.hpp"
#include "upcxx_utils/version.h"

using namespace upcxx;
using namespace upcxx_utils;

void check_file_at(std::ifstream &in, size_t offset, const string &str, size_t base_offset = 0) {
  if (!in.is_open()) DIE("not open\n");
  if (str.size() == 0) return;
  DBG("Checking file at ", base_offset + offset, " for str ", str.size(), "\n");
  in.seekg(base_offset + offset);
  string fromfile;
  fromfile.resize(str.size() + 1);
  in.read(const_cast<char *>(fromfile.data()), str.size());
  if (in.fail()) DIE("Did not read ", str.size(), " bytes from file\n");
  fromfile.resize(str.size());
  if (str.compare(fromfile) != 0) DIE("compare failed at ", offset, " str='", str, "' fromfile='", fromfile, "'\n");
}

void check_ordered_file(string fname, const string mystr, uint64_t base_offset, const upcxx::team &team) {
  assert(!upcxx::in_progress());
  upcxx_utils::BarrierTimer bt(team, fname + " " + __FILEFUNC__);
  using DP = dist_object<promise<uint64_t> >;

  upcxx_utils::BaseTimer time_open("Opening " + fname + " to check");
  time_open.start();
  ifstream in(fname);
  int attempts = 0;
  while (++attempts < 130 && !in.is_open()) {
    // delay for eventually consistant file systems
    upcxx_utils::ThreadPool::sleep_ns(500000000);  // 1/2 sec
    in.close();
    in.open(fname);
  }
  if (!in.is_open())
    DIE("Could not open ", fname, " after ", attempts, " attempts over ", time_open.get_elapsed_since_start(), " s\n");
  time_open.stop();
  auto fut_log_open =
      upcxx_utils::min_sum_max_reduce_one(time_open.get_elapsed(), 0).then([&](upcxx_utils::MinSumMax<double> msm_open) {
        LOG("Opening of ", fname, ": ", msm_open.to_string(), "\n");
        if (rank_me() == 0 && msm_open.max > 1.0) WARN("Opening of ", fname, " took a while, but succeeded: ", msm_open.my, "\n");
      });
  DP dist_offset(team);

  auto fut = dist_offset->get_future().then([&in, &dist_offset, &mystr, &team](uint64_t offset) {
    DBG("Running offset=", offset, "\n");
    if (team.rank_me() + 1 < team.rank_n()) {
      rpc_ff(
          team, team.rank_me() + 1, [](DP &dp, size_t offset) { dp->fulfill_result(offset); }, dist_offset, offset + mystr.size());
    }
    check_file_at(in, offset, mystr);
  });
  if (team.rank_me() == 0) {
    dist_offset->fulfill_result(base_offset);
  }
  fut.wait();
  fut_log_open.wait();
}

int run_ofstream_test(string ofname, const upcxx::team &team, size_t block_size) {
  assert(!upcxx::in_progress());
  upcxx::barrier(world());
  barrier(team);
  DBG("Starting tests with ", ofname, " with block_size=", block_size, " team.rank_n=", team.rank_n(),
      " team.rank_me=", team.rank_me(), "\n");
  string mystr;
  {
    DBG("Opening ", ofname, "\n");
    upcxx_utils::dist_ofstream of(team, ofname, false, block_size);
    DBG("Opened ", ofname, "\n");
    for (int i = 0; i <= team.rank_me(); i++) {
      if (i) of << ", ";
      of << std::to_string(i);
    }
    of << "\n";
    mystr = of.str();
    DBG("Closing ", ofname, "\n");
    of.close();
    DBG("Implicit close and destroy out of scope\n");
  }
  check_ordered_file(ofname, mystr, 0, team);
  {
    DBG("Opening ", ofname, " again for append\n");
    upcxx_utils::dist_ofstream of(team, ofname, true, block_size);
    DBG("Opened ", ofname, "\n");
    of << mystr;
    DBG("Closing ", ofname, "\n");
    of.close();
    DBG("Closed ", ofname, "\n");
    auto s = upcxx::reduce_all(mystr.size(), upcxx::op_fast_add, team).wait();
    check_ordered_file(ofname, mystr, 0, team);
    check_ordered_file(ofname, mystr, s, team);
  }
  if (team.rank_me() == 0) {
    auto ret = unlink(ofname.c_str());
    if (ret != 0) DIE("After close, could not unlink ", ofname, " ", strerror(errno), "!");
  }
  barrier(team);

  ofname += ".2";
  future<> fut_close;
  {
    DBG("Opening ", ofname, "\n");
    upcxx_utils::dist_ofstream of(team, ofname, false, block_size);
    DBG("Opened ", ofname, "\n");
    for (int i = 0; i <= team.rank_me(); i += 2) {
      if (i) of << ", ";
      of << std::to_string(i);
    }
    of << "\n";
    mystr = of.str();
    DBG("Closing ", ofname, "\n");
    fut_close = of.close_async();
    DBG("Even lines\n");
  }
  fut_close.wait();
  check_ordered_file(ofname, mystr, 0, team);
  {
    DBG("Opening ", ofname, " again for append\n");
    upcxx_utils::dist_ofstream of(team, ofname, true, block_size);
    DBG("Opened ", ofname, "\n");
    of << mystr;
    DBG("Closing ", ofname, "\n");
    of.close();
    auto s = upcxx::reduce_all(mystr.size(), upcxx::op_fast_add, team).wait();
    check_ordered_file(ofname, mystr, 0, team);
    check_ordered_file(ofname, mystr, s, team);
  }
  if (team.rank_me() == 0) {
    auto ret = unlink(ofname.c_str());
    if (ret != 0) DIE("After close, could not unlink ", ofname, " ", strerror(errno), "!");
  }
  barrier(team);

  ofname += ".3";
  {
    DBG("Opening ", ofname, "\n");
    upcxx_utils::dist_ofstream of(team, ofname, false, block_size);
    DBG("Opened ", ofname, "\n");
    for (int i = 1; i <= team.rank_me(); i += 2) {
      if (i) of << ", ";
      of << std::to_string(i);
    }
    of << "\n";
    mystr = of.str();
    DBG("Closing ", ofname, "\n");
    of.close();
    DBG("Odd lines\n");
  }
  check_ordered_file(ofname, mystr, 0, team);
  {
    DBG("Opening ", ofname, " again for append\n");
    upcxx_utils::dist_ofstream of(team, ofname, true, block_size);
    DBG("Opened ", ofname, "\n");
    of << mystr;
    DBG("Closing ", ofname, "\n");
    of.close();
    DBG("Closed ", ofname, "\n");
    auto s = upcxx::reduce_all(mystr.size(), upcxx::op_fast_add, team).wait();
    check_ordered_file(ofname, mystr, 0, team);
    check_ordered_file(ofname, mystr, s, team);
  }
  if (team.rank_me() == 0) {
    auto ret = unlink(ofname.c_str());
    if (ret != 0) DIE("After close, could not unlink ", ofname, " ", strerror(errno), "!");
  }
  barrier(team);

  ofname += ".4";
  {
    DBG("Opening ", ofname, "\n");
    upcxx_utils::dist_ofstream of(team, ofname, false, block_size);
    DBG("Opened ", ofname, "\n");
    if (team.rank_me() % 2 == 0) {
      for (int i = 0; i <= team.rank_me(); i++) {
        if (i) of << ", ";
        of << std::to_string(i);
      }
      of << "\n";
    }

    mystr = of.str();
    DBG("Closing ", ofname, "\n");
    of.close();
    DBG("Even ranks\n");
  }
  check_ordered_file(ofname, mystr, 0, team);
  {
    DBG("Opening ", ofname, " again for append\n");
    upcxx_utils::dist_ofstream of(team, ofname, true, block_size);
    DBG("Opened ", ofname, "\n");
    of << mystr;
    DBG("Closing ", ofname, "\n");
    of.close();
    DBG("Closed ", ofname, "\n");
    auto s = upcxx::reduce_all(mystr.size(), upcxx::op_fast_add, team).wait();
    check_ordered_file(ofname, mystr, 0, team);
    check_ordered_file(ofname, mystr, s, team);
  }
  if (team.rank_me() == 0) {
    auto ret = unlink(ofname.c_str());
    if (ret != 0) DIE("After close, could not unlink ", ofname, " ", strerror(errno), "!");
  }
  barrier(team);

  ofname += ".5";
  {
    DBG("Opening ", ofname, "\n");
    upcxx_utils::dist_ofstream of(team, ofname, false, block_size);
    DBG("Opened ", ofname, "\n");
    if (team.rank_me() % 2 == 1) {
      for (int i = 0; i <= team.rank_me(); i++) {
        if (i) of << ", ";
        of << std::to_string(i);
      }
      of << "\n";
    }

    mystr = of.str();
    DBG("Closing ", ofname, " with ", mystr.size(), " mybytes\n");
    of.close();
    DBG("Odd ranks\n");
  }
  check_ordered_file(ofname, mystr, 0, team);
  {
    DBG("Opening ", ofname, " again for append\n");
    upcxx_utils::dist_ofstream of(team, ofname, true, block_size);
    DBG("Opened ", ofname, "\n");
    of << mystr;
    DBG("Closing ", ofname, "\n");
    of.close();
    DBG("Closed ", ofname, "\n");
    auto s = upcxx::reduce_all(mystr.size(), upcxx::op_fast_add, team).wait();
    check_ordered_file(ofname, mystr, 0, team);
    check_ordered_file(ofname, mystr, s, team);
  }
  if (team.rank_me() == 0) {
    auto ret = unlink(ofname.c_str());
    if (ret != 0) DIE("After close, could not unlink ", ofname, " ", strerror(errno), "!");
  }
  barrier(team);

  ofname += ".6";
  {
    DBG("Opening ", ofname, " again\n");
    upcxx_utils::dist_ofstream of(team, ofname, false, block_size);
    DBG("Opened ", ofname, "\n");
    for (int i = 0; i <= team.rank_me(); i++) {
      if (i) of << ", ";
      of << std::to_string(i);
    }
    of << "\n";
    mystr = of.str();
    DBG("Closing ", ofname, "\n");
    of.close();
    DBG("Implicit destroy out of scope after close\n");
  }
  check_ordered_file(ofname, mystr, 0, team);

  using WritePos = struct {
    string str;
    uint64_t pos;
  };
  WritePos wp1, wp2, wp3;
  {
    DBG("Opening ", ofname, " again for append\n");
    upcxx_utils::dist_ofstream of(team, ofname, true, block_size);
    DBG("Opened ", ofname, "\n");
    of << "rank" << team.rank_me();
    for (int i = 0; i <= team.rank_me(); i++) {
      of << ", ";
      of << std::to_string(i);
    }
    of << "\n";
    wp1.str = of.str();
    auto fut = of.flush_async();
    auto fut2 = fut.then([&wp1, &of]() { wp1.pos = of.get_last_known_tellp() - wp1.str.size(); });
    fut2.wait();
    of << "rank" << team.rank_me() << "flushed\n";
    wp2.str = of.str();
    fut = of.flush_async();
    fut2 = fut.then([&wp2, &of]() { wp2.pos = of.get_last_known_tellp() - wp2.str.size(); });
    fut2.wait();
    of << "rank" << std::setw(10) << team.rank_me() << " is qdone\n";
    wp3.str = of.str();
    DBG("Closing ", ofname, "\n");
    of.close();
    DBG("Implicit destroy out of scope after close\n");
  }
  {
    barrier(team);
    ifstream in(ofname);
    check_file_at(in, wp1.pos, wp1.str);
    check_file_at(in, wp2.pos, wp2.str);
    auto fut_base = upcxx::reduce_all(wp2.pos + wp2.str.size(), upcxx::op_fast_max, team).wait();
    check_file_at(in, team.rank_me() * wp3.str.size(), wp3.str, fut_base);
    barrier(team);
  }
  {
    DBG("Opening ", ofname, " again for append again\n");
    upcxx_utils::dist_ofstream of(team, ofname, true, block_size);
    DBG("Opened ", ofname, "\n");
    for (int i = 0; i <= team.rank_me(); i++) {
      if (i) of << ", ";
      of << std::to_string(i);
    }
    of << "\n";
    mystr = of.str();
    DBG("Closing ", ofname, "\n");
    of.close();
    DBG("Implicit close and destroy out of scope\n");
  }
  check_ordered_file(ofname, mystr, 0, team);
  if (team.rank_me() == 0) {
    auto ret = unlink(ofname.c_str());
    if (ret != 0) DIE("After close, could not unlink ", ofname, " ", strerror(errno), "!");
  }
  barrier(team);

  DBG("I'm all done with block_size=", block_size, "\n");
  upcxx::barrier(team);

  upcxx::barrier(world());
  return 0;
}

int run_test_ofstream() {
  string ofname("test_ofstream.txt");
  int ret = 0;
  ret += run_ofstream_test(ofname, world(), 7);
  ret += run_ofstream_test(ofname, world(), 32);
  ret += run_ofstream_test(ofname, world(), 63);
  ret += run_ofstream_test(ofname, world(), 64);
  ret += run_ofstream_test(ofname, world(), 16 * 1024 * 1024);
  ret += run_ofstream_test(ofname, world(), 0);
  ret += run_ofstream_test(ofname, world(), 1);
  ret += run_ofstream_test(ofname, world(), 2);
  ret += run_ofstream_test(ofname, world(), 3);
  ret += run_ofstream_test(ofname, world(), 4);
  ret += run_ofstream_test(ofname, world(), 5);
  ret += run_ofstream_test(ofname, world(), 6);
  ret += run_ofstream_test(ofname, world(), 8);
  ret += run_ofstream_test(ofname, world(), 11);
  ret += run_ofstream_test(ofname, world(), 13);
  ret += run_ofstream_test(ofname, world(), 16);
  ret += run_ofstream_test(ofname, world(), 21);
  ret += run_ofstream_test(ofname, world(), 23);

  {
    upcxx_utils::split_team split;
    DBG("Split with nodes=", split.node_n(), ", thread_n=", split.thread_n(), " thread_team n=", split.thread_team().rank_n(),
        " me=", split.thread_team().rank_me(), "\n");
    ofname += std::to_string(split.node_me()) + "of" + std::to_string(split.node_n());
    ret += run_ofstream_test(ofname, split.thread_team(), 7);
    ret += run_ofstream_test(ofname, split.thread_team(), 32);
    ret += run_ofstream_test(ofname, split.thread_team(), 63);
    ret += run_ofstream_test(ofname, split.thread_team(), 64);
    ret += run_ofstream_test(ofname, split.thread_team(), 16 * 1024 * 1024);
    ret += run_ofstream_test(ofname, split.thread_team(), 0);
    ret += run_ofstream_test(ofname, split.thread_team(), 1);
    ret += run_ofstream_test(ofname, split.thread_team(), 2);
    ret += run_ofstream_test(ofname, split.thread_team(), 3);
    ret += run_ofstream_test(ofname, split.thread_team(), 4);
    ret += run_ofstream_test(ofname, split.thread_team(), 5);
    ret += run_ofstream_test(ofname, split.thread_team(), 6);
    ret += run_ofstream_test(ofname, split.thread_team(), 8);
    ret += run_ofstream_test(ofname, split.thread_team(), 11);
    ret += run_ofstream_test(ofname, split.thread_team(), 13);
    ret += run_ofstream_test(ofname, split.thread_team(), 16);
    ret += run_ofstream_test(ofname, split.thread_team(), 21);
    ret += run_ofstream_test(ofname, split.thread_team(), 23);
  }
  return ret;
}

int test_several_asyncs(int argc, char **argv) {
  SLOG_VERBOSE("test_several_asyncs\n");
  string line(
      "0000000000 "
      "abcdefghijklmnopqrstuvwxyz00112233445566778899aabbccddeeffgghhiijjkkllmmnnooppqqrrssttuuvvwwxxyyzz00011122233344455566677788"
      "8999aaabbbcccdddeeefffggghhhiiijjjkkklllmmmnnnooopppqqqrrrssstttuuuvvvwwwxxxyyyzzz; ,./?~!@#$%^&*()_+-=[]{}:01234567890\n");
  string one_k = line + line + line + line;
  barrier();
  string fname("test_several_asyncs.txt");
  future<> all_done = make_future();
  upcxx_utils::dist_ofstream f(fname);
  for (int i = 0; i < 5; i++) {
    f << "rank " << rank_me() << " iteration " << i << "\n";
    for (int j = 0; j < rank_me() + 1; j++) {
      f << one_k;
    }
    f << "\n";
    all_done = when_all(all_done, f.flush_collective());
  }
  all_done = when_all(all_done, f.close_async());
  all_done.wait();
  f.close_and_report_timings().wait();
  barrier();
  if (!rank_me()) unlink(fname.c_str());
  SLOG_VERBOSE("Done test_several_async\n");
  return 0;
}

using ShTimer = shared_ptr<upcxx_utils::BaseTimer>;
using ShTimings = upcxx_utils::ShTimings;
int run_large_test(int argc, char **argv) {
  assert(argc >= 2);
  size_t kb = atoi(argv[1]);
  size_t iterations = 1;
  if (argc >= 3) iterations = atoi(argv[2]);
  SLOG_VERBOSE("Running ", iterations, " iterations writing ", upcxx_utils::get_size_str(kb * ONE_KB), " per rank, ",
               upcxx_utils::get_size_str(kb * ONE_KB * rank_n()), " total\n");

  string line(
      "0000000000 "
      "abcdefghijklmnopqrstuvwxyz00112233445566778899aabbccddeeffgghhiijjkkllmmnnooppqqrrssttuuvvwwxxyyzz00011122233344455566677788"
      "8999aaabbbcccdddeeefffggghhhiiijjjkkklllmmmnnnooopppqqqrrrssstttuuuvvvwwwxxxyyyzzz; ,./?~!@#$%^&*()_+-=[]{}:01234567890\n");
  string one_k = line + line + line + line;
  barrier();
  future<> all_done = make_future();
  for (int i = 0; i < iterations; i++) {
    string fname("test-" + upcxx_utils::get_size_str(kb * ONE_KB) + "-" + std::to_string(i) + ".txt");
    SLOG_VERBOSE("Writing ", fname, "\n");
    ShTimer t1 = make_shared<upcxx_utils::BaseTimer>("Opening " + fname);
    ShTimer t2 = make_shared<upcxx_utils::BaseTimer>("Building " + fname);
    ShTimer t3 = make_shared<upcxx_utils::BaseTimer>("Closing " + fname);

    t1->start();
    upcxx_utils::dist_ofstream f(fname);
    t1->stop();
    auto t1_t = t1->reduce_timings();

    t2->start();
    for (int j = 0; j < kb; j++) f << one_k;
    t2->stop();
    auto t2_t = t2->reduce_timings();

    t3->start();
    future<> fut;
    if (i % 2 == 1) {
      fut = f.close_async();
      assert(!fut.is_ready());
      fut = fut.then([t3]() { t3->stop(); });
    } else {
      fut = make_future();
      f.close();
      t3->stop();
    }

    fut = fut.then([t1, t2, t3, fname]() {
      auto t1_t = t1->get_elapsed(), t2_t = t2->get_elapsed(), t3_t = t3->get_elapsed();
      SLOG_VERBOSE(fname, " - Total ", t1_t + t2_t + t3_t, " Open ", t1_t, " Build ", t2_t, " Close ", t3_t, "\n");
    });

    all_done = when_all(all_done, fut);
    // fut.wait();
    // barrier();
  }
  SLOG_VERBOSE("Completed output, waiting for file close\n");
  all_done.wait();
  barrier();
  SLOG_VERBOSE("Closed all files\n");
  return 0;
}

int test_ofstream(int argc, char **argv) {
  if (!upcxx::rank_me()) std::cout << "Found upcxx_utils version " << UPCXX_UTILS_VERSION << std::endl;
  upcxx_utils::open_dbg("test_ofstream");

  if (argc <= 1) {
    LOG_TRY_CATCH(if (run_test_ofstream() != 0 || test_several_asyncs(argc, argv) != 0) DIE("Failed\n"););
  } else {
    LOG_TRY_CATCH(if (run_large_test(argc, argv) != 0) DIE("Failed\n"););
  }

  upcxx_utils::dist_ofstream::sync_all_files();  // clean up straggling atomic domains
  DBG("All done\n");
  upcxx_utils::close_dbg();

  return 0;
}
