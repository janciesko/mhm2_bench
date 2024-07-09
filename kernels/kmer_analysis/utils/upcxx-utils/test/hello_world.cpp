// upcxx -g -std=c++17 hello_world.cpp -o hello_world
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <upcxx/upcxx.hpp>
#include <utility>
#include <vector>

using namespace std;
using duration_seconds = std::chrono::duration<double>;

std::string &left_trim(std::string &str) {
  auto it = std::find_if(str.begin(), str.end(), [](char ch) { return !std::isspace<char>(ch, std::locale()); });
  str.erase(str.begin(), it);
  return str;
}
std::string get_proc_pin() {
  std::ifstream f("/proc/self/status");
  std::string line;
  std::string prefix = "Cpus_allowed_list:";
  while (getline(f, line)) {
    if (line.substr(0, prefix.length()) == prefix) {
      line = line.substr(prefix.length(), line.length() - prefix.length());
      return left_trim(line);
      break;
    }
  }
  return "";
}

auto get_current_time() {
  auto t = std::time(nullptr);
  return std::put_time(localtime(&t), "%Y%m%d %H:%M:%S");
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

double get_free_mem(void) {
  vector<string> no_fields;
  auto ret = get_free_mem(no_fields);
  return ret.first;
}

string get_self_stat(void) {
  std::stringstream buffer;
  std::ifstream i("/proc/self/stat");
  buffer << i.rdbuf();
  return buffer.str();
}

int main(int argc, char **argv) {
  duration_seconds dur;
  auto start_t = std::chrono::high_resolution_clock::now();
  auto starting_free_mem = get_free_mem();
  char *proc_id = getenv("SLURM_PROCID");
  int my_rank = -1;
  if (proc_id) {
    my_rank = atol(proc_id);
  }
  auto pid = getpid();
  char hnbuf[64];
  gethostname(hnbuf, sizeof(hnbuf) - 1);
  if (!my_rank)
    std::cout << "Starting Rank0 with free_mem=" << starting_free_mem / 1048576.0 << " MB on pid=" << pid << " host=" << hnbuf
              << " at " << get_current_time() << std::endl;

  auto pre_init_t = std::chrono::high_resolution_clock::now();
  dur = pre_init_t - start_t;
  auto pre_init_s = dur.count();
  upcxx::init();
  auto post_init_t = std::chrono::high_resolution_clock::now();
  dur = post_init_t - pre_init_t;
  auto post_init_s = dur.count();

  upcxx::barrier(upcxx::local_team());
  auto post_init_free_mem = get_free_mem();
  upcxx::barrier(upcxx::local_team());

  auto upcxx_mem_per_rank = (starting_free_mem - post_init_free_mem) / 1048576.0 / upcxx::local_team().rank_n();

  std::stringstream ss;
  std::string pinnings = get_proc_pin();
  ss << "Hello from " << upcxx::rank_me() << " of " << upcxx::rank_n() << " on " << hnbuf << " pid " << pid << " bound to "
     << pinnings << " upcxx consumed " << (starting_free_mem - post_init_free_mem) / 1048576.0 / upcxx::local_team().rank_n()
     << " MB/rank start=" << starting_free_mem / 1048576.0 << " MB post_init=" << post_init_free_mem / 1048576.0
     << " MB pre_init=" << pre_init_s << " post_init=" << post_init_s << "\n";

  upcxx::barrier(upcxx::local_team());
  auto self_stat = get_self_stat();
  upcxx::barrier(upcxx::local_team());

  vector<string> fields{"Huge", "Direct", "Map", "Shmem", "Mem"};
  auto free_mem = get_free_mem(fields);
  ss << "R" << upcxx::rank_me() << ": L:" << upcxx::local_team().rank_me() << " stat=" << self_stat
     << " free=" << size_t(free_mem.first / 1024 / 1024) << "MB /proc/meminfo=" << free_mem.second << "\n";
  string fname("hello_world-upcxx.out");
  if (upcxx::rank_me() == 0) std::cout << "writing to " << fname << std::endl;

  auto pre_min = upcxx::reduce_one(pre_init_s, upcxx::op_fast_min, 0).wait();
  auto pre_sum = upcxx::reduce_one(pre_init_s, upcxx::op_fast_add, 0).wait();
  auto pre_max = upcxx::reduce_one(pre_init_s, upcxx::op_fast_max, 0).wait();
  auto post_min = upcxx::reduce_one(post_init_s, upcxx::op_fast_min, 0).wait();
  auto post_sum = upcxx::reduce_one(post_init_s, upcxx::op_fast_add, 0).wait();
  auto post_max = upcxx::reduce_one(post_init_s, upcxx::op_fast_max, 0).wait();
  auto mem_min = upcxx::reduce_one(upcxx_mem_per_rank, upcxx::op_fast_min, 0).wait();
  auto mem_sum = upcxx::reduce_one(upcxx_mem_per_rank, upcxx::op_fast_add, 0).wait();
  auto mem_max = upcxx::reduce_one(upcxx_mem_per_rank, upcxx::op_fast_max, 0).wait();
  upcxx::barrier();
  auto reduce_t = std::chrono::high_resolution_clock::now();
  if (upcxx::rank_me() == 0) {
    std::cout << "pre-init: " << pre_min << "/" << pre_sum / upcxx::rank_n() << "/" << pre_max << " s" << std::endl;
    std::cout << "post-init: " << post_min << "/" << post_sum / upcxx::rank_n() << "/" << post_max << " s" << std::endl;
    std::cout << "upcxx_mem_per_rank: " << mem_min << "/" << mem_sum / upcxx::rank_n() << "/" << mem_max << " MB" << std::endl;
    dur = reduce_t - post_init_t;
    std::cout << "reduction time: " << dur.count() << "s" << std::endl;
  }

  upcxx::barrier();
  if (upcxx::local_team().rank_me() == 0) {
    std::stringstream outss;
    outss << "Post upcxx::init on " << upcxx::rank_me() << " of " << upcxx::rank_n() << " on " << hnbuf << " pid " << pid
          << " upcxx consumed " << upcxx_mem_per_rank << " MB/rank at " << get_current_time() << "\n";
    std::cout << outss.str() << std::flush;
  }

  auto msg = ss.str();

  using dist_ofstream_t = upcxx::dist_object<std::ofstream>;
  {
    dist_ofstream_t dist_outfile(upcxx::world());
    if (upcxx::rank_me() == 0) dist_outfile->open(fname);

    rpc(
        0, [](dist_ofstream_t &df, string msg) { *df << msg; }, dist_outfile, msg)
        .wait();

    auto post_rpc_t = std::chrono::high_resolution_clock::now();
    dur = post_rpc_t - reduce_t;
    auto post_rpc_s = dur.count();

    upcxx::barrier();
    if (upcxx::rank_me() == 0) {
      dist_outfile->close();
    }
    upcxx::barrier();

    auto post_print_t = std::chrono::high_resolution_clock::now();
    dur = post_print_t - post_rpc_t;
    auto post_print_s = dur.count();

    auto post_rpc_min = upcxx::reduce_one(post_rpc_s, upcxx::op_fast_min, 0).wait();
    auto post_rpc_sum = upcxx::reduce_one(post_rpc_s, upcxx::op_fast_add, 0).wait();
    auto post_rpc_max = upcxx::reduce_one(post_rpc_s, upcxx::op_fast_max, 0).wait();
    auto post_print_min = upcxx::reduce_one(post_print_s, upcxx::op_fast_min, 0).wait();
    auto post_print_sum = upcxx::reduce_one(post_print_s, upcxx::op_fast_add, 0).wait();
    auto post_print_max = upcxx::reduce_one(post_print_s, upcxx::op_fast_max, 0).wait();

    if (upcxx::rank_me() == 0) {
      std::cout << "post-rpc: " << post_rpc_min << "/" << post_rpc_sum / upcxx::rank_n() << "/" << post_rpc_max << " s"
                << std::endl;
      std::cout << "post-print: " << post_print_min << "/" << post_print_sum / upcxx::rank_n() << "/" << post_print_max << " s"
                << std::endl;
      std::cout << "Done. " << std::endl;
    }
  }

  upcxx::finalize();
  exit(0);
}
