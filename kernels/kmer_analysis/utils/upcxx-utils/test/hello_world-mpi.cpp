// mpicxx -std=c++17 hello_world-mpi.cpp -o hello_world-mpi
#include <mpi.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

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
  int status, err = 100;
  status = MPI_Init(NULL, NULL);
  if (status != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, err);
  err++;
  int world_size;
  status = MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (status != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, err);
  err++;
  int world_rank;
  status = MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (status != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, err);
  err++;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  status = MPI_Get_processor_name(processor_name, &name_len);
  if (status != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, err);
  err++;

  char hnbuf[64];
  gethostname(hnbuf, sizeof(hnbuf) - 1);
  auto pid = getpid();
  std::string pinnings = get_proc_pin();
  std::stringstream ss;

  status = MPI_Barrier(MPI_COMM_WORLD);
  if (status != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, err);
  err++;

  ss << "Hello from " << world_rank << " of " << world_size << " on " << hnbuf << " as " << processor_name << " pid " << pid
     << " bound to " << pinnings << "\n";

  status = MPI_Barrier(MPI_COMM_WORLD);
  if (status != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, err);
  err++;

  auto self_stat = get_self_stat();

  status = MPI_Barrier(MPI_COMM_WORLD);
  if (status != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, err);
  err++;

  vector<string> fields{"Huge", "Direct", "Map", "Shmem", "Mem"};
  auto free_mem = get_free_mem(fields);
  ss << "R" << world_rank << ": stat=" << self_stat << " free=" << size_t(free_mem.first / 1024 / 1024)
     << "MB /proc/meminfo=" << free_mem.second << " ";

  status = MPI_Barrier(MPI_COMM_WORLD);
  if (status != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, err);
  err++;

  string fname("hello_world-mpi.out");
  if (world_rank == 0) {
    std::cout << "Writing hello and memory stats to " << fname << std::endl;
  }
  auto msg = ss.str();
  int len = msg.length();
  int max_len = 0;
  status = MPI_Allreduce(&len, &max_len, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (status != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, err);
  err++;
  if (world_rank == 0) {
    std::cout << "max_len=" << max_len << std::endl;
  }
  max_len++;
  msg.resize(max_len, ' ');
  msg[max_len - 1] = '\n';
  string output;
  if (world_rank == 0) {
    output.resize(max_len * world_size, ' ');
  }
  status = MPI_Gather(msg.data(), msg.length(), MPI_BYTE, (char *)output.data(), max_len, MPI_CHAR, 0, MPI_COMM_WORLD);
  if (status != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, err);
  err++;

  if (world_rank == 0) {
    std::cout << "writing " << max_len * world_size << " bytes" << std::endl;
    std::ofstream of(fname);
    of << output;
    of.close();
  }

  status = MPI_Barrier(MPI_COMM_WORLD);
  if (status != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, err);
  err++;

  status = MPI_Finalize();
  if (world_rank == 0) {
    std::cout << "Done" << std::endl;
  }
  exit(status);
}
