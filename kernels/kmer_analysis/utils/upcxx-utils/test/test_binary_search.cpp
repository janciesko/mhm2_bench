#include <upcxx/upcxx.hpp>

#include "upcxx_utils/binary_search.hpp"

using upcxx::barrier;
using upcxx::dist_object;
using upcxx::intrank_t;
using upcxx::rank_me;
using upcxx::rank_n;
using upcxx::world;

using upcxx_utils::binary_search_fetch;
using upcxx_utils::binary_search_rpc;
using upcxx_utils::binary_search_rpc_obj;

class Int {
  int _;

 public:
  Int(int x)
      : _(x) {}

  int get() const { return _; }

  int cmp(const Int &other) const {
    int me = get();
    int o = other.get();
    if (me > o)
      return -1;
    else if (me == o)
      return 0;
    else
      return 1;
  }

  int operator()(const dist_object<Int> &distrank, const Int other) const { return cmp(other); }

  operator std::string() const { return std::to_string(_); }
};

void run_binary_search() {
  {
    // test uniform array
    dist_object<Int> da(upcxx::rank_me());
    barrier();
    if (true) {
      DBG_VERBOSE("Finding me rpc\n");
      auto res = binary_search_rpc_obj(da, Int(rank_me())).wait();
      if (std::get<0>(res) != rank_me() || std::get<1>(res).get() != rank_me()) DIE("");
      DBG_VERBOSE("Finding 0 rpc\n");
      res = binary_search_rpc_obj(da, Int(0)).wait();
      if (std::get<0>(res) != 0 || std::get<1>(res).get() != 0) DIE("");
      DBG_VERBOSE("Finding last rpc\n");
      res = binary_search_rpc_obj(da, Int(rank_n() - 1)).wait();
      if (std::get<0>(res) != rank_n() - 1 || std::get<1>(res).get() != rank_n() - 1) DIE("");
      DBG_VERBOSE("Finding mid rpc\n");
      res = binary_search_rpc_obj(da, Int(rank_n() / 2)).wait();
      if (std::get<0>(res) != rank_n() / 2 || std::get<1>(res).get() != rank_n() / 2) DIE("");
      DBG_VERBOSE("Finding -1 rpc\n");
      res = binary_search_rpc_obj(da, Int(-1)).wait();
      if (std::get<0>(res) != rank_n()) DIE("");
      DBG_VERBOSE("Finding rank_n rpc\n");
      res = binary_search_rpc_obj(da, Int(rank_n())).wait();
      if (std::get<0>(res) != rank_n()) DIE("");
      DBG_VERBOSE("Finding -1000 rpc\n");
      res = binary_search_rpc_obj(da, Int(-1000)).wait();
      if (std::get<0>(res) != rank_n()) DIE("");
      DBG_VERBOSE("Finding *1000 rpc\n");
      res = binary_search_rpc_obj(da, Int(rank_n() * 1000)).wait();
      if (std::get<0>(res) != rank_n()) DIE("");
    }
    barrier();
    {
      DBG_VERBOSE("Finding me rpc\n");
      auto res = binary_search_rpc(da, Int(rank_me())).wait();
      if ((res) != rank_me()) DIE("");
      DBG_VERBOSE("Finding 0 rpc\n");
      res = binary_search_rpc(da, Int(0)).wait();
      if ((res) != 0) DIE("");
      DBG_VERBOSE("Finding last rpc\n");
      res = binary_search_rpc(da, Int(rank_n() - 1)).wait();
      if ((res) != rank_n() - 1) DIE("");
      DBG_VERBOSE("Finding mid rpc\n");
      res = binary_search_rpc(da, Int(rank_n() / 2)).wait();
      if ((res) != rank_n() / 2) DIE("");
      DBG_VERBOSE("Finding -1 rpc\n");
      res = binary_search_rpc(da, Int(-1)).wait();
      if ((res) != rank_n()) DIE("");
      DBG_VERBOSE("Finding rank_n rpc\n");
      res = binary_search_rpc(da, Int(rank_n())).wait();
      if ((res) != rank_n()) DIE("");
      DBG_VERBOSE("Finding -1000 rpc\n");
      res = binary_search_rpc(da, Int(-1000)).wait();
      if ((res) != rank_n()) DIE("");
      DBG_VERBOSE("Finding *1000 rpc\n");
      res = binary_search_rpc(da, Int(rank_n() * 1000)).wait();
      if ((res) != rank_n()) DIE("");
    }
    barrier();
    {
      DBG_VERBOSE("Finding me fetch\n");
      auto res = binary_search_fetch(da, Int(rank_me())).wait();
      if (std::get<0>(res) != rank_me() || std::get<1>(res).get() != rank_me()) DIE("");
      DBG_VERBOSE("Finding 0 fetch\n");
      res = binary_search_fetch(da, Int(0)).wait();
      if (std::get<0>(res) != 0 || std::get<1>(res).get() != 0) DIE("");
      DBG_VERBOSE("Finding last fetch\n");
      res = binary_search_fetch(da, Int(rank_n() - 1)).wait();
      if (std::get<0>(res) != rank_n() - 1 || std::get<1>(res).get() != rank_n() - 1)
        DIE("got ", std::get<0>(res), " and ", std::get<1>(res).get(), " not ", rank_n() - 1, "\n");
      DBG_VERBOSE("Finding mid fetch\n");
      res = binary_search_fetch(da, Int(rank_n() / 2)).wait();
      if (std::get<0>(res) != rank_n() / 2 || std::get<1>(res).get() != rank_n() / 2) DIE("");
      DBG_VERBOSE("Finding -1 fetch\n");
      res = binary_search_fetch(da, Int(-1)).wait();
      if (std::get<0>(res) != rank_n()) DIE("");
      DBG_VERBOSE("Finding rank_n fetch\n");
      res = binary_search_fetch(da, Int(rank_n())).wait();
      if (std::get<0>(res) != rank_n()) DIE("");
      DBG_VERBOSE("Finding -1000 fetch\n");
      res = binary_search_fetch(da, Int(-1000)).wait();
      if (std::get<0>(res) != rank_n()) DIE("");
      DBG_VERBOSE("Finding *1000 fetch\n");
      res = binary_search_fetch(da, Int(rank_n() * 1000)).wait();
      if (std::get<0>(res) != rank_n()) DIE("");
    }
    barrier();
    for (intrank_t i = 0; i < rank_n(); i++) {
      DBG_VERBOSE("Finding ", i, " rpc\n");
      auto res = binary_search_rpc_obj(da, Int(i)).wait();
      if (std::get<0>(res) != i || std::get<1>(res).get() != i)
        DIE("got ", std::get<0>(res), " and ", std::get<1>(res).get(), " not ", i, "\n");

      auto res_noobj = binary_search_rpc(da, Int(i)).wait();
      if ((res_noobj) != i) DIE("got ", res_noobj, " not ", i, "\n");

      DBG_VERBOSE("Finding ", i, " fetch\n");
      res = binary_search_fetch(da, Int(i)).wait();
      if (std::get<0>(res) != i || std::get<1>(res).get() != i)
        DIE("got ", std::get<0>(res), " and ", std::get<1>(res).get(), " not ", i, "\n");
    }

    barrier();
  }
  {
    barrier();
    // test sparse array with missing vals
    dist_object<Int> da(upcxx::rank_me() * 2);
    barrier();
    DBG_VERBOSE("Finding me*2 rpc\n");
    auto res = binary_search_rpc_obj(da, Int(rank_me() * 2)).wait();
    if (std::get<0>(res) != rank_me() || std::get<1>(res).get() != rank_me() * 2) DIE("");
    DBG_VERBOSE("Finding 0 rpc\n");
    res = binary_search_rpc_obj(da, Int(0)).wait();
    if (std::get<0>(res) != 0 || std::get<1>(res).get() != 0) DIE("");
    DBG_VERBOSE("Finding last*2 rpc\n");
    auto last = (rank_n() - 1) * 2;
    res = binary_search_rpc_obj(da, Int(last)).wait();
    if (std::get<0>(res) != rank_n() - 1 || std::get<1>(res).get() != last) DIE("");
    auto mid = rank_n();
    DBG_VERBOSE("Finding mid*2 rpc: ", mid, " my=", da->get(), "\n");
    res = binary_search_rpc_obj(da, Int(mid)).wait();
    if (mid % 2 == 0)
      if (std::get<0>(res) != rank_n() / 2 || std::get<1>(res).get() != mid)
        DIE("expected ", rank_n() / 2, " got ", std::get<0>(res), " / ", std::get<1>(res).get(), " for mid=", mid, "\n");
    DBG_VERBOSE("Finding -1 rpc\n");
    res = binary_search_rpc_obj(da, Int(-1)).wait();
    if (std::get<0>(res) != rank_n()) DIE("");
    DBG_VERBOSE("Finding rank_n*2 rpc\n");
    res = binary_search_rpc_obj(da, Int(rank_n() * 2)).wait();
    if (std::get<0>(res) != rank_n()) DIE("");
    DBG_VERBOSE("Finding -1000 rpc\n");
    res = binary_search_rpc_obj(da, Int(-1000)).wait();
    if (std::get<0>(res) != rank_n()) DIE("");
    DBG_VERBOSE("Finding *1000 rpc\n");
    res = binary_search_rpc_obj(da, Int(rank_n() * 1000)).wait();
    if (std::get<0>(res) != rank_n()) DIE("");

    DBG_VERBOSE("Finding me*2 fetch\n");
    res = binary_search_fetch(da, Int(rank_me() * 2)).wait();
    if (std::get<0>(res) != rank_me() || std::get<1>(res).get() != rank_me() * 2) DIE("");
    DBG_VERBOSE("Finding 0 fetch\n");
    res = binary_search_fetch(da, Int(0)).wait();
    if (std::get<0>(res) != 0 || std::get<1>(res).get() != 0) DIE("");
    DBG_VERBOSE("Finding last fetch\n");
    res = binary_search_fetch(da, Int(last)).wait();
    if (std::get<0>(res) != rank_n() - 1 || std::get<1>(res).get() != last)
      DIE("got ", std::get<0>(res), " and ", std::get<1>(res).get(), " not ", rank_n() - 1, "\n");
    DBG_VERBOSE("Finding mid fetch\n");
    res = binary_search_fetch(da, Int(mid)).wait();
    if (mid % 2 == 0)
      if (std::get<0>(res) != rank_n() / 2 || std::get<1>(res).get() != mid) DIE("");
    DBG_VERBOSE("Finding -1 fetch\n");
    res = binary_search_fetch(da, Int(-1)).wait();
    if (std::get<0>(res) != rank_n()) DIE("");
    DBG_VERBOSE("Finding rank_n*2 fetch\n");
    res = binary_search_fetch(da, Int(rank_n() * 2)).wait();
    if (std::get<0>(res) != rank_n()) DIE("");
    DBG_VERBOSE("Finding -1000 fetch\n");
    res = binary_search_fetch(da, Int(-1000)).wait();
    if (std::get<0>(res) != rank_n()) DIE("");
    DBG_VERBOSE("Finding *1000 fetch\n");
    res = binary_search_fetch(da, Int(rank_n() * 1000)).wait();
    if (std::get<0>(res) != rank_n()) DIE("");

    DBG_VERBOSE("Finding me*2 rpc\n");
    auto res2 = binary_search_rpc(da, Int(rank_me() * 2)).wait();
    if ((res2) != rank_me()) DIE("");
    DBG_VERBOSE("Finding 0 rpc\n");
    res2 = binary_search_rpc(da, Int(0)).wait();
    if (res2 != 0) DIE("");
    DBG_VERBOSE("Finding last*2 rpc\n");
    last = (rank_n() - 1) * 2;
    res2 = binary_search_rpc(da, Int(last)).wait();
    if (res2 != rank_n() - 1) DIE("");
    DBG_VERBOSE("Finding mid*2 rpc\n");
    mid = rank_n();
    res2 = binary_search_rpc(da, Int(mid)).wait();
    if (mid % 2 == 0)
      if (res2 != rank_n() / 2) DIE("");
    DBG_VERBOSE("Finding -1 rpc\n");
    res2 = binary_search_rpc(da, Int(-1)).wait();
    if (res2 != rank_n()) DIE("");
    DBG_VERBOSE("Finding rank_n*2 rpc\n");
    res2 = binary_search_rpc(da, Int(rank_n() * 2)).wait();
    if (res2 != rank_n()) DIE("");
    DBG_VERBOSE("Finding -1000 rpc\n");
    res2 = binary_search_rpc(da, Int(-1000)).wait();
    if (res2 != rank_n()) DIE("");
    DBG_VERBOSE("Finding *1000 rpc\n");
    res2 = binary_search_rpc(da, Int(rank_n() * 1000)).wait();
    if (res2 != rank_n()) DIE("");

    for (intrank_t i = 0; i < rank_n() * 2; i++) {
      DBG_VERBOSE("Finding ", i, " rpc\n");
      res = binary_search_rpc_obj(da, Int(i)).wait();
      if (i % 2 == 0) {
        if (std::get<0>(res) != i / 2 || std::get<1>(res).get() != i)
          DIE("got ", std::get<0>(res), " and ", std::get<1>(res).get(), " not ", i, "\n");
      } else {
        if (std::get<0>(res) != rank_n()) DIE("got ", std::get<0>(res), " and ", std::get<1>(res).get(), " not ", rank_n(), "\n");
      }

      DBG_VERBOSE("Finding ", i, " fetch\n");
      res = binary_search_fetch(da, Int(i)).wait();
      if (i % 2 == 0) {
        if (std::get<0>(res) != i / 2 || std::get<1>(res).get() != i)
          DIE("got ", std::get<0>(res), " and ", std::get<1>(res).get(), " not ", i, "\n");
      } else {
        if (std::get<0>(res) != rank_n()) DIE("got ", std::get<0>(res), " and ", std::get<1>(res).get(), " not ", rank_n(), "\n");
      }

      res2 = binary_search_rpc(da, Int(i)).wait();
      if (i % 2 == 0) {
        if (res2 != i / 2) DIE("got ", res2, " not ", i, "\n");
      } else {
        if (res2 != rank_n()) DIE("got ", res2, " not ", rank_n(), "\n");
      }
    }

    barrier();
  }
}

int test_binary_search(int argc, char **argv) {
  if (!upcxx::rank_me()) std::cout << "Found upcxx_utils version " << UPCXX_UTILS_VERSION << std::endl;
  upcxx_utils::open_dbg("test_binary_search");

  LOG_TRY_CATCH(run_binary_search(););

  DBG("All done\n");
  upcxx_utils::close_dbg();

  return 0;
}
