#include <iostream>
#include <unordered_map>
#include <upcxx/upcxx.hpp>

#include "upcxx_utils/flat_aggr_store.hpp"
#include "upcxx_utils/log.hpp"
#include "upcxx_utils/version.h"

struct KV {
  char key;
  int val;
  UPCXX_SERIALIZED_FIELDS(key, val);
};

int test_flat_aggr_store(int argc, char **argv) {
  if (!upcxx::rank_me())
    SOUT(argv[0], ": Found upcxx_utils version ", UPCXX_UTILS_VERSION_DATE, " on ", UPCXX_UTILS_BRANCH, " with ", upcxx::rank_n(),
         " procs.", "\n");

  upcxx_utils::open_dbg("test_flat_aggr_store");

  using raw_map_t = std::unordered_map<char, size_t>;
  using map_t = upcxx::dist_object<raw_map_t>;

  for (int i = 0; i < 2; i++) {
    upcxx::barrier();
    map_t myMap(upcxx::world());

    upcxx_utils::FlatAggrStore<KV> flatStore;
    flatStore.set_size("char counter", i * 128 * upcxx::rank_n(), 100);
    flatStore.set_update_func([&m = myMap](KV kv) {
      assert(kv.key >= ' ' && kv.key <= 'z');
      assert(kv.val == 1);
      const auto it = m->find(kv.key);
      if (it == m->end()) {
        m->insert({kv.key, kv.val});
      } else {
        it->second += kv.val;
      }
    });

    string data("The quick brown fox jumped over the lazy dog's tail...");

    raw_map_t expected;

    for (char &c : data) {
      KV kv = {c, 1};
      intrank_t target_rank = ((int)c) % upcxx::rank_n();
      if (target_rank == upcxx::rank_me()) expected[c] += upcxx::rank_n();
      flatStore.update(target_rank, kv);
    }
    flatStore.flush_updates();
    int count = 0;
    for (auto &kv : *myMap) {
      int exp = 0;
      switch (kv.first) {
        case '.': exp = 3; break;
        case ' ': exp = 9; break;
        case 'r':
        case 'd':
        case 'u':
        case 'a':
        case 'h':
        case 'i':
        case 't':
        case 'l': exp = 2; break;
        case 'e':
        case 'o': exp = 4; break;
        case 'T':
        case '\'': exp = 1; break;
        default: exp = 0;
      }
      if (exp == 0 && (kv.first >= 'a' && kv.first <= 'z')) {
        exp = 1;
      }
      if (expected.find(kv.first) != expected.end()) {
        if (expected[kv.first] != exp * upcxx::rank_n()) {
          DIE("Wrong expectation for ", (int)kv.first, "'", kv.first, "' expected ", exp * upcxx::rank_n(), " got ",
              expected[kv.first], "\n");
        }
      }
      if (kv.second != exp * upcxx::rank_n()) {
        DIE("Wrong count for ", (int)kv.first, "'", kv.first, "': ", kv.second, " - expected:", exp * upcxx::rank_n(), "\n");
      }
      if ((int)kv.first % upcxx::rank_n() != upcxx::rank_me())
        DIE("Got key for wrong rank! ", kv.first, " should be on ", (int)kv.first % upcxx::rank_n(), "\n");
      assert(kv.second == exp * upcxx::rank_n());
      count++;
      OUT("rank=", upcxx::rank_me(), " c='", kv.first, "' ", kv.second, "\n");
    }
    if (expected.size() != myMap->size()) {
      DIE("Incorrect number of keys for this rank: ", expected.size(), " vs got ", myMap->size(), "\n");
    }

    int total = upcxx::reduce_one(count, upcxx::op_fast_add, 0).wait();
    if (!upcxx::rank_me()) {
      if (total != 30) {
        DIE("Got wrong size not 30:", total, "\n");
      }
    }

    if (i == 0) continue;
  }

  upcxx_utils::Timings::wait_pending();
  upcxx_utils::close_dbg();
  return 0;
}
