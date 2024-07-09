#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <upcxx/upcxx.hpp>

#include "upcxx_utils/log.hpp"
#include "upcxx_utils/progress_bar.hpp"
#include "upcxx_utils/three_tier_aggr_store.hpp"
#include "upcxx_utils/version.h"

struct KV3 {
  char key;
  uint8_t val;
  UPCXX_SERIALIZED_FIELDS(key, val);
};

struct KV4_sub1 {
  int v1;
  string s1;
  UPCXX_SERIALIZED_FIELDS(v1, s1);
};

struct KV4_sub {
  int v1;
  string s1;
  string s2;
  UPCXX_SERIALIZED_FIELDS(v1, s1, s2);
};
struct KV4 {
  string key;
  int val;
  KV4_sub sub;
  UPCXX_SERIALIZED_FIELDS(key, val, sub);
};
#define TTAS_ALLOW_NON_TRIVIAL
int test_three_tier_aggr_store(int argc, char **argv) {
  upcxx_utils::open_dbg("test_three_tier_aggr_store");

  if (!upcxx::rank_me())
    SOUT(argv[0], ": Found upcxx_utils version ", UPCXX_UTILS_VERSION_DATE, " on ", UPCXX_UTILS_BRANCH, " with ", upcxx::rank_n(),
         " procs.", "\n");

  if (true) {
    SOUT("Starting view of view messages\n");
    string longmsg;
    for (int i = 0; i < 10; i++)
      longmsg += "adsflkdasfdasfasdjkfndsajfnadsfkjasdk;fjdsakjfhadsjkfhsadjkfsadfkjlhdsfkjlhadsfkdsajfl;dksfjdslk;"
                 "fadskfjhadskjfhasdfkjaer23qr";
    upcxx::dist_object<string> dummy(world(), longmsg);
    // ping pong
    vector<string> msgs;
    for (int i = 0; i < 100; i++) msgs.push_back(to_string(rank_me()) + ":" + to_string(i) + longmsg);
    future<> all_complete = make_future();
    for (auto _tgt = 0; _tgt < rank_n(); _tgt++) {
      auto tgt = (rank_me() + _tgt) % rank_n();
      auto fut = rpc(
          tgt,
          [](dist_object<string> &dummy, int from, upcxx::view<string> msgs) {
            auto block_size = (msgs.size() + upcxx::local_team().rank_n() - 1) / upcxx::local_team().rank_n();
            future<> all_done = make_future();
            auto iter = msgs.begin();
            for (int j = 0; j < upcxx::local_team().rank_n(); j++) {
              auto start = iter;
              auto end = iter;
              for (int k = 0; k < block_size; k++) {
                if (end == msgs.end()) break;
                end = ++iter;
              }
              // split this view into several partial views
              auto partial = upcxx::make_view(start, end);
              auto fut = rpc(
                  from,
                  [](dist_object<string> &dummy, int returned_by, int startidx, upcxx::view<string> msgs) {
                    for (const auto &msg : msgs) {
                      assert(msg == to_string(rank_me()) + ":" + to_string(startidx) + *dummy);
                      startidx++;
                    }
                  },
                  dummy, rank_me(), j * block_size, partial);
              all_done = when_all(all_done, fut);
            }
            return all_done;
          },
          dummy, rank_me(), upcxx::make_view(msgs));
      fut.wait();
    }
    all_complete.wait();
    barrier();
    SOUT("Completed view of view\n");
  }
#ifdef TTAS_ALLOW_NON_TRIVIAL
  if (true) {
    int num_messages = 1000;
    // test using a struct with strings KV4_sub
    std::unordered_map<string, KV4_sub1> myMap;
    barrier();
    upcxx_utils::ThreeTierAggrStore<KV4_sub1> ttStore;
    ttStore.set_update_func([&myMap](const KV4_sub1 &elem) {
      DBG_VERBOSE2("Inserting ", elem.s1, "\n");
      assert(myMap.find(elem.s1) == myMap.end());
      myMap[elem.s1] = elem;
    });
    ttStore.set_size("map assigner1", 1024 * 512);
    for (intrank_t tgt = 0; tgt < rank_n(); tgt++) {
      for (int i = 0; i < num_messages; i++) {
        KV4_sub1 sub{tgt, to_string(tgt) + ":" + to_string(rank_me()) + "-" + to_string(i)};
        DBG_VERBOSE2("Sending to ", tgt, " ", sub.s1, "\n");
        ttStore.update(tgt, sub);
      }
      DBG_VERBOSE("Sent ", num_messages, " to ", tgt, "\n");
    }
    ttStore.flush_updates();
    ttStore.clear();
    barrier();
    if (myMap.size() != rank_n() * num_messages) DIE("Wrong count of elements ", myMap.size(), "!=", rank_n() * num_messages);
    for (int i = 0; i < num_messages; i++) {
      for (intrank_t src = 0; src < rank_n(); src++) {
        // KV4_sub sub{rank_me(), to_string(rank_me()) + to_string(src) + "-" + to_string(i), to_string(src)};

        string k = to_string(rank_me()) + ":" + to_string(src) + "-" + to_string(i);
        auto it = myMap.find(k);
        if (it == myMap.end()) DIE("Could not find ", k);
        KV4_sub1 &val = it->second;
        if (val.s1 != k) DIE("s1!=k");
        if (val.v1 != rank_me()) DIE("v1!=me");
      }
    }
  }
  if (true) {
    int num_messages = 1000;
    // test using a struct with strings KV4_sub
    std::unordered_map<string, KV4_sub> myMap;
    barrier();
    upcxx_utils::ThreeTierAggrStore<KV4_sub> ttStore;
    ttStore.set_update_func([&myMap](const KV4_sub &elem) {
      DBG_VERBOSE2("Inserting ", elem.s1, "\n");
      assert(myMap.find(elem.s1) == myMap.end());
      myMap[elem.s1] = elem;
    });
    ttStore.set_size("map assigner", 1024 * 512);
    for (intrank_t tgt = 0; tgt < rank_n(); tgt++) {
      for (int i = 0; i < num_messages; i++) {
        KV4_sub sub{tgt, to_string(tgt) + to_string(rank_me()) + "-" + to_string(i), to_string(rank_me())};
        DBG_VERBOSE2("Sending to ", tgt, " ", sub.s1, "\n");
        ttStore.update(tgt, sub);
      }
      DBG_VERBOSE("Sent ", num_messages, " to ", tgt, "\n");
    }
    ttStore.flush_updates();
    ttStore.clear();
    barrier();
    if (myMap.size() != rank_n() * num_messages) DIE("Wrong count of elements ", myMap.size(), "!=", rank_n() * num_messages);
    for (int i = 0; i < num_messages; i++) {
      for (intrank_t src = 0; src < rank_n(); src++) {
        // KV4_sub sub{rank_me(), to_string(rank_me()) + to_string(src) + "-" + to_string(i), to_string(src)};

        string k = to_string(rank_me()) + to_string(src) + "-" + to_string(i);
        auto it = myMap.find(k);
        if (it == myMap.end()) DIE("Could not find ", k);
        KV4_sub &val = it->second;
        if (val.s1 != k) DIE("s1!=k");
        if (val.s2 != to_string(src)) DIE("s2!=src");
        if (val.v1 != rank_me()) DIE("v1!=me");
      }
    }
  }
  if (true) {
    int num_messages = 1000;
    // test using a nested struct KV4 (string + int + anotherstruct)
    std::unordered_map<string, KV4> myMap;
    barrier();
    upcxx_utils::ThreeTierAggrStore<KV4> ttStore;
    ttStore.set_update_func([&myMap](const KV4 &elem) {
      DBG_VERBOSE2("Inserting ", elem.key, "\n");
      assert(myMap.find(elem.key) == myMap.end());
      myMap[elem.key] = elem;
    });
    ttStore.set_size("map assigner2", 1024 * 512);
    for (intrank_t tgt = 0; tgt < rank_n(); tgt++) {
      KV4_sub sub{tgt, to_string(rank_me()), to_string(tgt)};
      for (int i = 0; i < num_messages; i++) {
        string k = to_string(rank_me()) + "-" + to_string(i);
        DBG_VERBOSE2("Sending to ", tgt, " ", k, "\n");
        ttStore.update(tgt, {k, i, sub});
      }
      DBG_VERBOSE("Sent ", num_messages, " to ", tgt, "\n");
    }
    ttStore.flush_updates();
    ttStore.clear();
    barrier();
    if (myMap.size() != rank_n() * num_messages) DIE("Wrong count of elements ", myMap.size(), "!=", rank_n() * num_messages);
    if (myMap.find("0-0") == myMap.end() || myMap.find("0-999") == myMap.end()) DIE("");
    for (int i = 0; i < num_messages; i++) {
      for (intrank_t src = 0; src < rank_n(); src++) {
        // KV4_sub sub{rank_me(), to_string(src), to_string(rank_me())};

        string k = to_string(src) + "-" + to_string(i);
        auto it = myMap.find(k);
        if (it == myMap.end()) DIE("Could not find ", k);
        KV4 &val = it->second;
        if (val.key != k) DIE("key!=k");
        if (val.val != i) DIE("val!=i");
        KV4_sub &kv4 = val.sub;
        if (kv4.v1 != rank_me()) DIE("v1!=me");
        if (kv4.s2 != to_string(rank_me())) DIE("s2!=me");
        if (kv4.s1 != to_string(src)) DIE("s1!=src");
      }
    }
  }
#endif
  // trivial structs only now
  using raw_map_t = std::unordered_map<char, size_t>;
  using map_t = upcxx::dist_object<raw_map_t>;

  for (int i = 0; i < 2; i++) {
    SOUT("Starting round ", i, " with TTAS\n");
    upcxx::barrier();
    map_t myMap(upcxx::world());
    raw_map_t expected;
    upcxx_utils::ThreeTierAggrStore<KV3> ttStore;
    size_t max_bytes = i * sizeof(KV3) * 64 * upcxx::rank_n();
    LOG("Created TTAS, setting size as ", upcxx_utils::get_size_str(max_bytes), ", and 1 rpc in flight\n");
    ttStore.set_size("char counter", max_bytes, 1);
    ttStore.set_update_func([&m = myMap](KV3 kv) {
      DBG("updatefunc: key=", (int)kv.key, "'", kv.key, "', val=", (int)kv.val, "\n");
      assert(kv.key >= ' ' && kv.key <= 'z');
      assert(kv.val == 1);
      const auto it = m->find(kv.key);
      if (it == m->end()) {
        m->insert({kv.key, kv.val});
      } else {
        it->second += kv.val;
      }
    });

    // test a bunch of letters...
    string data("The quick brown fox jumped over the lazy dog's tail...");

    LOG("Sending updates\n");

    for (char &c : data) {
      KV3 kv = {c, 1};
      intrank_t target_rank = ((int)c) % upcxx::rank_n();
      if (target_rank == upcxx::rank_me()) expected[c] += upcxx::rank_n();
      ttStore.update(target_rank, kv);
    }
    LOG("Flushing\n");
    ttStore.flush_updates();

    upcxx_utils::BarrierTimer bt("Starting clear");
    LOG("Clearing\n");
    ttStore.clear();

    upcxx_utils::BarrierTimer("Starting to validate");

    LOG("Validating\n");
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
      DBG("rank=", upcxx::rank_me(), " c='", kv.first, "' ", kv.second, "\n");
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

    barrier();
    SOUT("Testing a large dataset.\n");

    size_t data_size = 4 * 1024 * 1024;
    max_bytes = data_size / 4.5;
    barrier();
    myMap->clear();
    ttStore.clear();
    ttStore.set_size("large char counter", max_bytes);
    // new update func
    ttStore.set_update_func([&m = myMap](KV3 kv) {
      assert(kv.key >= 32 && kv.key <= 126);
      assert(kv.val == 1);
      const auto it = m->find(kv.key);
      if (it == m->end()) {
        m->insert({kv.key, kv.val});
      } else {
        it->second += kv.val;
      }
    });

    SOUT("Initializing\n");

    char *large_data = new char[data_size];
    srand(upcxx::rank_n());
    for (size_t i = 0; i < data_size; i++) {
      large_data[i] = (char)((rand() % 95) + 32);
      assert(large_data[i] >= 32 && large_data[i] <= 126);
    }
    expected.clear();
    SOUT("Sending updates\n");
    upcxx_utils::ProgressBar pb(data_size, "large update");
    for (size_t i = 0; i < data_size; i++) {
      auto c = large_data[i];
      assert(c >= 32 && c <= 126);
      KV3 kv = {c, 1};
      intrank_t target_rank = ((int)c) % upcxx::rank_n();
      if (target_rank == rank_me()) expected[c] += upcxx::rank_n();
      ttStore.update(target_rank, kv);
      pb.update();
    }
    pb.done();
    SOUT("Flushing\n");
    ttStore.flush_updates();
    SOUT("Validating\n");
    upcxx_utils::flush_logger();

    for (auto kv : *myMap) {
      if (expected[kv.first] != kv.second)
        DIE("Incorrect size for ", kv.first, " expected ", expected[kv.first], " got ", kv.second, "\n");
      if ((int)kv.first % upcxx::rank_n() != upcxx::rank_me())
        DIE("Got key for another rank: ", (int)kv.first, " should be on ", (int)kv.first % upcxx::rank_n(), "\n");
    }
    if (expected.size() != myMap->size()) {
      DIE("Different sizes between expected and myMap: ", expected.size(), " vs ", myMap->size(), "\n");
    }
    delete[] large_data;
  }
  upcxx_utils::Timings::wait_pending();

  SOUT("Done\n");
  upcxx_utils::close_dbg();
  return 0;
}
