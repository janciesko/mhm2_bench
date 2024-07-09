#include <unistd.h>

#include <iostream>
#include <string>
#include <upcxx/upcxx.hpp>

#include "upcxx_utils/fixed_size_cache.hpp"

using std::string;
using upcxx_utils::FixedSizeCache;

int test_fixed_size_cache(int argc, char **argv) {
  if (!upcxx::rank_me()) std::cout << "Found upcxx_utils version " << UPCXX_UTILS_VERSION << std::endl;

  FixedSizeCache<int, int> cache;
  assert(cache.size() == 0);

  int did_cache = 0;
  for (int i = 0; i < 100; i++) {
    auto x = cache.insert({i, i * i});
    if (x.second || x.first != cache.end()) {
      did_cache++;
      auto val = *x.first;
      assert(val.first == i);
      assert(val.second == i * i);
    } else {
      assert(x.first == cache.end());
    }
  }
  assert(did_cache == 0 && "empty cache cannot cache anything");
  int found_cache = 0;
  for (int i = 0; i < 100; i++) {
    auto x = cache.find(i);
    if (x != cache.end()) {
      found_cache++;
      auto val = *x;
      assert(val.first == i);
      assert(val.second == i * i);
    }
  }
  assert(found_cache == 0 && "empty cache cannot cache anything");

  cache.reserve(16);
  for (int i = 0; i < 100; i++) {
    auto x = cache.insert({i, i * i});
    if (x.second || x.first != cache.end()) {
      did_cache++;
      auto val = *x.first;
      assert(val.first == i);
      assert(val.second == i * i);
    } else {
      assert(x.first == cache.end());
    }
  }
  assert(did_cache != 0 && "non empty cache should cache some");

  found_cache = 0;
  for (int i = 0; i < 100; i++) {
    auto x = cache.find(i);
    if (x != cache.end()) {
      found_cache++;
      auto val = *x;
      assert(val.first == i);
      assert(val.second == i * i);
    }
  }
  assert(found_cache != 0 && "non-empty cache should cache some");

  cache.clear();
  found_cache = 0;
  for (int i = 0; i < 100; i++) {
    auto x = cache.find(i);
    if (x != cache.end()) {
      found_cache++;
      auto val = *x;
      assert(val.first == i);
      assert(val.second == i * i);
    }
  }
  assert(found_cache == 0 && "cleared cache should not have anything cached");

  return 0;
}
