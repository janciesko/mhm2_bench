#pragma once

// A fixed size map table with just simple find, insert and replace methods mimicking unordered_map
// find returns the iterator if a match is found
// inserts, replace and erase may invalidate outstanding iterators -- checking it->first == key verifies the iterator
// replace overwrites entries (key+value) on collision, optionally only replacing empty buckets
// insert replaces existing entries for the same bucket but having a different key (existing values remain unchanged)
// Key must be nonzero, or initialized to a nonsensical invalid_key
// lookups of the invalid_key will always fail
// insertions / replacements of the invalid_key is undefined and causes an assertion error
// deviation from std::map:
//   the insert() return value of:
//      {end(), false}, means the cache was *NOT* updated,
//      { valid iterator, false } means it was
//      { valid iteraror, true} means it was inserted, and may have replaced/clobbered an existing entry
//   updated
//
// effectively a Random Replacement cache
#include <functional>
#include <memory>
#include <utility>

#include "upcxx_utils/log.hpp"

namespace upcxx_utils {

template <class Key, class T, class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>,
          class Allocator = std::allocator<std::pair<Key, T>>, size_t LINEAR_PROBING = 6>
class FixedSizeCache {
  static const size_t linear_probe = LINEAR_PROBING;

 public:
  using key_type = Key;
  using mapped_type = T;
  using value_type = std::pair<Key, T>;
  using allocator_type = Allocator;
  using buckets_type = std::vector<value_type, allocator_type>;

  FixedSizeCache()
      : buckets{}
      , buckets_minus_one{}
      , num_entries(0)
      , clobberings(0)
      , inserts(0)
      , hits(0)
      , misses(0)
      , empty_key{}
      , hasher()
      , key_equal() {}

  FixedSizeCache(size_t max_count, Key invalid = {})
      : buckets{}
      , buckets_minus_one{}
      , num_entries(0)
      , clobberings(0)
      , inserts(0)
      , hits(0)
      , misses(0)
      , empty_key{invalid}
      , hasher()
      , key_equal() {
    resize(max_count);
  }
  ~FixedSizeCache() {
    size_t lookups = hits + misses;
    LOG(__PRETTY_FUNCTION__, ": hits ", perc_str(hits, lookups), " misses ", misses, " entries ", perc_str(num_entries, capacity()),
        " clobberings ", perc_str(clobberings, inserts), "\n");
  }
  class iterator {
    FixedSizeCache* fsc;
    size_t idx;

   public:
    iterator()
        : fsc(nullptr)
        , idx(0) {}
    iterator(const FixedSizeCache& _fsc, size_t pos = 0)
        : fsc(const_cast<FixedSizeCache*>(&_fsc))
        , idx(pos) {
      assert(pos <= fsc->buckets.size());
      current_or_next_non_empty();
    }
    iterator(const iterator& copy) = default;
    iterator& operator=(const iterator& copy) = default;
    bool operator==(const iterator& other) const { return fsc == other.fsc && idx == other.idx; }
    bool operator!=(const iterator& other) const { return !(*this == other); }
    iterator& operator++() {
      assert(fsc && idx < fsc->buckets.size());
      ++idx;
      current_or_next_non_empty();
      return *this;
    }
    iterator operator++(int) {
      auto copy = *this;
      ++(*this);
      return copy;
    }
    value_type* operator->() const {
      assert(fsc && idx < fsc->buckets.size());
      return &fsc->buckets[idx];
    }
    value_type& operator*() const {
      assert(fsc && idx < fsc->buckets.size());
      return fsc->buckets[idx];
    }

   protected:
    void current_or_next_non_empty() {
      assert(fsc);
      while (idx < fsc->buckets.size() && fsc->key_equal(fsc->buckets[idx].first, fsc->empty_key)) idx++;
    }
  };
  iterator begin() const { return iterator(*this, 0); }
  iterator end() const { return iterator(*this, buckets.size()); }

  iterator find(const Key& key, size_t hash) const {
    if (buckets.empty()) return end();
    assert(buckets.size());
    assert(hasher(key) == hash);
    if (key_equal(key, empty_key)) return end();
    for (size_t i = 0; i < linear_probe; i++) {
      auto idx = (hash + i) & buckets_minus_one;
      auto& bucket = buckets[idx];
      if (key_equal(bucket.first, key)) {
        const_cast<FixedSizeCache*>(this)->hits++;
        return iterator(*this, idx);
      } else if (key_equal(bucket.first, empty_key))
        break;
    }
    const_cast<FixedSizeCache*>(this)->misses++;
    return end();
  }
  iterator find(const Key& key) const {
    auto hash = hasher(key);
    return find(key, hash);
  }
  iterator replace(const Key& key, const T& val, size_t hash, bool only_empty = false) {
    if (buckets.empty()) return end();
    assert(buckets.size());
    assert(hasher(key) == hash);
    assert(!key_equal(key, empty_key));
    for (size_t i = 0; i < linear_probe; i++) {
      auto idx = (hash + i) & buckets_minus_one;
      auto& bucket = buckets[idx];
      auto no_existing = key_equal(bucket.first, empty_key);
      if (no_existing | key_equal(bucket.first, key)) {
        if (no_existing) {
          bucket.first = key;
          num_entries++;
          inserts++;
        }
        bucket.second = val;
        return iterator(*this, idx);
      }
    }
    if (only_empty) return end();
    // clobber existing
    clobberings++;
    inserts++;
    auto idx = hash & buckets_minus_one;
    auto& bucket = buckets[idx];
    assert(!key_equal(bucket.first, empty_key));
    assert(!key_equal(bucket.first, key));
    bucket = {.first = key, .second = val};
    return iterator(*this, idx);
  }
  iterator replace(const Key& key, const T& val, bool only_empty = false) {
    auto hash = hasher(key);
    return replace(key, val, hash, only_empty);
  }
  std::pair<iterator, bool> insert(const value_type& value, size_t hash) {
    if (buckets.empty()) return {end(), false};
    assert(buckets.size());
    assert(hasher(value.first) == hash);
    assert(!key_equal(value.first, empty_key));
    for (size_t i = 0; i < linear_probe; i++) {
      auto idx = (hash + i) & buckets_minus_one;
      auto& bucket = buckets[idx];
      if (key_equal(bucket.first, value.first)) {
        // same key.  Do not replace existing
        return {iterator(*this, idx), false};
      } else if (key_equal(bucket.first, empty_key)) {
        // clobber empty
        num_entries++;
        inserts++;
        bucket = value;
        assert(key_equal(bucket.first, value.first));
        return {iterator(*this, idx), true};
      }
    }
    // clobber existing
    clobberings++;
    inserts++;
    auto idx = hash & buckets_minus_one;
    auto& bucket = buckets[idx];
    assert(!key_equal(bucket.first, empty_key));
    assert(!key_equal(bucket.first, value.first));
    bucket = value;
    return {iterator(*this, idx), true};
  }
  std::pair<iterator, bool> insert(const value_type& value) {
    if (key_equal(value.first, empty_key)) return {end(), false};
    auto hash = hasher(value.first);
    return insert(value, hash);
  }

  void set_invalid_key(const Key invalid) {
    if (buckets.size() && !key_equal(empty_key, invalid)) {
      // replace any existing empty slots with the new invalid key
      for (auto& v : buckets) {
        if (key_equal(v.first, empty_key)) v.first = invalid;
      }
    }
    empty_key = invalid;
  }
  void erase(iterator it) {
    if (it == end() || buckets.empty()) DIE("Cannot erase an empty entry!");
    it->first = empty_key;
    num_entries--;
  }

  size_t size() const { return num_entries; }
  size_t capacity() const { return buckets.size(); }
  size_t get_clobberings() const { return clobberings; }
  void clear() {
    // just clears entries does not free memory
    for (auto& v : buckets) {
      v.first = empty_key;
    }
    num_entries = 0;
  }
  void resize(size_t max_count) {
    size_t num_buckets = 0;
    if (max_count > 0) {
      // round up to nearest power of 2, then possibly again for a decent load_factor
      num_buckets = upper_power_of_two(max_count);
      while (num_buckets > 0 && 100 * max_count / num_buckets > 37) num_buckets = num_buckets << 1;
      if (num_buckets <= max_count) DIE("Invalid round up of ", max_count, " to ", num_buckets, "\n");
    }
    set_num_buckets(num_buckets);
  }
  void reserve(size_t max_count) { resize(max_count); }
  void reserve_max_memory(size_t max_bytes) {
    assert(max_bytes > sizeof(value_type));
    size_t num_buckets = max_bytes / sizeof(value_type);
    num_buckets = upper_power_of_two(num_buckets);
    num_buckets >>= 1;  // one less power
    assert(num_buckets * sizeof(value_type) <= max_bytes);
    set_num_buckets(num_buckets);
  }

 protected:
  buckets_type buckets;
  size_t buckets_minus_one;
  size_t num_entries;
  size_t clobberings;
  size_t inserts;
  size_t hits;
  size_t misses;
  Key empty_key;
  Hash hasher;
  KeyEqual key_equal;

  void set_num_buckets(size_t num_buckets) {
    LOG(__PRETTY_FUNCTION__, ": Set size to ", num_buckets, " ", get_size_str(num_buckets * sizeof(value_type)), "\n");
    buckets.clear();
    value_type empty_value(empty_key, {});
    buckets.resize(num_buckets, empty_value);
    buckets_minus_one = num_buckets - 1;
    num_entries = 0;
    clobberings = 0;
    inserts = 0;
    hits = 0;
    misses = 0;
  }

  size_t upper_power_of_two(size_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v++;
    return v;
  }
};

};  // namespace upcxx_utils
