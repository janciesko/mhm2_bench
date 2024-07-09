#pragma once
/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   shared_global_ptr.hpp
 * Author: regan
 *
 * Created on April 29, 2020, 2:58 PM
 */

#include <cassert>
#include <atomic>
#include <exception>
#include <upcxx/upcxx.hpp>
#include <cassert>

#include "upcxx_utils/log.hpp"

using std::atomic;

using upcxx::global_ptr;
using upcxx::intrank_t;
using upcxx::memory_kind;
using upcxx::rank_me;
using upcxx::rank_n;

#define UPCXX_UTILS_DOUBLE_COPY_RPC

namespace upcxx_utils {

template <class Type, memory_kind Kind = memory_kind::host>
struct shared_global_ptr {
  using global_ptr_T = global_ptr<Type, Kind>;
  using global_ptr_Count = global_ptr<atomic<int64_t> >;
  using element_type = typename global_ptr_T::element_type;

  using Pair = std::pair<atomic<int64_t>, global_ptr_T>;  // negative for delete_array
  using GPair = global_ptr<Pair>;

 protected:
  GPair _gpair;

 public:
  shared_global_ptr()
      : _gpair() {
    assert(use_count() == 0);
    // DBG_VERBOSE("shared_global_ptr() this=", *this, "\n");
  }

  shared_global_ptr(global_ptr_T gptr, bool is_array = false)
      : _gpair() {
    if (gptr) {
      reset(gptr, is_array);
      assert(use_count() == 1);
    }
    // DBG_VERBOSE("shared_global_ptr(", gptr, ", is_array=", is_array, ") this=", *this, "\n");
  }

  shared_global_ptr(const shared_global_ptr &copy)
      : _gpair(copy._gpair) {
    // DBG_VERBOSE("shared_global_ptr(copy=", copy, ")\n");
    assert(this != &copy);
    if (_gpair) {
      assert(copy._gpair.is_local());
      assert(copy._gpair.local()->second);
      assert(_gpair.is_local());
      assert(_gpair.local()->second);
      add_count(_gpair);
    }
    // DBG_VERBOSE("shared_global_ptr(copied=", copy, ") this=", *this, "\n");
  }

  shared_global_ptr(shared_global_ptr &&move)
      : _gpair(std::move(move._gpair)) {
    assert(this != &move);
    // DBG_VERBOSE("shared_global_ptr(move=", move, ") this=", *this, "\n");
    move._gpair = nullptr;
  }

  shared_global_ptr &operator=(const shared_global_ptr &copy) {
    // DBG_VERBOSE("shared_global_ptr assign= copied=", copy, ") this=", *this, "\n");
    if (this == &copy) return *this;
    shared_global_ptr new_ptr(copy);
    this->swap(new_ptr);
    return *this;
  }

  shared_global_ptr &operator=(shared_global_ptr &&move) {
    // DBG_VERBOSE("shared_global_ptr assign= moved=", move, ") this=", *this, "\n");
    if (this == &move) return *this;
    shared_global_ptr new_ptr(std::move(move));
    this->swap(new_ptr);
    return *this;
  }

  ~shared_global_ptr() {
    // DBG_VERBOSE("~shared_global_ptr() this=", *this, "\n");
    if (!_gpair) return;
    assert(is_local());
    if (use_count()) {
      auto old_count = subtract_count(_gpair);
      // DBG_VERBOSE("old_count=", old_count, "\n");
#ifdef UPCXX_UTILS_DOUBLE_COPY_RPC
      int64_t last_count = 2;
#else
      int64_t last_count = 1;
#endif
      if (old_count == last_count || old_count == -last_count) {
        assert(_gpair.local()->first.load() == 0);
        destroy(_gpair, old_count == -last_count);
        assert(!*this);
      }
    }
    _gpair = nullptr;
  }

  void swap(shared_global_ptr &other) {
    // DBG_VERBOSE("shared_global_ptr::swap(this=", *this, ", other=", other, "\n");
    std::swap(_gpair, other._gpair);
  }

  void reset() {
    // DBG_VERBOSE("shared_global_ptr::reset() this=", *this, "\n");
    if (!_gpair) return;
    shared_global_ptr sgp;
    assert(!sgp._gpair);
    assert(!sgp);
    swap(sgp);
    assert(!_gpair);
    assert(!*this);
  }

  void reset(global_ptr_T gptr, bool is_array = false) {
    // DBG_VERBOSE("shared_global_ptr::reset(gptr=", gptr, ", is_array=", is_array, ") this=", *this, "\n");
    if (_gpair) reset();
    if (gptr) {
      assert(!_gpair);
      _gpair = construct(gptr, is_array);
      assert(use_count() == 1);
      assert(*this);
    }
  }

  bool is_null() const {
    // DBG_VERBOSE("shared_global_ptr::is_null() this=", *this, "\n");
    if (_gpair) {
      assert(!_gpair.is_null());
      assert(_gpair.is_local());
      return _gpair.local()->second.is_null();
    }
    return true;
  }

  explicit operator bool() const { return !is_null(); }

  bool is_local() const {
    if (_gpair) return _gpair.local()->second.is_local();
    return true;
  }

  Type *local() const {
    if (_gpair) {
      assert(is_local());
      return _gpair.local()->second.local();
    }
    return nullptr;
  }

  // cast to a global_ptr<Type>

  const global_ptr_T get() const {
    if (_gpair) return _gpair.local()->second;
    return global_ptr_T();
  }

  global_ptr_T get() {
    if (_gpair) return _gpair.local()->second;
    return global_ptr_T();
  }

  explicit operator const global_ptr_T() const { return get(); }

  explicit operator global_ptr_T() { return get(); }

  intrank_t where() const {
    if (_gpair) return _gpair.local()->second.where();
    return rank_n();
  }

  bool operator==(const shared_global_ptr &other) const { return _gpair == other._gpair; }

  bool operator!=(const shared_global_ptr &other) const { return _gpair != other._gpair; }

  size_t use_count() const {
    // counts are stored double because upcxx internals call increment twice during transit via UPCXX_SERIALIZED_VALUES
    if (_gpair) {
      assert(_gpair.is_local());
      assert(!_gpair.is_null());
      int64_t count = _gpair.local()->first.load();
#ifdef UPCXX_UTILS_DOUBLE_COPY_RPC
      if (count > 0) {
        count = (count + 1) / 2;  // atomic count can be odd in the middle of an RPC, but count it as a usage in the middle
      } else {
        count = (count - 1) / 2;
      }
#endif
      if (count < 0)
        return 0 - count;
      else
        return count;
    }
    return 0;
  }

  bool unique() const { return use_count() == 1; }

  friend std::ostream &operator<<(std::ostream &os, const shared_global_ptr &sgp) {
    os << "shared_global_ptr(this=" << &sgp << ", use_count=" << sgp.use_count() << ", " << sgp._gpair;
    if (sgp._gpair) {
      os << " - first=" << sgp._gpair.local()->first << ", second=" << sgp._gpair.local()->second;
    } else {
      os << " (nil) ";
    }
    os << ")";
    return os;
  }

  static const GPair &increment_partial(const GPair &gpair) {
    DBG_VERBOSE("increment copy on ", gpair, "\n");
    if (gpair) add_subtract_count(const_cast<GPair &>(gpair), true, false);
    return gpair;
  }

  UPCXX_SERIALIZED_VALUES(shared_global_ptr::increment_partial(_gpair));

  shared_global_ptr(GPair &&gpair)
      : _gpair(std::move(gpair)) {
    // DBG_VERBOSE("shared_global_ptr(&&gpair=", gpair, ") this=", *this, "\n");
  }

 protected:
  static int64_t add_subtract_count(GPair &gpair, bool add, bool two = true) {
    // DBG_VERBOSE("add_subtract_count on ", gpair, " ", add ? "add" : "subtract", "\n");
    assert(gpair);
    assert(gpair.is_local());
    Pair &pair = *(gpair.local());
    int64_t old_count = pair.first.load();
#ifdef UPCXX_UTILS_DOUBLE_COPY_RPC
    int64_t val = two ? 2 : 1;  // counts are always double, unless in transit during serialization, since increment is called twice
                                // by upcxx internals
#else
    int64_t val = 1;
#endif
    if (old_count == 0) return 0;
    bool positive = old_count > 0;
    old_count = pair.first.fetch_add(positive ? (add ? val : -val) : (add ? -val : val));
    return old_count;
  }

  static int64_t add_count(GPair &gpair) { return add_subtract_count(gpair, true, true); }

  static int64_t subtract_count(GPair &gpair) { return add_subtract_count(gpair, false, true); }

  static void _delete(GPair &gpair, bool is_array) {
    assert(!gpair.is_null());
    assert(gpair.is_local());
    assert(gpair.where() == rank_me());
    Pair &pair = *(gpair.local());
    DBG("shared_global_ptr::_delete gpair=", gpair, ", is_array=", is_array, ", pair.first=", pair.first,
        ", pair.second=", pair.second, "\n");
    assert(pair.first.load() == 0);
    assert(!pair.second.is_null());
    assert(pair.second.is_local());
    assert(pair.second.where() == rank_me());
    if (is_array)
      upcxx::delete_array(pair.second);
    else
      upcxx::delete_(pair.second);
    pair.second = nullptr;
    upcxx::delete_(gpair);
    gpair = nullptr;
  }

  static void destroy(GPair &gpair, bool is_array) {
    DBG("shared_global_ptr::destroy(gpair=", gpair, ", is_array=", is_array, "\n");
    // last use_count deallocate
    assert(gpair);
    assert(!gpair.is_null());
    assert(gpair.is_local());
    assert(gpair.local()->first.load() == 0);
    assert(gpair.local()->second);
    assert(gpair.local()->second.is_local());
    Pair &pair = *(gpair.local());
    if (gpair.where() == rank_me()) {
      _delete(gpair, is_array);
      assert(gpair.is_null());
    } else {
      DBG("Firing rpc to ", gpair.where(), " to delete ", gpair, "\n");
      rpc_ff(
          gpair.where(),
          [](GPair gpair, bool is_array) {
            DBG("rpc received to delete ", gpair, "\n");
            assert(gpair.is_local());
            assert(gpair.local()->first.load() == 0);
            assert(!gpair.local()->second.is_null());
            assert(gpair.local()->second.is_local());
            shared_global_ptr::_delete(gpair, is_array);
            return upcxx::make_future();
          },
          gpair, is_array);
      gpair = nullptr;
      upcxx::progress();
    }
    assert(gpair.is_null());
  }

  static GPair construct(global_ptr_T gptr, bool is_array) {
    // first use_count
    assert(gptr);
#ifdef UPCXX_UTILS_DOUBLE_COPY_RPC
    auto gpair = upcxx::new_<Pair>(is_array ? -2 : 2, gptr);  // negative counts for array
#else
    auto gpair = upcxx::new_<Pair>(is_array ? -1 : 1, gptr);  // negative counts for array
#endif
    DBG("shared_global_ptr::construct gpair=", gpair, ", is_array=", is_array, ", pair.first=", gpair.local()->first,
        ", pair.second=", gpair.local()->second, "\n");
    return gpair;
  }
};

};  // namespace upcxx_utils
