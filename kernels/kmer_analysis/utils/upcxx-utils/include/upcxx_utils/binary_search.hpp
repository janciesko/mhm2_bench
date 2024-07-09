#pragma once

/*
 * File:   binary_search.hpp
 * Author: regan
 *
 * Created on June 19, 2020, 9:29 PM
 */

#include <cassert>
#include <functional>
#include <memory>

using std::function;
using std::make_shared;
using std::shared_ptr;

#include <upcxx/upcxx.hpp>

using upcxx::dist_object;
using upcxx::intrank_t;
using upcxx::make_future;
using upcxx::rank_me;
using upcxx::rank_n;

#include "upcxx_utils/log.hpp"
#include "upcxx_utils/version.h"

namespace upcxx_utils {

struct binary_search_pos {
  intrank_t n, low, high, mid;
  UPCXX_SERIALIZED_FIELDS(n, low, high, mid);

  binary_search_pos(intrank_t n = rank_n(), intrank_t low = 0, intrank_t high = rank_n() - 1);

  binary_search_pos(const upcxx::team &team);

  // calculates the new mid to test
  // return true if found at test_rank
  // may cause state to be !is_valid() or is_found() or is_nowhere()
  bool apply_cmp(int cmp, intrank_t test_rank);
  void set_found(intrank_t rank);
  intrank_t found() const;
  bool is_found() const;
  void set_nowhere();
  bool is_nowhere() const;
  bool is_valid() const;

  // returns a new binary_search_pos and Object.  Check is_found() for a valid return object
  // Object must have a member comparison operator - int Object::operator()(const dist_object<Object> &, const Elem &) const
  template <typename Object, typename Elem = Object>
  upcxx::future<binary_search_pos, Object> apply_fetch(dist_object<Object> &dist_obj, const Elem test) {
    DBG_VERBOSE(*this, "\n");
    if (is_valid()) {
      auto test_rank = mid;
      upcxx::future<Object> fut_obj;
      if (test_rank == dist_obj.team().rank_me())
        fut_obj = make_future(*dist_obj);
      else
        fut_obj = dist_obj.fetch(test_rank);
      return upcxx::when_all(make_future(*this), fut_obj)
          .then([&dist_obj, test, test_rank](binary_search_pos bsp, const Object &obj) -> upcxx::future<binary_search_pos, Object> {
            DBG_VERBOSE(bsp, "\n");
            assert(bsp.mid == test_rank);
            auto cmp = obj(dist_obj, test);
            auto found = bsp.apply_cmp(cmp, test_rank);
            if ((found && bsp.is_found()) || !bsp.is_valid()) return make_future(bsp, obj);
            return bsp.apply_fetch(dist_obj, test);
          });
    } else {
      return make_future(*this, *dist_obj);
    }
  }

  // returns a new binary_search_pos and Object.  Check is_found() for a valid return object
  // Object must have a member comparison operator - int Object::operator()(const dist_object<Object> &, const Elem &) const
  template <typename Object, typename Elem = Object, typename Prom = upcxx::promise<intrank_t, Object> >
  void apply_rpc(dist_object<Object> &dist_obj, const Elem test, intrank_t origin_rank, Prom *prom_ptr) {
    assert(mid < dist_obj.team().rank_n());
    DBG_VERBOSE(*this, " origin_rank=", origin_rank, " prom_ptr=", prom_ptr, "\n");
    rpc_ff(
        dist_obj.team(), mid,
        [](dist_object<Object> &dist_obj, const Elem &test, binary_search_pos bsp, Prom *prom_ptr, intrank_t origin_rank) {
          DBG_VERBOSE("binary_search_rpc_recursive::rpc, origin_rank=", origin_rank, "\n");
          const upcxx::team &team = dist_obj.team();
          const Object &obj = *dist_obj;
          auto cmp = obj(dist_obj, test);
          auto found = bsp.apply_cmp(cmp, team.rank_me());
          if ((found && bsp.is_found()) || bsp.is_nowhere()) {
            // return back to origin and fulfill its promise
            if constexpr (std::is_same<Prom, upcxx::promise<intrank_t> >::value) {
              rpc_ff(
                  team, origin_rank,
                  [](dist_object<Object> &dist_obj, Prom *prom_ptr, const binary_search_pos &bsp, intrank_t from_rank) {
                    DBG_VERBOSE("binary_search_rpc_recursive::rpc - recieved result from rank=", from_rank, " bsp=", bsp, "\n");
                    Prom &prom = *prom_ptr;
                    prom.fulfill_result(bsp.mid);
                  },
                  dist_obj, prom_ptr, bsp, dist_obj.team().rank_me());
            } else {
              static_assert(std::is_same<Prom, upcxx::promise<intrank_t, Object> >::value);
              rpc_ff(
                  team, origin_rank,
                  [](dist_object<Object> &dist_obj, Prom *prom_ptr, const Object &obj, const binary_search_pos &bsp,
                     intrank_t from_rank) {
                    DBG_VERBOSE("binary_search_rpc_recursive::rpc - recieved result from rank=", from_rank, " bsp=", bsp, "\n");
                    Prom &prom = *prom_ptr;
                    prom.fulfill_result(bsp.mid, obj);
                  },
                  dist_obj, prom_ptr, *dist_obj, bsp, dist_obj.team().rank_me());
            }
          } else {
            // keep on searching
            bsp.apply_rpc(dist_obj, test, origin_rank, prom_ptr);
          }
        },
        dist_obj, test, *this, prom_ptr, origin_rank);
  };

  string to_string() const;
  friend std::ostream &operator<<(std::ostream &os, const binary_search_pos &bsp);
};
std::ostream &operator<<(std::ostream &os, const binary_search_pos &bsp);

template <typename Object, typename Elem = Object>
upcxx::future<intrank_t, Object> binary_search_recurse_fetch_dist_obj(dist_object<Object> &dist_obj, const Elem test,
                                                                      binary_search_pos &bsp) {
  DBG_VERBOSE("bsp=", bsp, "\n");
  const upcxx::team &team = dist_obj.team();
  upcxx::future<binary_search_pos, Object> fut = bsp.apply_fetch(dist_obj, test);

  return fut.then([](const binary_search_pos &bsp, const Object &obj) {
    DBG_VERBOSE("bsp=", bsp, "\n");
    if (bsp.is_found()) {
      return make_future(bsp.mid, obj);
    } else {
      return make_future(bsp.n, obj);
    }
  });
};

// returns the rank and object for a match. if rank == n, then there was no hit
// Object must have a member comparison operator - int Object::operator()(const dist_object<Object> &, const Elem &) const
template <typename Object, typename Elem = Object>
upcxx::future<intrank_t, Object> binary_search_fetch(dist_object<Object> &dist_obj, const Elem test) {
  DBG_VERBOSE("\n");
  const upcxx::team &team = dist_obj.team();
  binary_search_pos bsp(team);
  return binary_search_recurse_fetch_dist_obj(dist_obj, test, bsp);
};

// returns the rank and object for a match. if rank == n, then there was no hit
// Object must have a member comparison operator - int Object::operator()(const dist_object<Object> &, const Elem &) const
template <typename Object, typename Elem = Object>
upcxx::future<intrank_t, Object> binary_search_rpc_obj(dist_object<Object> &dist_obj, const Elem test) {
  DBG_VERBOSE("\n");
  auto sh_prom = make_shared<upcxx::promise<intrank_t, Object> >(1);
  auto &prom = *sh_prom;
  const upcxx::team &team = dist_obj.team();

  // first check self for match
  binary_search_pos bsp(team);
  const Object &obj = *dist_obj;
  auto cmp = obj(dist_obj, test);
  auto found = bsp.apply_cmp(cmp, dist_obj.team().rank_me());
  if ((found && bsp.is_found()) || bsp.is_nowhere()) {
    // matched self or nowhere
    DBG_VERBOSE("Found match without rpc bsp=", bsp, "\n");
    prom.fulfill_result(bsp.mid, *dist_obj);
  } else {
    // no match to me, start rpc_ff loop, that will return
    bsp.apply_rpc(dist_obj, test, team.rank_me(), sh_prom.get());
  }
  // wait on the promise, and keep the promise in scope until fulfilled
  return sh_prom->get_future().then([sh_prom](intrank_t rank, const Object &obj) { return make_future(rank, obj); });
};

template <typename Object, typename Elem = Object>
upcxx::future<intrank_t> binary_search_rpc(dist_object<Object> &dist_obj, const Elem test) {
  DBG_VERBOSE("\n");
  auto sh_prom = make_shared<upcxx::promise<intrank_t> >(1);
  auto &prom = *sh_prom;
  const upcxx::team &team = dist_obj.team();

  // first check self for match
  binary_search_pos bsp(team);
  const Object &obj = *dist_obj;
  auto cmp = obj(dist_obj, test);
  auto found = bsp.apply_cmp(cmp, dist_obj.team().rank_me());
  if ((found && bsp.is_found()) || bsp.is_nowhere()) {
    // matched self or nowhere
    DBG_VERBOSE("Found match without rpc bsp=", bsp, "\n");
    prom.fulfill_result(bsp.mid);
  } else {
    // no match to me, start rpc_ff loop, that will return
    bsp.apply_rpc(dist_obj, test, team.rank_me(), sh_prom.get());
  }
  // wait on the promise, and keep the promise in scope until fulfilled
  return sh_prom->get_future().then([sh_prom](intrank_t rank) { return make_future(rank); });
};

};  // namespace upcxx_utils
