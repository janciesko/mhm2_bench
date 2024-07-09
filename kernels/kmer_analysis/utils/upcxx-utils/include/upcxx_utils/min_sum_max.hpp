#pragma once

#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <upcxx/upcxx.hpp>
#include <vector>

using std::make_shared;
using std::ostream;
using std::ostringstream;
using std::shared_ptr;
using std::string;
using std::vector;

using upcxx::intrank_t;
using upcxx::world;

#include "upcxx_utils/log.hpp"

namespace upcxx_utils {

template <typename T>
class MinSumMax {
 public:
  MinSumMax() { reset(); }
  MinSumMax(T my_val, bool is_active = true) { reset(my_val, is_active); }

  // construct a MinSumMax from another type and factor
  template <typename T2>
  MinSumMax(const T2 &other, T factor)
      : my(other.my * factor)
      , min(other.min * factor)
      , sum(other.sum * factor)
      , max(other.max * factor)
      , avg(other.avg * factor)
      , min_rank(other.min_rank)
      , max_rank(other.max_rank) {}

  void reset(T my_val, bool is_active = true) {
    my = my_val;
    min = my_val;
    sum = my_val;
    max = my_val;
    avg = 0.0;
    if (is_active) {
      min_rank = max_rank = upcxx::rank_me();
      active_ranks = 1;
    } else {
      min_rank = upcxx::rank_n();
      max_rank = -1;
      active_ranks = 0;
    }
  }

  // inactive
  void reset() { reset(T{}, false); }

  string to_string(bool include_rank_info = true) const {
    if (!is_valid()) return "NOT VALID MinSumMax";
    ostringstream oss;
    oss << std::setprecision(2) << std::fixed;
    double bal = (max != 0 ? avg / max : 1.0);
    oss << min << "/" << my << "/" << avg << "/" << max << ", ";
    oss << "bal=" << bal;
    if (include_rank_info) {
      if (active_ranks > 1 && min != max && max_rank >= 0) oss << " " << min_rank << ":" << max_rank;
      oss << " active=" << active_ranks;
    }
    auto s = oss.str();
    return s;
  }

  bool is_valid() const { return min <= my && min <= max && min <= avg && max >= my && max >= avg && sum >= fabs(avg); }

  struct op_MinSumMax {
    template <typename Ta, typename Tb>
    MinSumMax operator()(Ta a, Tb &&b) const {
      // skip .myq
      // DBG("before a.min=", a.min, " a.max=", a.max, " ", a.min_rank, " ", a.max_rank, " b.min=", b.min, " b.max=", b.max, " ",
      // b.min_rank, " ", b.max_rank, a.min <= b.min ? " a.min <= b.min " : " a.min > b.min ", a.max >= b.max ? " a.max >= b.max " :
      // " a.max < b.max ", "\n");
      a.min = upcxx::op_fast_min(a.min, std::forward<T>(b.min));
      a.sum = upcxx::op_fast_add(a.sum, std::forward<T>(b.sum));
      a.max = upcxx::op_fast_max(a.max, std::forward<T>(b.max));
      a.min_rank = a.min == std::forward<T>(b.min) ? std::forward<int>(b.min_rank) : a.min_rank;
      a.max_rank = a.max == std::forward<T>(b.max) ? std::forward<int>(b.max_rank) : a.max_rank;
      a.active_ranks = upcxx::op_fast_add(a.active_ranks, std::forward<int>(b.active_ranks));
      // DBG("after a.min=", a.min, " a.max=", a.max, " ", a.min_rank, " ", a.max_rank, "\n");

      return static_cast<MinSumMax &&>(a);
    };
  };
  void apply_avg(intrank_t n = 0) {
    if (n <= 0) n = active_ranks;
    avg = n > 0 ? ((double)sum) / n : sum;
  }

  void apply_avg(const upcxx::team &team) { apply_avg(team.rank_n()); }

  T my, min, sum, max;
  int min_rank, max_rank, active_ranks;
  double avg;
};

template <typename T>
ostream &operator<<(ostream &os, const MinSumMax<T> &msm) {
  return os << msm.to_string();
};

template <typename T>
upcxx::future<> min_sum_max_reduce_one(const MinSumMax<T> *msm_in, MinSumMax<T> *msm_out, int count, intrank_t root = 0,
                                       const upcxx::team &team = world()) {
  using MSM = MinSumMax<T>;

  auto sh_op = make_shared<typename MSM::op_MinSumMax>();
  typename MSM::op_MinSumMax &op = *sh_op;
  vector<T> orig_vals(count);
  for (int i = 0; i < count; i++) {
    orig_vals[i] = msm_in[i].my;
  }
  return upcxx::reduce_one(msm_in, msm_out, count, op, root, team)
      .then([msm_out, count, root, &team, orig_vals = std::move(orig_vals), sh_op]() {
        for (auto i = 0; i < count; i++) {
          if (team.rank_me() == root) {
            assert(msm_out[i].active_ranks >= 0 && msm_out[i].active_ranks <= team.rank_n());
            msm_out[i].apply_avg(msm_out[i].active_ranks);
            msm_out[i].my = orig_vals[i];
          }
        }
      });
};

template <typename T>
upcxx::future<MinSumMax<T>> min_sum_max_reduce_one(const T my, bool is_active, intrank_t root = 0,
                                                   const upcxx::team &team = world()) {
  using MSM = MinSumMax<T>;
  auto sh_msm = make_shared<MSM>(my);
  return min_sum_max_reduce_one(sh_msm.get(), sh_msm.get(), 1, root, team).then([sh_msm, &team]() { return *sh_msm; });
};

template <typename T>
upcxx::future<MinSumMax<T>> min_sum_max_reduce_one(const T my, intrank_t root = 0, const upcxx::team &team = world()) {
  return min_sum_max_reduce_one(my, true, root, team);
};

template <typename T>
upcxx::future<> min_sum_max_reduce_all(const MinSumMax<T> *msm_in, MinSumMax<T> *msm_out, int count,
                                       const upcxx::team &team = world()) {
  using MSM = MinSumMax<T>;
  auto sh_op = make_shared<typename MSM::op_MinSumMax>();
  typename MSM::op_MinSumMax &op = *sh_op;
  vector<T> orig_vals(count);
  for (int i = 0; i < count; i++) {
    orig_vals[i] = msm_in[i].my;
  }

  auto fut = upcxx::reduce_all(msm_in, msm_out, count, op, team);
  auto fut2 = fut.then([msm_out, count, &team, orig_vals = std::move(orig_vals), sh_op]() {
    for (auto i = 0; i < count; i++) {
      assert(msm_out[i].active_ranks <= team.rank_n());
      msm_out[i].apply_avg(msm_out[i].active_ranks);
      msm_out[i].my = orig_vals[i];
    }
  });
  return fut2;
};

template <typename T>
upcxx::future<MinSumMax<T>> min_sum_max_reduce_all(const T my, bool is_active, const upcxx::team &team = world()) {
  using MSM = MinSumMax<T>;
  auto sh_msm = make_shared<MSM>(my);
  return min_sum_max_reduce_all(sh_msm.get(), sh_msm.get(), 1, team).then([sh_msm, &team]() { return *sh_msm; });
};

template <typename T>
upcxx::future<MinSumMax<T>> min_sum_max_reduce_all(const T my, const upcxx::team &team = world()) {
  return min_sum_max_reduce_all(my, true, team);
};

  // Reduce compile time by declaring extern templates of common types
  // template instantiations happen in src/CMakeList and timers-extern-template.in.cpp

#define MACRO_MIN_SUM_MAX(TYPE, MODIFIER)                                                                                          \
  MODIFIER class MinSumMax<TYPE>;                                                                                                  \
  MODIFIER upcxx::future<> min_sum_max_reduce_one<TYPE>(const MinSumMax<TYPE> *, MinSumMax<TYPE> *, int, intrank_t,                \
                                                        const upcxx::team &);                                                      \
  MODIFIER upcxx::future<> min_sum_max_reduce_all<TYPE>(const MinSumMax<TYPE> *, MinSumMax<TYPE> *, int, const upcxx::team &team); \
  MODIFIER upcxx::future<MinSumMax<TYPE>> min_sum_max_reduce_one<TYPE>(const TYPE, intrank_t, const upcxx::team &);                \
  MODIFIER upcxx::future<MinSumMax<TYPE>> min_sum_max_reduce_all<TYPE>(const TYPE, const upcxx::team &team);

MACRO_MIN_SUM_MAX(float, extern template);
MACRO_MIN_SUM_MAX(double, extern template);
MACRO_MIN_SUM_MAX(int64_t, extern template);
MACRO_MIN_SUM_MAX(uint64_t, extern template);
MACRO_MIN_SUM_MAX(int, extern template);

};  // namespace upcxx_utils
