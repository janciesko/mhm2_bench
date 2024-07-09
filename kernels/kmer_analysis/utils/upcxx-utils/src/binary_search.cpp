#include "upcxx_utils/binary_search.hpp"

using upcxx::make_future;

namespace upcxx_utils {

binary_search_pos::binary_search_pos(intrank_t n, intrank_t low, intrank_t high)
    : n(n)
    , low(low)
    , high(high) {
  assert(low >= 0);
  assert(high < n);
  mid = (high + low) / 2;
}
binary_search_pos::binary_search_pos(const upcxx::team &team)
    : n(team.rank_n())
    , low(0)
    , high(team.rank_n() - 1)
    , mid(team.rank_me()) {}

// calculates the new mid to test
// return true if found at test_rank
// may cause state to be !is_valid() or is_found() or is_nowhere()
bool binary_search_pos::apply_cmp(int cmp, intrank_t test_rank) {
  DBG_VERBOSE("cmp=", cmp, " test_rank=", test_rank, ", this=", *this, "\n");
  if (test_rank < low || test_rank > high || low >= n || high >= n || low > high) {
    set_nowhere();
    return false;
  }
  if (cmp == 0) {
    DBG_VERBOSE("found\n");
    set_found(test_rank);  // it is this
    assert(is_found());
    return true;
  } else if (cmp < 0) {
    // lower if it exists
    if (test_rank <= low) {
      set_nowhere();
    } else {
      high = test_rank - 1;
      DBG_VERBOSE("lower high=", high, "\n");
    }
  } else {
    // higher if it exists
    low = test_rank + 1;
    DBG_VERBOSE("higher low=", low, "\n");
  }
  if (low <= high && high < n) {
    mid = (low + high) / 2;
    assert(mid != test_rank);
    DBG_VERBOSE("mid=", mid, "\n");
  } else {
    set_nowhere();
  }
  assert(!is_found());
  return false;
}
void binary_search_pos::set_found(intrank_t rank) {
  // only mid != n
  mid = rank;
  low = high = n;
  assert(is_found());
  assert(!is_nowhere());
}
intrank_t binary_search_pos::found() const {
  assert(is_found());
  assert(!is_nowhere());
  return mid;
}
bool binary_search_pos::is_found() const {
  // only mid != n
  return mid != n && low == n && high == n;
}
void binary_search_pos::set_nowhere() {
  DBG_VERBOSE("is nowhere\n");
  // everything is n
  low = high = mid = n;
  assert(is_nowhere());
  assert(!is_found());
}
bool binary_search_pos::is_nowhere() const {
  // everything is n
  return mid == n && low == n && high == n;
}
bool binary_search_pos::is_valid() const {
  // nothing is n
  return mid < n && low < n && high < n && low <= high;
}

string binary_search_pos::to_string() const {
  stringstream ss;
  _logger_recurse(ss, "binary_search_pos(mid=", mid, ", low=", low, ", high=", high, ", n=", n, " is_found()=", is_found(),
                  " is_nowhere()=", is_nowhere(), ")");
  return ss.str();
}

std::ostream &operator<<(std::ostream &os, const binary_search_pos &bsp) { return os << bsp.to_string(); }

};  // namespace upcxx_utils
