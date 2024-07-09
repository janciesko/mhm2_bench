#include "upcxx_utils/split_rank.hpp"

#include <exception>
#include <random>
#include <upcxx/upcxx.hpp>

#include "upcxx_utils/log.hpp"

using namespace upcxx;
using std::vector;

namespace upcxx_utils {

intrank_t foreach_rank_by_rank::operator()(const intrank_t position, const upcxx::team &t) const {
  assert(position >= 0);
  assert(position < t.rank_n());
  return (position + t.rank_me()) % t.rank_n();
}
intrank_t foreach_rank_by_node::operator()(const intrank_t position, const upcxx::team &t) const {
  assert(position >= 0);
  assert(position < t.rank_n());
  auto nlranks = local_team().rank_n();
  auto nnodes = t.rank_n() / nlranks;
  // auto ret = ((position % nnodes) * nlranks + position / nnodes + t.rank_me()) % t.rank_n();
  auto node = ((t.rank_me() / nlranks) + (position % nnodes)) % nnodes;
  auto lrank = ((t.rank_me() % nlranks) + (position / nnodes)) % nlranks;
  auto ret = node * nlranks + lrank;
  DBG_VERBOSE("position=", position, " ret=", ret, "\n");
  return ret;
}

void foreach_rank_by_random::init(const upcxx::team &t) {
  assert(nodes.empty());
  assert(lranks.empty());
  auto nlranks = local_team().rank_n();
  if (t.rank_n() % nlranks != 0)
    DIE("Can not call foreach_rank_by_random if the team is irregular with respect to local_team: rank_n=", t.rank_n(),
        " local=", local_team().rank_n());
  auto nnodes = t.rank_n() / nlranks;
  nodes.reserve(nnodes);
  lranks.reserve(nlranks);
  for (intrank_t i = 0; i < nnodes; i++) nodes.push_back(i);
  for (intrank_t i = 0; i < nlranks; i++) lranks.push_back(i);
  std::mt19937 g(t.rank_me());  // seed based on rank_me
  std::shuffle(nodes.begin(), nodes.end(), g);
  std::shuffle(lranks.begin(), lranks.end(), g);
}
intrank_t foreach_rank_by_random::operator()(const intrank_t position, const upcxx::team &t) const {
  assert(position >= 0);
  assert(position < t.rank_n());
  auto nlranks = local_team().rank_n();
  if (nodes.empty() || lranks.empty() || (t.rank_n() + nlranks - 1) / nlranks != nodes.size() || nlranks != lranks.size())
    DIE("Must first initialize foreach_rank_by_random, nodes=", nodes.size(), " t_n=", t.rank_n(), " nlranks=", nlranks,
        ", lranks=", lranks.size());
  auto nnodes = nodes.size();
  auto nodei = position % nnodes;  // round robin by node first
  assert(nodei < nnodes);
  auto node = nodes[nodei];
  auto lranki = (position / nnodes + node) % nlranks;
  assert(lranki < lranks.size());
  auto lrank = lranks[lranki];
  return node * nlranks + lrank;
}
void foreach_rank_by_random_full::init(const upcxx::team &t) {
  assert(ranks.empty());
  auto nranks = t.rank_n();
  ranks.reserve(nranks);
  for (intrank_t i = 0; i < nranks; i++) ranks.push_back(i);
  std::mt19937 g(t.rank_me());  // seed based on rank_me
  std::shuffle(ranks.begin(), ranks.end(), g);
}
intrank_t foreach_rank_by_random_full::operator()(const intrank_t position, const upcxx::team &t) const {
  assert(position >= 0);
  assert(position < t.rank_n());
  if (ranks.empty() || position >= ranks.size()) DIE("Must first initialize foreach_rank_by_random_full\n");
  return ranks[position];
}

foreach_rank_t::foreach_rank_t(const upcxx::team &myteam)
    : myteam(myteam) {
  DBG_VERBOSE("Constructed for myteam_rank_n()=", myteam.rank_n(), " me=", myteam.rank_me(), "\n");
}

foreach_rank_t::iterator::iterator(const foreach_rank_t *ordering, const upcxx::team &myteam, intrank_t pos)
    : myteam(myteam)
    , pos(pos)
    , ordering(ordering) {}
foreach_rank_t::iterator::iterator(const iterator &copy)
    : myteam(copy.myteam)
    , pos(copy.pos)
    , ordering(copy.ordering) {}

foreach_rank_t::iterator &foreach_rank_t::iterator::operator=(const iterator &copy) {
  iterator newit(copy);
  std::swap(*this, newit);
  return *this;
}
intrank_t foreach_rank_t::iterator::operator*() const {
  assert(pos < myteam.rank_n());
  auto ret = (*ordering)(pos, myteam);
  DBG_VERBOSE("pos=", pos, " ret=", ret, "\n");
  return ret;
}
intrank_t foreach_rank_t::iterator::operator->() const { return *(*this); }
foreach_rank_t::iterator &foreach_rank_t::iterator::operator++() {
  assert(pos < myteam.rank_n());
  pos++;
  return *this;
}
foreach_rank_t::iterator foreach_rank_t::iterator::operator++(int) {
  auto copy = *this;
  ++(*this);
  return copy;
}
bool foreach_rank_t::iterator::operator==(const iterator other) const { return pos == other.pos && ordering == other.ordering; }
bool foreach_rank_t::iterator::operator!=(const iterator other) const { return !(*this == other); }
foreach_rank_t::iterator foreach_rank_t::begin() const { return iterator(this, myteam, 0); }
foreach_rank_t::iterator foreach_rank_t::end() const { return iterator(this, myteam, myteam.rank_n()); }

pair<intrank_t, intrank_t> lead_and_last_in_local_team(const upcxx::team &team, bool quick) {
  intrank_t lme = local_team().rank_me();
  intrank_t ln = local_team().rank_n();
  intrank_t tme = team.rank_me();
  intrank_t tn = team.rank_n();

  pair<intrank_t, intrank_t> lead_last;
  if (team.id() == world().id()) {
    lead_last.first = tme - lme;
    lead_last.second = tme + (ln - 1 - lme);
    return lead_last;
  } else {
    lead_last.first = tme;
    lead_last.second = tme;
    if (ln == 1) return lead_last;
  }
  assert(ln > 1);

  // optimistically assume team contains all local_team and the ranks are in the same order
  bool is_regular = quick && tn % ln == 0 && tme % ln == lme;
  if (is_regular) {
    if (tme >= lme) {
      intrank_t test_rank = tme - lme;
      if (local_team_contains(team[test_rank])) {
        lead_last.first = test_rank;
        if (test_rank > 0 && local_team_contains(team[test_rank - 1])) {
          is_regular = false;
        }
        if (test_rank < tn - 1 && !local_team_contains(team[test_rank + 1])) {
          is_regular = false;
        }
      } else {
        is_regular = false;
      }
    } else {
      is_regular = false;
    }

    if (is_regular && ln - 1 >= tme) {
      intrank_t test_rank = tme + (ln - 1 - tme);
      if (test_rank < tn && local_team_contains(team[test_rank])) {
        lead_last.second = test_rank;
        if (test_rank > 0 && !local_team_contains(team[test_rank - 1])) {
          is_regular = false;
        }
        if (test_rank < tn - 1 && local_team_contains(team[test_rank + 1])) {
          is_regular = false;
        }
      } else {
        is_regular = false;
      }
    } else {
      is_regular = false;
    }
  }

  if (!is_regular) {
    // exhaustively search through all team ranks
    for (intrank_t test_rank = 0; test_rank < tn; test_rank++) {
      if (local_team_contains(team[test_rank])) {
        if (test_rank < lead_last.first) lead_last.first = test_rank;
        if (test_rank > lead_last.second) lead_last.second = test_rank;
      }
    }
  }
  return lead_last;
}

intrank_t next_in_local_team(const upcxx::team &team) {
  intrank_t next = team.rank_me() + 1;
  if (next < team.rank_n() && local_team_contains(team[next]))
    return next;
  else
    return team.rank_n();
}

intrank_t prev_in_local_team(const upcxx::team &team) {
  intrank_t prev = team.rank_me();
  if (prev > 0 && local_team_contains(team[--prev]))
    return prev;
  else
    return team.rank_n();
}

future<intrank_t> regular_local_team_n(const upcxx::team &team, bool contains_full_local_team) {
  intrank_t tme = team.rank_me();
  intrank_t tn = team.rank_n();
  intrank_t next = next_in_local_team(team);
  intrank_t prev = prev_in_local_team(team);
  intrank_t lme = local_team().rank_me();
  intrank_t ln = local_team().rank_n();
  uint64_t my_result = ln;
  if (contains_full_local_team) {
    my_result = tme % ln == lme ? ln : 0;
  }
  if (my_result && ln > 1) {
    if (lme == 0 && prev != tn) my_result = 0;
    if (lme != 0 && prev == tn) my_result = 0;
    if (lme < ln - 1 && next == tn) my_result = 0;
    if (lme == ln - 1 && next != tn) my_result = 0;
  }
  return upcxx::reduce_all(my_result, upcxx::op_fast_add, team).then([&team](uint64_t sum) {
    if (sum == (uint64_t)team.rank_n() * (uint64_t)local_team().rank_n()) {
      return local_team().rank_n();
    } else {
      return 0;
    }
  });
}

//
// split_rank
//
intrank_t split_rank::get_split_node_rank_n() {
  int split_node_rank_n = local_team().rank_n();
  if (local_team().id() == world().id() && world().rank_n() > 1) {
    intrank_t test_mod = 2;
    intrank_t ln = local_team().rank_n();
    split_node_rank_n = 1;
    while (test_mod <= ln) {
      if (ln % test_mod == 0) {
        split_node_rank_n = ln / test_mod;
        break;
      }
      test_mod++;
    }
  }
  return split_node_rank_n;
}

const upcxx::team &split_rank::split_local_team() {
#ifdef DEBUG
  /* if the local team == world and >1 ranks, split the local team so there are >=2 nodes in a debug run */
  auto split_node_rank_n = get_split_node_rank_n();
  static upcxx::team localTeam = upcxx::world().split(rank_me() / split_node_rank_n, rank_me());
  return localTeam;
#else   // not def DEBUG
  return upcxx::local_team();
#endif  // def DEBUG
}

split_rank::split_rank(intrank_t world_rank)
    : node(0)
    , thread(0) {
  set(world_rank);
}

void split_rank::set(intrank_t world_rank) {
  node = get_node_from_rank(world_rank);
  thread = world_rank - node * split_local_team().rank_n();
  assert(world_rank == get_rank());
}

// returns the world rank

intrank_t split_rank::get_rank() const { return node * split_local_team().rank_n() + thread; }

// returns the node -1 .. nodes-2

node_num_t split_rank::get_node() { return node; }

// returns the thread 0 .. threads-1 or -1 if it is rank_me()

thread_num_t split_rank::get_thread() { return thread; }

bool split_rank::is_local() { return node == get_my_node(); }

// store some statics to cache often calculated data relative to me

node_num_t split_rank::get_my_node() {
  static node_num_t _ = rank_me() / split_local_team().rank_n();
  return _;
}

thread_num_t split_rank::get_my_thread() {
  static thread_num_t _ = split_local_team().rank_me();
  return _;
}

intrank_t split_rank::get_first_local() {
  static intrank_t _ = get_my_node() * split_local_team().rank_n();
  return _;
}

intrank_t split_rank::get_last_local() {
  static intrank_t _ = get_first_local() + split_local_team().rank_n() - 1;
  return _;
}

node_num_t split_rank::get_node_from_rank(intrank_t target_rank) {
  node_num_t target_node = target_rank / split_local_team().rank_n();
  return target_node;
}

intrank_t split_rank::get_rank_from_thread(thread_num_t target_thread) { return get_first_local() + target_thread; }

intrank_t split_rank::get_rank_from_node(node_num_t target_node) {
  return target_node * split_local_team().rank_n() + get_my_thread();
}

split_rank split_rank::from_thread(thread_num_t target_thread) {
  split_rank sr;
  sr.node = get_my_node();
  sr.thread = target_thread;
  assert(sr.get_rank() == get_rank_from_thread(target_thread));
  return sr;
}

split_rank split_rank::from_node(node_num_t target_node) {
  split_rank sr;
  sr.node = target_node;
  sr.thread = get_my_thread();
  assert(sr.get_rank() == get_rank_from_node(target_node));
  return sr;
}

node_num_t split_rank::num_nodes() {
  static node_num_t _ = rank_n() / split_local_team().rank_n();
  assert(_ * split_local_team().rank_n() == upcxx::rank_n());
  return _;
}

thread_num_t split_rank::num_threads() {
  static thread_num_t _ = split_local_team().rank_n();
  assert(_ * num_nodes() == upcxx::rank_n());
  return _;
}

bool split_rank::all_local_ranks_are_local_castable() {
  static bool _ = upcxx::local_team_contains(get_first_local()) & upcxx::local_team_contains(get_last_local());
  return _;
}

bool split_rank::all_ranks_are_local_castable() {
  static bool _ = upcxx::local_team_contains(0) & upcxx::local_team_contains(upcxx::rank_n() - 1);
  return _;
}

//
// split_team
//

split_team::split_team(const upcxx::team &full_, const upcxx::team &thread_)
    : _full_team(full_)
    , _thread_team(thread_)
    , _node_team(_full_team.split(_thread_team.rank_me(), _full_team.rank_me())) {
  if (_full_team.rank_n() % _thread_team.rank_n() != 0) throw std::runtime_error("Mismatch between full and thread teams");
  assert(_node_team.rank_n() == _full_team.rank_n() / _thread_team.rank_n());
  DBG_VERBOSE("full=", full_me(), "of", full_n(), " node=", node_me(), "of", node_n(), " thread=", thread_me(), "of", thread_n(),
              "\n");
}

split_team::~split_team() {
  // do not destroy if finalize has already been called
  if (upcxx::initialized()) {
    _node_team.destroy();
  } else {
    SWARN("~split_team called after finalize\n");
  }
}

// *_from_thread returns ranks within this thread_team (i.e. local)

intrank_t split_team::world_from_thread(thread_num_t thread_rank) const {
  assert(thread_rank < thread_n());
  auto r = _thread_team[thread_rank];
  assert(local_team_contains(r));
  assert(r < rank_n());
  return r;
}

intrank_t split_team::full_from_thread(thread_num_t thread_rank) const {
  assert(thread_rank < thread_n());
  auto r = _full_team.rank_me() - _thread_team.rank_me() + thread_rank;
  assert(local_team_contains(world_from_full(r)));
  assert(r < full_n());
  return r;
}

intrank_t split_team::local_from_thread(thread_num_t thread_rank) const {
  assert(thread_rank < thread_n());
  auto w = world_from_thread(thread_rank);
  assert(local_team_contains(w));
  auto r = local_team().rank_me() - _thread_team.rank_me() + thread_rank;
  assert(r < local_team().rank_n());
  return r;
}

bool split_team::thread_team_contains_world(intrank_t world_rank) const {
  assert(world_rank < rank_n());
  auto r = rank_me() - _thread_team.rank_me();
  return world_rank >= r && world_rank < r + _thread_team.rank_n();
}

bool split_team::thread_team_contains_full(intrank_t full_rank) const {
  assert(full_rank < full_n());
  auto r = _full_team.rank_me() - _thread_team.rank_me();
  return full_rank >= r && full_rank < r + _thread_team.rank_n();
}

bool split_team::thread_team_contains_node(node_num_t node_rank) const {
  assert(node_rank < node_n());
  return node_rank == _node_team.rank_me();
}

// *_from_full returns ranks within the full_team (i.e. world)

thread_num_t split_team::thread_from_full(intrank_t full_rank) const {
  assert(full_rank < full_n());
  auto r = (thread_num_t)(full_rank % _thread_team.rank_n());
  assert(r < thread_n());
  return r;
}

intrank_t split_team::world_from_full(intrank_t full_rank) const {
  assert(full_rank < full_n());
  auto r = _full_team[full_rank];
  assert(r < rank_n());
  return r;
}

node_num_t split_team::node_from_full(intrank_t full_rank) const {
  assert(full_rank < full_n());
  auto r = (node_num_t)(full_rank / _thread_team.rank_n());
  assert(r < node_n());
  return r;
}

// inefficient

bool split_team::full_team_contains_world(intrank_t world_rank) const {
  assert(world_rank < rank_n());
  return _full_team.from_world(world_rank, rank_n()) != rank_n();
}

// throws if full is not from world
// inefficient

thread_num_t split_team::thread_from_world(intrank_t world_rank) const {
  assert(world_rank < rank_n());
  if (!full_team_contains_world(world_rank)) throw std::runtime_error("invalid call to thread_from_world");
  auto r = (thread_num_t)world_rank % _thread_team.rank_n();
  assert(r < thread_n());
  return r;
}

// throws if full is not from world
// inefficient

intrank_t split_team::full_from_world(intrank_t world_rank) const {
  assert(world_rank < rank_n());
  if (!full_team_contains_world(world_rank)) throw std::runtime_error("invalid call to thread_from_world");
  auto r = world_rank % _full_team.rank_n();
  assert(r < full_n());
  return r;
}

// throws if full is not from world
// returns a rank from the node (with the same thread_me())
// inefficient

node_num_t split_team::node_from_world(intrank_t world_rank) const {
  assert(world_rank < rank_n());
  auto r = (node_num_t)full_from_world(world_rank) / _thread_team.rank_n();
  assert(r < node_n());
  return r;
}

// *_from_node returns ranks within the full_team (i.e. world)
// returns the world_rank with the same thread_rank offset as me

intrank_t split_team::world_from_node(node_num_t node_rank) const {
  assert(node_rank < node_n());
  return _node_team[node_rank];
}

intrank_t split_team::full_from_node(node_num_t node_rank) const {
  assert(node_rank < node_n());
  auto r = node_rank * _thread_team.rank_n() + _thread_team.rank_me();
  assert(r < full_n());
  return r;
}

bool split_team::local_team_contains_thread(thread_num_t thread_rank) const {
  assert(thread_rank < thread_n());
  return local_team_contains(world_from_thread(thread_rank));
}

bool split_team::local_team_contains_full(intrank_t full_rank) const {
  assert(full_rank < full_n());
  return local_team_contains(world_from_full(full_rank));
}

bool split_team::local_team_contains_node(node_num_t node_rank) const {
  assert(node_rank < node_n());
  return local_team_contains(world_from_node(node_rank));
}

};  // namespace upcxx_utils
