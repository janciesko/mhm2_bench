#pragma once
// split_rank.h

#include <iterator>
#include <memory>
#include <upcxx/upcxx.hpp>
#include <utility>
#include <vector>

using node_num_t = uint32_t;   /* i.e .0 to 4b */
using thread_num_t = uint16_t; /* i.e. 0 to 64k */

using std::make_shared;
using std::pair;
using std::shared_ptr;
using std::vector;

using upcxx::intrank_t;
using upcxx::local_team;
using upcxx::rank_me;
using upcxx::rank_n;
using upcxx::world;

namespace upcxx_utils {

class foreach_rank_t {
  // use as a replacement iterator over a (load balanced) ordering of a team's ranks

  // usage
  // for(auto rank : foreach_rank()) { ... }
  // for(intrank_t rank : foreach_rank_by_random(local_team())) { ... }

 public:
  virtual intrank_t operator()(const intrank_t position, const upcxx::team &) const = 0;

  foreach_rank_t(const upcxx::team &myteam = world());

  class iterator {
   protected:
    const upcxx::team &myteam;
    intrank_t pos;
    const foreach_rank_t *ordering;

   public:
    iterator(const foreach_rank_t *, const upcxx::team &, intrank_t);
    iterator(const iterator &);
    iterator &operator=(const iterator &);
    intrank_t operator*() const;
    intrank_t operator->() const;
    iterator &operator++();
    iterator operator++(int);
    bool operator==(const iterator other) const;
    bool operator!=(const iterator other) const;
  };

  iterator begin() const;
  iterator end() const;

 protected:
  const upcxx::team &myteam;
};

//
// specialized foreach_rank orderings
//

class foreach_rank_by_rank : public foreach_rank_t {
 public:
  // just stagger by rank_me(), order by rank
  foreach_rank_by_rank(const upcxx::team &myteam = world())
      : foreach_rank_t(myteam) {}
  intrank_t operator()(const intrank_t position, const upcxx::team &) const;
};

class foreach_rank_by_node : public foreach_rank_t {
 public:
  // (default) stagger by rank_me(), round robin order by node first, then by local rank
  intrank_t operator()(const intrank_t position, const upcxx::team &) const;
  foreach_rank_by_node(const upcxx::team &myteam = world())
      : foreach_rank_t(myteam) {}
};

class foreach_rank_by_random : public foreach_rank_t {
 public:
  // fully shuffled set of nodes, then a constant ordering of shuffled ranks
  // round robin by node (random order), then by rank (also random order)
  // (consumes a little memory O(rank_n()/local_rank_n() + local_rank_n())
  foreach_rank_by_random(const upcxx::team &myteam = world())
      : foreach_rank_t(myteam) {
    init(myteam);
  }
  intrank_t operator()(const intrank_t position, const upcxx::team &) const;
  void init(const upcxx::team &t);

 protected:
  vector<intrank_t> nodes;
  vector<intrank_t> lranks;
};

class foreach_rank_by_random_full : public foreach_rank_t {
 public:
  // fully shuffled set of ranks
  // (consumes some more memory O(rank_n()))
  foreach_rank_by_random_full(const upcxx::team &myteam = world())
      : foreach_rank_t(myteam) {
    init(myteam);
  }
  intrank_t operator()(const intrank_t position, const upcxx::team &) const;
  void init(const upcxx::team &t);

 protected:
  vector<intrank_t> ranks;
};

//
// default is by node as that is implicitly load balancing in most cases
// and consumes no extra memory
//
typedef foreach_rank_by_node foreach_rank;

// returns the lowest and highest team ranks that are members the local_team for this process
// if quick is true, then only make a few checks for regular monotonicity of team with the local_team
// otherwise test all members of team

pair<intrank_t, intrank_t> lead_and_last_in_local_team(const upcxx::team &team, bool quick = true);

// returns the next rank in the team if it is also in the local team, team.rank_n() otherwise
intrank_t next_in_local_team(const upcxx::team &team);

// returns the previous rank in the team if it is also in the local team, team.rank_n() otherwise
intrank_t prev_in_local_team(const upcxx::team &team);

// performs a reduction and ensures the local_team is monotonic and regular with the team
// returns the number of team members in every local_team or 0 if it is not regular
upcxx::future<intrank_t> regular_local_team_n(const upcxx::team &team, bool contains_full_local_team = false);

// node and threads are the mappings between intrank_t divided and modulo by the local_team
// node is the inclusive range (0, (rank_n() / local_team().rank_n() - 1))
// thread is the inclusive range (0, (local_team().rank_n() - 1))
class split_rank {
  // get the node and thread indexes, excluding local node
  node_num_t node;
  thread_num_t thread;

 public:
  // primary use case is to help with debugging on a single node
  // returns the local_team() in release builds
  // but in 1 node DEBUG builds, returns the local_team() split in 2
  static const upcxx::team &split_local_team();
  // returns the number of ranks in each of the split nodes
  static intrank_t get_split_node_rank_n();

  split_rank(intrank_t world_rank = rank_me());

  void set(intrank_t world_rank);

  // returns the world rank
  intrank_t get_rank() const;

  // returns the node -1 .. nodes-2
  node_num_t get_node();

  // returns the thread 0 .. threads-1 or -1 if it is rank_me()
  thread_num_t get_thread();

  bool is_local();

  // store some statics to cache often calculated data relative to me
  static node_num_t get_my_node();

  static thread_num_t get_my_thread();

  static intrank_t get_first_local();

  static intrank_t get_last_local();

  // returns the node for a given rank
  static node_num_t get_node_from_rank(intrank_t target_rank);

  // return the intrank_t on this node given a thread
  static intrank_t get_rank_from_thread(thread_num_t target_thread);

  // returns the parallel thread on a remote node with the same split_local_team().rank_me()
  static intrank_t get_rank_from_node(node_num_t target_node);

  // returns the split_rank for the rank on this node with this thread (0 to split_local_team().rank_n() - 1)
  static split_rank from_thread(thread_num_t target_thread);

  // returns split_rank for the parallel thread on a remote node with the same split_local_team().rank_me()
  static split_rank from_node(node_num_t target_node);

  static node_num_t num_nodes();

  static thread_num_t num_threads();

  static bool all_local_ranks_are_local_castable();

  static bool all_ranks_are_local_castable();
};

class split_team {
  // split_rank version 2
  // creates a team of nodes where thread_team.rank_me() are the same across each node
  // and there are thread_team.rank_n() distinct node_team instances
  const upcxx::team &_full_team;
  const upcxx::team &_thread_team;  // usually local_team(), except when debugging
  upcxx::team _node_team;           // one rank per node
 public:
  split_team(const upcxx::team &full_ = world(), const upcxx::team &thread_ = split_rank::split_local_team());
  virtual ~split_team();

  inline const upcxx::team &full_team() const { return _full_team; }  // all ranks in this split_team
  inline const upcxx::team &thread_team() const {
    return _thread_team;
  }                                                                   // only local ranks from this split_team / split_local_team()
  inline const upcxx::team &node_team() const { return _node_team; }  // only 1 rank per node from split_team();

  inline intrank_t full_n() const { return _full_team.rank_n(); }
  inline thread_num_t thread_n() const { return _thread_team.rank_n(); }
  inline node_num_t node_n() const { return _node_team.rank_n(); }

  inline intrank_t full_me() const { return _full_team.rank_me(); }
  inline thread_num_t thread_me() const { return _thread_team.rank_me(); }
  inline node_num_t node_me() const { return _node_team.rank_me(); }

  // *_from_thread returns ranks within this thread_team (i.e. local)
  intrank_t world_from_thread(thread_num_t thread_rank) const;
  intrank_t full_from_thread(thread_num_t thread_rank) const;
  intrank_t local_from_thread(thread_num_t thread_rank) const;

  bool thread_team_contains_world(intrank_t world_rank) const;
  bool thread_team_contains_full(intrank_t full_rank) const;
  bool thread_team_contains_node(node_num_t node_rank) const;

  // *_from_full returns ranks within the full_team (i.e. world)
  thread_num_t thread_from_full(intrank_t full_rank) const;
  intrank_t world_from_full(intrank_t full_rank) const;
  node_num_t node_from_full(intrank_t full_rank) const;

  // returns the full_rank with the same thread_rank offset as me
  intrank_t full_from_node(node_num_t node_rank) const;

  // *_from_node returns ranks within the full_team (i.e. world)
  // returns the world_rank with the same thread_rank offset as me
  intrank_t world_from_node(node_num_t node_rank) const;

  bool local_team_contains_thread(thread_num_t thread_rank) const;
  bool local_team_contains_full(intrank_t full_rank) const;
  bool local_team_contains_node(node_num_t node_rank) const;

 protected:
  //
  // These are potentially expensive and will be deprecated
  //

  bool full_team_contains_world(intrank_t world_rank) const;

  // throws if full is not from world
  // returns the thread from world.
  // NOTE if not on this node this rank will NOT be from the thread_team, but rather a parallel team on another node
  thread_num_t thread_from_world(intrank_t world_rank) const;
  // throws if full is not from world
  intrank_t full_from_world(intrank_t world_rank) const;
  // throws if full is not from world
  // returns a rank from the node (with the same thread_me())
  node_num_t node_from_world(intrank_t world_rank) const;
};

};  // namespace upcxx_utils
