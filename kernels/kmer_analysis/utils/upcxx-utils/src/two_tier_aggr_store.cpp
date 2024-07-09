#include "upcxx_utils/two_tier_aggr_store.hpp"

using std::deque;
using std::list;
using std::make_shared;
using std::pair;
using std::shared_ptr;
using std::vector;

using upcxx::barrier;
using upcxx::dist_object;
using upcxx::future;
using upcxx::global_ptr;
using upcxx::intrank_t;
using upcxx::make_future;
using upcxx::make_view;
using upcxx::op_fast_add;
using upcxx::op_fast_max;
using upcxx::progress;
using upcxx::promise;
using upcxx::rank_me;
using upcxx::rank_n;
using upcxx::reduce_all;
using upcxx::reduce_one;
using upcxx::rget;
using upcxx::rpc;
using upcxx::to_future;
using upcxx::view;

namespace upcxx_utils {
//
// TrackRPCs
//

bool TrackRPCs::empty() const {
  bool is_empty = sent_rpcs == returned_rpcs && rpcs_in_flight.empty() && sent_rpcs == returned_rpcs;
  return is_empty;
}

void TrackRPCs::clear() {
  if (sent_rpcs != returned_rpcs) DIE("sent ", sent_rpcs, " but ", returned_rpcs, " returned\n");
  if (!rpcs_in_flight.empty()) DIE("Clear called with RPCs pending: ", rpcs_in_flight.size(), "\n");
  t_prog.print_out();
  rpc_timer.print_reduce_timings(description + string("-outer-rpc"));
  rpc_inner_timer.print_reduce_timings(description + string("-inner-rpc"));
  rpc_relay_timer.print_reduce_timings(description + string("-relay-rpc"));
  rpcs_in_flight.clear();
}

void TrackRPCs::push(future_ack_t fut) {
  sent_rpcs++;
  rpcs_in_flight.push_back(fut);
}

size_t TrackRPCs::pop_finished() {
  size_t remaining = 0, popped = 0;
  for (auto it = rpcs_in_flight.begin(); it != rpcs_in_flight.end();) {
    if (it->is_ready()) {
      it = rpcs_in_flight.erase(it);
      popped++;
      returned_rpcs++;
    } else {
      it++;
      remaining++;
    }
  }
  assert(rpcs_in_flight.size() == sent_rpcs - returned_rpcs);
  DBG_VERBOSE("pop_finished_rpcs: popped=", popped, " remaining=", remaining, "\n");
  return remaining;
}

size_t TrackRPCs::count_pending() {
  assert(rpcs_in_flight.size() == sent_rpcs - returned_rpcs);
  return sent_rpcs - returned_rpcs;
}

void TrackRPCs::flush(size_t max_pending) {
  DBG("TrackRPCs flush max_pending=", max_pending, " count_pending=", count_pending(), " -- ", to_string(), "\n");
  pop_finished();
  StallTimer is_stalled(description + string("-flush"));
  while (count_pending() > max_pending) {
    is_stalled.check();
    t_prog.progress();
    pop_finished();
  }
}

string TrackRPCs::to_string() const {
  ostringstream os;
  os << description << "-";
  os << "TrackRPCs(";
  os << ",rpcs_in_flight=" << rpcs_in_flight.size();
  os << ",sent_rpcs=" << sent_rpcs;
  os << ",returned_rpcs=" << returned_rpcs;
  os << ")";
  return os.str();
}

};  // namespace upcxx_utils