#include "upcxx_utils/flat_aggr_store.hpp"

#include <cassert>

namespace upcxx_utils {

TargetRPCCounts::TargetRPCCounts() { reset(); }

void TargetRPCCounts::reset() {
  rpcs_sent = 0;
  rpcs_expected = 0;
  rpcs_processed = 0;
  rpcs_progressed = 0;
  imbalance_factor = 1;
}

FASRPCCounts::FASRPCCounts(const upcxx::team &tm)
    : total()
    , targets()
    , rpc_prep_timer("FlatAggrStore::prep_rpc")
    , rpc_inner_timer("FlatAggrStore::inner_rpc")
    , rpc_outer_timer("FlatAggrStore::outer_rpc")
    , wait_for_rpcs_timer("FlatAggrStore::wait_for_rpcs")
    , append_store_timer("FlatAggrStore::append_store")
    , progress_timer("FlatAggrStore::progress")
    , tm(tm) {}

void FASRPCCounts::init() {
  targets.resize(tm.rank_n());
  reset();
}

void FASRPCCounts::reset() {
  // explicitly world reduction not aggr_team!
  auto has_counts = upcxx::reduce_all(total.rpcs_sent + total.rpcs_processed, upcxx::op_fast_add, world()).wait();
  if (has_counts) print_out();
  total.reset();
  for (auto &t : targets) {
    t.reset();
  }
  rpc_prep_timer.clear();
  rpc_inner_timer.clear();
  rpc_outer_timer.clear();
  wait_for_rpcs_timer.clear();
  append_store_timer.clear();
  progress_timer.clear();
}

void FASRPCCounts::print_out() {
  // explicitly world reduction not aggr_team!
  wait_for_rpcs_timer.print_out(world());
  rpc_prep_timer.print_out(world());
  rpc_outer_timer.print_out(world());
  rpc_inner_timer.print_out(world());
  append_store_timer.print_out(world());
  progress_timer.print_out(world());
}

void FASRPCCounts::increment_sent_counters(intrank_t target_rank) {
  assert((intrank_t)targets.size() > target_rank);
  // increment the counters
  total.rpcs_sent++;
  targets[target_rank].rpcs_sent++;
}

void FASRPCCounts::increment_processed_counters(intrank_t source_rank) {
  assert((intrank_t)targets.size() > source_rank);
  // increment the counters
  total.rpcs_processed++;
  targets[source_rank].rpcs_processed++;
}

void FASRPCCounts::set_progressed_count(intrank_t source_rank, CountType count) {
  assert((intrank_t)targets.size() > source_rank);
  auto &rpcs_progressed = targets[source_rank].rpcs_progressed;
  if (count > rpcs_progressed) {
    total.rpcs_progressed += count - rpcs_progressed;
    rpcs_progressed = count;
  }
}

void FASRPCCounts::wait_for_rpcs(intrank_t target_rank, CountType max_rpcs_in_flight) {
  assert(!upcxx::in_progress());
  assert(target_rank < (intrank_t)targets.size());
  DBG("wait_for_rpcs(): target_rank=", target_rank, ", tgt sent=", targets[target_rank].rpcs_sent,
      ", proc/prog=", targets[target_rank].rpcs_processed, "/", targets[target_rank].rpcs_progressed, " tot sent=", total.rpcs_sent,
      " proc/prog=", total.rpcs_processed, "/", total.rpcs_progressed, "\n");
  // limit the number in flight by making sure we don't have too many more sent than received (with good load balance,
  // every process is sending and receiving about the same number)
  // we don't actually want to check every possible rank's count while waiting, so just check the target rank

  wait_for_rpcs_timer.start();
  bool imbalanced = false;
  CountType max_per_rank = (max_rpcs_in_flight + targets.size() - 1) / targets.size();
  max_per_rank = std::min(max_per_rank, (CountType)4);  // always allow a minimum of 4 outstanding per rank.
  auto &tgt = targets[target_rank];
  auto &tgt_balance_factor = tgt.imbalance_factor;
  auto &tgt_rpcs_expected = tgt.rpcs_expected;  // set when target is in flush
  if (max_rpcs_in_flight) {
    auto &rpcs_sent = total.rpcs_sent;                // total sent by this rank
    auto &rpcs_processed = total.rpcs_processed;      // total processed by this rank
    auto &rpcs_progressed = total.rpcs_progressed;    // total known processed by others from this rank (delayed)
    auto &tgt_rpcs_sent = tgt.rpcs_sent;              // sent to target
    auto &tgt_rpcs_processed = tgt.rpcs_processed;    // received from target and processed
    auto &tgt_rpcs_progressed = tgt.rpcs_progressed;  // sent to target and processed (delayed)

    size_t iter = 0;

    StallTimer stall_t("Flat wait for rpcs ");

    while (tgt_rpcs_sent - tgt_rpcs_processed > max_per_rank && tgt_rpcs_sent - tgt_rpcs_progressed > max_per_rank &&
           rpcs_sent - rpcs_processed > max_rpcs_in_flight && rpcs_sent - rpcs_progressed > max_rpcs_in_flight) {
      stall_t.check();
      iter++;
      progress_timer.progress();
      bool target_is_in_flush = iter > max_per_rank && tgt_rpcs_expected > 0;
      bool target_is_imbalanced = !target_is_in_flush && iter > (max_rpcs_in_flight + 1) * tgt_balance_factor;
      if ((target_is_in_flush || target_is_imbalanced) && !upcxx::progress_required()) {
        if ((!target_is_in_flush && target_is_imbalanced) && tgt_balance_factor >= 4) {
          DBG("Breaking out of wait_for_rpc FAS - iter=", iter, " rpcs sent=", rpcs_sent, " processed=", rpcs_processed,
              " progressed=", rpcs_progressed, ", balance_factor=", tgt_balance_factor, " target=", target_rank,
              " sent=", tgt_rpcs_sent, " processed=", tgt_rpcs_processed, " progressed=", tgt_rpcs_progressed, "\n");
        } else {
          DBG("Breaking out of wait_for_rpc tgt=", target_rank, " iter=", iter, " flush=", target_is_in_flush,
              " imbalfact=", tgt_balance_factor, "\n");
        }
        if (target_is_imbalanced) {
          tgt_balance_factor <<= 1;  // call progress twice as much next time.
          imbalanced = true;
        }
        break;  // escape eventually if load is imbalanced
      }
    }
  }
  if (!imbalanced) tgt_balance_factor = 1;
  wait_for_rpcs_timer.stop();
}

void FASRPCCounts::update_progressed_count(DistFASRPCCounts &dist_fas_rpc_counts, intrank_t target_rank) {
  assert(this == &(*dist_fas_rpc_counts));
  const upcxx::team &t = dist_fas_rpc_counts.team();
  assert(target_rank < t.rank_n());
  intrank_t me = t.rank_me();

  upcxx::rpc_ff(
      t, target_rank,
      [](DistFASRPCCounts &dfas_rpc_counts, intrank_t source_rank, CountType known_progressed) {
        CountType processed = dfas_rpc_counts->targets[source_rank].rpcs_processed;
        DBG("request for progress source_rank=", source_rank, " known_progressed=", known_progressed, " processed=", processed,
            "\n");
        assert(known_progressed <= processed);
        if (known_progressed == processed) return;
        // send update back
        rpc_ff(
            dfas_rpc_counts.team(), source_rank,
            [](DistFASRPCCounts &dfas_rpc_counts, intrank_t return_rank, CountType processed) {
              DBG("Updating progressed for ", return_rank, " to ", processed, "\n");
              FASRPCCounts &rpc_counts = *dfas_rpc_counts;
              rpc_counts.set_progressed_count(return_rank, processed);
            },
            dfas_rpc_counts, dfas_rpc_counts.team().rank_me(), processed);
      },
      dist_fas_rpc_counts, me, targets[target_rank].rpcs_progressed);
  if (!upcxx::in_progress()) this->progress_timer.progress();  // call progress after every rpc
}

};  // namespace upcxx_utils
