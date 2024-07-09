#include "upcxx_utils.hpp"
#include "kcount.hpp"
#include "kmer_dht.hpp"
#include "devices_gpu.hpp"

#include "gpu-utils/gpu_utils.hpp"
#include "kcount-gpu/parse_and_pack.hpp"
#include "kcount-gpu/gpu_hash_table.hpp"



using namespace std;
using namespace upcxx_utils;
using namespace upcxx;
using namespace kcount_gpu;

template <int MAX_K>
struct SeqBlockInserter<MAX_K>::SeqBlockInserterState {
    ParseAndPackGPUDriver *pnp_gpu_driver;
    int64_t num_pnp_gpu_waits = 0;
    int num_block_calls = 0;
    int64_t num_kmers = 0;
    int64_t bytes_kmers_sent = 0;
    int64_t bytes_supermers_sent = 0;
    string seq_block;
    vector<kmer_count_t> depth_block;

    SeqBlockInserterState()
            : pnp_gpu_driver(nullptr) {
        seq_block.reserve(KCOUNT_SEQ_BLOCK_SIZE);
        depth_block.reserve(KCOUNT_SEQ_BLOCK_SIZE);
    }
};

template <int MAX_K>
SeqBlockInserter<MAX_K>::SeqBlockInserter(int qual_offset, int minimizer_len) {
    double init_time;
    state = new SeqBlockInserterState();
    state->pnp_gpu_driver = new ParseAndPackGPUDriver(rank_me(), rank_n(), qual_offset, Kmer<MAX_K>::get_k(),
                                                      Kmer<MAX_K>::get_N_LONGS(), minimizer_len, init_time);
}

template <int MAX_K>
SeqBlockInserter<MAX_K>::~SeqBlockInserter() {
    if (state->pnp_gpu_driver) {
        delete state->pnp_gpu_driver;
        state->pnp_gpu_driver = nullptr;
    }
    if (state) delete state;
}

template <int MAX_K>
static void process_block(SeqBlockInserter<MAX_K> *seq_block_inserter, dist_object<KmerDHT<MAX_K>> &kmer_dht) {
   unsigned int num_valid_kmers = 0;
    auto state = seq_block_inserter->state;
//    bool from_ctgs = !state->depth_block.empty();
//    state->num_block_calls++;
    future<bool> fut = execute_in_thread_pool(
            [&state, &num_valid_kmers] { return state->pnp_gpu_driver->process_seq_block(state->seq_block, num_valid_kmers); });
//    while (!fut.is_ready()) {
//        state->num_pnp_gpu_waits++;
//        progress();
//    }
    bool success = fut.wait();
    if (!success) DIE("seq length is too high, ", state->seq_block.length(), " >= ", KCOUNT_SEQ_BLOCK_SIZE);
//    state->bytes_kmers_sent += sizeof(KmerAndExt<MAX_K>) * num_valid_kmers;
    future<> fut_pnp = execute_in_thread_pool([&state] { state->pnp_gpu_driver->pack_seq_block(state->seq_block); });
//    while (!fut_pnp.is_ready()) {
//        state->num_pnp_gpu_waits++;
//        progress();
//    }
    fut_pnp.wait();
    int num_targets = (int)state->pnp_gpu_driver->supermers.size();
    for (int i = 0; i < num_targets; i++) {
        auto target = state->pnp_gpu_driver->supermers[i].target;
        auto offset = state->pnp_gpu_driver->supermers[i].offset;
        auto len = state->pnp_gpu_driver->supermers[i].len;
        Supermer supermer;
        int packed_len = len / 2;
        if (offset % 2 || len % 2) packed_len++;
        supermer.seq = state->pnp_gpu_driver->packed_seqs.substr(offset / 2, packed_len);
        if (offset % 2) supermer.seq[0] &= 15;
        if ((offset + len) % 2) supermer.seq[supermer.seq.length() - 1] &= 240;
//        supermer.count = (from_ctgs ? state->depth_block[offset + 1] : (kmer_count_t)1);
        supermer.count = (kmer_count_t)1;
//        state->bytes_supermers_sent += supermer.get_bytes();
        kmer_dht->add_supermer(supermer, target);
//        state->num_kmers += (2 * supermer.seq.length() - Kmer<MAX_K>::get_k());
        progress();
    }
}

template <int MAX_K>
void SeqBlockInserter<MAX_K>::process_seq(string &seq, kmer_count_t depth, dist_object<KmerDHT<MAX_K>> &kmer_dht) {
    if (seq.length() >= KCOUNT_SEQ_BLOCK_SIZE)
        DIE("Oh dear, my laziness is revealed: the ctg seq is too long ", seq.length(), " for this GPU implementation ",
            KCOUNT_SEQ_BLOCK_SIZE);
    if (state->seq_block.length() + 1 + seq.length() >= KCOUNT_SEQ_BLOCK_SIZE) {
        process_block(this, kmer_dht);
        state->seq_block.clear();
        state->depth_block.clear();
    }
    state->seq_block += seq;
    state->seq_block += "_";
//    if (depth) state->depth_block.insert(state->depth_block.end(), seq.length() + 1, depth);
}

template <int MAX_K>
struct HashTableInserter<MAX_K>::HashTableInserterState {
  HashTableGPUDriver<MAX_K> ht_gpu_driver;

  HashTableInserterState()
      : ht_gpu_driver({}) {}
};

template <int MAX_K>
HashTableInserter<MAX_K>::HashTableInserter() {}

template <int MAX_K>
HashTableInserter<MAX_K>::~HashTableInserter() {
    if (state != nullptr) delete state;
}

template <int MAX_K>
void HashTableInserter<MAX_K>::init(size_t max_elems, size_t max_ctg_elems, bool use_qf) {
  barrier(local_team());
#ifdef USE_TCF
  this->use_qf = use_qf;
#else
  this->use_qf = false;
#endif
  state = new HashTableInserterState();
  // calculate total slots for hash table. Reserve space for parse and pack
  size_t bytes_for_pnp = KCOUNT_SEQ_BLOCK_SIZE * (2 + Kmer<MAX_K>::get_N_LONGS() * sizeof(uint64_t) + sizeof(int));
//   DBG("Finding available memory on GPU ", gpu_utils::get_gpu_uuid(), "\n");
  auto init_gpu_mem = gpu_utils::get_gpu_avail_mem();
  auto gpu_avail_mem_per_rank = (get_gpu_avail_mem_per_rank() - bytes_for_pnp) * 0.9;
//   SLOG_GPU("Available GPU memory per rank for kmers hash table is ", get_size_str(gpu_avail_mem_per_rank), 
//            " accounting for PnP of ", get_size_str(bytes_for_pnp/0.9), "\n");
//   SLOG_GPU("Initializing read kmers hash table with max ", max_elems, " elems (with max ", max_ctg_elems,
//            " elems for ctg hash table}\n");
//   assert(state != nullptr);
  string driver_msgs, driver_warnings;
//   BaseTimer t;
//   t.start();
    int num_errors = 0;
  state->ht_gpu_driver.init(rank_me(), rank_n(), Kmer<MAX_K>::get_k(), max_elems, max_ctg_elems, num_errors, gpu_avail_mem_per_rank,
                            driver_msgs, driver_warnings, use_qf);
//   t.stop();
//   SLOG_GPU(driver_msgs);
//   if (!driver_warnings.empty()) SWARN(driver_warnings);
//   SLOG_GPU("Initialized hash table GPU driver in ", fixed, setprecision(3), t.get_elapsed(), " s\n");
  barrier(local_team());
  auto gpu_used_mem = init_gpu_mem - gpu_utils::get_gpu_avail_mem();
  barrier(local_team());
//   SLOG_GPU("GPU read kmers hash table used ", get_size_str(gpu_used_mem), " memory on GPU out of ",
//            get_size_str(gpu_utils::get_gpu_tot_mem()), "\n");
}

template <int MAX_K>
void HashTableInserter<MAX_K>::insert_supermer(const std::string &supermer_seq, kmer_count_t supermer_count) {
  assert(state != nullptr);
  state->ht_gpu_driver.insert_supermer(supermer_seq, supermer_count);
}

template <int MAX_K>
void HashTableInserter<MAX_K>::insert_into_local_hashtable(dist_object<KmerMap<MAX_K>> &local_kmers) {
  barrier();
//   IntermittentTimer insert_timer("gpu insert to cpu timer");
//   insert_timer.start();
//   if (state->ht_gpu_driver.pass_type == CTG_KMERS_PASS) {
//     LOG_MEM("Before done_ctg_kmer_inserts");
//     uint64_t attempted_inserts = 0, dropped_inserts = 0, new_inserts = 0;
//     state->ht_gpu_driver.done_ctg_kmer_inserts(attempted_inserts, dropped_inserts, new_inserts);
//     barrier();
//     LOG_MEM("After done_ctg_kmer_inserts");
//     auto num_dropped_elems = reduce_one((uint64_t)dropped_inserts, op_fast_add, 0).wait();
//     auto num_attempted_inserts = reduce_one((uint64_t)attempted_inserts, op_fast_add, 0).wait();
//     auto num_new_inserts = reduce_one((uint64_t)new_inserts, op_fast_add, 0).wait();
//     SLOG_GPU("GPU ctg kmers hash table: inserted ", new_inserts, " new elements into read kmers hash table\n");
//     auto all_capacity = reduce_one((uint64_t)state->ht_gpu_driver.get_capacity(), op_fast_add, 0).wait();
//     if (num_dropped_elems) {
//       if (num_dropped_elems > num_attempted_inserts / 10000)
//         SWARN("GPU read kmers hash table: failed to insert ", perc_str(num_dropped_elems, num_attempted_inserts),
//               " ctg kmers; total capacity ", all_capacity);
//       else
//         SLOG_GPU("GPU read kmers hash table: failed to insert ", perc_str(num_dropped_elems, num_attempted_inserts),
//                  " ctg kmers; total capacity ", all_capacity, "\n");
//     }
//   }
  barrier();
//   LOG_MEM("before done_all_inserts");
//   Timings::wait_pending();
  uint64_t num_dropped = 0, num_entries = 0, num_purged = 0;
  state->ht_gpu_driver.done_all_inserts(num_dropped, num_entries, num_purged);
  barrier();
//   LOG_MEM("after done_all_inserts");
  Timings::wait_pending();
  auto msm_num_dropped = min_sum_max_reduce_one(num_dropped).wait();
  auto msm_num_entries = min_sum_max_reduce_one(num_entries).wait();
  auto msm_pct_dropped =
      min_sum_max_reduce_one((float)(num_entries == 0 ? 0.0 : ((float)num_dropped) / ((float)num_entries))).wait();
//   if (msm_num_dropped.max > 0)
//     SWARN("GPU dropped ", msm_pct_dropped.to_string(), " entries. dropped: ", msm_num_dropped.to_string(),
//           " total: ", msm_num_entries.to_string(), " when compacting to output hash table\n");

  auto all_capacity = reduce_one((uint64_t)state->ht_gpu_driver.get_final_capacity(), op_fast_add, 0).wait();
  auto all_num_purged = reduce_one((uint64_t)num_purged, op_fast_add, 0).wait();
  auto all_num_entries = reduce_one((uint64_t)num_entries, op_fast_add, 0).wait();
  auto prepurge_num_entries = all_num_entries + all_num_purged;
//   SLOG_GPU("GPU hash table: purged ", perc_str(all_num_purged, prepurge_num_entries), " singleton kmers out of ",
//            prepurge_num_entries, "\n");
//   SLOG_GPU("GPU hash table final size is ", (all_num_entries / rank_n()), " entries and final load factor is ",
//            ((double)all_num_entries / all_capacity), "\n");
  int64_t max_kmer_count = 0;
  state->ht_gpu_driver.begin_iterate();
  while (true) {
    auto [kmer_array, count_exts] = state->ht_gpu_driver.get_next_entry();
    if (!kmer_array) break;
    if (count_exts->count > max_kmer_count) max_kmer_count = count_exts->count;
  }

  auto msm_max_kmer_count = upcxx_utils::min_sum_max_reduce_all(max_kmer_count).wait();
//   if (!rank_me()) LOG("High count (max) for kmers: ", msm_max_kmer_count.to_string(), "\n");
  int64_t high_count_threshold = msm_max_kmer_count.avg;
  barrier();

  // add some space for the ctg kmers
  local_kmers->reserve(num_entries * 1.5);
//   LOG_MEM("After insert_into_local_hashtable reserve");
  uint64_t invalid = 0;
  uint64_t sum_kmer_counts = 0;
  state->ht_gpu_driver.begin_iterate();
  while (true) {
    auto [kmer_array, count_exts] = state->ht_gpu_driver.get_next_entry();
    if (!kmer_array) break;
    // empty slot
    if (!count_exts->count) continue;
    // kmers with these extensions are not used in the dbjg traversal
    if ((char)count_exts->left == 'X' || (char)count_exts->right == 'X' || (char)count_exts->left == 'F' ||
        (char)count_exts->right == 'F') {
      invalid++;
      continue;
    }
    if ((count_exts->count < 2)) {
    //   WARN("Found a kmer that should have been purged, count is ", count_exts->count);
      invalid++;
      continue;
    }
    if (count_exts->count >= high_count_threshold) {
      Kmer<MAX_K> kmer(kmer_array->longs);
    //   NET_LOG("High count kmer: k = ", Kmer<MAX_K>::get_k(), " count = ", count_exts->count, " kmer = ", kmer.to_string(), "\n");
    }

    KmerCounts kmer_counts = {
                              .count = static_cast<kmer_count_t>(min(count_exts->count, static_cast<count_t>(UINT16_MAX))),
                              .left = (char)count_exts->left,
                              .right = (char)count_exts->right};
    Kmer<MAX_K> kmer(reinterpret_cast<const uint64_t *>(kmer_array->longs));
    const auto it = local_kmers->find(kmer);
    // if (it != local_kmers->end())
    //   WARN("Found a duplicate kmer ", kmer.to_string(), " - shouldn't happen: existing count ", it->second.count, " new count ",
    //        kmer_counts.count);
    local_kmers->insert({kmer, kmer_counts});
    sum_kmer_counts += kmer_counts.count;
  }
//   insert_timer.stop();

//   auto all_avg_elapsed_time = reduce_one(insert_timer.get_elapsed(), op_fast_add, 0).wait() / rank_n();
//   auto all_max_elapsed_time = reduce_one(insert_timer.get_elapsed(), op_fast_max, 0).wait();
//   SLOG_GPU("Inserting kmers from GPU to cpu hash table took ", all_avg_elapsed_time, " avg, ", all_max_elapsed_time, " max\n");
//   auto all_kmers_size = reduce_all((uint64_t)local_kmers->size(), op_fast_add).wait();
//   if (local_kmers->size() != (num_entries - invalid))
//     WARN("kmers->size() is ", local_kmers->size(), " != ", (num_entries - invalid), " num_entries");
//   SLOG_GPU("Compact hash table has ", local_kmers->size(), " elements and load factor ", local_kmers->load_factor(), "\n");
//   auto all_invalid = reduce_one((uint64_t)invalid, op_fast_add, 0).wait();
//   if (all_kmers_size != all_num_entries - all_invalid)
//     SWARN("CPU kmer counts not equal to gpu kmer counts: ", all_kmers_size, " != ", (all_num_entries - all_invalid),
//           " all_num_entries: ", all_num_entries, " all_invalid: ", all_invalid);
//   auto all_sum_kmer_counts = reduce_all(sum_kmer_counts, op_fast_add).wait();
//   double avg_kmer_count = (double)all_sum_kmer_counts / all_kmers_size;
//   SLOG_GPU("For ", all_kmers_size, " kmers, average kmer count (depth): ", fixed, setprecision(2), avg_kmer_count, "\n");
//   double gpu_insert_time = 0, gpu_kernel_time = 0;
//   state->ht_gpu_driver.get_elapsed_time(gpu_insert_time, gpu_kernel_time);
//   auto avg_gpu_insert_time = reduce_one(gpu_insert_time, op_fast_add, 0).wait() / rank_n();
//   auto max_gpu_insert_time = reduce_one(gpu_insert_time, op_fast_max, 0).wait();
//   auto avg_gpu_kernel_time = reduce_one(gpu_kernel_time, op_fast_add, 0).wait() / rank_n();
//   auto max_gpu_kernel_time = reduce_one(gpu_kernel_time, op_fast_max, 0).wait();
//   SLOG_GPU("Elapsed GPU time for kmer hash tables:\n");
//   double load_balance = (double)avg_gpu_insert_time / max_gpu_insert_time;
//   SLOG_GPU("  insert: ", fixed, setprecision(3), (load_balance < 0.5 ? KLRED : ""), avg_gpu_insert_time, " avg, ",
//            max_gpu_insert_time, " max, load balance ", load_balance, KNORM, "\n");
//   SLOG_GPU("  kernel: ", fixed, setprecision(3), avg_gpu_kernel_time, " avg, ", max_gpu_kernel_time, " max\n");
  barrier();
//   LOG_MEM("After insert_into_local_hashtable inserts");
//   return avg_kmer_count;}
}

#define seq_block_inserter_K(KMER_LEN) template struct SeqBlockInserter<KMER_LEN>;
#define HASH_TABLE_INSERTER_K(KMER_LEN) template class HashTableInserter<KMER_LEN>;

seq_block_inserter_K(32);
HASH_TABLE_INSERTER_K(32);
#if MAX_BUILD_KMER >= 64
seq_block_inserter_K(64);
HASH_TABLE_INSERTER_K(64);
#endif
#if MAX_BUILD_KMER >= 96
seq_block_inserter_K(96);
HASH_TABLE_INSERTER_K(96);
#endif
#if MAX_BUILD_KMER >= 128
seq_block_inserter_K(128);
HASH_TABLE_INSERTER_K(128);
#endif
#if MAX_BUILD_KMER >= 160
seq_block_inserter_K(160);
HASH_TABLE_INSERTER_K(160);
#endif
#undef seq_block_inserter_K
#undef HASH_TABLE_INSERTER_K