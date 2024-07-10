#include <stdarg.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

/*#include "upcxx_utils/log.hpp"
#include "upcxx_utils/mem_profile.hpp"
#include "upcxx_utils/progress_bar.hpp"
#include "upcxx_utils/timers.hpp"*/

#include "kmer_dht.hpp"
#include "../kmer.hpp"

template <int MAX_K>
KmerDHT<MAX_K>::KmerDHT(uint64_t my_num_kmers, size_t max_kmer_store_bytes, int max_rpcs_in_flight, bool use_qf)
    : local_kmers(KmerMap<MAX_K>{})
    , ht_inserter(HashTableInserter<MAX_K>{})
    , kmer_store()
    , max_kmer_store_bytes(max_kmer_store_bytes)
    , my_num_kmers(my_num_kmers)
    , max_rpcs_in_flight(max_rpcs_in_flight)
    , num_supermer_inserts(0) {

    minimizer_len = Kmer<MAX_K>::k * 2 / 3 + 1;
    if (minimizer_len < 15) minimizer_len = 15;
    if (minimizer_len > 27) minimizer_len = 27;

    kmer_store.set_size("kmers", max_kmer_store_bytes, max_rpcs_in_flight, my_num_kmers);
    barrier();
    kmer_store.set_update_func(
            [&ht_inserter = this->ht_inserter, &num_supermer_inserts = this->num_supermer_inserts](Supermer supermer) {
                num_supermer_inserts++;
                ht_inserter->insert_supermer(supermer.seq, supermer.count);
            });
    // setting contig kmers to 0 because there are no contig kmers to use
    auto my_num_ctg_kmers = 0;
    ht_inserter->init(my_num_kmers, use_qf);
    barrier();
}

template <int MAX_K>
KmerDHT<MAX_K>::~KmerDHT() {
    local_kmers->clear();
    KmerMap<MAX_K>().swap(*local_kmers);
    kmer_store.clear();
}

template <int MAX_K>
int KmerDHT<MAX_K>::get_minimizer_len() {
  return minimizer_len;
}

template <int MAX_K>
upcxx::intrank_t KmerDHT<MAX_K>::get_kmer_target_rank(const Kmer<MAX_K> &kmer, const Kmer<MAX_K> *kmer_rc) const {
    return kmer.minimizer_hash_fast(minimizer_len, kmer_rc) % rank_n();
}

template <int MAX_K>
int64_t KmerDHT<MAX_K>::get_num_supermer_inserts() {
  return num_supermer_inserts;
}

template <int MAX_K>
void KmerDHT<MAX_K>::add_supermer(Supermer &supermer, int target_rank) {
    kmer_store.update(target_rank, supermer);
}

template <int MAX_K>
void KmerDHT<MAX_K>::flush_updates() {
    kmer_store.flush_updates();
    barrier();
}

template <int MAX_K>
void KmerDHT<MAX_K>::finish_updates() {
    ht_inserter->insert_into_local_hashtable(local_kmers);
}

template <int MAX_K>
typename KmerMap<MAX_K>::iterator KmerDHT<MAX_K>::local_kmers_begin() {
    return local_kmers->begin();
}

template <int MAX_K>
typename KmerMap<MAX_K>::iterator KmerDHT<MAX_K>::local_kmers_end() {
    return local_kmers->end();
}

#define KMER_DHT_K(KMER_LEN) template class KmerDHT<KMER_LEN>

KMER_DHT_K(32);
#if MAX_BUILD_KMER >= 64
KMER_DHT_K(64);
#endif
#if MAX_BUILD_KMER >= 96
KMER_DHT_K(96);
#endif
#if MAX_BUILD_KMER >= 128
KMER_DHT_K(128);
#endif
#if MAX_BUILD_KMER >= 160
KMER_DHT_K(160);
#endif

#undef KMER_DHT_K
