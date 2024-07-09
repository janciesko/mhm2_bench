/**
 * CPU implementation of SeqBlockInserter, HashTableInserter, helper functions/structs
 */

#include "upcxx_utils.hpp"
#include "kcount.hpp"
#include "kmer_dht.hpp"
#include "kmer.hpp"
#include "prime.hpp"

using namespace std;
using namespace upcxx;
using namespace upcxx_utils;

template<int MAX_K>
struct SeqBlockInserter<MAX_K>::SeqBlockInserterState {
    vector<Kmer<MAX_K>> kmers;
};

template<int MAX_K>
SeqBlockInserter<MAX_K>::SeqBlockInserter(int qual_offset, int minimizer_len) {
    state = new SeqBlockInserterState();
}

template<int MAX_K>
SeqBlockInserter<MAX_K>::~SeqBlockInserter() {
    if (state) delete state;
}

/**
 * get_kmers() converts sequence string to vector of Kmer objects
 *
 * Supermer starts as a kmer with left and right extensions
 * If the next kmer in the string has the same target rank as the current supermer, it is added to the supermer
 * ie, the supermer is extended by one base
 *
 * once the next kmer has a new target rank, the current supermer is added to the ThreeTierAggrStore kmer_store
 *
 * @param seq
 * @param depth when inserting from read, depth is 1
 * @param kmer_dht
 */
template<int MAX_K>
void SeqBlockInserter<MAX_K>::process_seq(string &seq, kmer_count_t depth, dist_object<KmerDHT<MAX_K>> &kmer_dht) {
    if (!depth) depth = 1;

    int kmer_len = Kmer<MAX_K>::get_k();

    Kmer<MAX_K>::get_kmers(kmer_len, seq, state->kmers);
    for (size_t i = 0; i < state->kmers.size(); i++) {
        Kmer<MAX_K> kmer_rc = state->kmers[i].revcomp();
        if (kmer_rc < state->kmers[i]) state->kmers[i] = kmer_rc;
    }

    Supermer supermer{.seq = seq.substr(0, kmer_len + 1), .count = (kmer_count_t )depth};
    auto prev_target_rank = kmer_dht->get_kmer_target_rank(state->kmers[1]);
    for (int i = 1; i < (int)(seq.length() - kmer_len); i++) {
        auto &kmer = state->kmers[i];
        auto target_rank = kmer_dht->get_kmer_target_rank(kmer);
        if (target_rank == prev_target_rank) {
            supermer.seq += seq[i + kmer_len];
        } else {
            kmer_dht->add_supermer(supermer, prev_target_rank);
            supermer.seq = seq.substr(i - 1, kmer_len + 2);
            prev_target_rank = target_rank;
        }
    }

    // last supermer
    if (supermer.seq.length() >= kmer_len + 2) {
        kmer_dht->add_supermer(supermer, prev_target_rank);
    }


}

/**
 * Stores counts of bases for a kmer extension
 * get_ext() sorts the counts and returns the most frequent base
 * if there is a tie, it returns the lexicographically highest base of the tie
 */
struct ExtCounts {
    kmer_count_t count_A;
    kmer_count_t count_C;
    kmer_count_t count_G;
    kmer_count_t count_T;

    void inc(char ext, int count) {
        switch (ext) {
            case 'A': count_A += count; break;
            case 'C': count_C += count; break;
            case 'G': count_G += count; break;
            case 'T': count_T += count; break;
        }
    }

    char get_ext() {
        std::array<std::pair<char, int>, 4> counts = {std::make_pair('A', (int)count_A), std::make_pair('C', (int)count_C),
                                                      std::make_pair('G', (int)count_G), std::make_pair('T', (int)count_T)};

        std::sort(std::begin(counts), std::end(counts), [](const auto &elem1, const auto &elem2) {
            if (elem1.second == elem2.second)
                return elem1.first > elem2.first;
            else
                return elem1.second > elem2.second;
        });

        return counts[0].first;
    }

};

/**
 * Stores kmer count and both extensions
 *
 * This struct is used in the KmerMapExts class defined below.
 */
struct KmerExtsCounts {
    ExtCounts left_exts;
    ExtCounts right_exts;
    kmer_count_t count;

    char get_left_ext() { return left_exts.get_ext(); }

    char get_right_ext() { return right_exts.get_ext(); }
};


/**
 * Hash map esque data structure consisting of a vector of Kmers and a vector KmerExtsCounts
 *
 * in kcount_cpu, HashTableInserterState is a KmerMapExts - in kcount_gpu, HashTableInserterState is implemented differently
 *
 * serves as an intermediate data structure before the distributed hash table is populated
 *
 * insert_supermer adds kmers and extensions to KmerMapExts
 * insert_into_local_hashtable adds kmers and extensions from KmerMapExts to local_kmers (distributed unordered map)
 */
template<int MAX_K>
class KmerMapExts {
public:
    vector<Kmer<MAX_K>> ht_kmers;
    vector<KmerExtsCounts> counts;
    size_t num_elems = 0;
    size_t capacity = 0;
    size_t iter_pos = 0;
    const int N_LONGS = Kmer<MAX_K>::get_N_LONGS();
    const uint64_t KEY_EMPTY = 0xffffffffffffffff;

    void reserve(size_t max_elems) {
        primes::Prime prime;
        prime.set(max_elems, true);
        capacity = prime.get();
        ht_kmers.resize(capacity);
        memset((void *)ht_kmers.data(), 0xff, sizeof(Kmer<MAX_K>) * capacity);
        counts.resize(capacity, {0});
    }

    KmerExtsCounts * insert(const Kmer<MAX_K> &kmer) {

        size_t slot = kmer.hash() % capacity;
        const size_t MAX_PROBE = capacity;
        for (size_t i = 1; i <= MAX_PROBE; i++) {

            if (ht_kmers[slot].get_longs()[N_LONGS] == KEY_EMPTY) {
                ht_kmers[slot] = kmer;
                num_elems++;

                return &(counts[slot]);
            } else if (kmer == ht_kmers[slot]) {

                return &(counts[slot]);
            }
            slot = (slot + 1) % capacity;
        }

        return nullptr;

    }

    void begin_iterate() { iter_pos = 0; }

    pair<Kmer<MAX_K> *, KmerExtsCounts *> get_next() {
        for (; iter_pos < capacity; iter_pos++) {
            if (ht_kmers[iter_pos].get_longs()[N_LONGS - 1] != KEY_EMPTY) {
                iter_pos++;
                return {&ht_kmers[iter_pos - 1], &counts[iter_pos - 1]};
            }
        }
        return {nullptr, nullptr};
    }
};

/**
 * Decompose a supermer into kmers and left/right single base extensions
 *
 * @param supermer
 * @param kmers_and_exts
 */
template<int MAX_K>
static void get_kmers_and_exts(Supermer &supermer, vector<KmerAndExt<MAX_K>> &kmers_and_exts) {
    auto kmer_len = Kmer<MAX_K>::k;
    vector<Kmer<MAX_K>> kmers;

    Kmer<MAX_K>::get_kmers(kmer_len, supermer.seq, kmers);

    for (int i = 1; i < (int) (supermer.seq.length() - kmer_len); i++) {
        Kmer<MAX_K> kmer = kmers[i];
        char left_ext = supermer.seq[i - 1];
        char right_ext = supermer.seq[i + kmer_len];

        kmers_and_exts.push_back({.kmer = kmer, .count = supermer.count, .left = left_ext, .right = right_ext});

    }
}


template<int MAX_K>
struct HashTableInserter<MAX_K>::HashTableInserterState {
    dist_object<KmerMapExts<MAX_K>> kmers;

    HashTableInserterState()
        : kmers(KmerMapExts<MAX_K>{}) {}
};

template<int MAX_K>
HashTableInserter<MAX_K>::HashTableInserter() {}

template<int MAX_K>
HashTableInserter<MAX_K>::~HashTableInserter() {
    if (state) delete state;
}

template<int MAX_K>
void HashTableInserter<MAX_K>::init(size_t max_elems, bool use_qf) {
    state = new HashTableInserterState;
    size_t max_read_kmers = max_elems * rank_n();
//    cout << "rank " << upcxx::rank_me() << " max elems: " << ma
    state->kmers->reserve(max_read_kmers);
}

/**
 * get kmers and extensions from a supermer, add them to KmerMapExts (HashTableInserterState)
 *
 * @param supermer_seq
 * @param supermer_count
 */
template<int MAX_K>
void HashTableInserter<MAX_K>::insert_supermer(const std::string &supermer_seq, kmer_count_t supermer_count) {

    Supermer supermer = {.seq = supermer_seq, .count = supermer_count};
    auto kmer_len = Kmer<MAX_K>::k;
    vector<KmerAndExt<MAX_K>> kmers_and_exts;
    kmers_and_exts.reserve(supermer.seq.length() - kmer_len);

    get_kmers_and_exts(supermer, kmers_and_exts);

    for (auto &kmer_and_ext : kmers_and_exts) {

        auto exts_counts = state->kmers->insert(kmer_and_ext.kmer);

        exts_counts->count += kmer_and_ext.count;
        exts_counts->left_exts.inc(kmer_and_ext.left, kmer_and_ext.count);
        exts_counts->right_exts.inc(kmer_and_ext.right, kmer_and_ext.count);
    }
}

/**
 * iterate KmerMapExts (HashTableInserterState)
 * add kmers and counts to distributed hash table local_kmers
 * @param local_kmers
 */
template<int MAX_K>
void HashTableInserter<MAX_K>::insert_into_local_hashtable(dist_object<KmerMap<MAX_K>> &local_kmers) {
    state->kmers->begin_iterate();

    while (true) {
        auto [kmer, kmer_ext_counts] = state->kmers->get_next();

        if (!kmer) break;

        KmerCounts kmer_counts = {.count = kmer_ext_counts->count,
                                  .left = kmer_ext_counts->get_left_ext(),
                                  .right = kmer_ext_counts->get_right_ext()};

        local_kmers->insert({*kmer, kmer_counts});
    }
    barrier();
}

#define SEQ_BLOCK_INSERTER_K(KMER_LEN) template struct SeqBlockInserter<KMER_LEN>;
#define HASH_TABLE_INSERTER_K(KMER_LEN) template class HashTableInserter<KMER_LEN>;

SEQ_BLOCK_INSERTER_K(32);
HASH_TABLE_INSERTER_K(32);
#if MAX_BUILD_KMER >= 64
SEQ_BLOCK_INSERTER_K(64);
HASH_TABLE_INSERTER_K(64);
#endif
#if MAX_BUILD_KMER >= 96
SEQ_BLOCK_INSERTER_K(96);
HASH_TABLE_INSERTER_K(96);
#endif
#if MAX_BUILD_KMER >= 128
SEQ_BLOCK_INSERTER_K(128);
HASH_TABLE_INSERTER_K(128);
#endif
#if MAX_BUILD_KMER >= 160
SEQ_BLOCK_INSERTER_K(160);
HASH_TABLE_INSERTER_K(160);
#endif
#undef SEQ_BLOCK_INSERTER_K
#undef HASH_TABLE_INSERTER_K

