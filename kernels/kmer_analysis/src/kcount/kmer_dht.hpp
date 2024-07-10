#pragma once

#include <map>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
//#include <upcxx/upcxx.hpp>

#include "../kmer.hpp"
#include <upcxx_utils/flat_aggr_store.hpp>
#include <upcxx_utils/three_tier_aggr_store.hpp>

using std::pair;
using std::string;
using std::string_view;
using std::vector;

using kmer_count_t = uint16_t;

/**
 * in MHM2, this struct also holds a global pointer to a "FragElem" involved in assembly
 * not needed for this implementation
 * this is the value of KV pairs in local_kmers map below
*/
struct KmerCounts {
    // global_ptr<FragElem> uutig_frag;
    kmer_count_t count;
    char left, right;
};

/**
 * struct used by get_kmers_and_exts and insert_supermer
 * primarily used in CPU build
 */
template <int MAX_K>
struct KmerAndExt {
    Kmer<MAX_K> kmer;
    kmer_count_t count;
    char left, right;
    UPCXX_SERIALIZED_FIELDS(kmer, count, left, right);
};

/**
 * DNA sequence comprised of kmers that are processed by the same rank
 * a supermer can have only one kmer, but it also includes that kmers left and right extensions,
 * therefore minimum supermer length will be kmer length + 2
 */
struct Supermer {
    string seq;
    kmer_count_t count;
    UPCXX_SERIALIZED_FIELDS(seq, count);
};

// define the local hash table that makes up the distributed hash table
#include <unordered_map>
#define HASH_TABLE std::unordered_map
template <int MAX_K>
using KmerMap = HASH_TABLE<Kmer<MAX_K>, KmerCounts>;

/**
 * Intermediate data structure used to process kmers before going to final hash table
 *
 * kcount_cpu implements HashTableInserterState as a hash table: KmerMapExts
 * kcount_gpu has a different implementation
 *
 * insert_supermer adds kmers to HashTableInserterState
 * insert_into_local_hashtable adds kmers to the local_kmers member of KmerDHT
 */
template <int MAX_K>
class HashTableInserter {
    struct HashTableInserterState;
    HashTableInserterState *state = nullptr;
    bool use_qf;

public:
    HashTableInserter();

    ~HashTableInserter();

    void init(size_t max_elems, bool use_qf);

    void insert_supermer(const std::string &supermer_seq, kmer_count_t supermer_count);

    void insert_into_local_hashtable(dist_object<KmerMap<MAX_K>> &local_kmers);

};

/**
 * Distributed hash table
 * local_kmers is the hash table used for assembly
 * ht_inserter is the intermediate data structure used to populate local_kmers
 * kmer_store is a custom UPC++ data structure used to aggregate and communicate updates between processes - in this case, supermers
 *
 * NOTE: num_supermer_inserts is not strictly necessary for the KmerDHT to function.
 * Its one of the few stats variables left in from MHM2 because I thought it was very likely to come back anyway for testing purposes
 *
 * minimizers are used for determining kmer length - see kmer.hpp
 * minimum minimizer_len is 15, but can be modified at runtime based on kmer length
 */
template <int MAX_K>
class KmerDHT {
public:
    dist_object<KmerMap<MAX_K>> local_kmers;
    dist_object<HashTableInserter<MAX_K>> ht_inserter;

    upcxx_utils::ThreeTierAggrStore<Supermer> kmer_store;
    int64_t max_kmer_store_bytes;
    int64_t my_num_kmers;
    int max_rpcs_in_flight;
    int64_t num_supermer_inserts;

    int minimizer_len = 15;


    // initializes KmerDHT members
    // set ThreeTierAggrStore kmer_store update function to insert_supermer (with lambda function?)
    KmerDHT(uint64_t my_num_kmers, size_t max_kmer_store_bytes, int max_rpcs_in_flight, bool use_qf);

    ~KmerDHT();

    int get_minimizer_len();

    // gets kmer's destination process rank
    upcxx::intrank_t get_kmer_target_rank(const Kmer<MAX_K> &kmer, const Kmer<MAX_K> *kmer_rc = nullptr) const;

    int64_t get_num_supermer_inserts();

    // adds supermer to kmer_store
    void add_supermer(Supermer &supermer, int target_rank);

    // sends pending updates to ranks
    void flush_updates();

    // calls HashTableInserter->insert_into_local_hashtable
    void finish_updates();

    typename KmerMap<MAX_K>::iterator local_kmers_begin();

    typename KmerMap<MAX_K>::iterator local_kmers_end();

};

