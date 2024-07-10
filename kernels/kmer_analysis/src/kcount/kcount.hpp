#pragma once

#include <vector>

#include"kmer_dht.hpp"
//#include <upcxx/upcxx.hpp>

using std::vector;
using upcxx::dist_object;

using count_t = uint32_t;

/**
 * Converts reads->kmers->supermers, adds supermers to ThreeTierAggrStore kmer_store
 */
template <int MAX_K>
struct SeqBlockInserter {
    struct SeqBlockInserterState;
    SeqBlockInserterState *state = nullptr;

    SeqBlockInserter(int qual_offset, int minimizer_len);
    ~SeqBlockInserter();

    void process_seq(string &seq, kmer_count_t depth, dist_object<KmerDHT<MAX_K>> &kmer_dht);
};

template <int MAX_K>
void analyze_kmers(unsigned kmer_len, vector<string> &local_reads, dist_object<KmerDHT<MAX_K>> &kmer_dht);

#define __MACRO_KCOUNT__(KMER_LEN, MODIFIER)                                                        \
  MODIFIER void analyze_kmers<KMER_LEN>(unsigned, vector<string> &, dist_object<KmerDHT<KMER_LEN>> &)

// Reduce compile time by instantiating templates of common types
// extern template declarations are in in kcount.hpp
// template instantiations each happen in src/CMakeLists via kcount-extern-template.in.cpp

__MACRO_KCOUNT__(32, extern template);
#if MAX_BUILD_KMER >= 64
__MACRO_KCOUNT__(64, extern template);
#endif
#if MAX_BUILD_KMER >= 96
__MACRO_KCOUNT__(96, extern template);
#endif
#if MAX_BUILD_KMER >= 128
__MACRO_KCOUNT__(128, extern template);
#endif
#if MAX_BUILD_KMER >= 160
__MACRO_KCOUNT__(160, extern template);
#endif

