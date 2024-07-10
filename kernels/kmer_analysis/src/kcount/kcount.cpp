//#include "upcxx_utils.hpp"
#include "kcount.hpp"

using namespace std;
using namespace upcxx_utils;
using namespace upcxx;

template <int MAX_K>
static void count_kmers(unsigned kmer_len, vector<string> &local_reads, dist_object<KmerDHT<MAX_K>> &kmer_dht) {

    // initialize SeqBlockInserter
    SeqBlockInserter<MAX_K> seq_block_inserter(0, kmer_dht->get_minimizer_len());

    // read in sequence, get kmers, supermers sent to ThreeTierAggrStore (add_supermer)
    // MHM2: kcount.cpp count_kmers()
    for (string &read : local_reads) {
        seq_block_inserter.process_seq(read, 0, kmer_dht);
    }

    // sends pending updates to ranks
    // presumably triggers update_func set in KmerDHT (insert_supermer)
    // supermers decomposed back to kmers, inserted into HashTableInserterState
    kmer_dht->flush_updates();
}


template <int MAX_K>
void analyze_kmers(unsigned kmer_len, vector<string> &local_reads, dist_object<KmerDHT<MAX_K>> &kmer_dht) {
    count_kmers(kmer_len, local_reads, kmer_dht);
    barrier();

    // insert kmers from HashTableInserterState into KmerDHT->local_kmers
    kmer_dht->finish_updates();
    barrier();
}