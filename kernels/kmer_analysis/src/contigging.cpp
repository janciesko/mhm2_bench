#include "contigging.hpp"

#include "kcount/kcount.hpp"
#include "kmer_dht.hpp"
#include <upcxx_utils/mem_profile.hpp>

using namespace upcxx;
using namespace upcxx_utils;
using namespace std;

template<int MAX_K>
void contigging(int kmer_len, vector<string> &local_reads, int num_reads, int read_len) {

    // set kmer length
    Kmer<MAX_K>::set_k(kmer_len);

    // prepare args for KmerDHT constructor
    double max_kmer_store_mb = upcxx_utils::get_free_mem(true) / 1024 / 1024 / 100;
    max_kmer_store_mb = upcxx::reduce_all(max_kmer_store_mb / upcxx::local_team().rank_n(), upcxx::op_fast_min).wait();
    auto max_kmer_store_bytes = max_kmer_store_mb * ONE_MB;
    int max_rpcs_in_flight = 100;

    int kmers_per_read = read_len - kmer_len + 1;
    int64_t my_num_kmers = num_reads * kmers_per_read / rank_n();

    // initialize KmerDHT
    // MHM2: contigging.cpp
    dist_object<KmerDHT<MAX_K>> kmer_dht(world(), my_num_kmers, max_kmer_store_bytes, max_rpcs_in_flight, true);

    analyze_kmers(kmer_len, local_reads, kmer_dht);

    upcxx::barrier();

    // check local_kmers
    for (int i = 0; i < upcxx::rank_n(); i++) {
        if (upcxx::rank_me() == i) {
            int num_local_kmers = 0;
            int total_kmer_count = 0;
            for (auto it = kmer_dht->local_kmers_begin(); it != kmer_dht->local_kmers_end(); it++) {
                   auto kmer = it->first;
                   auto kmer_counts = &it->second;
//                    int kmer_len = Kmer<MAX_K>::k;
//                    char *kmer_string = new char[kmer_len + 1];
//                    kmer.to_string(kmer_string);
//                    cout << "kmer: " << kmer_string << endl;
//                    cout << "\tcount: " << kmer_counts->count << endl;
//                    cout << "\tleft: " << kmer_counts->left << endl;
//                    cout << "\tright: " << kmer_counts->right << endl;
//                    delete[] kmer_string;
                    total_kmer_count += kmer_counts->count;


                num_local_kmers++;

            }
            cout << "\nRank " << i << " num kmers: " << num_local_kmers << ", total kmer count: " << total_kmer_count << ", num supermer inserts: " << kmer_dht->get_num_supermer_inserts() << endl;

        }
        upcxx::barrier();
    }

}