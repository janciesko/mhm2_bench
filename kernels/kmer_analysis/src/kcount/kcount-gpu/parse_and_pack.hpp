#pragma once

#include <vector>

namespace kcount_gpu {

    struct ParseAndPackDriverState;

    struct SupermerInfo {
        int target;
        int offset;
        uint16_t len;
    };

    class ParseAndPackGPUDriver {
        // this opaque data type stores CUDA specific variables
        ParseAndPackDriverState *dstate = nullptr;

        int upcxx_rank_me;
        int upcxx_rank_n;
        int max_kmers;
        int kmer_len;
        int qual_offset;
        int num_kmer_longs;
        int minimizer_len;
        double t_func = 0, t_malloc = 0, t_cp = 0, t_kernel = 0;
        char *dev_seqs;
        int *dev_kmer_targets;

        SupermerInfo *dev_supermers;
        char *dev_packed_seqs;
        unsigned int *dev_num_supermers;
        unsigned int *dev_num_valid_kmers;

    public:
        std::vector<SupermerInfo> supermers;
        std::string packed_seqs;

        ParseAndPackGPUDriver(int upcxx_rank_me, int upcxx_rank_n, int qual_offset, int kmer_len, int num_kmer_longs, int minimizer_len,
                              double &init_time);
        ~ParseAndPackGPUDriver();
        bool process_seq_block(const std::string &seqs, unsigned int &num_valid_kmers);
        void pack_seq_block(const std::string &seqs);
        std::tuple<double, double> get_elapsed_times();
    };

}