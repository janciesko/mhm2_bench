#include "kcount/kcount.hpp"
#include "kcount/kmer_dht.hpp"
#include "kmer.hpp"
#include "contigging.hpp"
#include <upcxx_utils/mem_profile.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <upcxx/upcxx.hpp>
#include <chrono>
#include <filesystem>


using namespace upcxx;
using namespace upcxx_utils;
using namespace std;

void init_devices();
void done_init_devices();
void teardown_devices();

void print_results(const std::string& output_csv, const std::string& result) {
    string directory = "../timing_results";
    if (!filesystem::exists(directory)) {
        if (!filesystem::create_directory(directory)) {
            cout << "Error: Cannot create results folder" << endl;
            exit(1);
        }
    }

    string path = directory + "/" + output_csv;


    std::ifstream ifs(path);
    bool file_exists = ifs.good();
    ifs.close();

    std::ofstream file(path, std::ios::app);

    if (!file.is_open()) {
        cout << "Error: Cannot open " << path << endl;
        exit(1);
    }

    if (!file_exists) {
        file << "Num reads,Read len,Kmer len,DHT time" << endl;
    }

    file << result << std::endl;

    file.close();
}

void generate_reads(const string& filename, vector<string> &local_reads, int read_len, int num_reads) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Error: Cannot open file " << filename << endl;
        exit(1);
    }

    string genome;
    string line;
    while (getline(file, line)) {
        genome += line;
    }
    file.close();


    int genome_len = genome.length();
    srand(upcxx::rank_me());

    int my_genome_portion = genome_len / upcxx::rank_n();
    int genome_remainder = genome_len % upcxx::rank_n();
    int my_start;
    if (upcxx::rank_me() < genome_remainder) {
        my_genome_portion++;
        my_start = upcxx::rank_me() * my_genome_portion;
    } else if (genome_remainder) {
        my_start = (genome_remainder * (my_genome_portion + 1)) + (upcxx::rank_me() - genome_remainder) * my_genome_portion;
    } else {
        my_start = upcxx::rank_me() * my_genome_portion;
    }

    int my_reads_portion = num_reads / upcxx::rank_n();
    int reads_remainder = num_reads % upcxx::rank_n();
    if (upcxx::rank_me() < reads_remainder) {
        my_reads_portion++;
    }

    for (int i = 0; i < my_reads_portion; i++) {
        int start_pos = my_start + rand() % (my_genome_portion - read_len + 1);
        local_reads.push_back(genome.substr(start_pos, read_len));
    }

    if (!upcxx::rank_me()) cout << "genome length: " << genome_len << "\ngenome remainder: " << genome_remainder << "\nreads remainder: " << reads_remainder << endl;
    upcxx::barrier();
    cout << "rank " << upcxx::rank_me() << ", genome portion: " << my_genome_portion << ", start: " << my_start << ", reads portion: " << my_reads_portion << endl;
}

int main(int argc, char **argv) {

    upcxx::init();

    if (argc != 5 && argc != 6) {
        cout << "Usage: kmer_dht_exercise <genome file> <num reads> <read len> <kmer len> <output csv (optional)>" << endl;
        exit(1);
    }
    string filename = argv[1];
    int num_reads = stoi(argv[2]);
    int read_len = stoi(argv[3]);
    int kmer_len = stoi(argv[4]);

    vector<string> local_reads;

    auto start = std::chrono::high_resolution_clock::now();

    generate_reads(filename, local_reads, read_len, num_reads);

    upcxx::barrier();
    init_devices();
    done_init_devices();
    upcxx::barrier();

    auto mid = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> first_duration = mid - start;

    auto max_k = (kmer_len / 32 + 1) * 32;

#define CONTIG_K(KMER_LEN)                                                                                                         \
    case KMER_LEN:                                                                                                                   \
        contigging<KMER_LEN>(kmer_len, local_reads, num_reads, read_len);                                                                                                 \
        break

    switch (max_k) {
        CONTIG_K(32);
#if MAX_BUILD_KMER >= 64
        CONTIG_K(64);
#endif
#if MAX_BUILD_KMER >= 96
        CONTIG_K(96);
#endif
#if MAX_BUILD_KMER >= 128
        CONTIG_K(128);
#endif
#if MAX_BUILD_KMER >= 160
        CONTIG_K(160);
#endif
        default: DIE("Built for max k = ", MAX_BUILD_KMER, " not k = ", max_k);
    }
#undef CONTIG_K

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> second_duration = end - start;
    if (upcxx::rank_me() == 0) {
        cout << "Read simulation time: " << first_duration.count() << " ms" << std::endl;
        cout << "DHT time: " << second_duration.count() << " ms" << std::endl;
    }


    if (argc == 6 && upcxx::rank_me() == 0) {
        string result = string(argv[2]) + "," + string(argv[3]) + "," + string(argv[4]) + "," + to_string(second_duration.count());
        print_results(argv[5], result);
    }


    upcxx::barrier();
    upcxx::finalize();

    return 0;
}