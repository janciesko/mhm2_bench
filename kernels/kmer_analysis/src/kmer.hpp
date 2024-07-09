/*
 HipMer v 2.0, Copyright (c) 2020, The Regents of the University of California,
 through Lawrence Berkeley National Laboratory (subject to receipt of any required
 approvals from the U.S. Dept. of Energy).  All rights reserved."

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 (1) Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.

 (2) Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 (3) Neither the name of the University of California, Lawrence Berkeley National
 Laboratory, U.S. Dept. of Energy nor the names of its contributors may be used to
 endorse or promote products derived from this software without specific prior
 written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 DAMAGE.

 You are under no obligation whatsoever to provide any bug fixes, patches, or upgrades
 to the features, functionality or performance of the source code ("Enhancements") to
 anyone; however, if you choose to make your Enhancements available either publicly,
 or directly to Lawrence Berkeley National Laboratory, without imposing a separate
 written license agreement for such Enhancements, then you hereby grant the following
 license: a  non-exclusive, royalty-free perpetual license to install, use, modify,
 prepare derivative works, incorporate into other computer software, distribute, and
 sublicense such enhancements or derivative works thereof, in binary and source code
 form.
*/


#pragma once

#include <array>
#include <iostream>
#include <string>
#include <vector>

#include <upcxx/upcxx.hpp>

using longs_t = uint64_t;

template<int MAX_K>
class Kmer {
public:
    inline static unsigned int k = 0;
    inline static const int N_LONGS = (MAX_K + 31) / 32;
    std::array<longs_t, N_LONGS> longs;

    Kmer();

    Kmer(const longs_t *other_longs);

    static void set_k(unsigned int k);

    static unsigned int get_k();

    static unsigned int get_N_LONGS();

    static void get_kmers(unsigned int kmer_len, const std::string_view &seq, std::vector<Kmer> &kmers);

    bool operator<(const Kmer &o) const;

    uint64_t hash() const;

    // returns the *greatest* least-complement m-mer of this k-mer
    // -- greatest in order to avoid the trivial poly-A mer prevalent as errors in Illumina reads
    // -- least compliment should help smooth the distribution space of m-mers
    // -- returns a m-mer between the minimizer of the trivial fwd and rc for this kmer
    uint64_t get_minimizer_fast(int m, const Kmer *revcomp) const;

    uint64_t get_minimizer_fast(int m, bool least_complement = true) const;

    uint64_t minimizer_hash_fast(int m, const Kmer *revcomp = nullptr) const;

    Kmer revcomp() const;

    bool operator==(const Kmer &o) const;

    void to_string(char *s) const;

    std::string to_string() const;

    static void mer_to_string(char *s, const longs_t mmer, const int m);

    const uint64_t *get_longs() const;

};

template <int MAX_K>
struct KmerMinimizerHash {
    size_t operator()(const Kmer<MAX_K> &km) const { return km.minimizer_hash_fast(MINIMIZER_LEN); }
};

// specialization of std::Hash

namespace std {

template<int MAX_K>
struct hash<Kmer<MAX_K>> {
    size_t operator()(Kmer<MAX_K> const &km) const { return km.hash(); }
};

}  // namespace std

template <int MAX_K>
std::ostream &operator<<(std::ostream &out, const Kmer<MAX_K> &k);

#define __MACRO_KMER__(KMER_LEN, MODIFIER) \
  MODIFIER class Kmer<KMER_LEN>;           \
  MODIFIER struct KmerMinimizerHash<KMER_LEN>

// Reduce compile time by instantiating templates of common types
// extern template declarations are in kmer.hpp
// template instantiations each happen in src/CMakeLists via kmer-extern-template.in.cpp

__MACRO_KMER__(32, extern template);
#if MAX_BUILD_KMER >= 64
__MACRO_KMER__(64, extern template);
#endif
#if MAX_BUILD_KMER >= 96
__MACRO_KMER__(96, extern template);
#endif
#if MAX_BUILD_KMER >= 128
__MACRO_KMER__(128, extern template);
#endif
#if MAX_BUILD_KMER >= 160
__MACRO_KMER__(160, extern template);
#endif
