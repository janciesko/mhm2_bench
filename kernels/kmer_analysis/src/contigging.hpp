#pragma once

#include <array>
#include <string>
#include <vector>

using std::string;
using std::vector;


template <int MAX_K>
void contigging(int kmer_len, vector<string> &local_reads, int num_reads, int read_len);

#define __MACRO_CONTIGGING__(KMER_LEN, MODIFIER) \
  MODIFIER void contigging<KMER_LEN>(int, vector<string> &, int, int);

// Reduce compile time by instantiating templates of common types
// extern template declarations are in contigging.hpp
// template instantiations each happen in src/CMakeLists via contigging-extern-template.in.cpp

__MACRO_CONTIGGING__(32, extern template);
#if MAX_BUILD_KMER >= 64
__MACRO_CONTIGGING__(64, extern template);
#endif
#if MAX_BUILD_KMER >= 96
__MACRO_CONTIGGING__(96, extern template);
#endif
#if MAX_BUILD_KMER >= 128
__MACRO_CONTIGGING__(128, extern template);
#endif
#if MAX_BUILD_KMER >= 160
__MACRO_CONTIGGING__(160, extern template);
#endif