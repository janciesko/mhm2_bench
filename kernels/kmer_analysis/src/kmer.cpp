#include <stdint.h>
#include <stdio.h>

#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>

#include <endian.h>
#define H2BE htobe64
#define BE2H be64toh


#include "hash_funcs.hpp"
#include "kmer.hpp"

using namespace std;

inline const longs_t TWIN_TABLE[256] = {
        0xFF, 0xBF, 0x7F, 0x3F, 0xEF, 0xAF, 0x6F, 0x2F, 0xDF, 0x9F, 0x5F, 0x1F, 0xCF, 0x8F, 0x4F, 0x0F, 0xFB, 0xBB, 0x7B, 0x3B,
        0xEB, 0xAB, 0x6B, 0x2B, 0xDB, 0x9B, 0x5B, 0x1B, 0xCB, 0x8B, 0x4B, 0x0B, 0xF7, 0xB7, 0x77, 0x37, 0xE7, 0xA7, 0x67, 0x27,
        0xD7, 0x97, 0x57, 0x17, 0xC7, 0x87, 0x47, 0x07, 0xF3, 0xB3, 0x73, 0x33, 0xE3, 0xA3, 0x63, 0x23, 0xD3, 0x93, 0x53, 0x13,
        0xC3, 0x83, 0x43, 0x03, 0xFE, 0xBE, 0x7E, 0x3E, 0xEE, 0xAE, 0x6E, 0x2E, 0xDE, 0x9E, 0x5E, 0x1E, 0xCE, 0x8E, 0x4E, 0x0E,
        0xFA, 0xBA, 0x7A, 0x3A, 0xEA, 0xAA, 0x6A, 0x2A, 0xDA, 0x9A, 0x5A, 0x1A, 0xCA, 0x8A, 0x4A, 0x0A, 0xF6, 0xB6, 0x76, 0x36,
        0xE6, 0xA6, 0x66, 0x26, 0xD6, 0x96, 0x56, 0x16, 0xC6, 0x86, 0x46, 0x06, 0xF2, 0xB2, 0x72, 0x32, 0xE2, 0xA2, 0x62, 0x22,
        0xD2, 0x92, 0x52, 0x12, 0xC2, 0x82, 0x42, 0x02, 0xFD, 0xBD, 0x7D, 0x3D, 0xED, 0xAD, 0x6D, 0x2D, 0xDD, 0x9D, 0x5D, 0x1D,
        0xCD, 0x8D, 0x4D, 0x0D, 0xF9, 0xB9, 0x79, 0x39, 0xE9, 0xA9, 0x69, 0x29, 0xD9, 0x99, 0x59, 0x19, 0xC9, 0x89, 0x49, 0x09,
        0xF5, 0xB5, 0x75, 0x35, 0xE5, 0xA5, 0x65, 0x25, 0xD5, 0x95, 0x55, 0x15, 0xC5, 0x85, 0x45, 0x05, 0xF1, 0xB1, 0x71, 0x31,
        0xE1, 0xA1, 0x61, 0x21, 0xD1, 0x91, 0x51, 0x11, 0xC1, 0x81, 0x41, 0x01, 0xFC, 0xBC, 0x7C, 0x3C, 0xEC, 0xAC, 0x6C, 0x2C,
        0xDC, 0x9C, 0x5C, 0x1C, 0xCC, 0x8C, 0x4C, 0x0C, 0xF8, 0xB8, 0x78, 0x38, 0xE8, 0xA8, 0x68, 0x28, 0xD8, 0x98, 0x58, 0x18,
        0xC8, 0x88, 0x48, 0x08, 0xF4, 0xB4, 0x74, 0x34, 0xE4, 0xA4, 0x64, 0x24, 0xD4, 0x94, 0x54, 0x14, 0xC4, 0x84, 0x44, 0x04,
        0xF0, 0xB0, 0x70, 0x30, 0xE0, 0xA0, 0x60, 0x20, 0xD0, 0x90, 0x50, 0x10, 0xC0, 0x80, 0x40, 0x00};

inline const longs_t ZERO_MASK[32] = {
        0x0000000000000000, 0xC000000000000000, 0xF000000000000000, 0xFC00000000000000, 0xFF00000000000000, 0xFFC0000000000000,
        0xFFF0000000000000, 0xFFFC000000000000, 0xFFFF000000000000, 0xFFFFC00000000000, 0xFFFFF00000000000, 0xFFFFFC0000000000,
        0xFFFFFF0000000000, 0xFFFFFFC000000000, 0xFFFFFFF000000000, 0xFFFFFFFC00000000, 0xFFFFFFFF00000000, 0xFFFFFFFFC0000000,
        0xFFFFFFFFF0000000, 0xFFFFFFFFFC000000, 0xFFFFFFFFFF000000, 0xFFFFFFFFFFC00000, 0xFFFFFFFFFFF00000, 0xFFFFFFFFFFFC0000,
        0xFFFFFFFFFFFF0000, 0xFFFFFFFFFFFFC000, 0xFFFFFFFFFFFFF000, 0xFFFFFFFFFFFFFC00, 0xFFFFFFFFFFFFFF00, 0xFFFFFFFFFFFFFFC0,
        0xFFFFFFFFFFFFFFF0, 0xFFFFFFFFFFFFFFFC};

template <int MAX_K>
Kmer<MAX_K>::Kmer() {
  assert(Kmer::k > 0);
  longs.fill(0);
}

template <int MAX_K>
Kmer<MAX_K>::Kmer(const longs_t *other_longs) {
  assert(Kmer::k > 0);
  memcpy(&(longs[0]), other_longs, N_LONGS * sizeof(longs_t));
}

template <int MAX_K>
void Kmer<MAX_K>::set_k(unsigned int k) {
    Kmer::k = k;
}

template <int MAX_K>
unsigned int Kmer<MAX_K>::get_k() {
  return Kmer::k;
}

template <int MAX_K>
unsigned int Kmer<MAX_K>::get_N_LONGS() {
    return Kmer::N_LONGS;
}

template <int MAX_K>
void Kmer<MAX_K>::get_kmers(unsigned kmer_len, const std::string_view &seq, std::vector<Kmer> &kmers) {

    kmers.clear();
    int bufsize = max((int)N_LONGS, (int)(seq.size() + 31) / 32) + N_LONGS;
    int lastLong = N_LONGS - 1;
    kmers.resize(seq.size() - kmer_len + 1, {});
    longs_t *buf = new longs_t[bufsize];
    uint8_t *bufPtr = (uint8_t *)buf;
    memset(buf, 0, bufsize * 8);
    const char *s = seq.data();

    for (unsigned i = 0; i < seq.size(); ++i) {
        int j = i % 32;
        int l = i / 32;
        assert(*s != '\0');
        longs_t x = ((*s) & 4) >> 1;
        buf[l] |= ((x + ((x ^ (*s & 2)) >> 1)) << (2 * (31 - j)));
        s++;
    }

    // fix to big endian
    for (int l = 0; l < bufsize; l++) buf[l] = H2BE(buf[l]);
    const longs_t mask = ((int64_t)0x3);
    longs_t endmask = 0;
    if (Kmer::k % 32) {
        endmask = (((longs_t)2) << (2 * (31 - (Kmer::k % 32)) + 1)) - 1;
        // k == 0 :                0x0000000000000000
        // k == 1 : 2 << 61  - 1 : 0x3FFFFFFFFFFFFFFF
        // k == 31: 2 << 1   - 1 : 0x0000000000000003
    }
    endmask = ~endmask;

    // make 4 passes for each phase of 2 bits per base
    for (int shift = 0; shift < 4; shift++) {
        if (shift > 0) {
            // shift all bits by 2 in the buffer of longs
            for (int l = 0; l < bufsize - 1; l++) {
                buf[l] = (~mask & (BE2H(buf[l]) << 2)) | (mask & (BE2H(buf[l + 1]) >> 62));
                buf[l] = H2BE(buf[l]);
            }
        }
        // enumerate the kmers in the phase
        for (unsigned i = shift; i < kmers.size(); i += 4) {
            int byteOffset = i / 4;
            assert(byteOffset + N_LONGS * 8 <= bufsize * 8);
            for (int l = 0; l < N_LONGS; l++) {
                kmers[i].longs[l] = BE2H(*((longs_t *)(bufPtr + byteOffset + l * 8)));
            }
            // set remaining bits to 0
            kmers[i].longs[lastLong] &= endmask;
        }
    }
    delete[] buf;

}

template <int MAX_K>
bool Kmer<MAX_K>::operator<(const Kmer<MAX_K> &o) const {
    for (size_t i = 0; i < N_LONGS; ++i) {
        if (longs[i] < o.longs[i]) return true;
        if (longs[i] > o.longs[i]) return false;
    }
    return false;
}

template <int MAX_K>
uint64_t Kmer<MAX_K>::hash() const {
    return MurmurHash3_x64_64(reinterpret_cast<const void *>(longs.data()), N_LONGS * sizeof(longs_t));
}

template <int MAX_K>
Kmer<MAX_K> Kmer<MAX_K>::revcomp() const {
    Kmer<MAX_K> km;
    auto last_long = (k + 31) / 32;
    assert(last_long <= N_LONGS);
    for (size_t i = 0; i < last_long; i++) {
        longs_t v = longs[i];
        km.longs[last_long - 1 - i] = (TWIN_TABLE[v & 0xFF] << 56) | (TWIN_TABLE[(v >> 8) & 0xFF] << 48) |
                                      (TWIN_TABLE[(v >> 16) & 0xFF] << 40) | (TWIN_TABLE[(v >> 24) & 0xFF] << 32) |
                                      (TWIN_TABLE[(v >> 32) & 0xFF] << 24) | (TWIN_TABLE[(v >> 40) & 0xFF] << 16) |
                                      (TWIN_TABLE[(v >> 48) & 0xFF] << 8) | (TWIN_TABLE[(v >> 56)]);
    }
    longs_t shift = (Kmer::k % 32) ? 2 * (32 - (Kmer::k % 32)) : 0;
    longs_t shiftmask = (Kmer::k % 32) ? (((((longs_t)1) << shift) - 1) << (64 - shift)) : ((longs_t)0);
    km.longs[0] = km.longs[0] << shift;
    for (size_t i = 1; i < last_long; i++) {
        km.longs[i - 1] |= (km.longs[i] & shiftmask) >> (64 - shift);
        km.longs[i] = km.longs[i] << shift;
    }
    return km;
}

// returns the *greatest* least-complement m-mer of this k-mer
// -- greatest in order to avoid the trivial poly-A mer prevalent as errors in Illumina reads
// -- least compliment should help smooth the distribution space of m-mers
// -- returns a m-mer between the minimizer of the trivial fwd and rc for this kmer
template <int MAX_K>
uint64_t Kmer<MAX_K>::get_minimizer_fast(int m, const Kmer<MAX_K> *revcomp) const {
    assert(m <= Kmer::k);
    assert(m <= 28);
    // chunk is a uint64_t whose bases fully containing a series of chunk_step candidate minimizers
    const int chunk = 32;
    const int chunk_step = chunk - ((m + 3) / 4) * 4;  // chunk_step is a multiple of 4;

    int base;
    const int num_candidates = revcomp == nullptr ? 1 : Kmer::k - m + 1;
    uint64_t revcomp_candidates[num_candidates];
    if (revcomp != nullptr) {
        // calculate and temporarily store all revcomp minimizer candidates on the stack
        for (base = 0; base <= Kmer::k - m; base += chunk_step) {
            int shift = base % 32;
            int l = base / 32;
            uint64_t tmp = revcomp->longs[l];
            if (shift) {
                tmp = (tmp << (shift * 2));
                if (l < N_LONGS - 1) tmp |= revcomp->longs[l + 1] >> (64 - shift * 2);
            }
            for (int j = 0; j < chunk_step; j++) {
                if (base + j + m > Kmer::k) break;
                assert(base + j < num_candidates);
                revcomp_candidates[base + j] = ((tmp << (j * 2)) & ZERO_MASK[m]);
            }
        }
    }

    uint64_t minimizer = 0;
    // calculate and compare minimizers from revcomp
    const uint8_t *data = (uint8_t *)longs.data();
    for (base = 0; base <= Kmer::k - m; base += chunk_step) {
        int shift = base % 32;
        int l = base / 32;
        uint64_t tmp = longs[l];
        if (shift) {
            tmp = (tmp << (shift * 2));
            if (l < N_LONGS - 1) tmp |= longs[l + 1] >> (64 - shift * 2);
        }
        for (int j = 0; j < chunk_step; j++) {
            if (base + j + m > Kmer::k) break;
            uint64_t fwd_candidate = ((tmp << (j * 2)) & ZERO_MASK[m]);
            auto &revcomp_candidate = revcomp == nullptr ? fwd_candidate : revcomp_candidates[num_candidates - base - j - 1];
            uint64_t &least_candidate = (fwd_candidate < revcomp_candidate) ? fwd_candidate : revcomp_candidate;
            if (least_candidate > minimizer) minimizer = least_candidate;
        }
    }
    return minimizer;
}

template <int MAX_K>
uint64_t Kmer<MAX_K>::get_minimizer_fast(int m, bool least_complement) const {
    if (least_complement) {
        Kmer<MAX_K> revcomp = this->revcomp();
        return get_minimizer_fast(m, &revcomp);
    } else {
        return get_minimizer_fast(m, nullptr);
    }
}

template <int MAX_K>
uint64_t Kmer<MAX_K>::minimizer_hash_fast(int m, const Kmer<MAX_K> *revcomp) const {
    uint64_t minimizer;
    if (revcomp == nullptr) {
        minimizer = get_minimizer_fast(m, true);
    } else {
        minimizer = get_minimizer_fast(m, revcomp);
    }
    return quick_hash(minimizer);
}

template <int MAX_K>
bool Kmer<MAX_K>::operator==(const Kmer &o) const {
    return longs == o.longs;
}

template <int MAX_K>
void Kmer<MAX_K>::to_string(char *s) const {
    size_t i, j, l = 0;
    for (i = 0; i < Kmer::k; i += 32) {
        assert(l < longs.size());
        const longs_t &mer = longs[l++];
        j = (i + 32 <= Kmer::k) ? 32 : Kmer::k % 32;
        mer_to_string(s, mer, j);
        s += j;
    }
    *s = '\0';
}

template <int MAX_K>
string Kmer<MAX_K>::to_string() const {
    string buf(Kmer::k, 'N');
    to_string(buf.data());
    return buf;
}

template <int MAX_K>
void Kmer<MAX_K>::mer_to_string(char *s, const longs_t mmer, const int m) {
    for (int j = 0; j < m; j++) {
        switch ((mmer >> (2 * (31 - j))) & 0x03) {
            case 0x00: *s = 'A'; break;
            case 0x01: *s = 'C'; break;
            case 0x02: *s = 'G'; break;
            case 0x03: *s = 'T'; break;
        }
        ++s;
    }
}



template <int MAX_K>
const uint64_t *Kmer<MAX_K>::get_longs() const {
    return longs.data();
}


template <int MAX_K>
ostream &operator<<(ostream &out, const Kmer<MAX_K> &k) {
    return out << k.to_string();
}

#define KMER_K(KMER_LEN) template ostream &operator<< <KMER_LEN>(ostream &out, const Kmer<KMER_LEN> &k);

KMER_K(32);
#if MAX_BUILD_KMER >= 64

KMER_K(64);
#endif
#if MAX_BUILD_KMER >= 96
KMER_K(96);
#endif
#if MAX_BUILD_KMER >= 128
KMER_K(128);
#endif
#if MAX_BUILD_KMER >= 160
KMER_K(160);
#endif

#undef KMER_K

