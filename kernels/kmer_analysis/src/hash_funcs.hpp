#pragma once

#include <stdint.h>
#include <iostream>

using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

void MurmurHash3_x64_128(const void *key, const uint32_t len, const uint32_t seed, void *out);
uint64_t MurmurHash3_x64_64(const void *key, uint32_t len);
uint64_t quick_hash(uint64_t v);

#ifdef __cplusplus
}
#endif