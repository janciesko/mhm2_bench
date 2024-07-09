#pragma once
// bin_hash.hpp

#include <functional>

#include "version.h"

namespace upcxx_utils {

template <class T>
struct BinHash {
  // a simple class that takes the contents of a class and hashes it
  // assumes the class is a simple type, not a container

  std::hash<uint64_t> uint64_t_hash;
  std::hash<uint32_t> uint32_t_hash;
  std::hash<uint16_t> uint16_t_hash;
  std::hash<uint8_t> uint8_t_hash;

  std::size_t operator()(T const &s) const noexcept {
    size_t h1 = 0xDEADBEEF;
    const uint8_t *p = (const uint8_t *)&s;
    int bytes = sizeof(T), pos = 0;
    // likely to be unrolled
    for (; pos + 8 <= bytes; pos += 8) {
      size_t h2 = uint64_t_hash(*((uint64_t *)(p + pos)));
      h1 ^= h2 << 1;
    }
    if (pos + 4 <= bytes) {
      size_t h2 = uint32_t_hash(*((uint32_t *)(p + pos)));
      h1 ^= h2 << 1;
      pos += 4;
    }
    if (pos + 2 <= bytes) {
      size_t h2 = uint16_t_hash(*((uint16_t *)(p + pos)));
      h1 ^= h2 << 1;
      pos += 2;
    }
    if (pos < bytes) {
      size_t h2 = uint8_t_hash(*(p + pos));
      h1 ^= h2 << 1;
    }
    return h1;
  }
};

};  // namespace upcxx_utils
