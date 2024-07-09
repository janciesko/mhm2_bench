// Allocator.cpp

#include "upcxx_utils/memory-allocators/Allocator.h"

#include <cassert>  //assert

namespace upcxx_utils {

Allocator::Allocator(const std::size_t totalSize)
    : m_totalSize(totalSize) {}

Allocator::~Allocator() { m_totalSize = 0; }

};  // namespace upcxx_utils
