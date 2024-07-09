#pragma once
// Utils.h
#include <cstddef>

namespace upcxx_utils {

class Utils {
 public:
  static std::size_t CalculatePadding(const std::size_t baseAddress, const std::size_t alignment);

  static std::size_t CalculatePaddingWithHeader(const std::size_t baseAddress, const std::size_t alignment,
                                                const std::size_t headerSize);
};

};  // namespace upcxx_utils
