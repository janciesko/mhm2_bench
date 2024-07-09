#pragma once
// upcxx_defs.h

#include <upcxx/upcxx.hpp>

namespace upcxx_utils {

using global_byte_t = uint8_t;
using global_byte_ptr = upcxx::global_ptr<global_byte_t>;
using global_byte_ptr_ptr = upcxx::global_ptr<global_byte_ptr>;

};  // namespace upcxx_utils
