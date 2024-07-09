#include <iostream>
#include <upcxx/upcxx.hpp>

#include "upcxx_utils.hpp"

int test_upcxx_utils(int argc, char **argv) {
  upcxx_utils::open_dbg("test_upcxx_utils");

  if (!upcxx::rank_me()) std::cout << "Found upcxx_utils version " << UPCXX_UTILS_VERSION << std::endl;

  upcxx_utils::close_dbg();

  return 0;
}
