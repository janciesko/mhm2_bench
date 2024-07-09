#include <unistd.h>

#include <iostream>
#include <string>
#include <upcxx/upcxx.hpp>

#include "upcxx_utils/version.h"

int test_version(int argc, char **argv) {
  if (!upcxx::rank_me()) std::cout << "Found upcxx_utils version " << UPCXX_UTILS_VERSION << std::endl;
  char hnbuf[64];
  gethostname(hnbuf, sizeof(hnbuf) - 1);
  if (upcxx::local_team().rank_me() == 0) {
    std::cout << "proc " << upcxx::rank_me() << " on " << hnbuf << std::endl;
  }

  return 0;
}
