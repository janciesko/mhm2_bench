#include <upcxx/upcxx.hpp>

#include "upcxx_utils/log.hpp"
#include "upcxx_utils/progress_bar.hpp"

int test_progress_bar(int argc, char **argv) {
  upcxx_utils::open_dbg("test_progress_bar");

  int total = 1000;
  upcxx_utils::ProgressBar prog(total, "TestProgress");
  for (int i = 0; i < total; i++) prog.update();
  prog.done();

  upcxx_utils::close_dbg();

  return 0;
}
