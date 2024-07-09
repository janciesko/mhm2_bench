/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include "upcxx/upcxx.hpp"

#ifndef MAIN
#error "MAIN must be passed as a definition"
#endif

int MAIN(int argc, char **argv);

int main(int argc, char **argv) {
  upcxx::init();
  upcxx::barrier();
  int ret = MAIN(argc, argv);
  if (!upcxx::rank_me()) std::cout << "Done ret=" << ret << std::endl;
  auto all_ret = upcxx::reduce_one(ret, upcxx::op_fast_add, 0).wait();
  upcxx::barrier();
  if (!upcxx::rank_me()) std::cout << "All done ret=" << all_ret << std::endl;
  upcxx::finalize();
  return ret;
}
