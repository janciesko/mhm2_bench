#include <iostream>
#include <upcxx/upcxx.hpp>
#include <vector>

using std::vector;

using upcxx::barrier;
using upcxx::rank_me;
using upcxx::rank_n;
using upcxx::world;

#include "upcxx_utils/log.hpp"
#include "upcxx_utils/reduce_prefix.hpp"
#include "upcxx_utils/split_rank.hpp"
#include "upcxx_utils/version.h"

#define VALIDATE(RESULT, ANSWER, msg)                                                     \
  do {                                                                                    \
    auto dummy = RESULT;                                                                  \
    DBG("Validating ", msg, "\n");                                                        \
    if ((RESULT) != (ANSWER)) DIE(msg, " got:", (RESULT), " expected:", (ANSWER), "!\n"); \
  } while (0)

#define VALIDATE_ARR(RESULTS, ANSWERS, COUNT, msg)                                                                   \
  do {                                                                                                               \
    DBG("Validating array of count=", COUNT, " ", msg, "\n");                                                        \
    for (size_t i = 0; i < (COUNT); i++) {                                                                           \
      if ((RESULTS)[i] != (ANSWERS)[i]) DIE(msg, "i=", i, " got:", (RESULTS)[i], " expected:", (ANSWERS)[i], "!\n"); \
    }                                                                                                                \
  } while (0)

int test_reduce_prefix(int argc, char **argv) {
  if (!upcxx::rank_me()) std::cout << "Found upcxx_utils version " << UPCXX_UTILS_VERSION << std::endl;

  upcxx_utils::open_dbg("test_reduce_prefix");
  {
    const int max_count = 100;
    vector<int> vals(rank_n() * max_count), answers(rank_n() * max_count);
    for (intrank_t rank = 0; rank < rank_n(); rank++) {
      for (int i = 0; i < max_count; i++) {
        vals[rank * max_count + i] = rank + i + 1;
        // SOUT("rank=", rank, ", i=", i, ", vals[", rank * max_count + i, "]=", vals[rank * max_count + i], "\n");
      }
    }
    for (int i = 0; i < max_count; i++) {
      answers[i] = vals[i];
    }
    for (intrank_t rank = 1; rank < rank_n(); rank++) {
      for (int i = 0; i < max_count; i++) {
        answers[rank * max_count + i] = answers[(rank - 1) * max_count + i] + vals[rank * max_count + i];
        // SOUT("rank=", rank, ", i=", i, ", answers[", rank * max_count + i, "]=", answers[rank * max_count + i], "\n");
      }
    }

    vector<int> results(max_count);

    auto &lteam = upcxx_utils::split_rank::split_local_team();

    auto fut1 = upcxx_utils::reduce_prefix(vals[rank_me() * max_count], upcxx::op_fast_add, world(), false);
    VALIDATE(fut1.wait(), answers[rank_me() * max_count], "reduce_prefix 1");

    fut1 = upcxx_utils::reduce_prefix_one_ring(vals[rank_me() * max_count], upcxx::op_fast_add, world(), false);
    VALIDATE(fut1.wait(), answers[rank_me() * max_count], "reduce_prefix_ring 1");

    fut1 = upcxx_utils::reduce_prefix_one_binomial(vals[rank_me() * max_count], upcxx::op_fast_add, world(), false);
    VALIDATE(fut1.wait(), answers[rank_me() * max_count], "reduce_prefix_binomial 1");

    fut1 = upcxx_utils::reduce_prefix_one_binomial(vals[rank_me() * max_count], upcxx::op_fast_add, world(), false);
    VALIDATE(fut1.wait(), answers[rank_me() * max_count], "reduce_prefix_binomial 1");

    fut1 = upcxx_utils::reduce_prefix_one_binary_tree(vals[rank_me() * max_count], upcxx::op_fast_add, world(), false);
    VALIDATE(fut1.wait(), answers[rank_me() * max_count], "reduce_prefix_binary_tree 1");

    // now test the array API

    auto fut2 = upcxx_utils::reduce_prefix(vals.data() + rank_me() * max_count, results.data(), max_count, upcxx::op_fast_add,
                                           world(), false);
    fut2.wait();
    VALIDATE_ARR(results.data(), answers.data() + rank_me() * max_count, max_count, "reduce_prefix many");

    fut2 = upcxx_utils::reduce_prefix_ring(vals.data() + rank_me() * max_count, results.data(), max_count, upcxx::op_fast_add,
                                           world(), false);
    fut2.wait();
    VALIDATE_ARR(results.data(), answers.data() + rank_me() * max_count, max_count, "reduce_prefix_ring many");

    fut2 = upcxx_utils::reduce_prefix_binomial(vals.data() + rank_me() * max_count, results.data(), max_count, upcxx::op_fast_add,
                                               world(), false);
    fut2.wait();
    VALIDATE_ARR(results.data(), answers.data() + rank_me() * max_count, max_count, "reduce_prefix_binomial many");

    fut2 = upcxx_utils::reduce_prefix_binary_tree(vals.data() + rank_me() * max_count, results.data(), max_count,
                                                  upcxx::op_fast_add, world(), false);
    fut2.wait();
    VALIDATE_ARR(results.data(), answers.data() + rank_me() * max_count, max_count, "reduce_prefix_binary_tree many");

    fut2 = upcxx_utils::reduce_prefix_auto_choice_local(vals.data() + rank_me() * max_count, results.data(), max_count,
                                                        upcxx::op_fast_add, world(), false, upcxx::local_team());
    fut2.wait();
    VALIDATE_ARR(results.data(), answers.data() + rank_me() * max_count, max_count, "reduce_prefix_auto_local local_team() many");

    if (rank_n() % lteam.rank_n() == 0) {
      fut2 = upcxx_utils::reduce_prefix_auto_choice_local(vals.data() + rank_me() * max_count, results.data(), max_count,
                                                          upcxx::op_fast_add, world(), false, lteam);
      fut2.wait();
      VALIDATE_ARR(results.data(), answers.data() + rank_me() * max_count, max_count, "reduce_prefix_auto_local lteam many");
    }

    barrier();

    DBG("Changing answers for final to first\n");
    // now test with return final to first
    for (int i = 0; i < max_count; i++) {
      answers[i] = answers[(rank_n() - 1) * max_count + i];
      // SOUT("rank=", 0, ", i=", i, ", answers[", 0 * max_count + i, "]=", answers[0 * max_count + i], "\n");
    }

    auto fut3 = upcxx_utils::reduce_prefix(vals[rank_me() * max_count], upcxx::op_fast_add, world(), true);
    VALIDATE(fut3.wait(), answers[rank_me() * max_count], "reduce_prefix 1 return final");

    fut3 = upcxx_utils::reduce_prefix_one_ring(vals[rank_me() * max_count], upcxx::op_fast_add, world(), true);
    VALIDATE(fut3.wait(), answers[rank_me() * max_count], "reduce_prefix_ring 1 return final");

    fut3 = upcxx_utils::reduce_prefix_one_binomial(vals[rank_me() * max_count], upcxx::op_fast_add, world(), true);
    VALIDATE(fut3.wait(), answers[rank_me() * max_count], "reduce_prefix_binomial 1 return final");

    fut3 = upcxx_utils::reduce_prefix_one_binary_tree(vals[rank_me() * max_count], upcxx::op_fast_add, world(), true);
    VALIDATE(fut3.wait(), answers[rank_me() * max_count], "reduce_prefix_binary_tree 1 return final");

    // now test the array API with return final to first

    auto fut4 = upcxx_utils::reduce_prefix(vals.data() + rank_me() * max_count, results.data(), max_count, upcxx::op_fast_add,
                                           world(), true);
    fut4.wait();
    VALIDATE_ARR(results.data(), answers.data() + rank_me() * max_count, max_count, "reduce_prefix many return final");

    fut4 = upcxx_utils::reduce_prefix_ring(vals.data() + rank_me() * max_count, results.data(), max_count, upcxx::op_fast_add,
                                           world(), true);
    fut4.wait();
    VALIDATE_ARR(results.data(), answers.data() + rank_me() * max_count, max_count, "reduce_prefix_ring many return final");

    fut4 = upcxx_utils::reduce_prefix_binomial(vals.data() + rank_me() * max_count, results.data(), max_count, upcxx::op_fast_add,
                                               world(), true);
    fut4.wait();
    VALIDATE_ARR(results.data(), answers.data() + rank_me() * max_count, max_count, "reduce_prefix_binomial many return final");

    fut4 = upcxx_utils::reduce_prefix_binary_tree(vals.data() + rank_me() * max_count, results.data(), max_count,
                                                  upcxx::op_fast_add, world(), true);
    fut4.wait();
    VALIDATE_ARR(results.data(), answers.data() + rank_me() * max_count, max_count, "reduce_prefix_binary_tree many return final");

    fut4 = upcxx_utils::reduce_prefix_auto_choice_local(vals.data() + rank_me() * max_count, results.data(), max_count,
                                                        upcxx::op_fast_add, world(), true, local_team());
    fut4.wait();
    VALIDATE_ARR(results.data(), answers.data() + rank_me() * max_count, max_count,
                 "reduce_prefix_auto_choice_local many over local_team() return final");

    if (rank_n() % lteam.rank_n() == 0) {
      fut4 = upcxx_utils::reduce_prefix_auto_choice_local(vals.data() + rank_me() * max_count, results.data(), max_count,
                                                          upcxx::op_fast_add, world(), true, lteam);
      fut4.wait();
      VALIDATE_ARR(results.data(), answers.data() + rank_me() * max_count, max_count,
                   "reduce_prefix_auto_choice_local many over lteam return final");
    }

    barrier();
    DBG("Testing huge block\n");
    // test the pipelined version with > 16MB of data to reduce.

    const int huge_count = 17 * ONE_MB / sizeof(int);
    vals.resize(huge_count);
    answers.resize(huge_count);
    results.resize(huge_count);
    vector<int> final_vals(huge_count);

    for (int i = 0; i < huge_count; i++) {
      vals[i] = rank_me() + i + 1;
      results[i] = i + 1;  // for rank0
    }

    for (intrank_t rank = 0; rank < rank_n(); rank++) {
      for (int i = 0; i < huge_count; i++) {
        if (rank != 0) {
          results[i] += rank + i + 1;  // + val for rank
        }
        if (rank == rank_me()) {
          answers[i] = results[i];
        }
        if (rank == rank_n() - 1) {
          final_vals[i] = results[i];
        }
      }
    }
    DBG("vals[0]=", vals[0], " answers[0]=", answers[0], " final[0]=", final_vals[0], "\n");

    auto fut5 = upcxx_utils::reduce_prefix(vals.data(), results.data(), huge_count, upcxx::op_fast_add, world(), false);
    fut5.wait();
    VALIDATE_ARR(results.data(), answers.data(), huge_count, "reduce_prefix many");

    fut5 = upcxx_utils::reduce_prefix(vals.data(), results.data(), huge_count, upcxx::op_fast_add, world(), true);
    fut5.wait();
    VALIDATE_ARR(results.data(), rank_me() == 0 ? final_vals.data() : answers.data(), huge_count, "reduce_prefix many");
  }
  upcxx::barrier();
  upcxx_utils::close_dbg();
  return 0;
}
