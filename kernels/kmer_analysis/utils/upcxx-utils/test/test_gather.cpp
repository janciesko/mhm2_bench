#include <iostream>
#include <upcxx/upcxx.hpp>
#include <vector>

using std::vector;

using upcxx::barrier;
using upcxx::rank_me;
using upcxx::rank_n;
using upcxx::world;

#include "upcxx_utils/gather.hpp"
#include "upcxx_utils/log.hpp"
#include "upcxx_utils/split_rank.hpp"
#include "upcxx_utils/version.h"

#define VALIDATE(RESULT, ANSWER, msg)                                                     \
  do {                                                                                    \
    auto dummy = RESULT;                                                                  \
    DBG("Validating ", msg, "\n");                                                        \
    if ((RESULT) != (ANSWER)) DIE(msg, " got:", (RESULT), " expected:", (ANSWER), "!\n"); \
  } while (0)

#define VALIDATE_ARR(RESULTS, ANSWERS, COUNT, ROOT, RANK_N, msg)                                               \
  do {                                                                                                         \
    INFO("Validating array of count=", COUNT, " ", msg, "\n");                                                 \
    auto bad = 0;                                                                                              \
    for (size_t i = 0; i < (COUNT * RANK_N); i++) {                                                            \
      auto rotated = (i + ROOT * COUNT) % (COUNT * RANK_N);                                                    \
      if ((RESULTS)[i] != (ANSWERS)[rotated]) {                                                                \
        WARN(msg, " i=", i, " rot=", rotated, " got:", (RESULTS)[i], " expected:", (ANSWERS)[rotated], " at ", \
             (void *)&(RESULTS)[i], "!\n");                                                                    \
        bad++;                                                                                                 \
      }                                                                                                        \
    }                                                                                                          \
    if (bad) DIE("!!!\n");                                                                                     \
  } while (0)

int test_gather(int argc, char **argv) {
  if (!upcxx::rank_me()) std::cout << "Found upcxx_utils version " << UPCXX_UTILS_VERSION_DATE << std::endl;

  upcxx_utils::open_dbg("test_gather");
  {
    const int max_count = 100;
    vector<int> vals(rank_n() * max_count), answers(rank_n() * max_count), answers1(rank_n());
    for (intrank_t rank = 0; rank < rank_n(); rank++) {
      for (int i = 0; i < max_count; i++) {
        vals[rank * max_count + i] = rank + i + 1;
        // SOUT("rank=", rank, ", i=", i, ", vals[", rank * max_count + i, "]=", vals[rank * max_count + i], "\n");
      }
    }
    for (int i = 0; i < rank_n() * max_count; i++) {
      answers[i] = vals[i];
    }
    for (int rank = 0; rank < rank_n(); rank++) {
      answers1[rank] = vals[rank * max_count];
    }
    int *my_vals = vals.data() + rank_me() * max_count;

    vector<int> results(max_count);

    auto &lteam = upcxx_utils::split_rank::split_local_team();

    barrier();
    LOG("Testing 1 value gather\n");
    for (intrank_t root = 0; root < rank_n(); root++) {
      LOG("Starting linear gather tests root=", root, "\n");

      assert(*my_vals == vals[rank_me() * max_count]);
      auto fut1 = upcxx_utils::linear_gather(*my_vals, root, world());
      auto res = fut1.wait();
      LOG("Completed linear gather\n");
      if (rank_me() == root) VALIDATE_ARR(res, answers1, 1, root, rank_n(), "linear_gather 1 root" + std::to_string(root));
      barrier();
    }
    for (intrank_t root = 0; root < rank_n(); root++) {
      LOG("Starting binomial gather tests root=", root, "\n");
      assert(*my_vals == vals[rank_me() * max_count]);
      auto fut1 = upcxx_utils::binomial_gather(*my_vals, root, world());
      auto res = fut1.wait();
      LOG("Completed binomial gather\n");
      if (rank_me() == root) VALIDATE_ARR(res, answers1, 1, root, rank_n(), "binomial_gather 1 root" + std::to_string(root));

      barrier();
    }
    for (intrank_t root = 0; root < rank_n(); root++) {
      LOG("Starting gather tests root=", root, "\n");

      assert(*my_vals == vals[rank_me() * max_count]);
      auto fut1 = upcxx_utils::gather(*my_vals, root, world());
      auto res = fut1.wait();
      LOG("Completed  gather\n");
      if (rank_me() == root) VALIDATE_ARR(res, answers1, 1, root, rank_n(), "gather 1 root" + std::to_string(root));
      barrier();
    }
    barrier();

    LOG("Testing multi value gather\n");
    for (intrank_t root = 0; root < rank_n(); root++) {
      LOG("Starting multi value linear gather tests root=", root, "\n");

      assert(*my_vals == vals[rank_me() * max_count]);
      auto fut1 = upcxx_utils::linear_gather(my_vals, max_count, root, world());
      auto res = fut1.wait();
      LOG("Completed linear gather\n");
      if (rank_me() == root) VALIDATE_ARR(res, answers, max_count, root, rank_n(), "linear_gather root" + std::to_string(root));
      barrier();
    }
    for (intrank_t root = 0; root < rank_n(); root++) {
      LOG("Starting multi value binomial gather tests root=", root, "\n");
      assert(*my_vals == vals[rank_me() * max_count]);
      auto fut1 = upcxx_utils::binomial_gather(my_vals, max_count, root, world());
      auto res = fut1.wait();
      LOG("Completed binomial gather\n");
      if (rank_me() == root) VALIDATE_ARR(res, answers, max_count, root, rank_n(), "binomial_gather root" + std::to_string(root));

      barrier();
    }
    for (intrank_t root = 0; root < rank_n(); root++) {
      LOG("Starting multi value gather tests root=", root, "\n");

      assert(*my_vals == vals[rank_me() * max_count]);
      auto fut1 = upcxx_utils::gather(my_vals, max_count, root, world());
      auto res = fut1.wait();
      LOG("Completed  gather\n");
      if (rank_me() == root) VALIDATE_ARR(res, answers, max_count, root, rank_n(), "gather root" + std::to_string(root));
      barrier();
    }

    for (int skip = 1; skip < rank_n(); skip++)
      if (rank_n() % skip == 0) {
        LOG("Testing linear_gather with skip_ranks: ", skip, "\n");
        intrank_t skip_n = rank_n() / skip;
        intrank_t skip_me = rank_me() / skip;
        for (intrank_t root = 0; root < skip; root++) {
          std::vector<int> skip_answers1(skip_n);
          for (int i = 0; i < skip_n; i++) {
            skip_answers1[i] = answers1[(i * skip + root) % (rank_n())];
            if (rank_me() == root)
              DBG_VERBOSE("Answer[", i, "] = ", skip_answers1[i], " not ", answers[(i) % rank_n()], " without skip and root\n");
          }
          LOG("Starting linear gather tests skip=", skip, " skip_n=", skip_n, " root=", root, "\n");
          auto skip_vals = rank_me() % skip == root ? vals.data() + rank_me() * max_count : nullptr;
          auto fut1 = upcxx_utils::linear_gather(*skip_vals, root, world(), skip);
          auto res = fut1.wait();
          LOG("Completed linear gather\n");
          if (rank_me() == root)
            VALIDATE_ARR(res, skip_answers1, 1, 0, skip_n,
                         "linear_gather 1 skip " + std::to_string(skip) + " root " + std::to_string(root));
          barrier();
        }
        for (intrank_t root = 0; root < skip; root++) {
          LOG("Starting multi value linear gather tests root=", root, "\n");
          std::vector<int> skip_answers(skip_n * max_count);
          for (int i = 0; i < skip_n; i++)
            for (int j = 0; j < max_count; j++)
              skip_answers[i * max_count + j] = answers[(i * skip * max_count + j + root * max_count) % (max_count * rank_n())];
          auto skip_vals = rank_me() % skip == root ? vals.data() + rank_me() * max_count : nullptr;
          auto fut1 = upcxx_utils::linear_gather(skip_vals, max_count, root, world(), skip);
          auto res = fut1.wait();
          LOG("Completed linear gather\n");
          if (rank_me() == root)
            VALIDATE_ARR(res, skip_answers, max_count, 0, skip_n,
                         "linear_gather skip " + std::to_string(skip) + " root" + std::to_string(root));
          barrier();
        }
      }

    const upcxx::team &split = upcxx_utils::split_rank::split_local_team();
    if (split.id() != local_team().id()) {
      auto ln = split.rank_n();
      LOG("Starting two stage gather test with split.rank_n()=", ln, "\n");
      for (intrank_t root = 0; root < split.rank_n(); root++) {
        LOG("Starting multi value two_stage gather tests root=", root, "\n");
        std::vector<int> twostage_answers(rank_n() * max_count);
        for (int i = 0; i < rank_n(); i++) {
          for (int j = 0; j < max_count; j++) {
            auto l_rank_i = i % ln;
            auto stage_i = i / ln;
            twostage_answers[i * max_count + j] =
                answers[(stage_i * ln * max_count + ((root + l_rank_i) % ln) * max_count + j) % (max_count * rank_n())];
          }
        }
        assert(*my_vals == vals[rank_me() * max_count]);
        auto fut1 = upcxx_utils::two_stage_gather(my_vals, max_count, root, world(), split);
        auto res = fut1.wait();
        LOG("Completed two_stage gather\n");
        if (rank_me() == root)
          VALIDATE_ARR(res, twostage_answers, max_count, 0, rank_n(), "two_stage_gather root" + std::to_string(root));
        barrier();
      }
    }
  }
  SOUT("Complete with test_gather\n");
  upcxx::barrier();
  upcxx_utils::close_dbg();
  return 0;
}
