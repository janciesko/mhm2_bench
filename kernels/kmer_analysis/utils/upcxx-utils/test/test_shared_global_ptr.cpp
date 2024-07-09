/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <upcxx/upcxx.hpp>

#include "upcxx_utils/shared_global_ptr.hpp"

using upcxx::dist_object;
using upcxx::global_ptr;
using upcxx::local_team;
using upcxx::promise;
using upcxx_utils::shared_global_ptr;

int test_shared_global_ptr(int argc, char **argv) {
  SLOG("Found upcxx_utils version ", UPCXX_UTILS_VERSION, "\n");

  upcxx_utils::open_dbg("test_global_shared_ptr");

  {
    // test shared works like a shared_ptr on one rank:

    shared_global_ptr<int> intptr1, intptr2, intptr3;
    intptr1.reset(upcxx::new_<int>(rank_me()));
    assert(intptr1);
    assert(*intptr1.local() == rank_me());
    assert(intptr1.use_count() == 1);
    intptr2 = intptr1;
    assert(intptr1);
    assert(intptr2);
    assert(intptr1.use_count() == 2);
    assert(intptr2.use_count() == 2);
    intptr3 = std::move(intptr2);
    assert(intptr1.use_count() == 2);
    assert(intptr3.use_count() == 2);
    assert(intptr1 == intptr3);
    assert(intptr2 != intptr3);
    assert(intptr1 != intptr2);
    assert(intptr1);
    assert(!intptr2);
    assert(intptr3);
    intptr3.reset();
    assert(!intptr3);
    assert(!intptr2);
    assert(intptr2 == intptr3);
    assert(intptr1 != intptr2);
    assert(intptr1 != intptr3);
    assert(intptr1);
    assert(intptr1.use_count() == 1);
    assert(intptr2.use_count() == 0);
    assert(intptr3.use_count() == 0);
    intptr1.reset();
    assert(!intptr1);
    assert(intptr1.use_count() == 0);
    {
      shared_global_ptr<int> intptra(upcxx::new_<int>());
      intptr1 = intptra;
      assert(intptra);
      assert(intptr1);
      assert(intptr1.use_count() == 2);
      assert(intptra.use_count() == 2);
      assert(intptra == intptr1);

      shared_global_ptr<int> intptrcpy(intptr1);
      assert(intptra.use_count() == 3);
      assert(intptrcpy == intptra);
      assert(intptrcpy == intptr1);
      assert(intptrcpy != intptr2);
    }
    assert(intptr1);
    assert(intptr1.use_count() == 1);
    intptr1.reset();
    assert(intptr1.use_count() == 0);
    assert(!intptr1);
  }
  upcxx::barrier();
  if (local_team().rank_n() > 1) {
    // test shared between local_team() ranks odd/even

    using GP = global_ptr<int>;
    using SGP = shared_global_ptr<int>;
    using DSGP = upcxx::dist_object<SGP>;
    DSGP dist_shptr(local_team());
    SGP intptr1, intptr2, intptr3;
    intptr1.reset(upcxx::new_<int>(rank_me()));
    upcxx::barrier();

    assert(intptr1);
    assert(!intptr2);
    assert(!intptr3);
    assert(!*dist_shptr);
    assert(intptr1.use_count() == 1);
    DBG("Init intptr1=", intptr1, ", intptr2=", intptr2, ", intptr3=", intptr3, "\n");
    DBG("Init dist_shptr=", *dist_shptr, "\n");
    upcxx::barrier();

    *dist_shptr = intptr1;
    assert(*dist_shptr);
    assert(intptr1.use_count() == 2);
    assert(dist_shptr->use_count() == 2);
    assert(intptr1 == *dist_shptr);
    assert(*(dist_shptr->local()) == rank_me());
    assert(intptr1.use_count() == 2);
    assert(*intptr1.local() == rank_me());
    assert(intptr1.get().where() == rank_me());

    upcxx::barrier();

    intrank_t other2 = (local_team().rank_me() + 1) % local_team().rank_n();
    intrank_t fromother = (local_team().rank_me() + local_team().rank_n() - 1) % local_team().rank_n();
    DBG("Sending to other2=", other2, " intptr1=", intptr1, ", gptr=", intptr1.get(), " val=", *intptr1.get().local(), "\n");

    using PromPtr = promise<global_ptr<int> >;
    dist_object<PromPtr> verify_value(local_team());
    rpc_ff(
        local_team()[other2],
        [](dist_object<PromPtr> &vv, global_ptr<int> gptr, intrank_t from_rank) {
          DBG("RPC for other2 gptr=", gptr, " from ", from_rank, ", val=", *gptr.local(), "\n");
          vv->fulfill_result(gptr);
        },
        verify_value, intptr1.get(), rank_me());
    auto ptr = verify_value->get_future().wait();
    DBG("Got gptr=", ptr, " from ", fromother, " val=", *ptr.local(), "\n");
    assert(ptr.where() == local_team()[fromother]);
    upcxx::barrier();

    intptr2 = rpc(
                  upcxx::local_team(), other2,
                  [](DSGP &dsgp, intrank_t from_rank) {
                    DBG("RPC for other2 dsgp=", *dsgp, " from ", from_rank, "\n");
                    return *dsgp;
                  },
                  dist_shptr, rank_me())
                  .wait();

    DBG("Got intptr2=", intptr2, " from ", other2, "\n");
    upcxx::barrier();
    int before = dist_shptr->use_count();
    int during = (*dist_shptr).use_count();
    int after = dist_shptr->use_count();
    DBG("before=", before, " during=", during, " after=", after, "\n");

    DBG("PostRPC intptr1=", intptr1, ", intptr2=", intptr2, ", intptr3=", intptr3, "\n");
    DBG("Init dist_shptr=", dist_shptr->use_count(), " intptr1=", intptr1, ", intptr2=", intptr2, ", intptr3=", intptr3, "\n");
    DBG("Init dist_shptr=", *dist_shptr, "\n");
    DBG("Init dist_shptr=", *dist_shptr, " intptr1=", intptr1, "\n");
    DBG("Init dist_shptr=", dist_shptr->use_count(), "\n");

    assert(intptr1.use_count() == 3);
    assert(intptr2.use_count() == 3);
    assert(intptr1 != intptr2);
    assert(intptr1.where() == rank_me());
    assert(intptr2.where() == local_team()[other2]);
    assert(dist_shptr->where() == rank_me());
    assert(*(intptr2.get().local()) == local_team()[other2]);
    DBG("barrier\n");
    upcxx::barrier();
    dist_shptr->reset();
    DBG("dist_shptr->reset()\n");
    upcxx::barrier();

    assert(intptr1.use_count() == 2);
    assert(intptr2.use_count() == 2);

    upcxx::barrier();
    *dist_shptr = intptr1;
    upcxx::barrier();
    assert(intptr1.use_count() == 3);
    assert(intptr2.use_count() == 3);

    upcxx::barrier();
    dist_shptr->reset();
    upcxx::barrier();
    assert(intptr1.use_count() == 2);
    assert(intptr2.use_count() == 2);
    upcxx::barrier();
    intptr1.reset();
    upcxx::barrier();
    assert(intptr1.use_count() == 0);
    assert(intptr2.use_count() == 1);
    upcxx::barrier();
    DBG("intptr1=", intptr1, " intptr2=", intptr2, "\n");
    upcxx::barrier();
    // intptr2.reset();
    DBG("intptr1=", intptr1, " intptr2=", intptr2, "\n");
    upcxx::barrier();

    /**
    if (local_team().rank_n() > 2) {
        intrank_t other3 = (local_team().rank_me() + local_team().rank_n() - 1) % local_team().rank_n();
        intptr3 = rpc(local_team(), other3, [](shared_global_ptr<int> shptr) {
            return shptr;
        }, intptr1).wait();
        upcxx::barrier();
        assert(intptr1.use_count() == 3);
        assert(intptr2.use_count() == 3);
        assert(intptr3.use_count() == 3);
        assert(intptr1 != intptr2);
        assert(intptr2 != intptr3);
        assert(intptr1 != intptr3);
        assert(intptr1.where() == rank_me());
        assert(*intptr1.local() == rank_me());
        assert(intptr2.where() == local_team()[other2]);
        assert(intptr3.where() == local_team()[other3]);

        upcxx::barrier();
        intptr3 = intptr2;
        upcxx::barrier();
        assert(intptr1.use_count() == 3);
        assert(intptr2.use_count() == 3);
        assert(intptr3.use_count() == 3);
        assert(intptr1 != intptr2);
        assert(intptr2 == intptr3);
        assert(intptr1 != intptr3);
        assert(intptr1.where() == rank_me());
        assert(*intptr1.local() == rank_me());
        assert(intptr2.where() == local_team()[other2]);
        assert(*intptr2.local() == local_team()[other2]);
        assert(intptr3.where() == local_team()[other2]);
        assert(*intptr3.local() == local_team()[other3]);
        upcxx::barrier();
    }
    upcxx::barrier();
    intptr3.reset();
    upcxx::barrier();

    assert(intptr1.use_count() == 2);
    assert(intptr2.use_count() == 2);
    assert(intptr1 != intptr2);
    assert(intptr1.where() == rank_me());
    assert(intptr2.where() == local_team()[other2]);
    assert(intptr3.use_count() == 0);
    assert(!intptr3);

    upcxx::barrier();
    intptr1.reset(); // the original
    upcxx::barrier();
    assert(!intptr1);
    assert(!intptr3);
    assert(intptr2);
    assert(intptr2.use_count() == 1);
    assert(intptr2.where() == local_team()[other2]);
     */
    upcxx::barrier();
  }

  upcxx_utils::close_dbg();
  return 0;
}
