//SNIPPET
#include <iostream>
#include "dmap.hpp"

using namespace std;

int main(int argc, char *argv[])
{
  upcxx::init();
  const long N = 1000;
  DistrMap dmap;
  // insert set of unique key, value pairs into hash map, wait for completion
  for (long i = 0; i < N; i++) {
    string key = to_string(upcxx::rank_me()) + ":" + to_string(i);
    string val = key;
    dmap.insert(key, val).wait();
  }
  // barrier to ensure all insertions have completed
  upcxx::barrier();

// EXERCISE 1:  Try full DHT implementation, as in https://upcxx.lbl.gov/docs/html/guide.html#implementing-a-distributed-hash-table

/* EXERCISE 2:  Try to implement a simple, small-scale version of a DHT for sequence data, following the pattern in mhm2 - Kmer Analysis 
   - Explain sequence input data
   - Describe recoding of sequence data into integer (?) representation, and why this step is necessary?
   - Describe early events, and reasons for them, in the construction of the DHT with sequence data
   - Are DHT used on the GPU, or some other data structure? Hint:  see `mhm2_proxy/src/kcount`

*/
  // now try to fetch keys inserted by neighbor
  for (long i = 0; i < N; i++) {
    string key = to_string((upcxx::rank_me() + 1) % upcxx::rank_n()) + ":" + to_string(i);
    string val = dmap.find(key).wait();
    // check that value is correct
    UPCXX_ASSERT(val == key);
  }
  upcxx::barrier(); // wait for finds to complete globally
  if (!upcxx::rank_me()) cout << "SUCCESS" << endl;
  upcxx::finalize();
  return 0;
}
//SNIPPET


