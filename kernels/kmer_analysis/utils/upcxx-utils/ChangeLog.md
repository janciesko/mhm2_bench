# ChangeLog.md for UPCXX-UTILS

This is the ChangeLog for UPCXX-UTILS with development at [bitbucket](https://bitbucket.org/berkeleylab/upcxx-utils)

### 0.4.0 2022-06-23
   * Refactored PromiseReduce and ProgressBar frameworks to avoid syncpoints
   * Fixed Issues #8, #12, #15, #16
   * Added support for continers in Aggregating Stores (with caveat the 3-tier mode is disabled)
   * Better support for ThreadPool and building without threads
   * Fixed outputs in timers
   * New Gather function
   * dist_ofstream exposes the ofstream API
   * better support of headers in other libraries (upcxx::future, etc)
   * Improved assertions enforcing proper restricted context usage
   * Improved memory monitoring and enforce periodic log flushes
   * Included Attentiveness measurments in the logs

### 0.3.5 2020-09-19
   * Fixed potential problems in starting collectives within the restricted context of upxxx::progress() callbacks
   * Improved efficienty of 3Tier aggr store
   * Fixed problems and efficienty of dist_ofstream - Issue#4 
   * Reformatted code
   * Improved compile time with extern templates
   * Support CMAKE_BUILD_TYPE=RelWithDebInfo that runs faster *and* includes upcxx runtime assertions
   * Build on MacOSX and updated contrib/install-upcxx.sh for 2020.3.2
   * Improved testing and CI deployment
   * Various other bugfixes

### 0.3.4 2020-09-02
   * added three_tier_aggr_store
      * Shows significant scaling improvements starting between 32 and 64 nodes
      * Shows same scaling between 1 and 32 nodes, using flat_aggr_store
      * Improved resiliance to load imbalanced update data
      * includes rate limited rpcs in flush_update functions (flat_aggr_store too)
      * shared_global_ptr has same interface as shared_ptr but works on global_ptrs within the local_team
   * added limit_oustanding_futures helper functions
   * fixed bug in ofstream with races to open
   * ofstream uses private not global memory for block write optimization
   * Timers now accept a team
   * Improved test suite

### 0.3.3 2020-08-20
   * ofstream fixes - API change moved team in overloaded constructor
      * fixed stall in atomic domain construction
      * fixed bug in blocked write optimizations
      * fixed race in closing

### 0.3.2 2020-07-09
   * added dist_ofstream class that supports optimal I/O to a single file across a team
      * in unordered batches by rank with 'future<> flush_async()' and 'dist_ofstream& flush()'
      * as rank-ordered batches collectively with 'future<> flush_collective()'
   * added prefix_reduce function with O(log(n)) latency
   * added binary_search function over a dist_object
   * minor changes to split_rank class for Debug build support
   * added split_team class for easy inter-node and local_team communication

### 0.3.1 2020-07-03
   * added make test and additional CI testing on hulk and cori
   * added LOG_TRY_CATCH macro and better logging and flush upon crashes
   * minor race fix to TwoTierAggrStore
   * various build fixes using the CMake Libary Interface better
   * fixed logging to use const reference, not copies of objects, and other fixes with regards to duplicate entries and calls
   * added abilities for memory profiling over time in logs

### 0.3.0 2020-04-06
   * refactored all functions into upcxx_utils namespace
   * added test
   * renamed aggr_store to two_tier_aggr_store, imported flat_aggr_store from mhmxx
   * added version calculations based on git state in CMake function
   * imported and renamed many mhmxx functions from utils.hpp progress_bar and log

### 0.2.2 2020-03-26
   * some CMake build changes
   * initial refactoring of code from hipmer, and mhmxx
