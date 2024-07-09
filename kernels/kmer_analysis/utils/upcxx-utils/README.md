______________________________________________________________________________

    UPCXX Utils v 0.1, Copyright (c) 2019, The Regents of the University of California,
    through Lawrence Berkeley National Laboratory (subject to receipt of any
    required approvals from the U.S. Dept. of Energy).  All rights reserved.
 
    If you have questions about your rights to use or distribute this software,
    please contact Berkeley Lab's Innovation & Partnerships Office at  IPO@lbl.gov.
 
    NOTICE.  This Software was developed under funding from the U.S. Department
    of Energy and the U.S. Government consequently retains certain rights. As such,
    the U.S. Government has been granted for itself and others acting on its behalf
    a paid-up, nonexclusive, irrevocable, worldwide license in the Software to
    reproduce, distribute copies to the public, prepare derivative works, and
    perform publicly and display publicly, and to permit other to do so.

______________________________________________________________________________

# * UPCXX Utils *

UPCXX Utils is NOT a part of the [UPC++](https://upcxx.lbl.gov/) distribution, but is
designed enable reuse of core useful components within that framework.  UPCXX Utils is
used as a submodule of [HipMer / MetaHipMer](https://sites.google.com/lbl.gov/exabiome/downloads)
and other related projects.

This repository is maintained by Rob Egan (RSEgan@lbl.gov)




## Requirements

  * [UPC++](https://upcxx.lbl.gov/) >= 2019.9.0
     * Either within the PATH or with UPCXX_INSTALL set in the environment 
  * CMake >= 3.10 (suggested >=3.13.0)
  * (optional) [zstr](https://github.com/JGI-Bioinformatics/zstr) to support progress on compressed files

## Building

In your project's CMakeList.txt when this project is a submodule include:
```
add_subdirectory(/path/to/upcxx-utils)
include_directories(/path/to/upcxx_utils/src)
```
And then be sure to include ${UPCXX_UTILS_LIBRARIES} in the target_link_libraries() for any target that uses this library

*NOTE* If your version of CMake is < 3.13.0, then you must compile directly with upcxx instead of using the library interface provided by this package, such as:

```
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.13 AND DEFINED UPCXX_LIBRARIES)
  message(STATUS "UPCXX_UTILS is using the UPCXX::upcxx library interface: ${UPCXX_LIBRARIES}")
else()
  find_program(UPCXX_EXEC upcxx)
  set(CMAKE_CXX_COMPILER ${UPCXX_EXEC})
  message(STATUS "UPCXX_UTILS is using upcxx directly as the UPCXX::upcxx library interface is not available in this low version of cmake: ${CMAKE_VERSION}")
endif()

```

Then to build simply configure and make

```
mkdir build && cd build && cmake .. && make

```

## Contents

CMake is assumed to be the build utility and including this top level directory will build this
library and dependencies.

### log.hpp

```
//
// optionally open a log file per process
//

// The file will be in a subdirectory with the pattern: ./per_thread/########/########/<name><time-seconds>.dbg
void open_dbg(const string name);
void close_dbg();

// The following macros take an arbitrary comma separated list of objects that can be used with a file stream << operator

// S-prefix -- only affects output if called by rank 0
SOUT(objects for rank 0 to output to stdout and flush)
SWARN(objects for rank 0 to output as a WARNING)
SDIE(objects for rank 0 to output and then terminate the application)

DBG(objects to write to debug log (if opened))

// The following will write to stdout for any calling rank and may be excessive at scale -- use caution
OUT(objects to output)
WARN(objects to output as a warning)
DIE(objects to output and then terminate the application)
INFO(objects to output with timestamp)

```

### progress_bar.hpp

```
  ProgressBar(int64_t total, string prefix = "", int pwidth = 20,
              int width = 50, char complete = '=', char incomplete = ' ');

  ProgressBar(const string &fname, istream *infile, string prefix = "", int pwidth = 20, int width = 50,
              char complete = '=', char incomplete = ' ');
              
  future<> set_done();

  void done();

  bool update(int64_t new_ticks = -1);
```


### timers.hpp

```
// basic timer
Timer(const string &name);
void start();
void stop();

// outputs warnings after max time has been spent while in scope
StallTimer(const string _name, double _max_seconds = 60.0, int64_t _max_count = -1);
void check();

// Timer when many start and stops happen 
IntermittentTimer(const string &name);
void start();
void stop();

// Timer that calls upcxx::progress (and times the duration of that call)
ProgressTimer(const string &name);
void progress(size_t run_every = 1);
void discharge(size_t run_every = 1);
void print_out();

// Timer that implemets a barrier on entrance and/or exit
BarrierTimer(const string &name, bool entrance_barrier = true, bool exit_barrier = true);

// Class that holds counts and timings for Active instances
ActiveCountTimer(const string _name = "");

// Times the active instances of Base given an ActiveCountTimer
template<typename Base> ActiveInstantiationTimer(ActiveCountTimer &act);

// Times the active instances of Base with a singleton ActiveCountTimer based on templated typer
temmplate<typename Base, typename ... DistinguishingArgs> InstantiationTimer();

```

### aggr_store.hpp

Aggregates many small (fine-grained) messages into larger ones.  Examples are to optimally construct a distributed hash table.

```
template<typename FuncDistObj, typename T>
AggrStore(FuncDistObj &f, const string description);

// set the fixed-size for communication buffers
void set_size(size_t max_store_bytes);

// call for as many elements need updating
bool update(intrank_t target_rank, T &elem);

// blocks until all updates have been processed and quiessence has been achieved
void flush_updates();

```

FuncDistObj is required to be a template<Functor> upcxx::dist_obj with an operator method on a const reference of T
and T must be Trivially Serializable

```
using std::map<Key,Val> = Map;
Map LocalObject;
class FuncObj {
  Map &obj;
public:
  FuncObj(Map &_): obj(_) {}
  void operator()(const Map::value_type &elem) { /* do something */ }
};
using upcxx::dist_object<FuncObj> = FuncDistObj;
FuncDistObj func_dist_obj;
```
