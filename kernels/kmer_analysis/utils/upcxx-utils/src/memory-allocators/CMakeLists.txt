message(STATUS "Building the Memory Allocators")

set(thr "")
if (UPCXX_UTILS_NO_THREADS)
  message(STATUS "EXCLUDING support for Threads")
else()
  find_package(Threads REQUIRED)
  set(thr Threads::Threads)
endif()

add_library(UPCXX_UTILS_MEMORY_ALLOCATORS OBJECT Allocator.cpp PoolAllocator.cpp UPCXXAllocator.cpp Utils.cpp)
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.13 AND DEFINED UPCXX_LIBRARIES)
    cmake_policy(SET CMP0079 NEW)
    target_link_libraries(UPCXX_UTILS_MEMORY_ALLOCATORS PUBLIC ${UPCXX_LIBRARIES} ${thr})
endif()

