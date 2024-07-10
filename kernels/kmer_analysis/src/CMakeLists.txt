FILE(GLOB SRCS *.cpp)

if(MHM2_BENCH_ENABLE_GPU)
set(SUBDIRECTORIES
  gpu-utils
  kcount)
else()
  set(SUBDIRECTORIES kcount)
endif()

if(MHM2_BENCH_ENABLE_GPU)
  FILE(GLOB SRCS_GPU kcount-gpu/*.cpp)
  set(GPU_UTILS_LIB gpu_utils)
else()
  list(FILTER SRCS EXCLUDE REGEX ".*gpu\\.cpp$")
  message(STATUS ${SRCS})
endif()

add_executable(kmer_analysis ${SRCS})

foreach(SUBDIR ${SUBDIRECTORIES})
  add_subdirectory(${SUBDIR})
endforeach()

# Build options
if(NOT MAX_BUILD_KMER)
set(MAX_BUILD_KMER 128 CACHE STRING "The maximum kmer allowed to use (must be multiple of 32)")
endif()
math(EXPR TEST_MAX_KMER_SIZE "((${MAX_BUILD_KMER} + 31) / 32) * 32")
if(NOT ${MAX_BUILD_KMER} EQUAL ${TEST_MAX_KMER_SIZE})
  message(
  FATAL_ERROR
  "MAX_BUILD_KMER must be a multiple of 32.  MAX_BUILD_KMER=${MAX_BUILD_KMER} is not"
  )
endif()
add_definitions(-DMAX_BUILD_KMER=${MAX_BUILD_KMER})

add_definitions(-DMINIMIZER_LEN=15)

target_include_directories(kmer_analysis PRIVATE ${libupcxx-utils_header_found} ${UPCXX_INCLUDE_DIRS} ./kcount/ )
target_link_libraries(kmer_analysis PRIVATE UPCXX::upcxx Kokkos::kokkos ${kcount_lib} ${libupcxx-utils_lib_found} "-L/home/jciesko/software/ompi/ompi/install/lib -lmpi" )

install(TARGETS kmer_analysis DESTINATION ${CMAKE_INSTALL_BINDIR})