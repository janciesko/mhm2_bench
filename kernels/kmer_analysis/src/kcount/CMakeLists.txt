FILE(GLOB SRCS *.cpp)

if(MHM2_BENCH_ENABLE_GPU)
  FILE(GLOB SRCS_GPU kcount-gpu/*.cpp)
  set(GPU_UTILS_LIB gpu_utils)
else()
  list(FILTER SRCS EXCLUDE REGEX ".*gpu\\.cpp$")
  message(STATUS ${SRCS})
endif()

add_definitions(-DMINIMIZER_LEN=15)

add_library(kcount_lib ${SRCS} ${SRCS_GPU})

target_include_directories(kcount_lib PRIVATE ${UPCXX_INCLUDE_DIRS} ${libupcxx-utils_header_found})
target_link_libraries(kcount_lib ${libupcxx-utils_lib_found})
