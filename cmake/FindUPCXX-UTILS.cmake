
find_library(libupcxx-utils_lib_found libUPCXX_UTILS_LIBRARY.a PATHS ${UPCXX-UTILS_ROOT} SUFFIXES lib lib64 NO_DEFAULT_PATHS)
find_path(libupcxx-utils_header_found upcxx_utils.hpp PATHS ${UPCXX-UTILS_ROOT}/include NO_DEFAULT_PATHS)

message(STATUS ${libupcxx-utils_lib_found})
message(STATUS ${libupcxx-utils_header_found})

if (libupcxx-utils_lib_found AND libupcxx-utils_header_found)
  add_library(libUPCXX_UTILS INTERFACE)
  target_link_libraries(libUPCXX_UTILS INTERFACE libupcxx-utils_lib_found)
  target_include_directories(libUPCXX_UTILS INTERFACE ${libupcxx-utils_header_found})
endif()