FILE(GLOB SRCS *.cpp)
add_executable(kmer_analysis ${SRCS})

set(SUBDIRECTORIES
  gpu-utils
  kcount)

foreach(SUBDIR ${SUBDIRECTORIES})
  add_subdirectory(${SUBDIR})
endforeach()


target_include_directories(kmer_analysis PRIVATE ${Kokkos_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${SUBDIRECTORIES})
target_link_libraries(kmer_analysis PRIVATE $<$<BOOL:${ENABLE_MPI}>:MPI::MPI_CXX> Kokkos::kokkos)

install(TARGETS kmer_analysis DESTINATION ${CMAKE_INSTALL_BINDIR})