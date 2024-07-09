cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(ExaMiniMD LANGUAGES CXX)

include(GNUInstallDirs)

find_package(Kokkos REQUIRED)
find_package(upcxx REQUIRED)
find_package(mpi REQUIRED)

add_subdirectory(src)
