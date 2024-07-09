# The name of our project is "UPCXX_UTILS". CMakeLists files in this project can
# refer to the root source directory of the project as ${UPCXX_UTILS_SOURCE_DIR} and
# to the root binary directory of the project as ${UPCXX_UTILS_BINARY_DIR}.
cmake_minimum_required (VERSION 3.10 FATAL_ERROR)

#----------------------------------------------------------------------------------------#
#   project specification
#----------------------------------------------------------------------------------------#
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/VERSION UPCXX_UTILS_BASE_VERSION)
project(UPCXX_UTILS VERSION ${UPCXX_UTILS_BASE_VERSION} DESCRIPTION "Utilties that can be used within the framework of UPC++" LANGUAGES C CXX)

# install exported targets
export(PACKAGE UPCXX_UTILS)
install(EXPORT UPCXX_UTILS DESTINATION "cmake")
#install(FILES cmake/FindUPCXX_UTILS.cmake cmake/UPCXX_UTILSConfig.cmake DESTINATION "cmake")
install(FILES cmake/UPCXX_UTILSConfig.cmake DESTINATION "cmake")

# use PROJECT_SOURCE_DIR, not CMAKE_SOURCE_DIR
set(UPCXX_UTILS_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(UPCXX_UTILS_BINARY_DIR ${PROJECT_BINARY_DIR})

# include modules for this package
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# check compiler and set appropriate linker flags
include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)
include(CheckCXXSourceRuns)
include(GetGitVersion)

# enable testing
option(UPCXX_ENABLE_TESTING "Enable testing of upcxx-utils" ON)
if(UPCXX_ENABLE_TESTING)
  enable_testing()
endif()

# all c++ must be at least c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT APPLE)
    add_definitions(-D_XOPEN_SOURCE=700)
endif()

# prefer static to dynamic libraries when including third party libs (like ZLIB)
if (APPLE)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a" ".dylib" ".so")
else()
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a" ".so")
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo RelWithDebug MinSizeRel."
	    FORCE)
endif()
message(STATUS "Building for ${CMAKE_BUILD_TYPE}")
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    add_definitions(-DDEBUG)
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
  set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_C_FLAGS_RELWITHDEBINFO}")
  set(ENV{UPCXX_CODEMODE} "debug")  # build with all upcxx assertions turned on
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebug")
  # suggested by https://stackoverflow.com/questions/48754619/what-are-cmake-build-type-debug-release-relwithdebinfo-and-minsizerel
  # RelWithDebug [`-O2 -g -DDEBUG`] - aka RelWithDebInfo but `-DNDEBUG` removed and `-DDEBUG` added.
  # Additionally DBG logs are disabled BUT dbg files might be made for all ranks to capture all higher level log messages
  string(REGEX REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBUG "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DDEBUG" )
  string(REGEX REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELWITHDEBUG "${CMAKE_C_FLAGS_RELWITHDEBINFO} -DDEBUG" ) 
  add_definitions(-DNO_DBG_LOGS)
  set(ENV{UPCXX_CODEMODE} "debug")  # build with all upcxx assertions turned on
  set(CMAKE_CXX_FLAGS_RELWITHDEBUG "${CMAKE_CXX_FLAGS_RELWITHDEBUG}" CACHE STRING "CXX flags in RelWithDebug build mode" FORCE)
  set(CMAKE_C_FLAGS_RELWITHDEBUG "${CMAKE_C_FLAGS_RELWITHDEBUG}" CACHE STRING "C flags in RelWithDebug build mode" FORCE)
endif()

# set version
include_directories(include)
GET_GIT_VERSION(UPCXX_UTILS)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(BUILD_STATIC_LIBS "Build static libraries" ON)
option(CMAKE_VERBOSE_MAKEFILE "Verbose makefile" OFF)
option(UPCXX_UTILS_NO_THREADS "Disable thread libs" OFF)
option(UPCXX_UTILS_IO_NO_THREAD "Disable thread in dist_ofstream" OFF)

message(STATUS "Testing the environment..")
get_directory_property(hasParent PARENT_DIRECTORY)

# find upcxx, and bootstrap a recent copy of FindUPCXX.cmake if necessary
find_package(UPCXX 2020.3 REQUIRED)
message(STATUS "Using UPCXX version - ${UPCXX_VERSION} ${UPCXX_VERSION_STRING}")

if (UPCXX_UTILS_NO_THREADS)
  message(STATUS "EXCLUDING support for Threads")
  add_definitions(-DUPCXX_UTILS_NO_THREADS)
  add_definitions(-DUPCXX_UTILS_NO_THREAD_POOL)
  set(UPCXX_UTILS_IO_NO_THREAD OFF)
  message(STATUS "UPCXX_UTILS *disabling* thread libraries")
else()
  find_package(Threads REQUIRED)
endif()

if (UPCXX_UTILS_IO_NO_THREAD)
  add_definitions(-DUPCXX_UTILS_IO_NO_THREAD)
endif()

# necessary to avoid device link objects that may cause cuda conflicts with upcxx library dependencies
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

include_directories("${PROJECT_SOURCE_DIR}/include")
add_subdirectory("src")
if (UPCXX_ENABLE_TESTING)
  add_subdirectory("test")
endif()

# The interface library
add_library(UPCXX_UTILS_INTERFACE_LIB INTERFACE)
set_target_properties( UPCXX_UTILS_INTERFACE_LIB PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${UPCXX_UTILS_SOURCE_DIR}/include>;$<INSTALL_INTERFACE:include>"
    INTERFACE_LINK_LIBRARIES      UPCXX_UTILS_LIBRARY
    )

# include a version file
include(CMakePackageConfigHelpers)
message(STATUS "Installing package version ${UPCXX_UTILS_VERSION}")
#set(export_version_file "${CMAKE_CURRENT_BINARY_DIR}/makeVersionFile/UPCXX_UTILSVersion.cmake")
set(export_version_file "${CMAKE_CURRENT_BINARY_DIR}/makeVersionFile/UPCXX_UTILSConfigVersion.cmake")
write_basic_package_version_file(
  ${export_version_file}
  VERSION ${UPCXX_UTILS_VERSION}
  COMPATIBILITY AnyNewerVersion
)
install(FILES ${export_version_file} ${UPCXX_UTILS_VERSION_FILE} DESTINATION "cmake")

install(TARGETS UPCXX_UTILS_INTERFACE_LIB EXPORT UPCXX_UTILS)
if (hasParent)
  set(UPCXX_UTILS_LIBRARIES UPCXX_UTILS_INTERFACE_LIB PARENT_SCOPE)
endif()

# install the header files
FILE(GLOB UPCXX_UTILS_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/upcxx_utils/*.h*)
install(FILES include/upcxx_utils.hpp DESTINATION "include")
install(FILES ${UPCXX_UTILS_HEADERS} DESTINATION "include/upcxx_utils")
FILE(GLOB UPCXX_UTILS_MEM_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/upcxx_utils/memory-allocators/*.h*)
install(FILES ${UPCXX_UTILS_MEM_HEADERS} DESTINATION "include/upcxx_utils/memory-allocators")
install(FILES LICENSE.txt LEGAL.txt README.md DESTINATION ".") 

add_custom_target(clean-all
   COMMAND echo rm -rf ${CMAKE_INSTALL_PREFIX}/cmake ${CMAKE_INSTALL_PREFIX}/include ${CMAKE_INSTALL_PREFIX}/share ${CMAKE_INSTALL_PREFIX}/bin ${CMAKE_INSTALL_PREFIX}/lib 
   COMMAND ${CMAKE_BUILD_TOOL} clean
)

message(STATUS "UPCXX_UTILS-utils build type: ${CMAKE_BUILD_TYPE}")
