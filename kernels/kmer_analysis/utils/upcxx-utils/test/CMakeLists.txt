if(NOT UPCXX_FOUND)
  find_package(UPCXX)
endif()

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.13 AND DEFINED UPCXX_LIBRARIES)
  message(STATUS "UPCXX_UTILS is using the UPCXX::upcxx library interface to build tests: ${UPCXX_LIBRARIES}")
else()
  find_program(UPCXX_EXEC upcxx)
  set(CMAKE_CXX_COMPILER ${UPCXX_EXEC})
  message(STATUS "UPCXX_UTILS is using upcxx directly as the UPCXX::upcxx library interface is not available in this low version of cmake: ${CMAKE_VERSION}")
endif()

find_package(Threads REQUIRED)

SET(TEST_PROGS test_version test_fixed_size_cache test_upcxx_utils test_log test_gasnetvars test_timers test_progress_bar test_split_rank test_gather test_reduce_prefix test_binary_search
               test_ofstream test_flat_aggr_store test_three_tier_aggr_store test_shared_global_ptr test_allocators test_limit_outstanding test_promise_collectives
               test_thread_pool test_combined)


set(AND_THREADS)
if (THREADS_FOUND AND NOT UPCXX_UTILS_NO_THREADS)
  set(AND_THREADS Threads::Threads)
endif()

include(ProcessorCount)
ProcessorCount(N)
if(N EQUAL 0)
  set(N 2)
else()
  math(EXPR N "${N} / 2")
  if(${N} GREATER 4 AND NOT ("${CMAKE_BUILD_TYPE}" STREQUAL "Release"))
    set(N 4)
  endif()
endif()
message(STATUS "Running tests with ${N} ranks")

ADD_EXECUTABLE(hello_world hello_world.cpp)
TARGET_LINK_LIBRARIES(hello_world PUBLIC UPCXX_UTILS_LIBRARY)
INSTALL(TARGETS hello_world DESTINATION ${CMAKE_INSTALL_PREFIX}/test/bin/ )

SET(objs)
FOREACH(TEST ${TEST_PROGS})
  ADD_LIBRARY(${TEST}_obj OBJECT ${TEST}.cpp)
  IF(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.13)
     TARGET_LINK_LIBRARIES(${TEST}_obj PUBLIC ${UPCXX_LIBRARIES} UPCXX_UTILS_LIBRARY ${AND_THREADS})
  ENDIF()
  LIST(APPEND objs $<TARGET_OBJECTS:${TEST}_obj>)
ENDFOREACH()

FOREACH(TEST ${TEST_PROGS})
  IF(${TEST} STREQUAL "test_combined")
    ADD_EXECUTABLE(${TEST} main_shell.cpp ${objs})
  ELSE()
    ADD_EXECUTABLE(${TEST} main_shell.cpp $<TARGET_OBJECTS:${TEST}_obj>)
  ENDIF()
  TARGET_LINK_LIBRARIES(${TEST} PUBLIC UPCXX_UTILS_LIBRARY ${AND_THREADS})
  TARGET_COMPILE_DEFINITIONS(${TEST} PUBLIC MAIN=${TEST})
  ADD_TEST(${TEST} upcxx-run -n ${N} ${CMAKE_CURRENT_BINARY_DIR}/${TEST})
  set_tests_properties(${TEST} PROPERTIES ENVIRONMENT GASNET_BACKTRACE=1)
  install(TARGETS ${TEST} DESTINATION ${CMAKE_INSTALL_PREFIX}/test/bin/ )
ENDFOREACH()


install(PROGRAMS spawn_tests.sh DESTINATION ${CMAKE_INSTALL_PREFIX}/test/bin/ )
