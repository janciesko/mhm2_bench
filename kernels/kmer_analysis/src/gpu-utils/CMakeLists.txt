FILE(GLOB SRCS *.cpp)

add_library(gpu_utils ${SRCS})

target_link_libraries(gpu_utils ${UPCXX_LIBRARIES}
                      ${UPCXX_UTILS_LIBRARIES})
