cmake_minimum_required(VERSION 3.13)
project(KMeansClustering)

set(CMAKE_CXX_STANDARD 14)

add_executable(KMeansClustering main.cpp Point.h Node.cpp Node.h)

find_package(OpenMP REQUIRED)
if(NOT TARGET OpenMP::OpenMP_CXX)
    add_library(OpenMP_TARGET INTERFACE)
    add_library(OpenMP::OpenMP_CXX ALIAS OpenMP_TARGET)
    target_compile_options(OpenMP_TARGET INTERFACE ${OpenMP_CXX_FLAGS})
    find_package(Threads REQUIRED)
    target_link_libraries(OpenMP_TARGET INTERFACE Threads::Threads)
    target_link_libraries(OpenMP_TARGET INTERFACE ${OpenMP_CXX_FLAGS})
endif()

target_link_libraries(KMeansClustering PUBLIC OpenMP::OpenMP_CXX)

#find_package(MPI REQUIRED)

#include_directories(${MPI_INCLUDE_PATH})
#
#target_link_libraries(KMeansClustering ${MPI_LIBRARIES})
#
#if(MPI_COMPILE_FLAGS)
#    set_target_properties(hello PROPERTIES
#            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
#endif()
#
#if(MPI_LINK_FLAGS)
#    set_target_properties(hello PROPERTIES
#            LINK_FLAGS "${MPI_LINK_FLAGS}")
#endif()
