cmake_minimum_required(VERSION 3.13)
project(KMeansClustering)

set(CMAKE_CXX_STANDARD 14)

add_executable(KMeansClustering main.cpp Point.h Cluster.h KMeans.h KMeans.cpp)