cmake_minimum_required(VERSION 3.0)

project(bit-fitting CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(bit-fitting INTERFACE)
target_include_directories(bit-fitting INTERFACE include)
add_subdirectory(libbo EXCLUDE_FROM_ALL)
add_subdirectory(SimpleDataStructure EXCLUDE_FROM_ALL)
target_link_libraries(bit-fitting INTERFACE libbo sim_ds)

target_compile_options(bit-fitting INTERFACE -march=native)

add_executable(bench bench.cpp)
target_link_libraries(bench bit-fitting)
