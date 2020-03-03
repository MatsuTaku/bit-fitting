cmake_minimum_required(VERSION 3.0)

project(bit-fitting CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(bit-fitting INTERFACE)
target_include_directories(bit-fitting INTERFACE include)
add_subdirectory(libbo EXCLUDE_FROM_ALL)
add_subdirectory(SimpleDataStructure EXCLUDE_FROM_ALL)
find_package(Boost REQUIRED)
target_include_directories(bit-fitting INTERFACE Boost_INCLUDE_DIR)
target_link_directories(bit-fitting INTERFACE /usr/local/lib) # for fftw3
target_link_libraries(bit-fitting INTERFACE libbo sim_ds fftw3)

add_executable(bench bench.cpp)
target_link_libraries(bench bit-fitting)

add_executable(bench-simd bench.cpp)
target_link_libraries(bench-simd bit-fitting)
target_compile_options(bench-simd PRIVATE -march=native)
