cmake_minimum_required(VERSION 3.0)

project(bit-fitting CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(bit-fitting INTERFACE)
target_include_directories(bit-fitting INTERFACE include)
add_subdirectory(libbo EXCLUDE_FROM_ALL)
target_link_libraries(bit-fitting INTERFACE libbo)

add_executable(test test.cpp)
target_link_libraries(test bit-fitting)
