cmake_minimum_required(VERSION 3.22.1)

project(hpc-quiz VERSION 0.1 LANGUAGES CXX)

include(cmake/util.cmake)
include(GNUInstallDirs)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

# default to release
set(CMAKE_BUILD_TYPE Release)

if (BUILD_LEVEL STREQUAL "Debug")
    set(CMAKE_BUILD_TYPE Debug)
    add_definitions(-DDEBUG)
endif()

green_message("Build Level: ${CMAKE_BUILD_TYPE}")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

add_subdirectory(answer)
