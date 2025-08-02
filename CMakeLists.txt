cmake_minimum_required(VERSION 3.14)
project(mlscript)

set(CMAKE_CXX_STANDARD 17)

# Find pybind11
find_package(pybind11 REQUIRED)

# Include directories
include_directories(include)

# Python module target
pybind11_add_module(mlscript
    pybind/bindings.cpp
    src/evaluator.cpp
)

target_include_directories(mlscript PRIVATE include)
