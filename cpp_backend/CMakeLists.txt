cmake_minimum_required(VERSION 3.15)
project(mlscript)

set(CMAKE_CXX_STANDARD 17)
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0  # or latest stable
)
FetchContent_MakeAvailable(pybind11)


# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
#find_package(pybind11 REQUIRED)

# Create Python module
pybind11_add_module(mlscript bindings.cpp evaluator.cpp)
