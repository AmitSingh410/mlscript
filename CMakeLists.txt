cmake_minimum_required(VERSION 3.15)
project(mlscript LANGUAGES CXX)

message(STATUS "--- Configuring mlscript dependencies ---")

find_package(OpenMP REQUIRED)
message(STATUS "Found OpenMP: ${OpenMP_CXX_FOUND}")

find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Found Python interpreter: ${Python_EXECUTABLE}")

include_directories("${CMAKE_SOURCE_DIR}/third_party/eigen")
message(STATUS "Set Eigen include path to: ${CMAKE_SOURCE_DIR}/third_party/eigen")

add_subdirectory(third_party/pybind11)
message(STATUS "Configured pybind11 submodule")

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NUMPY_FOUND_RESULT
)
if(NOT NUMPY_FOUND_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to find NumPy headers. Make sure NumPy is installed.")
else()
    message(STATUS "Found NumPy headers at: ${NUMPY_INCLUDE_DIR}")
endif()

message(STATUS "--- Configuring mlscript C++ module ---")

pybind11_add_module(mlscript SHARED
    cpp_backend/bindings.cpp
    cpp_backend/evaluator.cpp
)

target_compile_features(mlscript PRIVATE cxx_std_17)

target_include_directories(mlscript PRIVATE
    "${CMAKE_SOURCE_DIR}/cpp_backend"
    ${NUMPY_INCLUDE_DIR}
)

target_link_libraries(mlscript PRIVATE
    pybind11::module
    OpenMP::OpenMP_CXX
)

message(STATUS "--- Configuration complete ---")