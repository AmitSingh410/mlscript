cmake_minimum_required(VERSION 3.15)
project(mlscript LANGUAGES CXX)

find_package(OpenMP REQUIRED)
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)

add_library(Eigen INTERFACE)
target_include_directories(Eigen INTERFACE
    "${CMAKE_SOURCE_DIR}/third_party/eigen"
)
add_subdirectory(third_party/pybind11)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NUMPY_FOUND_RESULT
)

if(NOT NUMPY_FOUND_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to find NumPy headers. Make sure NumPy is installed in the build environment.")
endif()

pybind11_add_module(mlscript SHARED
    cpp_backend/bindings.cpp
    cpp_backend/evaluator.cpp
)

target_compile_features(mlscript PRIVATE cxx_std_17)

target_link_libraries(mlscript PRIVATE
    Eigen
    pybind11::module
    OpenMP::OpenMP_CXX
)

target_include_directories(mlscript PRIVATE
    "${CMAKE_SOURCE_DIR}/cpp_backend"
    ${NUMPY_INCLUDE_DIR}
)