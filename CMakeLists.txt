cmake_minimum_required(VERSION 3.15)
project(mlscript)

# --- Find Dependencies ---

find_package(OpenMP REQUIRED)

add_subdirectory(third_party/pybind11)

# For Eigen, we create an INTERFACE library. This is the standard way to handle
# a header-only library. It just adds the include directory to the compiler's
# search path without trying to "build" Eigen itself.
add_library(Eigen INTERFACE)
target_include_directories(Eigen SYSTEM INTERFACE third_party/eigen)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# --- Configure the Main Project ---

# This places the final compiled .pyd file in the project's root directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

pybind11_add_module(mlscript SHARED
    cpp_backend/bindings.cpp
    cpp_backend/evaluator.cpp
)

# Link the necessary libraries and include directories
target_include_directories(mlscript PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp_backend"
    ${Python_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIR}
)

target_link_libraries(mlscript PRIVATE
    Eigen
    pybind11::embed
    OpenMP::OpenMP_CXX
)

# Set the C++ standard for consistency
target_compile_features(mlscript PRIVATE cxx_std_17)