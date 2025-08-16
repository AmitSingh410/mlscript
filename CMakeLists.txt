cmake_minimum_required(VERSION 3.15)
project(mlscript LANGUAGES CXX)

# This is the most critical fix. By setting the CXX_STANDARD here at the top
# level, we guarantee that our project AND all sub-projects (like pybind11)
# are configured using the same C++17 standard from the very beginning.
# This prevents the subtle configuration mismatch that was causing the build failure.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find Dependencies ---
find_package(OpenMP REQUIRED)
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(third_party/pybind11)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NUMPY_FOUND_RESULT
)
if(NOT NUMPY_FOUND_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to find NumPy headers.")
endif()

# --- Build Module ---

pybind11_add_module(mlscript SHARED
    cpp_backend/bindings.cpp
    cpp_backend/evaluator.cpp
)

target_include_directories(mlscript PRIVATE
    "${CMAKE_SOURCE_DIR}/cpp_backend"
    "${CMAKE_SOURCE_DIR}/third_party/eigen"
    ${NUMPY_INCLUDE_DIR}
)

target_link_libraries(mlscript PRIVATE
    pybind11::module
    OpenMP::OpenMP_CXX
)