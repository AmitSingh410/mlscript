cmake_minimum_required(VERSION 3.15)
project(mlscript LANGUAGES CXX)

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

target_compile_features(mlscript PRIVATE cxx_std_17)

# This is the most critical section. We are now explicitly giving our mlscript
# module the full, absolute paths to every header directory it needs.
# There is no more magic or automatic discovery for include paths.
target_include_directories(mlscript PRIVATE
    "${CMAKE_SOURCE_DIR}/cpp_backend"
    "${CMAKE_SOURCE_DIR}/third_party/eigen"
    ${NUMPY_INCLUDE_DIR}
)

# We link only to the libraries that need to be linked. Eigen is header-only,
# so we do not link to it; we only need its include directory (above).
target_link_libraries(mlscript PRIVATE
    pybind11::module
    OpenMP::OpenMP_CXX
)