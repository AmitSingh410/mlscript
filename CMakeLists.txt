cmake_minimum_required(VERSION 3.15)
project(mlscript)

# Add your local third_party directory to the module path
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party")

# OpenMP for multi-core processing
find_package(OpenMP REQUIRED)

# Add the subdirectories for Eigen and pybind11
# This builds them directly from your source files
add_subdirectory(third_party/eigen)
add_subdirectory(third_party/pybind11)

# Modern way to find Python and NumPy
find_package(Python COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set the output directory for build artifacts
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

foreach(config ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${config} CONFIG_UPPER)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_SOURCE_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_SOURCE_DIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_SOURCE_DIR})
endforeach()

message(STATUS "Build artifacts will be placed in: ${CMAKE_SOURCE_DIR}")

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

# Link against the targets provided by add_subdirectory
target_link_libraries(mlscript PRIVATE 
    Eigen3::Eigen 
    pybind11::embed
    OpenMP::OpenMP_CXX
)

# Set the C++ standard as a target property for consistency
target_compile_features(mlscript PRIVATE cxx_std_17)