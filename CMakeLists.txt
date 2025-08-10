cmake_minimum_required(VERSION 3.15)
project(mlscript)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party")

add_subdirectory(third_party/pybind11)
add_subdirectory(third_party/eigen)

# This sets the output path for any configuration that isn't explicitly set.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# This explicitly sets the output path for the "Release" configuration,
# which is what `cmake --build . --config Release` uses.
foreach(config ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${config} CONFIG_UPPER)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_SOURCE_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_SOURCE_DIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_SOURCE_DIR})
endforeach()

# Diagnostic message to confirm the path
message(STATUS "Build artifacts will be placed in: ${CMAKE_SOURCE_DIR}")

#include_directories(include)

pybind11_add_module(mlscript
    cpp_backend/bindings.cpp
    cpp_backend/evaluator.cpp
)

target_include_directories(mlscript PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/cpp_backend")

target_link_libraries(mlscript PRIVATE Eigen3::Eigen)