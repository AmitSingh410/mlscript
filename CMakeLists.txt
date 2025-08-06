cmake_minimum_required(VERSION 3.15)
project(mlscript)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)

# This sets the output path for any configuration that isn't explicitly set.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# --- THE CRITICAL FIX FOR WINDOWS/MSVC ---
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

include_directories(include)

pybind11_add_module(mlscript
    pybind/bindings.cpp
    src/evaluator.cpp
)

target_include_directories(mlscript PRIVATE include)