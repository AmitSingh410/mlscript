cmake_minimum_required(VERSION 3.15)
project(mlscript LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- Reject stray top-level compiled artifacts ----------
file(GLOB TOPLEVEL_MLSCRIPT "${CMAKE_SOURCE_DIR}/mlscript.*")
if(TOPLEVEL_MLSCRIPT)
    message(FATAL_ERROR "Remove top-level mlscript.* from source to avoid packaging it into the wheel: ${TOPLEVEL_MLSCRIPT}")
endif()
# --------------------------------------------------------------

# --- Find Dependencies ---
find_package(OpenMP REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
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
# Create a pybind11 target. Use a package-style name for clarity, but
# ensure output filename is 'mlscript' and that build outputs land in build/interpreter/
pybind11_add_module(interpreter.mlscript SHARED
    cpp_backend/bindings.cpp
    cpp_backend/evaluator.cpp
)

# Ensure the compiled filename is "mlscript" (so final file is mlscript*.pyd)
set_target_properties(interpreter.mlscript PROPERTIES
    PREFIX ""               # avoid platform prefix like 'lib'
    OUTPUT_NAME "mlscript"  # ensures filename: mlscript(.pyd/.so)
)

# Force runtime/library/archive output directories into build/interpreter
# so scikit-build will pick them up as package data under interpreter/
set_target_properties(interpreter.mlscript PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/interpreter
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/interpreter
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/interpreter
)

# Also set per-configuration output dirs (Visual Studio variants)
foreach(_cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
  set_target_properties(interpreter.mlscript PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_${_cfg} ${CMAKE_BINARY_DIR}/interpreter
    LIBRARY_OUTPUT_DIRECTORY_${_cfg} ${CMAKE_BINARY_DIR}/interpreter
    ARCHIVE_OUTPUT_DIRECTORY_${_cfg} ${CMAKE_BINARY_DIR}/interpreter
  )
endforeach()

target_include_directories(interpreter.mlscript PRIVATE
    "${CMAKE_SOURCE_DIR}/cpp_backend"
    "${CMAKE_SOURCE_DIR}/third_party/eigen"
    ${NUMPY_INCLUDE_DIR}
)

target_link_libraries(interpreter.mlscript PRIVATE
    pybind11::module
    OpenMP::OpenMP_CXX
)

# Install into 'interpreter' directory in CMake install tree so scikit-build
# places the extension inside the interpreter package in the wheel.
# On Windows, compiled extension ends up as RUNTIME; on Unix as LIBRARY.
install(TARGETS interpreter.mlscript
    RUNTIME DESTINATION interpreter
    LIBRARY DESTINATION interpreter
    ARCHIVE DESTINATION interpreter
)
