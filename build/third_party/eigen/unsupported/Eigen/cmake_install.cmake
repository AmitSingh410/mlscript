# Install script for directory: C:/Users/amits/Documents/Projects/mlscript/third_party/eigen/unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/mlscript")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "C:/Users/amits/Documents/Projects/mlscript/third_party/eigen/unsupported/Eigen/AdolcForward"
    "C:/Users/amits/Documents/Projects/mlscript/third_party/eigen/unsupported/Eigen/AlignedVector3"
    "C:/Users/amits/Documents/Projects/mlscript/third_party/eigen/unsupported/Eigen/ArpackSupport"
    "C:/Users/amits/Documents/Projects/mlscript/third_party/eigen/unsupported/Eigen/AutoDiff"
    "C:/Users/amits/Documents/Projects/mlscript/third_party/eigen/unsupported/Eigen/BVH"
    "C:/Users/amits/Documents/Projects/mlscript/third_party/eigen/unsupported/Eigen/EulerAngles"
    "C:/Users/amits/Documents/Projects/mlscript/third_party/eigen/unsupported/Eigen/FFT"
    "C:/Users/amits/Documents/Projects/mlscript/third_party/eigen/unsupported/Eigen/IterativeSolvers"
    "C:/Users/amits/Documents/Projects/mlscript/third_party/eigen/unsupported/Eigen/KroneckerProduct"
    "C:/Users/amits/Documents/Projects/mlscript/third_party/eigen/unsupported/Eigen/LevenbergMarquardt"
    "C:/Users/amits/Documents/Projects/mlscript/third_party/eigen/unsupported/Eigen/MatrixFunctions"
    "C:/Users/amits/Documents/Projects/mlscript/third_party/eigen/unsupported/Eigen/MoreVectorization"
    "C:/Users/amits/Documents/Projects/mlscript/third_party/eigen/unsupported/Eigen/MPRealSupport"
    "C:/Users/amits/Documents/Projects/mlscript/third_party/eigen/unsupported/Eigen/NonLinearOptimization"
    "C:/Users/amits/Documents/Projects/mlscript/third_party/eigen/unsupported/Eigen/NumericalDiff"
    "C:/Users/amits/Documents/Projects/mlscript/third_party/eigen/unsupported/Eigen/OpenGLSupport"
    "C:/Users/amits/Documents/Projects/mlscript/third_party/eigen/unsupported/Eigen/Polynomials"
    "C:/Users/amits/Documents/Projects/mlscript/third_party/eigen/unsupported/Eigen/Skyline"
    "C:/Users/amits/Documents/Projects/mlscript/third_party/eigen/unsupported/Eigen/SparseExtra"
    "C:/Users/amits/Documents/Projects/mlscript/third_party/eigen/unsupported/Eigen/SpecialFunctions"
    "C:/Users/amits/Documents/Projects/mlscript/third_party/eigen/unsupported/Eigen/Splines"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "C:/Users/amits/Documents/Projects/mlscript/third_party/eigen/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/Users/amits/Documents/Projects/mlscript/build/third_party/eigen/unsupported/Eigen/CXX11/cmake_install.cmake")

endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/amits/Documents/Projects/mlscript/build/third_party/eigen/unsupported/Eigen/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
