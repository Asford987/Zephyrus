cmake_minimum_required(VERSION 3.20.0)
project(Vortex LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17)

include(CMakeDependentOption)

#-------------------------------------------------------------------------------
# Options and definitions
#-------------------------------------------------------------------------------

option(VORTEX_BUILD_EMBEDDED "Build Vortex as part of another project" ON)
option(VORTEX_TOSA_USE_EIGEN "Enables use of Eigen library for some TOSA Ops." OFF)
option(VORTEX_INCLUDE_TESTS "Generate build targets for the MLIR Vortex unit tests." OFF)
cmake_dependent_option(VORTEX_TOSA_TEST_EIGEN "Enables testing of Eigen library for some TOSA Ops." ON "VORTEX_INCLUDE_TESTS;VORTEX_TOSA_USE_EIGEN" OFF)

include(CMake)